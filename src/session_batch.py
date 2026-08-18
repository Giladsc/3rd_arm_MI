#%%
"""
Leave-one-recording-out decoding: does a model generalise to an unseen recording?

The third peer of ``windowed_batch`` and ``full_epoch_batch``. Those two split over
TRIALS, blind to which recording each trial came from - so trials from every
recording sit in both the training and the test folds. This one holds out a whole
recording.

    Stage 1  run_session_batch()      unattended, every subject x split x mode x definition
             -> Analysis/<label>/Cache/<subj>_<variant>-epo.fif   (shared)
             -> Analysis/<label>/Session/Metrics/Individuals/<split>/<def>/<mode>/*.json
             -> Analysis/<label>/Session/Figures/<subj>/<split>/<def>/<mode>/*.png
    Stage 2  summarize_session()      group stats, group figures, the summary table
             -> Analysis/<label>/Session/Metrics/Group/*.json
             -> Analysis/<label>/Session/session_summary.csv  and  .json

WHY THIS EXISTS
---------------
Two things the trial-level analyses cannot answer.

1. CROSS-SESSION GENERALISATION. The live system runs on a session it has never
   seen. Splitting over trials measures nothing about that, because every
   recording is represented in the training set.

2. THE CENTERING QUESTION, STRUCTURALLY. With params_dict['CenterByClass'] on,
   EEG_Preprocessing centres per file over every trial in that file (this is what
   the 'centered' epoch variant is). Under a trial-level split those files' trials are
   spread across folds, so the class mean subtracted from a test trial was
   computed including that trial - the pooled-centering leak. Hold out a whole
   recording and the centered-test condition is not merely leaky, it is
   IMPOSSIBLE: you cannot subtract a class mean from an unseen recording without
   knowing its labels. So under the 'cross' split, mode 'c2u' is leak-free by
   construction, not by mitigation - the training recordings' class means are
   estimated inside files that contain no test trials at all.

THE TWO SPLITS
--------------
    'cross'   LeaveOneGroupOut grouped by recording: train on every recording but
              one, test on the held-out one. n_folds = n_recordings (3-5 here).
    'within'  StratifiedKFold(n_splits=n_recordings) over trials, recording-blind.

'within' is the control. Leave-one-recording-out trains on 2/3 to 4/5 of the data,
so a drop against the existing 5-fold numbers would partly be less training data
rather than session shift. Matching the fold count matches the training fraction,
which makes the difference interpretable:

    generalisation_gap = within - cross

That is the number this module exists to produce.

THE MODES
---------
Named <train>2<test>, where c = per-file class-mean centering (what EEG_Preprocessing
produces with CenterByClass=True, i.e. the 'centered' epoch variant) and u = not
centered. The same letters mean the same recipes in windowed_batch and
full_epoch_batch, so c2c/c2u refer to the same thing in all three.

    mode   train             test         meaning
    c2c    per-file          per-file     sizes the leak; NOT achievable live
    c2u    per-file          uncentered   REPORTABLE - leak-free by construction under 'cross'
    u2u    uncentered        uncentered   no-centering reference

CAVEATS (also written into session_summary.json)
------------------------------------------------
- The unit is one XDF RECORDING, not necessarily one session - several may be
  same-day runs. This is therefore a LOWER BOUND on true cross-session difficulty.
- 3-5 folds makes the per-subject estimate noisy; lead with between-subject spread.
- The held-out recording can have a different class balance from the training set,
  so macro F1 is reported beside accuracy.
- c2c exists only to size the leak and must never be quoted as achievable.

NOTE: importing this module imports ``src.preprocessing``, which runs
``%matplotlib qt`` at import time, so an IPython kernel is required. Use
``notebooks/Session_Analysis.ipynb``.
"""

import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from .analysis_common import (BALANCE_TRIGGER_CLASS, DESIRED_EVENTS, EPOCH_VARIANTS,
                              RANDOM_SEED, _assert_variants_aligned, _balance_indices,
                              _batch_backend, _capture_figures, _class_counts,
                              _merge_group_results, _run_provenance, _savefig,
                              assert_caveat_modes, load_or_build_epochs, subject_files,
                              subject_params)
from .analysis_common import project_paths as _common_paths
from .evaluation import plot_accuracy_group_full_epoch
from .full_epoch_batch import DEFINITIONS, _sum_fold_cms, full_epoch_params
from .group_analysis_full_epoch import (add_subject_to_group_full_epoch,
                                        load_group_results_full_epoch,
                                        save_group_results_full_epoch)
from .training import run_full_epoch_classification_cv
# re-exported so `from src.session_batch import *` gives the notebook these too
# re-exported so `from src.session_batch import *` reaches the parameters cell
from .analysis_common import (ELECTRODE_GROUP_NAMES, ELECTRODE_GROUPS,  # noqa: F401
                              PARAM_GROUPS, check_params, default_params,
                              describe_params, list_subjects, run_label)

#%%
# ============================================================
# Configuration
# ============================================================

FULL_EPOCH_WINDOW = (0.0, 5.0)      # matches full_epoch_batch; fixed by the paradigm

# {split: description}. 'cross' is the point; 'within' is its matched control.
SPLITS = {
    'cross': 'LeaveOneGroupOut by recording - train on all but one, test on it',
    'within': 'StratifiedKFold over trials, recording-blind, same fold count',
}

# The evaluation matrix: {mode: (train recipe, test recipe)}.
#
# Naming: <train>2<test>, where the letter is the centering recipe each side used -
# the same letters as windowed_batch and full_epoch_batch:
#
#     c   class means computed per XDF FILE - what EEG_Preprocessing produces with
#         CenterByClass=True (the 'centered' epoch variant), so it also removes
#         between-session drift. The recipe name behind the letter is 'file'.
#     u   not centered (the 'uncentered' variant)
#
# full_epoch_batch adds g = global pooled and l = fold-local, which need class means
# re-estimated on a chosen trial subset. This module does not implement those: it
# pairs whole epoch variants, so _variant_for below rejects them rather than
# quietly substituting one.
MODES = {
    'c2c': ('file', 'file'),    # sizes the leak; not achievable live
    'c2u': ('file', None),      # reportable: leak-free by construction under 'cross'
    'u2u': (None, None),        # no-centering reference
}

# {recipe: epoch variant}. MODES names recipes, the epoch cache is keyed on variants,
# and this is the only place the two vocabularies meet.
_RECIPE_VARIANTS = {'file': 'centered', None: 'uncentered'}


def _variant_for(recipe):
    """
    The epoch variant a centering recipe corresponds to.

    Raises on 'global'/'fold': full_epoch_batch implements those by re-estimating
    class means on a chosen trial subset, which this module has no machinery for.
    Falling back to a variant would silently answer a different question.
    """
    if recipe not in _RECIPE_VARIANTS:
        raise ValueError(
            f"recipe {recipe!r} has no epoch variant in this module - {list(MODES)} "
            f"pair whole variants, so only {list(_RECIPE_VARIANTS)} are supported. "
            f"'global'/'fold' live in full_epoch_batch.")
    return _RECIPE_VARIANTS[recipe]

# The headline, plus the two centering comparisons at matched training size.
SPLIT_DELTAS = {'generalisation_gap': ('within', 'cross')}
MODE_DELTAS = {
    'centering_benefit': ('c2u', 'u2u'),
    'leak_size': ('c2c', 'c2u'),
}

SUMMARY_CSV_NAME = 'session_summary.csv'
SUMMARY_JSON_NAME = 'session_summary.json'
GROUP_RESULTS_NAME = 'group_session_{split}_{definition}_{mode}.json'

CAVEATS = [
    "The held-out unit is one XDF RECORDING, not necessarily one session - several may be "
    "same-day runs. These numbers are therefore a LOWER BOUND on true cross-session difficulty.",
    "Under the 'cross' split, mode c2u is leak-free BY CONSTRUCTION: the training recordings' "
    "class means are estimated inside files that contain no test trials, so the per-file "
    "centering cannot see the held-out recording. This is the only configuration here that a "
    "live system could reproduce.",
    "c2c centres the held-out recording by its own class means, which requires knowing its "
    "labels. It is reported only to size the leak and must never be quoted as achievable.",
    "'within' is a matched control, not a result: StratifiedKFold with the same fold count as "
    "'cross' so the training fraction matches. generalisation_gap = within - cross is therefore "
    "the cost of an unseen recording with training-set size held constant.",
    "n_folds equals the subject's recording count (3-5 here), so per-subject estimates are "
    "noisy. Lead with the between-subject spread in the GROUP row.",
    "The held-out recording may have a different class balance from the training set, so macro "
    "F1 is reported beside accuracy.",
    f"Evaluation window {FULL_EPOCH_WINDOW}s, fixed a priori by the paradigm (cue at 0, trial "
    "ends at 5), not chosen because accuracy peaked there.",
]

# The caveats ship inside session_summary.json, so a mode named there but not defined
# here would be a wrong claim travelling with the numbers.
assert_caveat_modes(CAVEATS, MODES, 'session_batch')


def project_paths(root=None, label=None, params_dict=None):
    """This analysis's directories: ``Analysis/<label>/Session/`` plus the shared cache."""
    return _common_paths(root, label, params_dict, analysis='Session')


def _out_dir(parent, *parts):
    out = parent
    for part in parts:
        out = out / part
    out.mkdir(parents=True, exist_ok=True)
    return out


#%%
# ============================================================
# Splits
# ============================================================


def recording_groups(epochs, subject=''):
    """
    Per-trial recording labels, from the provenance attached at epoch-build time.

    File boundaries are NOT recoverable from concatenated epochs afterwards - event
    gaps reflect within-run block breaks, not files - so this is the only source of
    truth. A cache built before provenance existed is rebuilt automatically by
    ``load_or_build_epochs``; if that somehow did not happen, fail loudly rather
    than silently degrade to a trial-level split.
    """
    if epochs.metadata is None or 'recording' not in epochs.metadata:
        raise RuntimeError(
            f"[{subject}] epochs carry no 'recording' provenance, so a recording "
            f"cannot be held out. Delete the subject's cache under Analysis/<label>/"
            f"Cache/ and re-run to rebuild it.")
    return epochs.metadata['recording'].to_numpy()


def make_split(split, epochs, events, subject=''):
    """
    Fold list for one split scheme. Returns ``(folds, groups)``.

    Both schemes use the same number of folds - the subject's recording count - so
    their training-set sizes match and the difference between them isolates session
    shift rather than data volume.
    """
    groups = recording_groups(epochs, subject)
    n_folds = len(np.unique(groups))
    y = events[:, 2]
    X = np.zeros((len(y), 1))
    if split == 'cross':
        folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    elif split == 'within':
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=RANDOM_SEED).split(X, y))
    else:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    return folds, groups


def assert_recording_held_out(folds, groups, subject=''):
    """
    Every 'cross' fold must test exactly one recording, absent from its training set.

    This is the property the whole module rests on: if a test recording also
    contributed training trials, neither the generalisation claim nor the
    leak-freedom of c2u would hold.
    """
    for i, (train_idx, test_idx) in enumerate(folds):
        test_recordings = set(groups[test_idx])
        train_recordings = set(groups[train_idx])
        if len(test_recordings) != 1:
            raise AssertionError(
                f"[{subject}] fold {i} tests {len(test_recordings)} recordings, not 1")
        if test_recordings & train_recordings:
            raise AssertionError(
                f"[{subject}] fold {i}: {test_recordings & train_recordings} appears in "
                f"both train and test")
    return True


#%%
# ============================================================
# Per-subject worker
# ============================================================


def _prepare_subject(subject, params_dict=None, paths=None, force=False,
                     seed=RANDOM_SEED, balance_trigger_class=BALANCE_TRIGGER_CLASS):
    """Load both epoch variants, assert alignment, balance once with a shared selection."""
    paths = paths or project_paths()
    np.random.seed(seed)
    random.seed(seed)

    epochs_by_variant, params_by_variant, meta = {}, {}, None
    for variant in EPOCH_VARIANTS:
        params = subject_params(subject, variant, params_dict, paths)
        epochs_by_variant[variant], meta = load_or_build_epochs(
            subject, variant, params, paths, force)
        params_by_variant[variant] = params
    _assert_variants_aligned(subject, epochs_by_variant)

    reference = epochs_by_variant[EPOCH_VARIANTS[0]]
    counts_before, n_before = _class_counts(reference), len(reference)
    keep, balanced = _balance_indices(reference, balance_trigger_class)
    if keep is not None:
        epochs_by_variant = {v: e[keep] for v, e in epochs_by_variant.items()}
        reference = epochs_by_variant[EPOCH_VARIANTS[0]]
        print(f"  [{subject}] balanced {counts_before} -> {_class_counts(reference)}")

    events = reference.events
    for params in params_by_variant.values():
        params['events_trigger_dict'] = {k: v for k, v in reference.event_id.items()
                                         if k in params['desired_events']}
        params['desired_events'] = list(params['events_trigger_dict'])
    base = params_by_variant[EPOCH_VARIANTS[0]]
    assert set(base['events_trigger_dict'].values()) == set(np.unique(events[:, 2])), (
        f"[{subject}] label space mismatch")

    groups = recording_groups(reference, subject)
    counts_after = _class_counts(reference)
    shared = {
        'subject': subject,
        'classes': list(base['desired_events']),
        'n_classes': len(base['desired_events']),
        'n_channels': len(reference.ch_names),
        'n_files': len(meta['files']),
        'n_recordings': int(len(np.unique(groups))),
        'trials_per_recording': {str(r): int((groups == r).sum())
                                 for r in np.unique(groups)},
        'n_epochs': len(reference),
        'n_epochs_before': n_before,
        'counts_after': counts_after,
        'balanced': balanced,
        'majority_baseline': (max(counts_after.values()) / sum(counts_after.values())
                              if counts_after else None),
    }
    return epochs_by_variant, events, params_by_variant, shared


def run_subject_session(subject, splits=None, modes=None, definitions=None,
                        params_dict=None, paths=None, force=False, save_figs=True,
                        seed=RANDOM_SEED, balance_trigger_class=BALANCE_TRIGGER_CLASS):
    """
    Every split x definition x mode cell for one subject, all off the same epochs.

    Returns ``{(split, definition, mode): result dict}``.
    """
    paths = paths or project_paths()
    splits = list(splits or SPLITS)
    modes = list(modes or MODES)
    definitions = list(definitions or DEFINITIONS)
    started = time.perf_counter()

    epochs_by_variant, events, params_by_variant, shared = _prepare_subject(
        subject, params_dict, paths, force, seed, balance_trigger_class)
    print(f"  [{subject}] {shared['n_recordings']} recording(s), "
          f"{shared['trials_per_recording']}")

    results = {}
    for definition in definitions:
        params = full_epoch_params(definition, params_by_variant[EPOCH_VARIANTS[0]])
        cropped_by_variant = {
            v: e.copy().crop(params['classifier_window_s'], params['classifier_window_e'])
            for v, e in epochs_by_variant.items()}

        for split in splits:
            folds, groups = make_split(split, epochs_by_variant[EPOCH_VARIANTS[0]],
                                       events, subject)
            if split == 'cross':
                assert_recording_held_out(folds, groups, subject)

            for mode in modes:
                train_recipe, test_recipe = MODES[mode]
                train_variant = _variant_for(train_recipe)
                test_variant = _variant_for(test_recipe)
                started_cell = time.perf_counter()
                accs, cms = run_full_epoch_classification_cv(
                    epochs_by_variant[test_variant], cropped_by_variant[train_variant],
                    list(folds), params,
                    tmin=FULL_EPOCH_WINDOW[0], tmax=FULL_EPOCH_WINDOW[1])
                entry = dict(
                    shared, split=split, definition=definition, mode=mode,
                    fold_accuracies=accs, fold_confusion_matrices=cms, params=params,
                    n_folds=len(folds),
                    train_sizes=[int(len(tr)) for tr, _ in folds],
                    test_sizes=[int(len(te)) for _, te in folds],
                    runtime_s=round(time.perf_counter() - started_cell, 1))
                results[(split, definition, mode)] = entry
                print(f"  [{subject}/{split}/{definition}/{mode}] "
                      f"acc {np.mean(accs):.3f} over {len(folds)} fold(s) "
                      f"({entry['runtime_s']}s)")
                if save_figs:
                    save_subject_figures(subject, split, definition, mode, accs, cms, paths)

    print(f"  [{subject}] {len(results)} cell(s) in {time.perf_counter() - started:.0f}s")
    return results


def save_subject_figures(subject, split, definition, mode, fold_accuracies,
                         fold_confusion_matrices, paths=None):
    """Per-subject confusion matrix, pooled over folds."""
    paths = paths or project_paths()
    out_dir = _out_dir(paths.figures / subject, split, definition, mode)
    summed = _sum_fold_cms(fold_confusion_matrices)
    if summed is None:
        return
    cm, classes = summed
    fig, ax = plt.subplots(figsize=(7, 6))
    normed = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(normed, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{normed[i, j]:.2f}', ha='center', va='center',
                    color='white' if normed[i, j] > 0.5 else 'black')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'{subject} | {split} | {definition} | {mode}\n'
                 f'acc {np.mean(fold_accuracies):.3f} over {len(fold_accuracies)} folds')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _savefig(fig, out_dir / 'confusion.png', paths)
    plt.close(fig)


#%%
# ============================================================
# Summary
# ============================================================


def _entry_metrics(entry):
    """Accuracy, macro F1 and balanced accuracy for one cell."""
    accs = np.asarray(entry['fold_accuracies'], dtype=float)
    out = {'acc_mean': float(accs.mean()),
           'acc_sd': float(accs.std(ddof=1)) if len(accs) > 1 else float('nan'),
           'acc_fold_min': float(accs.min()), 'acc_fold_max': float(accs.max()),
           'n_folds': int(len(accs))}
    summed = _sum_fold_cms(entry['fold_confusion_matrices'])
    if summed is not None:
        cm, _ = summed
        with np.errstate(invalid='ignore', divide='ignore'):
            precision = np.diag(cm) / cm.sum(axis=0)
            recall = np.diag(cm) / cm.sum(axis=1)
            f1 = 2 * precision * recall / (precision + recall)
        out['f1_macro'] = float(np.nanmean(np.where(np.isnan(f1), 0.0, f1)))
        out['bal_acc'] = float(np.nanmean(recall))
    return out


def _add_deltas(row, definition):
    """generalisation_gap per (definition, mode), and the centering deltas per split."""
    for name, (left, right) in SPLIT_DELTAS.items():
        for mode in MODES:
            a = row.get(f'acc_mean_{left}_{definition}_{mode}')
            b = row.get(f'acc_mean_{right}_{definition}_{mode}')
            if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)):
                row[f'delta_{name}_{definition}_{mode}'] = a - b
    for name, (left, right) in MODE_DELTAS.items():
        for split in SPLITS:
            a = row.get(f'acc_mean_{split}_{definition}_{left}')
            b = row.get(f'acc_mean_{split}_{definition}_{right}')
            if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)):
                row[f'delta_{name}_{split}_{definition}'] = a - b
    return row


def build_summary(results_by_cell, subject_meta, failures=None, params_dict=None,
                  paths=None, write=True):
    """One row per subject plus a GROUP row. Returns ``(DataFrame, payload)``."""
    paths = paths or project_paths()
    failures = failures or {}
    from_results = {s for entries in results_by_cell.values() for s in entries}
    subjects = sorted(set(subject_meta) | from_results
                      | {k.split('/')[0] for k in failures})

    rows = []
    for subject in subjects:
        meta = subject_meta.get(subject, {})
        subj_failures = {k.split('/', 1)[1]: v for k, v in failures.items()
                         if k.split('/')[0] == subject}
        row = {'subject': subject,
               'status': 'failed' if subj_failures and not meta else (
                   'partial' if subj_failures else 'ok'),
               'error': '; '.join(f'{k}: {v}' for k, v in sorted(subj_failures.items())),
               'n_recordings': meta.get('n_recordings'),
               'n_epochs': meta.get('n_epochs'),
               'n_channels': meta.get('n_channels'),
               'classes': '|'.join(meta.get('classes') or []),
               'n_classes': meta.get('n_classes'),
               'chance': 1.0 / meta['n_classes'] if meta.get('n_classes') else None,
               'majority_baseline': meta.get('majority_baseline')}
        for (split, definition, mode), entries in results_by_cell.items():
            entry = entries.get(subject)
            if entry is None:
                continue
            for key, value in _entry_metrics(entry).items():
                row[f'{key}_{split}_{definition}_{mode}'] = value
            row[f'train_sizes_{split}'] = '|'.join(map(str, entry.get('train_sizes') or []))
            row[f'test_sizes_{split}'] = '|'.join(map(str, entry.get('test_sizes') or []))
        for definition in DEFINITIONS:
            _add_deltas(row, definition)
        rows.append(row)

    group_row = {'subject': 'GROUP', 'status': 'ok', 'error': ''}
    group_stats = {}
    for (split, definition, mode), entries in results_by_cell.items():
        vals = np.array([_entry_metrics(e)['acc_mean'] for e in entries.values()],
                        dtype=float)
        f1s = np.array([_entry_metrics(e).get('f1_macro', np.nan)
                        for e in entries.values()], dtype=float)
        if not len(vals):
            continue
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
        sem = sd / np.sqrt(len(vals)) if len(vals) > 1 else float('nan')
        key = f'{split}_{definition}_{mode}'
        group_row[f'acc_mean_{key}'] = float(vals.mean())
        group_row[f'acc_between_sd_{key}'] = sd
        group_row[f'f1_macro_{key}'] = float(np.nanmean(f1s))
        group_stats[f'{split}/{definition}/{mode}'] = {
            'n_subjects': int(len(vals)), 'mean': float(vals.mean()), 'between_sd': sd,
            'sem': sem, 'ci95': 1.96 * sem if len(vals) > 1 else float('nan'),
            'min': float(vals.min()), 'max': float(vals.max()),
            'f1_macro': float(np.nanmean(f1s))}
    for definition in DEFINITIONS:
        _add_deltas(group_row, definition)
    group_row['n_classes'] = next((r.get('n_classes') for r in rows if r.get('n_classes')),
                                  None)
    group_row['chance'] = (1.0 / group_row['n_classes']
                           if group_row.get('n_classes') else None)
    rows.append(group_row)

    df = pd.DataFrame(rows)
    payload = {'run': _run_provenance(params_dict, subjects, paths,
                                      splits=dict(SPLITS),
                                      modes={m: list(v) for m, v in MODES.items()},
                                      full_epoch_window=list(FULL_EPOCH_WINDOW)),
               'group': group_stats, 'per_subject': rows[:-1],
               'failures': failures, 'caveats': CAVEATS}
    if write:
        csv_path = paths.out / SUMMARY_CSV_NAME
        json_path = paths.out / SUMMARY_JSON_NAME
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(payload, indent=1, default=str))
        print(f"  saved {csv_path.relative_to(paths.root)}")
        print(f"  saved {json_path.relative_to(paths.root)}")
    _print_summary(df, group_stats)
    return df, payload


def _print_summary(df, group_stats):
    """Leads with the generalisation gap, because that is what the module is for."""
    subjects = df[df.subject != 'GROUP']
    print(f"\n{'=' * 78}\nLeave-one-recording-out, window {FULL_EPOCH_WINDOW}s\n{'=' * 78}")

    for definition in DEFINITIONS:
        cols = [f'acc_mean_{s}_{definition}_{m}' for s in SPLITS for m in MODES
                if f'acc_mean_{s}_{definition}_{m}' in df.columns]
        if not cols:
            continue
        print(f"\n--- definition '{definition}' ---")
        with pd.option_context('display.width', 250, 'display.max_columns', 60):
            print(df[['subject'] + cols].to_string(
                index=False, float_format=lambda v: f'{v:.3f}'))

    print(f"\n{'=' * 78}\nGENERALISATION GAP (within - cross, matched training size)\n{'=' * 78}")
    for definition in DEFINITIONS:
        for mode in MODES:
            col = f'delta_generalisation_gap_{definition}_{mode}'
            if col not in subjects.columns:
                continue
            d = subjects[col].dropna()
            if not len(d):
                continue
            flag = '  <== ' if mode == 'c2u' else '      '
            print(f"{flag}{definition:<7} {mode:<5}: {d.mean():+.4f} +/- {d.std(ddof=1):.4f}"
                  f"   {int((d > 0).sum())}/{len(d)} positive")

    print(f"\n{'=' * 78}\nGroup means (between-subject spread)\n{'=' * 78}")
    for key, st in group_stats.items():
        star = '  *' if key.endswith('c2u') and key.startswith('cross') else '   '
        print(f"{star}{key:<26} {st['mean']:.3f} +/- {st['between_sd']:.3f}  "
              f"(SEM {st['sem']:.3f}, F1 {st['f1_macro']:.3f}, n={st['n_subjects']})")
    print("\n  * cross/*/c2u is the leak-free, deployable configuration")


def save_group_figures(entries, split, definition, mode, n_classes, paths=None):
    """Group accuracy bar chart for one cell; returns the group_results list."""
    paths = paths or project_paths()
    out_dir = _out_dir(paths.group_figures, split, definition, mode)
    group_results = []
    for subject, entry in entries.items():
        add_subject_to_group_full_epoch(
            group_results, subject, entry['fold_accuracies'],
            entry['fold_confusion_matrices'], tmin=FULL_EPOCH_WINDOW[0],
            tmax=FULL_EPOCH_WINDOW[1], params_dict=entry.get('params') or None)
    _capture_figures(plot_accuracy_group_full_epoch, out_dir / 'accuracy_group.png',
                     paths, group_results_full_epoch=group_results,
                     n_classes=n_classes, figsize=(10, 5))
    return group_results


def summarize_session(results_by_cell=None, subject_meta=None, failures=None,
                      params_dict=None, paths=None, save_figs=True, write=True,
                      label=None):
    """Group statistics, group figures and the summary table."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    if results_by_cell is None:
        results_by_cell, subject_meta = load_session_results(paths)
    if not any(results_by_cell.values()):
        print("No session results found. Run run_session_batch() first.")
        return None, None

    # merge + persist ALWAYS, never behind save_figs
    for (split, definition, mode), entries in results_by_cell.items():
        if not entries:
            continue
        name = GROUP_RESULTS_NAME.format(split=split, definition=definition, mode=mode)
        group_results = []
        for subject, entry in entries.items():
            add_subject_to_group_full_epoch(
                group_results, subject, entry['fold_accuracies'],
                entry['fold_confusion_matrices'], tmin=FULL_EPOCH_WINDOW[0],
                tmax=FULL_EPOCH_WINDOW[1], params_dict=entry.get('params') or None)
        merged = _merge_group_results(group_results, paths.metrics_group / name,
                                      load_group_results_full_epoch)
        if write:
            save_group_results_full_epoch(merged, paths.metrics_group, filename=name)
        for e in merged:
            subject = e.get('subject_name')
            if subject and subject not in entries:
                entries[subject] = {
                    'fold_accuracies': e['fold_accuracies'],
                    'fold_confusion_matrices': e['fold_confusion_matrices'],
                    'params': {}, 'split': split, 'definition': definition, 'mode': mode,
                    'classes': list(e.get('desired_events') or []),
                    'n_classes': len(e.get('desired_events') or []) or None}

    if save_figs:
        with _batch_backend():
            for (split, definition, mode), entries in results_by_cell.items():
                if not entries:
                    continue
                n_classes = (next(iter(entries.values())).get('n_classes')
                             or len(DESIRED_EVENTS))
                save_group_figures(entries, split, definition, mode, n_classes, paths)

    return build_summary(results_by_cell, subject_meta or {}, failures, params_dict,
                         paths, write)


def load_session_results(paths=None, label=None, params_dict=None):
    """Reload saved group results, keyed by (split, definition, mode)."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    results_by_cell, subject_meta = {}, {}
    for split in SPLITS:
        for definition in DEFINITIONS:
            for mode in MODES:
                path = paths.metrics_group / GROUP_RESULTS_NAME.format(
                    split=split, definition=definition, mode=mode)
                entries = {}
                if path.exists():
                    for entry in load_group_results_full_epoch(path):
                        subject = entry['subject_name']
                        classes = list(entry.get('desired_events') or [])
                        entries[subject] = {
                            'fold_accuracies': entry['fold_accuracies'],
                            'fold_confusion_matrices': entry['fold_confusion_matrices'],
                            'params': {}, 'classes': classes,
                            'n_classes': len(classes) or None}
                        subject_meta.setdefault(subject, {}).setdefault(
                            'n_classes', len(classes) or None)
                results_by_cell[(split, definition, mode)] = entries
    return results_by_cell, subject_meta


#%%
# ============================================================
# Status and batch driver
# ============================================================


def session_status(paths=None, verbose=True, label=None, params_dict=None):
    """Per-subject progress across the split x definition x mode grid."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    rows = []
    for subject in list_subjects(paths):
        row = {'subject': subject, 'n_files': len(subject_files(subject, paths))}
        for split in SPLITS:
            for definition in DEFINITIONS:
                d = paths.metrics_individual / split / definition
                row[f'{split[:5]}_{definition[:4]}'] = bool(
                    list(d.rglob(f'{subject}_*.json'))) if d.exists() else False
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        print(f"Session root: {paths.out}   (label '{paths.label}')")
        print(df.to_string(index=False))
    return df


def run_session_batch(subjects=None, splits=None, modes=None, definitions=None,
                      force=False, save_figs=True, batch_backend='Agg',
                      params_dict=None, paths=None, seed=RANDOM_SEED,
                      balance_trigger_class=BALANCE_TRIGGER_CLASS, label=None):
    """
    Leave-one-recording-out for every subject, then summarise.

    A failing subject is reported and skipped, never fatal. The whole grid is
    computed per subject in one pass so every cell shares the same epochs, the same
    balancing selection and the same folds - which is what makes the deltas paired.
    """
    paths = paths or project_paths(label=label, params_dict=params_dict)
    print(f"output label: '{paths.label}'  ->  {paths.out.relative_to(paths.root)}")
    subjects = (list_subjects(paths) if subjects is None
                else [subjects] if isinstance(subjects, str) else list(subjects))
    splits = list(splits or SPLITS)
    modes = list(modes or MODES)
    definitions = list(definitions or DEFINITIONS)

    results_by_cell = {(s, d, m): {} for s in splits for d in definitions for m in modes}
    subject_meta, failures = {}, {}

    with _batch_backend(batch_backend):
        for i, subject in enumerate(subjects, 1):
            print(f"\n{'=' * 70}\n[{i}/{len(subjects)}] {subject}\n{'=' * 70}")
            try:
                results = run_subject_session(
                    subject, splits, modes, definitions, params_dict, paths, force,
                    save_figs, seed, balance_trigger_class)
                for key, entry in results.items():
                    results_by_cell[key][subject] = entry
                    split, definition, mode = key
                    add_subject_to_group_full_epoch(
                        [], subject, entry['fold_accuracies'],
                        entry['fold_confusion_matrices'],
                        tmin=FULL_EPOCH_WINDOW[0], tmax=FULL_EPOCH_WINDOW[1],
                        params_dict=entry['params'],
                        save_dir=_out_dir(paths.metrics_individual, split, definition, mode))
                    subject_meta.setdefault(subject, {}).update(
                        {k: entry[k] for k in
                         ('n_files', 'n_recordings', 'trials_per_recording', 'n_epochs',
                          'n_epochs_before', 'n_channels', 'classes', 'n_classes',
                          'counts_after', 'balanced', 'majority_baseline')})
            except Exception as err:
                for s in splits:
                    for d in definitions:
                        for m in modes:
                            failures[f'{subject}/{s}/{d}/{m}'] = f'{type(err).__name__}: {err}'
                print(f"  !! FAILED [{subject}]: {type(err).__name__}: {err}")
            finally:
                plt.close('all')

    if failures:
        rerun = sorted({k.split('/')[0] for k in failures})
        print(f"\nRerun the failures with run_session_batch({rerun}).")

    df, payload = summarize_session(results_by_cell, subject_meta, failures,
                                    params_dict, paths, save_figs)
    return {'results_by_cell': results_by_cell, 'subject_meta': subject_meta,
            'failures': failures, 'summary': df, 'payload': payload}
