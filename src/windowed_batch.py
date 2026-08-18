#%%
"""
Windowed decoding analysis: accuracy as a function of time, all subjects.

The counterpart of ``full_epoch_batch``. Where that one asks "given the whole
trial, how well can it be classified" - one number per subject, what a manuscript
reports - this asks **when** the information is there, by sliding a short window
across the epoch and cross-validating at every position.

    Stage 1  run_windowed_batch()    unattended, every subject x every mode
             -> Analysis/<label>/Cache/<subj>_<variant>-epo.fif (+ _meta.json)
             -> Analysis/<label>/Windowed/Metrics/Individuals/<mode>/*.json
             -> Analysis/<label>/Windowed/Figures/<subj>/<mode>/*.png
    Stage 2  summarize_windowed()    group stats, group figures, the summary table
             -> Analysis/<label>/Windowed/Metrics/Group/group_windowed_<mode>.json
             -> Analysis/<label>/Windowed/Figures/Group/<mode>/*.png
             -> Analysis/<label>/Windowed/windowed_summary.csv  and  .json

``<label>`` identifies the run's CLASS SET (e.g. 'MH_LH_RH_FR', 'LH_RH'), so
changing ``desired_events`` cannot overwrite a previous run, reuse its cached
epochs, or have skip_done hand back its results. The epoch cache is shared with
``full_epoch_batch`` - same epochs, no duplication. Everything common to the two
analyses lives in ``analysis_common``.

WHAT THIS MEASURES
------------------
Class-mean centering is LABEL-DEPENDENT: a trial's class must be known to centre
it, so it cannot be applied to an unlabelled trial and the live loop does not apply
it. So every subject is evaluated over a train/test centering matrix:

                    test centered      test uncentered
    train centered      c2c                 c2u     <- the deployment question
    train uncentered    u2c                 u2u

All four use the SAME loaded epochs, the SAME balancing selection and the SAME CV
folds, so the delta_* columns are exactly paired per subject and differ only in
which copy the training and test rows were drawn from.

    delta_cost_of_centering_at_inference = c2c - c2u
    delta_centered_vs_never_centered     = c2u - u2u

Caveats travel with every result via windowed_summary.json['caveats'] so they
cannot be separated from the numbers; see CAVEATS below.

NOTE: importing this module imports ``src.preprocessing``, which runs
``%matplotlib qt`` at import time, so an IPython kernel is required. Use
``notebooks/Windowed_Analysis.ipynb``.
"""

import io
import json
import random
import time
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_common import (BALANCE_TRIGGER_CLASS, CV_N_REPEATS, CV_N_SPLITS,
                              DESIRED_EVENTS, EPOCH_VARIANTS, RANDOM_SEED,
                              _assert_variants_aligned, _balance_indices,
                              _balanced_accuracy_from_cms, _batch_backend,
                              _cache_paths, _capture_figures, _class_counts,
                              _f1_macro_from_cms, _get_classes_initials, _make_cv_split,
                              assert_caveat_modes,
                              _merge_group_results, _meta_from_cache, _run_provenance,
                              _variant_dir, default_params, list_subjects,
                              load_or_build_epochs, subject_figure_dir,
                              subject_files, subject_params)
# re-exported so `from src.windowed_batch import *` gives the notebook these too
# re-exported so `from src.windowed_batch import *` reaches the parameters cell
from .analysis_common import (ELECTRODE_GROUP_NAMES, ELECTRODE_GROUPS,  # noqa: F401
                              PARAM_GROUPS, check_params, describe_params, run_label)
from .analysis_common import project_paths as _common_paths
from .evaluation import (plot_accuracy_over_time, plot_accuracy_over_time_group,
                         plot_average_confusion_fixed_cv,
                         plot_average_confusion_fixed_cv_group)
from .group_analysis import (add_subject_to_group, descriptive_stats_group,
                             load_group_results, save_group_results)
from .preprocessing import Split_training_validation, crop_the_data
from .training import run_windowed_classification_aug_cv


def project_paths(root=None, label=None, params_dict=None):
    """This analysis's directories: ``Analysis/<label>/Windowed/`` plus the shared cache."""
    return _common_paths(root, label, params_dict, analysis='Windowed')


#%%
# ============================================================
# Configuration specific to the windowed analysis
# ============================================================

# The evaluation matrix: {mode: (train recipe, test recipe)}.
#
# Naming: <train>2<test>, where the letter is the centering recipe each side used -
# the same letters as full_epoch_batch and session_batch:
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
#
# Centering is params_dict['CenterByClass'], applied inside EEG_Preprocessing once per
# recording - which is exactly what the two variants are: subject_params sets the flag
# from the variant name. So a cross cell is not built by centering part of one set; it
# takes training rows from one complete variant and test rows from the other. The two
# variants are trial-aligned (same events, same order), which
# run_windowed_classification_aug_cv requires and this module asserts.
#
# c2u is the deployment question: the live loop serves UNCENTERED data, because
# class-conditional centering cannot be applied to an unlabelled trial (you would
# have to know its class to pick the mean to subtract).
MODES = {
    'c2c': ('file', 'file'),   # what Main_Experiment measures today
    'u2u': (None, None),       # a pipeline that never centers
    'c2u': ('file', None),     # train centered, serve live  <- the question
    'u2c': (None, 'file'),     # symmetry check
}

# The two comparisons the matrix exists to produce.
#   cost_of_centering_at_inference = c2c - c2u
#       how much of a centered-trained model's measured accuracy evaporates once it
#       is served the uncentered data the live loop actually produces.
#   centered_vs_never_centered     = c2u - u2u
#       whether that centered-trained model, serving uncentered data, still beats a
#       model that never centered at all. Negative means train uncentered to deploy.
MODE_DELTAS = {
    'cost_of_centering_at_inference': ('c2c', 'c2u'),
    'centered_vs_never_centered': ('c2u', 'u2u'),
}

# {recipe: epoch variant}. The modes above name recipes, the epoch cache is keyed on
# variants, and this is the only place the two vocabularies meet.
_RECIPE_VARIANTS = {'file': 'centered', None: 'uncentered'}

# How a recipe is spelled in the printed tables, which talk about epoch variants.
_RECIPE_LABELS = {'file': 'centered', None: 'uncentered'}


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

# RepeatedStratifiedKFold, not StratifiedShuffleSplit: same 10 fits and the same
# 80/20 proportions, but each repeat is a true partition, so every trial is tested
# exactly n_repeats times and none is skipped. ShuffleSplit(10, 0.2) leaves ~11% of


EVAL_WINDOW = (2.0, 5.0)       # headline accuracy window
PRE_CUE_WINDOW = (-2.0, 0.0)   # must be ~chance; the leakage canary
CONFUSION_WINDOWS = ((0.0, 5.0), (2.0, 5.0), (-2.0, 0.0))
PRE_CUE_FLAG_MARGIN = 0.10     # flag when pre-cue accuracy exceeds chance by this


# One file per MODE, not per epoch variant - the filenames on disk are
# group_windowed_c2c.json and friends.
GROUP_RESULTS_NAME = 'group_windowed_{mode}.json'
SUMMARY_CSV_NAME = 'windowed_summary.csv'
SUMMARY_JSON_NAME = 'windowed_summary.json'


CAVEATS = [
    "The four modes are a train/test centering matrix: c2c/u2u are the diagonals, c2u trains "
    "on centered data and tests on uncentered data (the live case, since class-conditional "
    "centering cannot be applied to an unlabelled trial), and u2c is the symmetry check. All "
    "four use identical CV folds, so the delta_* columns are exactly paired per subject.",
    "POOLED CENTERING LIMITATION: the centered copy's class means were computed per XDF file "
    "over every trial of that class in that file - test trials included, which is the leak. "
    "So in c2u the TEST data is clean uncentered signal but the "
    "TRAINING data still carries the label leak. c2u therefore answers 'what happens if you "
    "train the way Main_Experiment does today and then deploy', not 'what happens if you train "
    "centering correctly and then deploy'. The leak-free version would centre inside each fold "
    "using training-fold class means only.",
    "Centering removes the per-class evoked response, so centered results are induced power.",
    "acc_precue_* covers windows entirely before the cue and must be at chance "
    f"(flagged when it exceeds chance by {PRE_CUE_FLAG_MARGIN}). Read it before the headline.",
    "acc_eval_within_sem / _within_ci95 are within-subject and ANTICONSERVATIVE: any two of "
    "the 10 CV training sets share ~78% of their trials, so fold scores are correlated and "
    "std/sqrt(n_folds) understates the variance. Use the GROUP row's between-subject spread.",
    f"CV is RepeatedStratifiedKFold({CV_N_SPLITS}, n_repeats={CV_N_REPEATS}); Main_Experiment "
    "used StratifiedShuffleSplit(10, 0.2), which left ~11% of trials never tested. Per-subject "
    "numbers therefore differ slightly from prior Metrics/ results.",
    f"Balancing applies only when a '{BALANCE_TRIGGER_CLASS}' class is present, in which case "
    "that class is subsampled to the smallest other one. The 4-class MI config has no "
    f"'{BALANCE_TRIGGER_CLASS}' and so runs UNBALANCED by design: raw accuracy sits against a "
    "per-subject majority_baseline rather than 1/n_classes, which is why macro F1 is reported "
    "beside it.",
    "Training uses only the [classifier_window_s, classifier_window_e] crop and the minimum "
    "inter-trial gap exceeds that window, so training segments from different trials never "
    "share samples, even though the full 11 s epochs do overlap at their edges.",
    "Subjects without an entry in get_subject_bad_electrodes keep all channels while others "
    "drop 1-5, so n_channels varies across subjects; see the n_channels / dropped_bads columns.",
    "Subjects lacking one of the desired classes are benchmarked on the classes they have and "
    "excluded from the group cohort (in_group_cohort=False) rather than compared across "
    "different class counts.",
]

# The caveats ship inside windowed_summary.json, so a mode named there but not defined
# here would be a wrong claim travelling with the numbers.
assert_caveat_modes(CAVEATS, MODES, 'windowed_batch')


#%%
# ============================================================
# Figures
# ============================================================

def save_subject_figures(subject, variant, scores_windows, folds_cm, w_times,
                         params_dict, paths=None):
    """Accuracy-over-time plus one confusion matrix per CONFUSION_WINDOWS entry."""
    paths = paths or project_paths()
    out_dir = subject_figure_dir(subject, variant, paths)
    _capture_figures(plot_accuracy_over_time, out_dir / 'accuracy_over_time.png', paths,
                     scores_windows=scores_windows, w_times=w_times,
                     params_dict=params_dict, axes_handle=None)
    for t_start, t_end in CONFUSION_WINDOWS:
        _capture_figures(plot_average_confusion_fixed_cv,
                         out_dir / f'confusion_{t_start}-{t_end}s.png', paths,
                         folds_conf_matrices_per_window=folds_cm, w_times=w_times,
                         t_start=t_start, t_end=t_end, normalize=True)


def save_group_figures(group_results, variant, n_classes, paths=None):
    """Group accuracy-over-time and group confusion matrix for one variant."""
    paths = paths or project_paths()
    out_dir = _variant_dir(paths.group_figures, variant)
    w_times = group_results[0]['w_times']
    _capture_figures(plot_accuracy_over_time_group,
                     out_dir / 'accuracy_over_time_group.png', paths,
                     group_results=group_results, n_classes=n_classes, figsize=(12, 5))
    t_start, t_end = EVAL_WINDOW
    _capture_figures(plot_average_confusion_fixed_cv_group,
                     out_dir / f'confusion_group_{t_start}-{t_end}s.png', paths,
                     group_results=group_results, w_times=w_times,
                     t_start=t_start, t_end=t_end, normalize=True)



#%%
# ============================================================
# Per-subject worker
# ============================================================

def run_subject_windowed(subject, modes=None, params_dict=None, paths=None, force=False,
                          save_figs=True, seed=RANDOM_SEED,
                          balance_trigger_class=BALANCE_TRIGGER_CLASS):
    """
    Cross-validate one subject across the train/test centering matrix.

    Loads both epoch variants once, balances them with a single shared selection,
    and evaluates every mode in MODES off the *same* folds, so the differences
    between modes are exactly paired and attributable only to the centering of the
    training and test sources.

    Returns ``{mode: result dict}`` plus a shared 'meta' entry.
    """
    paths = paths or project_paths()
    modes = list(modes or MODES)
    # balance/Split_training_validation use np.random and random, both unseeded.
    np.random.seed(seed)
    random.seed(seed)
    started = time.perf_counter()

    # --- load both variants -------------------------------------------------
    epochs_by_variant, params_by_variant, meta = {}, {}, None
    for variant in EPOCH_VARIANTS:
        params = subject_params(subject, variant, params_dict, paths)
        epochs_by_variant[variant], meta = load_or_build_epochs(
            subject, variant, params, paths, force)
        params_by_variant[variant] = params
    _assert_variants_aligned(subject, epochs_by_variant)

    reference = epochs_by_variant[EPOCH_VARIANTS[0]]
    counts_before = _class_counts(reference)
    n_epochs_before = len(reference)

    # --- balance ONCE, apply the same selection to both variants -------------
    keep, balanced = _balance_indices(reference, balance_trigger_class)
    if keep is not None:
        epochs_by_variant = {v: e[keep] for v, e in epochs_by_variant.items()}
        reference = epochs_by_variant[EPOCH_VARIANTS[0]]
        print(f"  [{subject}] balanced {counts_before} -> {_class_counts(reference)} "
              f"({n_epochs_before} -> {len(reference)} trials)")
    counts_after = _class_counts(reference)

    # --- label space, from THIS subject's epochs ----------------------------
    # classifier_training inverts events_trigger_dict to build string labels, so a
    # stale one yields a wrong label space rather than an error.
    events = reference.events
    for params in params_by_variant.values():
        params['events_trigger_dict'] = {k: v for k, v in reference.event_id.items()
                                         if k in params['desired_events']}
        params['desired_events'] = list(params['events_trigger_dict'])
    params = params_by_variant[EPOCH_VARIANTS[0]]
    assert set(params['events_trigger_dict'].values()) == set(np.unique(events[:, 2])), (
        f"[{subject}] label space mismatch: events_trigger_dict="
        f"{params['events_trigger_dict']} vs event codes {sorted(set(np.unique(events[:, 2])))}")

    # --- crop both variants --------------------------------------------------
    cropped_by_variant = {}
    for variant, epochs in epochs_by_variant.items():
        train_inds, validation_inds, _ = Split_training_validation(
            epochs, None, params['events_trigger_dict'])
        cropped_by_variant[variant] = crop_the_data(
            epochs.copy(), train_inds, validation_inds,
            params['classifier_window_s'], params['classifier_window_e'],
            use_all_for_training=False)['epochs_cropped']

    # --- evaluate every cell of the matrix on identical folds ----------------
    dropped = sorted(set(params['Electorde_Group']) - set(reference.ch_names))
    shared = {
        'subject': subject,
        'classes': list(params['desired_events']),
        'n_classes': len(params['desired_events']),
        'ch_names': list(reference.ch_names),
        'n_channels': len(reference.ch_names),
        'dropped_bads': dropped,
        'n_files': len(meta['files']),
        'n_epochs': len(reference),
        'n_epochs_before': n_epochs_before,
        'counts_before': counts_before,
        'counts_after': counts_after,
        'balanced': balanced,
    }

    results = {}
    for mode in modes:
        train_recipe, test_recipe = MODES[mode]
        train_variant = _variant_for(train_recipe)
        test_variant = _variant_for(test_recipe)
        mode_started = time.perf_counter()
        # A fresh generator per mode, but identical folds: same labels, same
        # random_state, so the modes are exactly paired.
        cv_split = _make_cv_split(cropped_by_variant[train_variant], events)
        scores_windows, folds_cm, w_times = run_windowed_classification_aug_cv(
            epochs_by_variant[test_variant], cropped_by_variant[train_variant],
            cv_split, params)

        if save_figs:
            save_subject_figures(subject, mode, scores_windows, folds_cm, w_times,
                                 params, paths)

        results[mode] = dict(
            shared, mode=mode, train_variant=train_variant, test_variant=test_variant,
            scores_windows=scores_windows,
            folds_confusion_matrices_per_window=folds_cm, w_times=w_times,
            params=params,
            runtime_s=round(time.perf_counter() - mode_started, 1))
        print(f"  [{subject}/{mode}] train={train_variant} test={test_variant} "
              f"-> {results[mode]['runtime_s']}s")

    print(f"  [{subject}] {len(modes)} mode(s) in "
          f"{round(time.perf_counter() - started, 1)}s")
    return results



#%%
# ============================================================
# Summary
# ============================================================

def _quiet_stats(entries, n_classes, tmin=None, tmax=None):
    """descriptive_stats_group without its printed table."""
    with redirect_stdout(io.StringIO()):
        return descriptive_stats_group(entries, n_classes=n_classes, tmin=tmin, tmax=tmax)



def _add_mode_deltas(row, metrics=('acc_eval_mean', 'f1_macro_eval')):
    """Paired mode differences. Identical folds across modes make these exact."""
    for name, (left, right) in MODE_DELTAS.items():
        for metric in metrics:
            a, b = row.get(f'{metric}_{left}'), row.get(f'{metric}_{right}')
            if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)):
                suffix = '' if metric == 'acc_eval_mean' else '_f1'
                row[f'delta_{name}{suffix}'] = a - b
    return row


def _cohorts(group_results):
    """
    Split entries into the largest comparable cohort and the rest.

    Comparability is (class tuple, n_windows): mixing a 3-class subject into
    4-class group stats would compare different chance levels, and unequal window
    counts break descriptive_stats_group's shared-shape assumption.
    """
    buckets = {}
    for entry in group_results:
        classes = tuple(entry.get('desired_events') or [])
        n_windows = len(np.asarray(entry['scores_windows'], dtype=float)[0])
        buckets.setdefault((classes, n_windows), []).append(entry)
    if not buckets:
        return [], {}
    key = max(buckets, key=lambda k: len(buckets[k]))
    cohort = buckets[key]
    excluded = {}
    for other_key, entries in buckets.items():
        if other_key == key:
            continue
        for entry in entries:
            excluded[entry['subject_name']] = (
                f"classes={list(other_key[0])}, n_windows={other_key[1]} "
                f"(cohort is {list(key[0])}, {key[1]})")
    return cohort, excluded



def _variant_columns(entry, n_classes, chance):
    """The per-variant metric block for one subject."""
    stats_full = _quiet_stats([entry], n_classes)['per_subject'][0]
    stats_eval = _quiet_stats([entry], n_classes, *EVAL_WINDOW)['per_subject'][0]
    stats_pre = _quiet_stats([entry], n_classes, *PRE_CUE_WINDOW)['per_subject'][0]
    return {
        'acc_full_mean': stats_full['mean'],
        'acc_full_peak': stats_full['peak_acc'],
        'acc_full_peak_t': stats_full['peak_time'],
        'acc_eval_mean': stats_eval['mean'],
        'acc_eval_within_sem': stats_eval['sem'],
        'acc_eval_within_ci95': stats_eval['ci95'],
        'acc_eval_fold_min': stats_eval['min'],
        'acc_eval_fold_max': stats_eval['max'],
        'acc_eval_above_chance': stats_eval['above_chance'],
        'f1_macro_eval': _f1_macro_from_cms(entry, *EVAL_WINDOW),
        'bal_acc_eval': _balanced_accuracy_from_cms(entry, *EVAL_WINDOW),
        'acc_precue_mean': stats_pre['mean'],
        'precue_flag': bool(stats_pre['mean'] > chance + PRE_CUE_FLAG_MARGIN),
        'n_folds': stats_eval['n_folds'],
        'n_windows': stats_full['n_windows'],
    }



def build_summary(results_by_variant, subject_meta, failures=None,
                            params_dict=None, paths=None, write=True):
    """
    Build, write and print the cross-subject summary.

    One row per subject with both variants side by side, plus a GROUP row carrying
    the between-subject statistics - the uncertainty that actually supports a claim
    about the pipeline. Returns ``(DataFrame, payload)``.
    """
    paths = paths or project_paths()
    failures = failures or {}
    cohorts = {v: _cohorts(entries) for v, entries in results_by_variant.items()}

    # failures are keyed '<subject>/<variant>'; rows are per subject
    failed_subjects = {key.split('/')[0] for key in failures}
    subjects = sorted({e['subject_name'] for entries in results_by_variant.values()
                       for e in entries} | failed_subjects | set(subject_meta))
    rows = []
    for subject in subjects:
        meta = subject_meta.get(subject, {})
        subject_failures = {k.split('/', 1)[1]: v for k, v in failures.items()
                            if k.split('/')[0] == subject}
        row = {'subject': subject,
               'status': ('failed' if len(subject_failures) >= len(results_by_variant)
                          else 'partial' if subject_failures else 'ok'),
               'error': '; '.join(f'{v}: {e}' for v, e in sorted(subject_failures.items())),
               'n_files': meta.get('n_files'),
               'n_epochs': meta.get('n_epochs'),
               'n_epochs_before': meta.get('n_epochs_before'),
               'n_channels': meta.get('n_channels'),
               'dropped_bads': '|'.join(meta.get('dropped_bads') or []),
               'classes': '|'.join(meta.get('classes') or []),
               'n_classes': meta.get('n_classes'),
               'balanced': meta.get('balanced'),
               }
        counts = meta.get('counts_after') or {}
        row['chance'] = 1.0 / meta['n_classes'] if meta.get('n_classes') else None
        row['majority_baseline'] = (max(counts.values()) / sum(counts.values())
                                    if counts else None)
        for event in DESIRED_EVENTS:
            row[f'n_{event}'] = counts.get(event)

        for variant in results_by_variant:
            entry = next((e for e in results_by_variant[variant]
                          if e['subject_name'] == subject), None)
            if entry is None or not meta.get('n_classes'):
                continue
            block = _variant_columns(entry, meta['n_classes'], row['chance'])
            for key, value in block.items():
                if key in ('n_folds', 'n_windows'):
                    row[key] = value
                else:
                    row[f'{key}_{variant}'] = value
            row[f'runtime_s_{variant}'] = meta.get(f'runtime_s_{variant}')

        # in the cohort only if EVERY variant agrees; assigning inside the loop
        # above would silently let the last variant win on a disagreement.
        row['in_group_cohort'] = bool(results_by_variant) and all(
            any(e['subject_name'] == subject for e in cohorts[v][0])
            for v in results_by_variant if results_by_variant[v])

        _add_mode_deltas(row)
        rows.append(row)

    group_row = {'subject': 'GROUP', 'status': 'ok', 'error': ''}
    group_stats = {}
    for variant, (cohort, excluded) in cohorts.items():
        if not cohort:
            continue
        n_classes = len(cohort[0].get('desired_events') or DESIRED_EVENTS)
        stats = _quiet_stats(cohort, n_classes, *EVAL_WINDOW)
        pre = _quiet_stats(cohort, n_classes, *PRE_CUE_WINDOW)
        group_stats[variant] = {'eval': stats['group'], 'precue': pre['group'],
                                'n_subjects': stats['group']['n_subjects'],
                                'classes': list(cohort[0].get('desired_events') or []),
                                'excluded': excluded}
        group_row['n_classes'] = n_classes
        group_row['chance'] = 1.0 / n_classes
        group_row[f'acc_eval_mean_{variant}'] = stats['group']['grand_mean']
        group_row[f'acc_eval_between_std_{variant}'] = stats['group']['between_std']
        group_row[f'acc_eval_between_sem_{variant}'] = stats['group']['sem']
        group_row[f'acc_eval_between_ci95_{variant}'] = stats['group']['ci95']
        group_row[f'acc_eval_above_chance_{variant}'] = stats['group']['above_chance']
        group_row[f'acc_precue_mean_{variant}'] = pre['group']['grand_mean']
        group_row['n_subjects'] = stats['group']['n_subjects']

        # Group macro F1 is the mean of the per-subject values (descriptive_stats_group
        # only knows about accuracy), with the between-subject spread beside it.
        f1s = np.array([r.get(f'f1_macro_eval_{variant}') for r in rows
                        if r.get('in_group_cohort')
                        and r.get(f'f1_macro_eval_{variant}') is not None], dtype=float)
        f1s = f1s[~np.isnan(f1s)]
        if len(f1s):
            group_row[f'f1_macro_eval_{variant}'] = float(f1s.mean())
            group_row[f'f1_macro_between_std_{variant}'] = (
                float(f1s.std(ddof=1)) if len(f1s) > 1 else float('nan'))
            group_stats[variant]['f1_macro'] = {
                'mean': float(f1s.mean()),
                'between_std': float(f1s.std(ddof=1)) if len(f1s) > 1 else float('nan'),
                'n_subjects': int(len(f1s))}
    _add_mode_deltas(group_row)
    rows.append(group_row)

    df = pd.DataFrame(rows)
    payload = {
        'run': _run_provenance(params_dict, subjects, paths,
                               modes={m: list(v) for m, v in MODES.items()},
                               eval_window=list(EVAL_WINDOW),
                               pre_cue_window=list(PRE_CUE_WINDOW)),
        'group': group_stats,
        'per_subject': rows[:-1],
        'failures': failures,
        'caveats': CAVEATS,
    }
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
    """The human-readable table, leading with the leakage canary."""
    print(f"\n{'=' * 78}\nWindowed summary\n{'=' * 78}")
    cols = ['subject', 'status', 'n_classes', 'n_channels', 'n_epochs', 'majority_baseline']
    for mode in MODES:
        cols += [c for c in (f'acc_eval_mean_{mode}', f'f1_macro_eval_{mode}')
                 if c in df.columns]
    cols += [c for c in df.columns if c.startswith('delta_')]
    with pd.option_context('display.width', 250, 'display.max_columns', 60):
        print(df[[c for c in cols if c in df.columns]].to_string(
            index=False, float_format=lambda v: f'{v:.3f}'))

    print(f"\n{'-' * 78}\nTrain/test centering matrix (accuracy {EVAL_WINDOW[0]}-"
          f"{EVAL_WINDOW[1]}s, macro F1 in brackets)\n{'-' * 78}")
    group = df[df.subject == 'GROUP']
    if len(group):
        g = group.iloc[0]
        print(f"{'':<16}{'test centered':>20}{'test uncentered':>20}")
        # Iterate RECIPES - what MODES holds - and label the rows with the variant
        # names the reader thinks in.
        for train in ('file', None):
            cells = []
            for test in ('file', None):
                mode = next((m for m, (tr, te) in MODES.items()
                             if tr == train and te == test), None)
                acc, f1 = g.get(f'acc_eval_mean_{mode}'), g.get(f'f1_macro_eval_{mode}')
                if acc is None or pd.isna(acc):
                    cells.append('-')
                elif f1 is None or pd.isna(f1):
                    cells.append(f"{acc:.3f}")
                else:
                    cells.append(f"{acc:.3f} [{f1:.3f}]")
            print(f"train {_RECIPE_LABELS[train]:<10}" + ''.join(f"{c:>20}" for c in cells))

    for mode, stats in group_stats.items():
        g = stats['eval']
        train, test = (_RECIPE_LABELS.get(r, '?') for r in MODES.get(mode, ('?', '?')))
        f1 = stats.get('f1_macro', {})
        print(f"\n[{mode}] train={train} test={test} | n={g['n_subjects']} subjects, "
              f"classes={stats['classes']}")
        print(f"   accuracy {EVAL_WINDOW[0]}-{EVAL_WINDOW[1]}s: {g['grand_mean']:.3f} "
              f"+/- {g['between_std']:.3f} (between-subject SD), "
              f"SEM {g['sem']:.3f}, 95% CI +/-{g['ci95']:.3f}")
        if f1:
            print(f"   macro F1: {f1['mean']:.3f} +/- {f1['between_std']:.3f} "
                  f"(between-subject SD)")
        print(f"   pre-cue {PRE_CUE_WINDOW[0]}-{PRE_CUE_WINDOW[1]}s: "
              f"{stats['precue']['grand_mean']:.3f}   <- must be near chance")
        if stats['excluded']:
            print(f"   excluded from cohort: {stats['excluded']}")

    # `is True` on purpose: a missing/failed variant leaves NaN here, and NaN is
    # truthy, which would flag every failed subject as a leak.
    flagged = [r['subject'] for _, r in df.iterrows()
               if any(r.get(f'precue_flag_{v}') is True for v in MODES)]
    if flagged:
        print(f"\n!! PRE-CUE ACCURACY ABOVE CHANCE for {flagged} - investigate before "
              f"trusting the headline numbers.")


def summarize_windowed(results_by_variant=None, subject_meta=None, failures=None,
                        params_dict=None, paths=None, save_figs=True, write=True,
                        label=None):
    """
    Group statistics, group figures and the summary table.

    Re-runnable from disk: with no arguments it loads
    ``Benchmark/Metrics/Group/group_windowed_<variant>.json``, so the summary can be
    rebuilt without re-classifying anything.
    """
    paths = paths or project_paths(label=label, params_dict=params_dict)
    if results_by_variant is None:
        results_by_variant, subject_meta = load_windowed_results(paths)
    if not any(results_by_variant.values()):
        print("No benchmark results found. Run run_windowed_batch() first.")
        return None, None

    if save_figs:
        with _batch_backend():
            for variant, entries in results_by_variant.items():
                cohort, _ = _cohorts(entries)
                if not cohort:
                    continue
                n_classes = len(cohort[0].get('desired_events') or DESIRED_EVENTS)
                save_group_figures(cohort, variant, n_classes, paths)

    return build_summary(results_by_variant, subject_meta or {},
                                   failures, params_dict, paths, write)



#%%
# ============================================================
# Status and batch driver
# ============================================================

def load_windowed_results(paths=None, label=None, params_dict=None):
    """Group results and per-subject meta previously written to Benchmark/<label>/."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    results_by_mode, subject_meta = {}, {}
    for mode in MODES:
        path = paths.metrics_group / GROUP_RESULTS_NAME.format(mode=mode)
        results_by_mode[mode] = load_group_results(path) if path.exists() else []
    for meta_file in sorted(paths.cache.glob('*_meta.json')):
        meta = json.loads(meta_file.read_text())
        subject = meta.get('subject')
        if not subject:
            continue
        entry = subject_meta.setdefault(subject, {})
        entry.setdefault('n_files', len(meta.get('files', [])))
        entry.setdefault('n_epochs', meta.get('n_epochs'))
        entry.setdefault('classes', list((meta.get('params') or {}).get('desired_events') or []))
        entry.setdefault('n_classes', len(entry['classes']) or None)
        entry.setdefault('n_channels', len(meta.get('ch_names') or []) or None)
    return results_by_mode, subject_meta


def windowed_status(paths=None, verbose=True, label=None, params_dict=None):
    """Per-subject progress: recordings, cached variants, metrics on disk."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    classes = _get_classes_initials((params_dict or default_params())['desired_events'])
    subjects = list_subjects(paths)
    rows = []
    for subject in subjects:
        row = {'subject': subject, 'n_files': len(subject_files(subject, paths))}
        # Two different things, and conflating them is what made these columns
        # useless: the epoch cache is keyed on EPOCH_VARIANTS ('<subj>_centered-epo.fif'),
        # while metrics are written one directory per MODE.
        for variant in EPOCH_VARIANTS:
            row[f'cached_{variant}'] = _cache_paths(subject, variant, paths).epochs.exists()
        for mode in MODES:
            metrics_dir = paths.metrics_individual / mode
            # class tag in the glob: a leftover file from another class set must not
            # make this subject look done
            row[f'metrics_{mode}'] = bool(
                list(metrics_dir.glob(f'{subject}_{classes}_*.json'))
            ) if metrics_dir.exists() else False
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        print(f"Windowed root: {paths.out}   (label '{paths.label}')")
        print(df.to_string(index=False))
        todo = [r['subject'] for r in rows
                if not all(r[f'metrics_{m}'] for m in MODES)]
        print(f"\n{len(subjects) - len(todo)}/{len(subjects)} subject(s) complete."
              + (f" Outstanding: {todo}" if todo else " Nothing outstanding."))
    return df



def _existing_entry(subject, mode, paths, desired_events=None):
    """
    A previously saved entry for this subject/mode, if it matches the class set.

    Matching on the subject alone is not enough. Label-scoped directories already
    keep different class sets apart, but an entry can still be stale within a label
    (same classes, different filters or electrode group). Comparing the stored
    desired_events turns a silently-wrong reused number into a recompute.
    """
    path = paths.metrics_group / GROUP_RESULTS_NAME.format(mode=mode)
    if not path.exists():
        return None
    for entry in load_group_results(path):
        if entry.get('subject_name') != subject:
            continue
        stored = entry.get('desired_events')
        # Compare as SETS: the stored list comes from epochs.event_id, which MNE
        # orders alphabetically, while the requested list is in config order. Only
        # the membership identifies the class set; ordering affects nothing
        # downstream (the classifier derives classes_ from the labels it sees).
        if desired_events is not None and stored is not None                 and set(stored) != set(desired_events):
            print(f"  [{subject}/{variant}] saved result is for classes {sorted(stored)}, "
                  f"not {sorted(desired_events)} - recomputing")
            return None
        return entry
    return None



def run_windowed_batch(subjects=None, modes=MODES, force=False, skip_done=True,
                        save_figs=True, batch_backend='Agg', params_dict=None,
                        paths=None, seed=RANDOM_SEED,
                        balance_trigger_class=BALANCE_TRIGGER_CLASS, label=None):
    """
    Benchmark every subject across the train/test centering matrix, then summarise.

    A failing subject is reported and skipped, never fatal. ``skip_done`` reuses a
    previously saved entry instead of recomputing, which both makes an interrupted
    run resumable and stops ``add_subject_to_group`` (which does not deduplicate)
    from double-weighting a subject.

    Note the whole matrix is computed per subject in one go, because all four modes
    share the same loaded epochs, the same balancing selection and the same folds -
    that sharing is what makes the delta_* columns exactly paired.
    """
    paths = paths or project_paths(label=label, params_dict=params_dict)
    requested_events = list((params_dict or default_params())['desired_events'])
    print(f"output label: '{paths.label}'  ->  {paths.out.relative_to(paths.root)}")
    subjects = (list_subjects(paths) if subjects is None
                else [subjects] if isinstance(subjects, str) else list(subjects))
    modes = [modes] if isinstance(modes, str) else list(modes)

    results_by_variant = {m: [] for m in modes}
    subject_meta, failures = {}, {}

    with _batch_backend(batch_backend):
        for i, subject in enumerate(subjects, 1):
            print(f"\n{'=' * 70}\n[{i}/{len(subjects)}] {subject}\n{'=' * 70}")
            try:
                pending = modes
                if skip_done and not force:
                    reused = {m: _existing_entry(subject, m, paths, requested_events)
                              for m in modes}
                    for mode, entry in reused.items():
                        if entry is not None:
                            print(f"  [{subject}/{mode}] reusing saved result "
                                  f"(skip_done=True)")
                            results_by_variant[mode].append(entry)
                            subject_meta.setdefault(
                                subject, _meta_from_cache(subject, entry, paths))
                    pending = [m for m, e in reused.items() if e is None]
                if not pending:
                    continue

                results = run_subject_windowed(
                    subject, pending, params_dict, paths, force, save_figs, seed,
                    balance_trigger_class)
                for mode, result in results.items():
                    add_subject_to_group(
                        results_by_variant[mode], subject, result['scores_windows'],
                        result['folds_confusion_matrices_per_window'], result['w_times'],
                        params_dict=result['params'],
                        save_dir=_variant_dir(paths.metrics_individual, mode))
                    meta = subject_meta.setdefault(subject, {})
                    meta.update({k: result[k] for k in
                                 ('n_files', 'n_epochs', 'n_epochs_before', 'n_channels',
                                  'dropped_bads',
                                  'classes', 'n_classes', 'counts_before', 'counts_after',
                                  'balanced')})
                    meta[f'runtime_s_{mode}'] = result['runtime_s']
            except Exception as err:
                # one failure takes the whole subject, since the modes share the
                # loaded epochs and folds
                for mode in modes:
                    failures[f'{subject}/{mode}'] = f'{type(err).__name__}: {err}'
                print(f"  !! FAILED [{subject}]: {type(err).__name__}: {err}")
            finally:
                plt.close('all')

    for mode, entries in results_by_variant.items():
        if not entries:
            continue
        name = GROUP_RESULTS_NAME.format(mode=mode)
        merged = _merge_group_results(entries, paths.metrics_group / name,
                                  load_group_results)
        if len(merged) > len(entries):
            print(f"  [{mode}] merged {len(entries)} from this run with "
                  f"{len(merged) - len(entries)} already on disk")
        results_by_variant[mode] = merged
        save_group_results(merged, paths.metrics_group, filename=name)
        # subjects carried in from disk have no meta from this run, so their summary
        # rows would come out blank; fill them from their cached epochs' metadata
        for entry in merged:
            name_ = entry.get('subject_name')
            if name_ and name_ not in subject_meta:
                subject_meta[name_] = _meta_from_cache(name_, entry, paths)

    # verify against disk rather than trusting in-memory state
    print(f"\n{'=' * 70}\nBatch: {len(subjects) - len({k.split('/')[0] for k in failures})}"
          f"/{len(subjects)} subject(s) without failures\n{'=' * 70}")
    print(f"{'subject':<10}" + ''.join(f"{m:>9}" for m in modes) + "   figures")
    for subject in subjects:
        cells = []
        for mode in modes:
            key = f'{subject}/{mode}'
            metrics_dir = paths.metrics_individual / mode
            ok = metrics_dir.exists() and bool(
                list(metrics_dir.glob(f'{subject}_*.json')))
            cells.append('FAIL' if key in failures else ('ok' if ok else '-'))
        n_figs = sum(len(list((paths.figures / subject / m).glob('*.png')))
                     for m in modes if (paths.figures / subject / m).exists())
        print(f"{subject:<10}" + ''.join(f"{c:>9}" for c in cells) + f"   {n_figs}")
    if failures:
        for key, err in failures.items():
            print(f"  {key}: {err}")
        rerun = sorted({k.split('/')[0] for k in failures})
        print(f"\nRerun the failures with run_windowed_batch({rerun}).")

    df, payload = summarize_windowed(results_by_variant, subject_meta, failures,
                                      params_dict, paths, save_figs)
    return {'results_by_variant': results_by_variant, 'subject_meta': subject_meta,
            'failures': failures, 'summary': df, 'payload': payload}

