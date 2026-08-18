#%%
"""
Full-epoch decoding analysis: one classification decision per trial, all subjects.

Where ``windowed_batch`` asks "what accuracy at each moment in time" (the live
question), this asks "given the whole trial, how well can it be classified" - a
single cross-validated number per subject, which is what a manuscript reports.

    Stage 1  run_full_epoch_batch()     unattended, every subject x mode x definition
             -> Analysis/<label>/FullEpoch/Metrics/Individuals/<definition>/<mode>/*.json
             -> Analysis/<label>/FullEpoch/Figures/<subject>/<definition>/<mode>/*.png
    Stage 2  summarize_full_epoch()     group stats, group figures, summary table
             -> Analysis/<label>/FullEpoch/Metrics/Group/group_full_epoch_<def>_<mode>.json
             -> Analysis/<label>/FullEpoch/Figures/Group/<definition>/<mode>/*.png
             -> Analysis/<label>/FullEpoch/full_epoch_summary.csv  and  .json
    Stage 3  run_permutation_pass()     opt-in, expensive; per-subject p-values

``<label>`` identifies the class set (e.g. 'MH_LH_RH_FR', 'LH_RH'), so runs with
different desired_events never overwrite or silently reuse each other - see
analysis_common.run_label. The preprocessed-epoch cache is SHARED with
windowed_batch (``Analysis/<label>/Cache/``): the epochs are identical, so there
is no reason to duplicate ~2.5 GB.

TWO DEFINITIONS OF "FULL EPOCH"
-------------------------------
Same code path, parameterised by the prediction window length:

    'vote'    train on the classifier_window crop, slide 2 s windows across the
              evaluation window and take the modal prediction per trial. This is
              what training.evaluate_full_epoch already does.
    'single'  train and test on ONE window spanning the whole evaluation window:
              a single covariance per trial, no sliding and no voting. Covariance
              is better conditioned over 5 s than over 2 s.

CENTERING: WHAT IS REPORTED, AND THE DIAGNOSTIC BEHIND IT
---------------------------------------------------------
The reported configuration is ``c2c``: per-file class-mean centering on both the
training and the test side - exactly what EEG_Preprocessing and Main_Experiment
produce with ``params_dict['CenterByClass'] = True``. That is the single mode in
MODES, so a default run computes only it.

Provenance note that belongs with any c2c number: per-file class-mean centering is
label-dependent (a trial's class must be known to pick the mean subtracted from
it), so c2c is a measure of class separability in centered data rather than an
accuracy an unlabelled-input decoder would achieve. Stated here so the question has
an answer on file.

DIAGNOSTIC_MODES keeps the full comparison available - pass
``modes=list(DIAGNOSTIC_MODES)`` to reproduce it. Measured 2026-08-02, n=10:

    per-file recipe over global   (c2c - g2g)  +0.063   9-10/10   p <= 0.003
    self-inclusion, test side     (g2g - l2l)  +0.008    6/10     p = 0.20  (n.s.)
    self-inclusion, train side    (g2u - l2u)  +0.011    9/10     p = 0.004
    label-dependence only         (l2l - l2u)  +0.005    5/10     p = 0.47  (n.s.)
    leak-free centering benefit   (l2u - u2u)  -0.010    6/10     p = 0.36  (n.s.)

i.e. the per-file recipe dominates, and it decomposes as
mean(file,class) = mean(file) + [mean(file,class) - mean(file)] - a label-free
session-drift term plus a label-dependent class term. Which of the two carries the
+0.063 was not measured; it needs per-file provenance, which the concatenated epoch
cache does not retain.

THE MODE TABLE (all of DIAGNOSTIC_MODES)
----------------------------------------
    mode   train            test             meaning
    c2c    per-file         per-file         REPORTED - Main_Experiment with CenterByClass=True
    c2u    per-file         uncentered       per-file training only, raw test
    g2g    global pooled    global pooled    recipe-matched baseline for the LOO test
    g2u    global pooled    uncentered       matched baseline, train side
    l2l    fold-local       fold-local       no self-inclusion, either side
    l2u    fold-local       uncentered       fully leak-free
    u2u    uncentered       uncentered       no centering anywhere

Naming: <train>2<test>, where the letter is the centering recipe each side used:

    c   class means computed per XDF FILE - what EEG_Preprocessing produces with
        CenterByClass=True (the 'centered' epoch variant), so it also removes
        between-session drift
    g   class means over every trial of the subject (global pooled)
    l   class means over the fold's TRAINING trials only (fold-local, no
        self-inclusion)
    u   not centered

The same letters mean the same recipes in windowed_batch and session_batch, so
c2c/c2u refer to the same thing in all three.

'file' means class means computed PER XDF FILE (what EEG_Preprocessing does with
CenterByClass=True, so it also removes between-session drift); 'global' means over every trial of the
subject; 'fold' means over the fold's TRAINING trials only. File boundaries are not
recoverable from the concatenated epochs - event gaps reflect within-run block
breaks, not files - which is why 'global' exists as the re-estimable stand-in.

All modes and definitions share the same loaded epochs, the same balancing
selection and the SAME folds, so every difference is exactly paired per subject.

REPORTING THIS IN A MANUSCRIPT
------------------------------
- Call it "offline single-trial decoding accuracy", NOT "optimal": it is this
  pipeline over this window, not an upper bound on decodability.
- The evaluation window must be fixed a priori by the paradigm (cue at 0, trial
  ends at 5), never chosen because accuracy peaked there.
- 'vote' aggregates 2 s windows at 0.25 s step, i.e. ~87% overlap. It is a
  reasonable aggregation rule but NOT n independent votes; do not describe it as
  averaging away noise.
- Report macro F1 and majority_baseline beside accuracy: the 4-class config runs
  unbalanced (baseline ~0.277 against a 0.25 chance level).
- For significance use run_permutation_pass with n_permutations >= 200.
  Main_Experiment's n_permutations=10 cannot produce a usable p-value.

NOTE: importing this module imports ``src.preprocessing``, which runs
``%matplotlib qt`` at import time, so an IPython kernel is required. Use
``notebooks/Full_Epoch_Analysis.ipynb``.
"""

import copy
import json
import random
import time

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .analysis_common import (BALANCE_TRIGGER_CLASS, CV_N_REPEATS, CV_N_SPLITS,
                              DESIRED_EVENTS, EPOCH_VARIANTS, RANDOM_SEED,
                              _assert_variants_aligned, _balance_indices,
                              _batch_backend, _capture_figures, _class_counts,
                              _make_cv_split, _merge_group_results, _meta_from_cache,
                              _run_provenance, _savefig, assert_caveat_modes, default_params,
                              load_or_build_epochs, subject_files, subject_params)
# re-exported so `from src.full_epoch_batch import *` gives the notebook these too -
# the parameters cell needs every one of them
from .analysis_common import (ELECTRODE_GROUP_NAMES, ELECTRODE_GROUPS,  # noqa: F401
                              PARAM_GROUPS, check_params, describe_params,
                              list_subjects, run_label)
from .analysis_common import project_paths as _common_paths
from .evaluation import (plot_accuracy_group_full_epoch,
                         plot_average_confusion_full_epoch_group)
from .group_analysis_full_epoch import (add_subject_to_group_full_epoch,
                                        load_group_results_full_epoch,
                                        save_group_results_full_epoch)
from .training import run_full_epoch_classification_cv, run_permutation_test

#%%
# ============================================================
# Configuration
# ============================================================

# The trial window the manuscript reports over. Fixed by the paradigm - cue at 0,
# trial ends at 5 - NOT selected because accuracy peaked here.
FULL_EPOCH_WINDOW = (0.0, 5.0)

# {definition: prediction window length in seconds, or None for "the whole window"}
DEFINITIONS = {
    'vote': 2.0,     # slide 2 s windows across FULL_EPOCH_WINDOW, majority vote
    'single': None,  # one window spanning FULL_EPOCH_WINDOW: no sliding, no voting
}

# Centering recipes. These are NOT interchangeable and the distinction is the whole
# reason the diagnostic needs care:
#
#   'file'   what EEG_Preprocessing actually produces with CenterByClass=True (the
#            'centered' epoch variant): class means computed PER XDF FILE, so it also
#            removes between-session drift. This is what
#            Main_Experiment trains on. File boundaries are not recoverable from the
#            concatenated epochs (event gaps reflect within-run block breaks, not
#            files), so this recipe cannot be reproduced with a different estimation
#            set - which is why the matched baseline below exists.
#   'global' the same idea but with class means over every trial of the subject.
#            Reproducible with any estimation set, so it is the recipe the
#            leave-one-out comparison is made against.
#   'fold'   class means from the fold's TRAINING trials only: no self-inclusion.
#   None     untouched.
#
# Naming: <train>2<test> using c=file, g=global, l=fold-local, u=uncentered - the
# same letters as windowed_batch and session_batch. Note c, not f: the letter names
# the CENTERED variant, the recipe name behind it is 'file'.
# The reported configuration: per-file class-mean centering on both sides, i.e.
# exactly what EEG_Preprocessing/Main_Experiment produce with CenterByClass=True.
# This is the analysis the study reports.
MODES = {
    'c2c': ('file', 'file'),
}

# Kept available but NOT run by default - these are the centering diagnostic. Pass
# modes=list(DIAGNOSTIC_MODES) to reproduce it. See the module docstring for what
# each isolates; the measured result is recorded in docs and in the run of
# 2026-08-02 (n=10): the per-file recipe accounts for +0.063 of accuracy, six times
# the self-inclusion effect, and leak-free centering (l2u) does not beat u2u.
DIAGNOSTIC_MODES = {
    'c2c': ('file', 'file'),
    'c2u': ('file', None),
    'g2g': ('global', 'global'),
    'g2u': ('global', None),
    'l2l': ('fold', 'fold'),
    'l2u': ('fold', None),
    'u2u': (None, None),
}

# Deltas for the diagnostic mode set; each compares two modes differing in exactly
# one thing. Inert when only 'c2c' is run.
MODE_DELTAS = {
    'self_inclusion_leak_test': ('g2g', 'l2l'),
    'self_inclusion_leak_train': ('g2u', 'l2u'),
    'label_dependence_only': ('l2l', 'l2u'),
    'per_file_vs_global': ('c2c', 'g2g'),
    'centering_benefit_deployable': ('l2u', 'u2u'),
}

PERMUTATION_N = 200          # >= 200 for a usable p-value; 10 (Main_Experiment) is not
FIGURE_DPI = 150
SUMMARY_CSV_NAME = 'full_epoch_summary.csv'
SUMMARY_JSON_NAME = 'full_epoch_summary.json'
GROUP_RESULTS_NAME = 'group_full_epoch_{definition}_{mode}.json'

CAVEATS = [
    "This is OFFLINE SINGLE-TRIAL DECODING ACCURACY for this pipeline over the fixed window "
    f"{FULL_EPOCH_WINDOW}s - not an upper bound on decodability. Do not call it 'optimal'.",
    "The evaluation window is fixed a priori by the paradigm (cue at 0, trial ends at 5). It "
    "was not chosen because accuracy peaked there.",
    "Definition 'vote' aggregates 2 s windows at 0.25 s step (~87% overlap). It is an "
    "aggregation rule, NOT n independent votes; it does not average away noise.",
    "Definition 'single' uses one covariance per trial over the whole window - no sliding, no "
    "voting - and is better conditioned than the 2 s windows.",
    "Class-mean centering is LABEL-DEPENDENT: a trial's class must be known to centre it, so "
    "any mode whose TEST source is centered (c2c, g2g, l2l) reports a quantity a decoder "
    "cannot achieve on unlabelled data. Those are separability measures, not decoder accuracy.",
    "Three centering recipes, and they are not interchangeable: 'file' is what "
    "EEG_Preprocessing produces with CenterByClass=True (class means PER XDF FILE, so it also "
    "removes between-session drift); 'global' uses class means over every trial; 'fold' uses the fold's training "
    "trials only. File boundaries cannot be recovered from the concatenated epochs, so the "
    "leave-one-out comparison is made against 'global' (g2g - l2l), and c2c - g2g reports "
    "separately what the per-file recipe adds.",
    "l2u is the only fully leak-free deployable number: training-side centering estimated from "
    "training trials only, test data untouched.",
    "The 4-class config runs unbalanced (balancing applies only when a 'Rest' class is "
    "present), so accuracy sits against a per-subject majority_baseline rather than "
    "1/n_classes. Macro F1 is reported beside it.",
    f"CV is RepeatedStratifiedKFold({CV_N_SPLITS}, n_repeats={CV_N_REPEATS}); the ~10 folds "
    "share ~78% of their training trials, so within-subject error bars are anticonservative. "
    "Use the between-subject spread in the GROUP row.",
    "Permutation p-values are only present if run_permutation_pass() was run; it is opt-in "
    f"because it costs n_permutations (default {PERMUTATION_N}) x the CV cost per subject.",
]

# The caveats ship inside full_epoch_summary.json, so a mode named there but not
# defined here would be a wrong claim travelling with the numbers. This caught 'f2f'.
assert_caveat_modes(CAVEATS, DIAGNOSTIC_MODES, 'full_epoch_batch')


def project_paths(root=None, label=None, params_dict=None):
    """
    This analysis's directories: ``Analysis/<label>/FullEpoch/`` plus the shared cache.

    ``paths.cache`` is the same directory ``windowed_batch`` uses - the preprocessed
    epochs are identical, so they are built once and read by both.
    """
    paths = _common_paths(root, label, params_dict, analysis='FullEpoch')
    paths.full_epoch = paths.out          # readable alias used throughout this module
    return paths


def _out_dir(parent, definition, mode):
    out = parent / definition / mode
    out.mkdir(parents=True, exist_ok=True)
    return out


def full_epoch_params(definition, params_dict=None):
    """
    Params for one full-epoch definition.

    'single' widens the training crop to the whole evaluation window and sets the
    prediction window to the same length, so exactly one window fits and the
    majority vote is trivial - which is how "train and test on the entire epoch"
    reduces to the existing sliding-window machinery.
    """
    if definition not in DEFINITIONS:
        raise ValueError(f"definition must be one of {list(DEFINITIONS)}, got {definition!r}")
    params = copy.deepcopy(params_dict or default_params())
    tmin, tmax = FULL_EPOCH_WINDOW
    win_len = DEFINITIONS[definition]
    if win_len is None:
        params['classifier_window_s'] = tmin
        params['classifier_window_e'] = tmax
        params['windowed_prediction_params'] = {'win_len': tmax - tmin, 'win_step': 0.25}
    else:
        params['windowed_prediction_params'] = dict(
            params.get('windowed_prediction_params') or {}, win_len=win_len)
        params['windowed_prediction_params'].setdefault('win_step', 0.25)
    return params


#%%
# ============================================================
# Fold-local (leave-one-out) centering
# ============================================================


def _fold_local_centered(data, codes, source_idx):
    """
    Subtract from every trial the mean of its own class, computed over ``source_idx``.

    This is the one genuinely new primitive here. With ``source_idx`` set to the
    fold's training trials it removes the self-inclusion that pooled centering has
    (where a trial contributes ~1/n of the mean subtracted from itself), while
    keeping the label-dependence - which is exactly the contrast the diagnostic
    needs. With ``source_idx`` set to every trial it reproduces pooled centering.

    Parameters
    ----------
    data : (n_trials, n_channels, n_times) array
    codes : (n_trials,) array of int event codes
    source_idx : array of int
        Rows the class means are estimated from.

    Returns
    -------
    (n_trials, n_channels, n_times) array
        A new array; ``data`` is not modified.
    """
    out = np.array(data, dtype=float, copy=True)
    source_idx = np.asarray(source_idx)
    source_codes = codes[source_idx]
    for code in np.unique(codes):
        rows = source_idx[source_codes == code]
        if len(rows) == 0:
            # class absent from the estimation set: leave it alone rather than
            # subtracting a mean derived from other classes
            print(f"  !! class code {code} absent from the centering source; not centered")
            continue
        out[codes == code] -= out[rows].mean(axis=0)
    return out


def _as_epochs(template, data):
    """A copy of ``template`` carrying ``data``, preserving events/ids/tmin."""
    return mne.EpochsArray(data, template.info, events=template.events,
                           event_id=template.event_id, tmin=template.tmin,
                           verbose='error')


def _static_source(recipe, epochs_by_variant, cropped_by_variant, codes, cache):
    """
    The (cropped, full) epochs for a recipe that does not depend on the fold.

    'global' is computed once per subject and cached, since its estimation set is
    every trial regardless of fold.
    """
    if recipe == 'file':
        return cropped_by_variant['centered'], epochs_by_variant['centered']
    if recipe is None:
        return cropped_by_variant['uncentered'], epochs_by_variant['uncentered']
    if recipe == 'global':
        if 'global' not in cache:
            all_idx = np.arange(len(codes))
            base_cropped = cropped_by_variant['uncentered']
            base_full = epochs_by_variant['uncentered']
            cache['global'] = (
                _as_epochs(base_cropped,
                           _fold_local_centered(base_cropped.get_data(), codes, all_idx)),
                _as_epochs(base_full,
                           _fold_local_centered(base_full.get_data(), codes, all_idx)))
        return cache['global']
    raise ValueError(f"recipe {recipe!r} is fold-dependent; use the fold-local driver")


def _run_cv_static(mode, epochs_by_variant, cropped_by_variant, events, params, cache):
    """Single call into run_full_epoch_classification_cv - no per-fold centering."""
    train_recipe, test_recipe = DIAGNOSTIC_MODES[mode]
    cropped, _ = _static_source(train_recipe, epochs_by_variant, cropped_by_variant,
                                events[:, 2], cache)
    _, full = _static_source(test_recipe, epochs_by_variant, cropped_by_variant,
                             events[:, 2], cache)
    tmin, tmax = FULL_EPOCH_WINDOW
    return run_full_epoch_classification_cv(
        full, cropped, _make_cv_split(cropped, events), params, tmin=tmin, tmax=tmax)


def _run_cv_fold_local(mode, epochs_by_variant, cropped_by_variant, events, params, cache):
    """
    Fold-local centering: rebuild the centered copies from each fold's training
    trials, then evaluate that fold.

    run_full_epoch_classification_cv reads its arrays once, before its own fold
    loop, so per-fold data means calling it once per fold with a single-fold split
    and concatenating. That reuses all of its evaluation logic instead of
    duplicating it.
    """
    train_recipe, test_recipe = DIAGNOSTIC_MODES[mode]
    tmin, tmax = FULL_EPOCH_WINDOW
    codes = events[:, 2]

    base_cropped = cropped_by_variant['uncentered']
    base_full = epochs_by_variant['uncentered']
    cropped_data = base_cropped.get_data()
    full_data = base_full.get_data()

    fold_accuracies, fold_confusion_matrices = [], []
    for train_idx, test_idx in _make_cv_split(base_cropped, events):
        if train_recipe == 'fold':
            train_source = _as_epochs(
                base_cropped, _fold_local_centered(cropped_data, codes, train_idx))
        else:
            train_source, _ = _static_source(train_recipe, epochs_by_variant,
                                             cropped_by_variant, codes, cache)
        if test_recipe == 'fold':
            test_source = _as_epochs(
                base_full, _fold_local_centered(full_data, codes, train_idx))
        else:
            _, test_source = _static_source(test_recipe, epochs_by_variant,
                                            cropped_by_variant, codes, cache)
        accs, cms = run_full_epoch_classification_cv(
            test_source, train_source, [(train_idx, test_idx)], params,
            tmin=tmin, tmax=tmax)
        fold_accuracies.extend(accs)
        fold_confusion_matrices.extend(cms)
    return fold_accuracies, fold_confusion_matrices


def _run_cv(mode, epochs_by_variant, cropped_by_variant, events, params, cache=None):
    """Dispatch to the static or fold-local driver."""
    cache = {} if cache is None else cache
    if 'fold' in DIAGNOSTIC_MODES[mode]:
        return _run_cv_fold_local(mode, epochs_by_variant, cropped_by_variant, events,
                                  params, cache)
    return _run_cv_static(mode, epochs_by_variant, cropped_by_variant, events, params, cache)


#%%
# ============================================================
# Per-subject worker
# ============================================================


def _prepare_subject(subject, params_dict=None, paths=None, force=False,
                     seed=RANDOM_SEED, balance_trigger_class=BALANCE_TRIGGER_CLASS):
    """
    Load both epoch variants, assert alignment, balance once with a shared selection.

    Returns ``(epochs_by_variant, events, params_by_variant, shared_meta)``. The
    shared balancing selection is what keeps the variants trial-aligned, which every
    cross-centering mode depends on.
    """
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

    counts_after = _class_counts(reference)
    shared = {
        'subject': subject,
        'classes': list(base['desired_events']),
        'n_classes': len(base['desired_events']),
        'n_channels': len(reference.ch_names),
        'dropped_bads': sorted(set(base['Electorde_Group']) - set(reference.ch_names)),
        'n_files': len(meta['files']),
        'n_epochs': len(reference),
        'n_epochs_before': n_before,
        'counts_before': counts_before,
        'counts_after': counts_after,
        'balanced': balanced,
        'majority_baseline': (max(counts_after.values()) / sum(counts_after.values())
                              if counts_after else None),
    }
    return epochs_by_variant, events, params_by_variant, shared


def run_subject_full_epoch(subject, definitions=None, modes=None, params_dict=None,
                           paths=None, force=False, save_figs=True, seed=RANDOM_SEED,
                           balance_trigger_class=BALANCE_TRIGGER_CLASS):
    """
    Full-epoch decoding for one subject, across every definition and centering mode.

    Returns ``{(definition, mode): result dict}``.
    """
    paths = paths or project_paths()
    definitions = list(definitions or DEFINITIONS)
    modes = list(modes or MODES)
    started = time.perf_counter()

    epochs_by_variant, events, params_by_variant, shared = _prepare_subject(
        subject, params_dict, paths, force, seed, balance_trigger_class)

    results = {}
    for definition in definitions:
        params = full_epoch_params(definition, params_by_variant[EPOCH_VARIANTS[0]])
        cropped_by_variant = {
            v: e.copy().crop(params['classifier_window_s'], params['classifier_window_e'])
            for v, e in epochs_by_variant.items()}
        # the 'global' recipe does not depend on the fold, so build it once per
        # definition and share it across modes
        recipe_cache = {}

        for mode in modes:
            mode_started = time.perf_counter()
            accs, cms = _run_cv(mode, epochs_by_variant, cropped_by_variant, events,
                                params, recipe_cache)
            entry = dict(shared, definition=definition, mode=mode,
                         train_centering=DIAGNOSTIC_MODES[mode][0],
                         test_centering=DIAGNOSTIC_MODES[mode][1],
                         fold_accuracies=accs, fold_confusion_matrices=cms,
                         params=params,
                         runtime_s=round(time.perf_counter() - mode_started, 1))
            results[(definition, mode)] = entry
            print(f"  [{subject}/{definition}/{mode}] acc {np.mean(accs):.3f} "
                  f"({entry['runtime_s']}s)")

            if save_figs:
                save_subject_figures(subject, definition, mode, accs, cms, paths)

    print(f"  [{subject}] {len(results)} cell(s) in {time.perf_counter() - started:.0f}s")
    return results


def save_subject_figures(subject, definition, mode, fold_accuracies,
                         fold_confusion_matrices, paths=None):
    """Per-subject confusion matrix, averaged over folds."""
    paths = paths or project_paths()
    out_dir = _out_dir(paths.figures / subject, definition, mode)
    total = _sum_fold_cms(fold_confusion_matrices)
    if total is None:
        return
    cm, classes = total
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
    ax.set_title(f'{subject} | {definition} | {mode}\n'
                 f'acc {np.mean(fold_accuracies):.3f}')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _savefig(fig, out_dir / 'confusion.png', paths)
    plt.close(fig)


def _sum_fold_cms(fold_confusion_matrices):
    """Sum the per-fold confusion matrices. Returns (cm, classes) or None."""
    total, classes = None, None
    for cm, labels in fold_confusion_matrices or []:
        cm = np.asarray(cm, dtype=float)
        total = cm if total is None else total + cm
        classes = labels
    if total is None:
        return None
    return total, list(classes)


#%%
# ============================================================
# Summary
# ============================================================


def _entry_metrics(entry):
    """Accuracy, macro F1 and balanced accuracy for one (definition, mode) cell."""
    accs = np.asarray(entry['fold_accuracies'], dtype=float)
    summed = _sum_fold_cms(entry['fold_confusion_matrices'])
    metrics = {
        'acc_mean': float(accs.mean()),
        'acc_within_sd': float(accs.std(ddof=1)) if len(accs) > 1 else float('nan'),
        'acc_fold_min': float(accs.min()),
        'acc_fold_max': float(accs.max()),
        'n_folds': int(len(accs)),
    }
    if summed is not None:
        cm, _ = summed
        with np.errstate(invalid='ignore', divide='ignore'):
            precision = np.diag(cm) / cm.sum(axis=0)
            recall = np.diag(cm) / cm.sum(axis=1)
            f1 = 2 * precision * recall / (precision + recall)
        metrics['f1_macro'] = float(np.nanmean(np.where(np.isnan(f1), 0.0, f1)))
        metrics['bal_acc'] = float(np.nanmean(recall))
    return metrics


def _add_deltas(row, definition):
    """The differences that decide what is reportable."""
    for name, (left, right) in MODE_DELTAS.items():
        a = row.get(f'acc_mean_{definition}_{left}')
        b = row.get(f'acc_mean_{definition}_{right}')
        if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)):
            row[f'delta_{name}_{definition}'] = a - b
    return row


def build_summary(results_by_cell, subject_meta, failures=None, params_dict=None,
                  paths=None, write=True):
    """
    Cross-subject summary: one row per subject, plus a GROUP row. Returns (df, payload).
    """
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
               'n_files': meta.get('n_files'), 'n_epochs': meta.get('n_epochs'),
               'n_channels': meta.get('n_channels'),
               'classes': '|'.join(meta.get('classes') or []),
               'n_classes': meta.get('n_classes'),
               'chance': 1.0 / meta['n_classes'] if meta.get('n_classes') else None,
               'majority_baseline': meta.get('majority_baseline'),
               'balanced': meta.get('balanced'),
               'dropped_bads': '|'.join(meta.get('dropped_bads') or [])}
        for (definition, mode), entries in results_by_cell.items():
            entry = entries.get(subject)
            if entry is None:
                continue
            for key, value in _entry_metrics(entry).items():
                if key == 'n_folds':
                    row['n_folds'] = value
                else:
                    row[f'{key}_{definition}_{mode}'] = value
            row[f'runtime_s_{definition}_{mode}'] = entry.get('runtime_s')
            if entry.get('permutation_p') is not None:
                row[f'perm_p_{definition}_{mode}'] = entry['permutation_p']
        for definition in DEFINITIONS:
            _add_deltas(row, definition)
        rows.append(row)

    group_row = {'subject': 'GROUP', 'status': 'ok', 'error': ''}
    group_stats = {}
    for (definition, mode), entries in results_by_cell.items():
        values = np.array([_entry_metrics(e)['acc_mean'] for e in entries.values()],
                          dtype=float)
        f1s = np.array([_entry_metrics(e).get('f1_macro', np.nan)
                        for e in entries.values()], dtype=float)
        if not len(values):
            continue
        sd = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
        sem = sd / np.sqrt(len(values)) if len(values) > 1 else float('nan')
        group_row[f'acc_mean_{definition}_{mode}'] = float(values.mean())
        group_row[f'acc_between_sd_{definition}_{mode}'] = sd
        group_row[f'acc_between_sem_{definition}_{mode}'] = sem
        group_row[f'f1_macro_{definition}_{mode}'] = float(np.nanmean(f1s))
        group_stats[f'{definition}/{mode}'] = {
            'n_subjects': int(len(values)), 'mean': float(values.mean()),
            'between_sd': sd, 'sem': sem, 'ci95': 1.96 * sem if len(values) > 1 else float('nan'),
            'min': float(values.min()), 'max': float(values.max()),
            'f1_macro': float(np.nanmean(f1s))}
    for definition in DEFINITIONS:
        _add_deltas(group_row, definition)
    group_row['n_classes'] = next((r.get('n_classes') for r in rows if r.get('n_classes')), None)
    group_row['chance'] = 1.0 / group_row['n_classes'] if group_row.get('n_classes') else None
    rows.append(group_row)

    df = pd.DataFrame(rows)
    payload = {'run': _run_provenance(params_dict, subjects, paths,
                                      full_epoch_window=list(FULL_EPOCH_WINDOW)),
               'full_epoch_window': list(FULL_EPOCH_WINDOW),
               'definitions': {d: v for d, v in DEFINITIONS.items()},
               'modes': {m: list(v) for m, v in MODES.items()},
               'diagnostic_modes': {m: list(v) for m, v in DIAGNOSTIC_MODES.items()},
               'group': group_stats, 'per_subject': rows[:-1],
               'failures': failures, 'caveats': CAVEATS}
    if write:
        csv_path = paths.full_epoch / SUMMARY_CSV_NAME
        json_path = paths.full_epoch / SUMMARY_JSON_NAME
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(payload, indent=1, default=str))
        print(f"  saved {csv_path.relative_to(paths.root)}")
        print(f"  saved {json_path.relative_to(paths.root)}")
    _print_summary(df, group_stats)
    return df, payload


def _print_summary(df, group_stats):
    """Leads with the diagnostic, because it decides what the rest means."""
    subjects = df[df.subject != 'GROUP']
    group = df[df.subject == 'GROUP']
    print(f"\n{'=' * 78}\nFull-epoch decoding, window {FULL_EPOCH_WINDOW}s\n{'=' * 78}")

    for definition in DEFINITIONS:
        cols = [f'acc_mean_{definition}_{m}' for m in MODES
                if f'acc_mean_{definition}_{m}' in df.columns]
        if not cols:
            continue
        print(f"\n--- definition '{definition}' "
              f"({'majority vote of 2s windows' if definition == 'vote' else 'one window per trial'}) ---")
        show = ['subject'] + cols + [c for c in df.columns
                                     if c.startswith('delta_') and c.endswith(definition)]
        with pd.option_context('display.width', 250, 'display.max_columns', 60):
            print(df[[c for c in show if c in df.columns]].to_string(
                index=False, float_format=lambda v: f'{v:.3f}'))

    print(f"\n{'=' * 78}\nDIAGNOSTIC - what each difference isolates\n{'=' * 78}")
    for definition in DEFINITIONS:
        print(f"\n[{definition}]")
        for name, (left, right) in MODE_DELTAS.items():
            col = f'delta_{name}_{definition}'
            if col not in subjects.columns:
                continue
            d = subjects[col].dropna()
            if not len(d):
                continue
            n_pos = int((d > 0).sum())
            print(f"  {name:<36} ({left} - {right}): {d.mean():+.4f} "
                  f"+/- {d.std(ddof=1):.4f}   {n_pos}/{len(d)} positive")

    if len(group):
        print(f"\n{'=' * 78}\nGroup (between-subject spread)\n{'=' * 78}")
        for key, stats in group_stats.items():
            print(f"  {key:<20} {stats['mean']:.3f} +/- {stats['between_sd']:.3f}  "
                  f"(SEM {stats['sem']:.3f}, F1 {stats['f1_macro']:.3f}, "
                  f"n={stats['n_subjects']})")


def save_group_figures(entries, definition, mode, n_classes, paths=None):
    """Group accuracy bar chart and pooled confusion matrix for one cell."""
    paths = paths or project_paths()
    out_dir = _out_dir(paths.group_figures, definition, mode)
    group_results = []
    for subject, entry in entries.items():
        add_subject_to_group_full_epoch(
            group_results, subject, entry['fold_accuracies'],
            entry['fold_confusion_matrices'], tmin=FULL_EPOCH_WINDOW[0],
            tmax=FULL_EPOCH_WINDOW[1], params_dict=entry['params'])
    _capture_figures(plot_accuracy_group_full_epoch, out_dir / 'accuracy_group.png', paths,
                     group_results_full_epoch=group_results, n_classes=n_classes,
                     figsize=(10, 5))
    _capture_figures(plot_average_confusion_full_epoch_group,
                     out_dir / 'confusion_group.png', paths,
                     group_results_full_epoch=group_results, normalize=True, figsize=(8, 7))
    return group_results


def summarize_full_epoch(results_by_cell=None, subject_meta=None, failures=None,
                         params_dict=None, paths=None, save_figs=True, write=True,
                         label=None):
    """Group statistics, group figures and the summary table."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    if results_by_cell is None:
        results_by_cell, subject_meta = load_full_epoch_results(paths)
    if not any(results_by_cell.values()):
        print("No full-epoch results found. Run run_full_epoch_batch() first.")
        return None, None

    # Merge with what is already on disk and persist, ALWAYS - this must not sit
    # behind save_figs, or a run with save_figs=False would silently skip saving
    # the group results and drop every subject it did not recompute.
    for (definition, mode), entries in results_by_cell.items():
        if not entries:
            continue
        name = GROUP_RESULTS_NAME.format(definition=definition, mode=mode)
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
        # subjects carried in from disk would otherwise have blank rows
        for e in merged:
            subject = e.get('subject_name')
            if subject and subject not in entries:
                entries[subject] = {
                    'fold_accuracies': e['fold_accuracies'],
                    'fold_confusion_matrices': e['fold_confusion_matrices'],
                    'params': {}, 'definition': definition, 'mode': mode,
                    'classes': list(e.get('desired_events') or []),
                    'n_classes': len(e.get('desired_events') or []) or None}

    if save_figs:
        with _batch_backend():
            for (definition, mode), entries in results_by_cell.items():
                if not entries:
                    continue
                n_classes = (next(iter(entries.values())).get('n_classes')
                             or len(DESIRED_EVENTS))
                save_group_figures(entries, definition, mode, n_classes, paths)

    return build_summary(results_by_cell, subject_meta or {}, failures, params_dict,
                         paths, write)


def load_full_epoch_results(paths=None, label=None, params_dict=None):
    """Reload previously saved group results, keyed by (definition, mode)."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    results_by_cell, subject_meta = {}, {}
    for definition in DEFINITIONS:
        for mode in MODES:
            path = paths.metrics_group / GROUP_RESULTS_NAME.format(
                definition=definition, mode=mode)
            entries = {}
            if path.exists():
                for entry in load_group_results_full_epoch(path):
                    subject = entry['subject_name']
                    entries[subject] = {
                        'fold_accuracies': entry['fold_accuracies'],
                        'fold_confusion_matrices': entry['fold_confusion_matrices'],
                        'params': {}, 'n_classes': len(entry.get('desired_events') or []) or None}
                    subject_meta.setdefault(subject, {}).setdefault(
                        'n_classes', entries[subject]['n_classes'])
            results_by_cell[(definition, mode)] = entries
    return results_by_cell, subject_meta


#%%
# ============================================================
# Batch driver and the opt-in permutation pass
# ============================================================


def run_full_epoch_batch(subjects=None, definitions=None, modes=None, force=False,
                         save_figs=True, batch_backend='Agg', params_dict=None,
                         paths=None, seed=RANDOM_SEED,
                         balance_trigger_class=BALANCE_TRIGGER_CLASS, label=None):
    """
    Full-epoch decoding for every subject, then summarise.

    A failing subject is reported and skipped, never fatal. The whole grid is
    computed per subject in one pass so that every cell shares the same epochs,
    balancing selection and folds - which is what makes the deltas exactly paired.
    """
    paths = paths or project_paths(label=label, params_dict=params_dict)
    print(f"output label: '{paths.label}'  ->  "
          f"{paths.full_epoch.relative_to(paths.root)}")
    subjects = (list_subjects(paths) if subjects is None
                else [subjects] if isinstance(subjects, str) else list(subjects))
    definitions = list(definitions or DEFINITIONS)
    modes = list(modes or MODES)

    results_by_cell = {(d, m): {} for d in definitions for m in modes}
    subject_meta, failures = {}, {}

    with _batch_backend(batch_backend):
        for i, subject in enumerate(subjects, 1):
            print(f"\n{'=' * 70}\n[{i}/{len(subjects)}] {subject}\n{'=' * 70}")
            try:
                results = run_subject_full_epoch(
                    subject, definitions, modes, params_dict, paths, force, save_figs,
                    seed, balance_trigger_class)
                for (definition, mode), entry in results.items():
                    results_by_cell[(definition, mode)][subject] = entry
                    add_subject_to_group_full_epoch(
                        [], subject, entry['fold_accuracies'],
                        entry['fold_confusion_matrices'],
                        tmin=FULL_EPOCH_WINDOW[0], tmax=FULL_EPOCH_WINDOW[1],
                        params_dict=entry['params'],
                        save_dir=_out_dir(paths.metrics_individual, definition, mode))
                    subject_meta.setdefault(subject, {}).update(
                        {k: entry[k] for k in
                         ('n_files', 'n_epochs', 'n_epochs_before', 'n_channels',
                          'dropped_bads', 'classes', 'n_classes', 'counts_after',
                          'balanced', 'majority_baseline')})
            except Exception as err:
                for definition in definitions:
                    for mode in modes:
                        failures[f'{subject}/{definition}/{mode}'] = (
                            f'{type(err).__name__}: {err}')
                print(f"  !! FAILED [{subject}]: {type(err).__name__}: {err}")
            finally:
                plt.close('all')

    if failures:
        rerun = sorted({k.split('/')[0] for k in failures})
        print(f"\nRerun the failures with run_full_epoch_batch({rerun}).")

    for entries in results_by_cell.values():
        for subject, entry in entries.items():
            if subject not in subject_meta:
                subject_meta[subject] = _meta_from_cache(
                    subject, {'desired_events': entry.get('classes')
                              or (DESIRED_EVENTS if entry.get('n_classes') == 4 else [])},
                    paths)

    df, payload = summarize_full_epoch(results_by_cell, subject_meta, failures,
                                       params_dict, paths, save_figs)
    return {'results_by_cell': results_by_cell, 'subject_meta': subject_meta,
            'failures': failures, 'summary': df, 'payload': payload}


def run_permutation_pass(subjects=None, definition='vote', mode='c2c',
                         n_permutations=PERMUTATION_N, params_dict=None, paths=None,
                         seed=RANDOM_SEED, balance_trigger_class=BALANCE_TRIGGER_CLASS,
                         label=None):
    """
    Per-subject permutation p-values for ONE cell. Opt-in: this is the expensive pass.

    Defaults to the reported cell ('c2c'/'vote'); pass mode='l2u' for the leak-free
    deployable one. Any DIAGNOSTIC_MODES key works, whether or not the diagnostic
    itself was run. n_permutations >= 200 is needed for a usable p-value.
    """
    paths = paths or project_paths(label=label, params_dict=params_dict)
    subjects = (list_subjects(paths) if subjects is None
                else [subjects] if isinstance(subjects, str) else list(subjects))
    if 'fold' in DIAGNOSTIC_MODES[mode]:
        print(f"  note: '{mode}' uses fold-local centering, which run_permutation_test "
              f"does not reproduce; permuting the pooled equivalent instead.")

    out = {}
    with _batch_backend():
        for i, subject in enumerate(subjects, 1):
            print(f"\n[{i}/{len(subjects)}] {subject} - {n_permutations} permutations")
            try:
                epochs_by_variant, events, params_by_variant, _ = _prepare_subject(
                    subject, params_dict, paths, False, seed, balance_trigger_class)
                params = full_epoch_params(definition, params_by_variant[EPOCH_VARIANTS[0]])
                cropped_by_variant = {
                    v: e.copy().crop(params['classifier_window_s'],
                                     params['classifier_window_e'])
                    for v, e in epochs_by_variant.items()}
                cache = {}
                cropped, _ = _static_source(
                    DIAGNOSTIC_MODES[mode][0] if DIAGNOSTIC_MODES[mode][0] != 'fold' else 'global',
                    epochs_by_variant, cropped_by_variant, events[:, 2], cache)
                _, full = _static_source(
                    DIAGNOSTIC_MODES[mode][1] if DIAGNOSTIC_MODES[mode][1] != 'fold' else 'global',
                    epochs_by_variant, cropped_by_variant, events[:, 2], cache)
                result = run_permutation_test(
                    full, cropped, params,
                    n_permutations=n_permutations, score_method='majority_vote',
                    eval_tmin=FULL_EPOCH_WINDOW[0], eval_tmax=FULL_EPOCH_WINDOW[1],
                    random_state=seed)
                true_score, perm_scores, p_value = result[0], result[1], result[2]
                out[subject] = {'true_score': float(true_score), 'p_value': float(p_value),
                                'n_permutations': int(n_permutations),
                                'perm_mean': float(np.mean(perm_scores))}
                print(f"  {subject}: score {true_score:.3f}, p = {p_value:.4g}")
            except Exception as err:
                out[subject] = {'error': f'{type(err).__name__}: {err}'}
                print(f"  !! FAILED [{subject}]: {out[subject]['error']}")
            finally:
                plt.close('all')

    path = paths.full_epoch / f'permutation_{definition}_{mode}.json'
    path.write_text(json.dumps(
        {'definition': definition, 'mode': mode, 'n_permutations': n_permutations,
         'window': list(FULL_EPOCH_WINDOW), 'subjects': out}, indent=1))
    print(f"\n  saved {path.relative_to(paths.root)}")
    return out


def full_epoch_status(paths=None, verbose=True, label=None, params_dict=None):
    """Per-subject progress across the definition x mode grid."""
    paths = paths or project_paths(label=label, params_dict=params_dict)
    rows = []
    for subject in list_subjects(paths):
        row = {'subject': subject, 'n_files': len(subject_files(subject, paths))}
        for definition in DEFINITIONS:
            for mode in MODES:
                d = paths.metrics_individual / definition / mode
                row[f'{definition[:4]}_{mode}'] = bool(
                    list(d.glob(f'{subject}_*.json'))) if d.exists() else False
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        print(f"FullEpoch root: {paths.full_epoch}   (label '{paths.label}')")
        print(df.to_string(index=False))
    return df
