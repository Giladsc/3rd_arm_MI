#%%
"""
Shared machinery for the per-subject batch analyses.

``windowed_batch`` and ``full_epoch_batch`` are peers: both preprocess the same
subjects the same way, cache the same epochs, balance them the same way and
cross-validate over the same folds. Everything that is genuinely common lives
here so neither analysis has to import the other, and so a change to the
preprocessing contract cannot silently apply to only one of them.

Nothing analysis-specific belongs in this module - no evaluation modes, no summary
tables, no plots of a particular metric.

OUTPUT LAYOUT
-------------
One tree per class set, both analyses inside it, cache shared between them:

    Analysis/<label>/
        Cache/              <subject>_<variant>-epo.fif (+ _meta.json)
        Windowed/           Figures/, Metrics/, windowed_summary.{csv,json}
        FullEpoch/          Figures/, Metrics/, full_epoch_summary.{csv,json}

``<label>`` identifies the class set (see run_label), so runs with different
``desired_events`` cannot overwrite, or silently reuse, each other's results.

NOTE: importing this module imports ``src.preprocessing``, which runs
``%matplotlib qt`` at import time, so an IPython kernel is required.
"""

import copy
import json
import re
import subprocess
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from pyxdf import load_xdf
from sklearn.model_selection import RepeatedStratifiedKFold

from . import params_spec
from .group_analysis import _get_classes_initials
from .preprocessing import EVENT_LABEL_ALIASES, get_subject_bad_electrodes
from .tfr_batch import ELECTRODE_GROUPS as _TFR_ELECTRODE_GROUPS
# list_subjects / subject_files are re-exported for the analysis modules and
# their notebooks, which import them from here rather than from tfr_batch.
from .tfr_batch import build_subject_epochs, list_subjects, subject_files  # noqa: F401

#%%
# ============================================================
# Configuration shared by every analysis
# ============================================================

ELECTRODE_GROUPS = dict(_TFR_ELECTRODE_GROUPS)
ELECTRODE_GROUPS['FC'] = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6']

ELECTRODE_GROUP_NAMES = 'FC+C+CP+P'
DESIRED_EVENTS = ['MiddleHand', 'LeftHand', 'RightHand', 'FixatedRest']

# Balancing applies only when this class is present, and subsamples it down to the
# smallest other class. 'Rest' is the over-represented class in the paradigms that
# have it, which is what the original class_to_subsample='Rest' was written for.
# The 4-class MI config has no 'Rest', so it runs unbalanced by design.
BALANCE_TRIGGER_CLASS = 'Rest'

# Two full preprocessing runs per subject; what the epoch CACHE is keyed on. A
# VARIANT is a preprocessing setting, not an evaluation mode: subject_params sets
# CenterByClass from the variant name, and each analysis module pairs the two
# variants into its own <train>2<test> modes (see the MODES dict in windowed_batch,
# full_epoch_batch and session_batch). Nothing about those modes belongs here.
EPOCH_VARIANTS = ('centered', 'uncentered')

CV_N_SPLITS = 5
CV_N_REPEATS = 2
CV_RANDOM_STATE = 22
RANDOM_SEED = 22          # np.random + random, both used unseeded upstream

FIGURE_DPI = 150

# Params that invalidate a cached epochs file when they change.
_META_PARAM_KEYS = ('PerformCsd', 'PerformAvgRef', 'AddRefChannel', 'CenterByClass',
                    'filter_method', 'epoch_tmin', 'epoch_tmax', 'LowPass', 'HighPass',
                    'Electorde_Group', 'bad_electrodes', 'desired_events')

# Every params key the pipeline reads, grouped by how a NOTEBOOK should treat it.
# The notebooks spell the whole dict out in a parameters cell, so the distinction
# between "edit this" and "editing this does nothing" has to be stated somewhere the
# reader can see; describe_params prints these groupings and check_params uses the
# key set to turn a typo from a silent no-op into an error.
PARAM_GROUPS = {
    # Edit freely - these reach the pipeline exactly as written.
    'active': ('desired_events', 'Electorde_Group', 'PerformAvgRef', 'AddRefChannel',
               'CenterByClass', 'PerformCsd', 'filter_method', 'LowPass', 'HighPass',
               'epoch_tmin', 'epoch_tmax', 'classifier_window_s', 'classifier_window_e',
               'windowed_prediction_params', 'augmentation_params', 'pipeline_name'),
    # Read only by the CSP / FBCSP pipelines - inert while pipeline_name is 'ts+FGDA'.
    'pipeline': ('n_components', 'n_components_fbcsp', 'filters_bands'),
    # Written by the pipeline itself; an edit in a notebook is silently discarded.
    'auto': ('bad_electrodes', 'events_trigger_dict',
             'epoch_tmins_and_maxes_grid', 'sfreq'),
}

PARAM_DOCS = {
    'desired_events': 'classes to decode; also names the Analysis/<label>/ tree',
    'Electorde_Group': 'channel picks, and the order the classifier sees them in',
    'PerformAvgRef': 'average re-reference',
    'AddRefChannel': 'reconstruct FCz (the online reference); requires PerformAvgRef',
    'CenterByClass': 'per-class mean removal; each epoch variant overrides it per build',
    'PerformCsd': 'current-source-density transform',
    'filter_method': "MNE filter method ('iir' / 'fir')",
    'LowPass': 'band-pass low edge, Hz',
    'HighPass': 'band-pass high edge, Hz',
    'epoch_tmin': 'epoch crop start, s relative to cue',
    'epoch_tmax': 'epoch crop end, s',
    'classifier_window_s': 'training crop start, s',
    'classifier_window_e': 'training crop end, s',
    'windowed_prediction_params': 'prediction sliding window {win_len, win_step}, s',
    'augmentation_params': 'training-set sliding window; win_len 0 disables it',
    'pipeline_name': 'classifier pipeline; also part of the saved metric filenames',
    'n_components': 'CSP components (CSP pipelines only)',
    'n_components_fbcsp': 'FBCSP components (fbcsp+lda only)',
    'filters_bands': 'FBCSP filter bank, Hz (fbcsp+lda only)',
    'bad_electrodes': 'per subject, from get_subject_bad_electrodes - set by subject_params',
    'events_trigger_dict': "per subject, from epochs.event_id - set by the batch",
    'epoch_tmins_and_maxes_grid': 'vestigial: read nowhere in this repo',
    'sfreq': 'sampling rate fallback, read by one training branch only',
}

KNOWN_PARAM_KEYS = frozenset(key for group in PARAM_GROUPS.values() for key in group)

_PARAM_GROUP_TITLES = {
    'active': 'ACTIVE',
    'pipeline': "NOT USED BY PIPELINE {pipeline!r}",
    'auto': 'SET AUTOMATICALLY - edits here are discarded',
}


#%%
# ============================================================
# Run label, paths and parameters
# ============================================================

def run_label(params_dict=None, label=None):
    """
    Directory-safe tag identifying a run's class set, e.g. 'MH_LH_RH_FR' or 'LH_RH'.

    Every output of a run is nested under this, so runs with different
    ``desired_events`` cannot overwrite or silently reuse each other's results.

    Built with ``group_analysis._get_classes_initials``, the same helper that already
    names the per-subject files inside ``Metrics/Individuals/``, so the folder and the
    filenames inside it agree.

    Derived from the **requested** ``desired_events``, not the per-subject resolved
    list: a subject missing one class must still land in the run's folder rather than
    scattering into one of its own. Pass ``label`` to override - useful when the same
    classes are run with different filters or electrode groups.
    """
    if label:
        return str(label)
    events = (params_dict or default_params())['desired_events']
    return _get_classes_initials(list(events))


def project_paths(root=None, label=None, params_dict=None, analysis=None):
    """
    Resolve this run's directories, creating the output ones if needed.

    The root is derived from this file's location rather than the working directory,
    so it is identical whether the caller runs from ``notebooks/`` or the repo root.

    Layout is one tree per class set, with both analyses inside it::

        Analysis/<label>/Cache/            shared by every analysis
        Analysis/<label>/Windowed/...
        Analysis/<label>/FullEpoch/...

    The cache sits beside the analyses rather than inside one of them, because both
    consume the identical preprocessed epochs and neither owns them.

    Without ``analysis`` you get the label-level paths (``analysis_root``, ``cache``);
    with ``analysis='Windowed'`` you additionally get ``out``, ``figures``,
    ``group_figures``, ``metrics_individual`` and ``metrics_group`` beneath it.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    label = run_label(params_dict, label)
    analysis_root = root / 'Analysis' / label
    paths = SimpleNamespace(
        root=root,
        recordings=root / 'Recordings',
        label=label,
        analysis=analysis,
        analysis_root=analysis_root,
        cache=analysis_root / 'Cache',
    )
    made = [paths.cache]
    if analysis:
        out = analysis_root / analysis
        paths.out = out
        paths.figures = out / 'Figures'
        paths.group_figures = out / 'Figures' / 'Group'
        paths.metrics_individual = out / 'Metrics' / 'Individuals'
        paths.metrics_group = out / 'Metrics' / 'Group'
        made += [paths.figures, paths.group_figures,
                 paths.metrics_individual, paths.metrics_group]
    for out_dir in made:
        out_dir.mkdir(parents=True, exist_ok=True)
    return paths


def subject_figure_dir(subject, variant, paths=None):
    """``Benchmark/<label>/Figures/<subject>/<variant>/``, created on demand."""
    paths = paths or project_paths()
    out_dir = paths.figures / subject / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _variant_dir(parent, variant):
    out_dir = parent / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def default_params(electrode_group_names=ELECTRODE_GROUP_NAMES, desired_events=None):
    """
    The benchmark's parameters, inherited from notebooks/Main_Experiment.ipynb.

    ``bad_electrodes`` is left empty here and filled in per subject by
    ``subject_params``, which is the only sanctioned way to build a per-subject dict.
    """
    params_dict = {}
    params_dict['PerformCsd'] = False
    params_dict['PerformAvgRef'] = True
    params_dict['AddRefChannel'] = True     # reconstruct FCz, the online reference
    params_dict['CenterByClass'] = True     # per-class mean removal; the epoch
                                            # variants override it per build
    params_dict['Electorde_Group'] = [
        elec for group in electrode_group_names.split('+')
        for elec in ELECTRODE_GROUPS[group]
    ]
    params_dict['bad_electrodes'] = {}
    params_dict['filter_method'] = 'iir'
    params_dict['epoch_tmins_and_maxes_grid'] = [-5, 6]
    params_dict['epoch_tmin'] = -5
    params_dict['epoch_tmax'] = 6
    params_dict['n_components'] = 8
    params_dict['LowPass'] = 8
    params_dict['HighPass'] = 32
    params_dict['filters_bands'] = [[7, 12], [12, 20], [20, 28], [28, 35]]
    params_dict['augmentation_params'] = {'win_len': 0, 'win_step': 0.25}
    params_dict['classifier_window_s'] = 0.2
    params_dict['classifier_window_e'] = 4
    params_dict['windowed_prediction_params'] = {'win_len': 2, 'win_step': 0.25}
    params_dict['pipeline_name'] = 'ts+FGDA'
    params_dict['n_components_fbcsp'] = 8
    params_dict['desired_events'] = list(desired_events or DESIRED_EVENTS)
    return params_dict


def subject_params(subject, variant, params_dict=None, paths=None):
    """
    A private params dict for one subject and one EPOCH VARIANT.

    ``variant`` is a preprocessing setting ('centered'/'uncentered'), not an
    evaluation mode - the modes pair these up afterwards.

    Deep-copies before touching anything: ``bad_electrodes``, ``events_trigger_dict``
    and ``desired_events`` are all per-subject, and a shared dict would let one
    subject silently contaminate the next. ``classifier_training`` inverts
    ``events_trigger_dict`` to build the label space, so a stale one produces a wrong
    label space rather than an error.

    ``CenterByClass`` is the one ACTIVE param this overrides: it IS the variant, so a
    params cell supplies the default (and the value single-variant callers get) while
    the variant being built wins here. Honouring the cell value instead would build
    both variants identically and collapse every cross-centering mode into u2u
    without an error - ``check_params`` warns when a cell sets it to anything but True.
    """
    if variant not in EPOCH_VARIANTS:
        raise ValueError(f"variant must be one of {EPOCH_VARIANTS}, got {variant!r}")
    params = copy.deepcopy(params_dict or default_params())
    params['bad_electrodes'] = get_subject_bad_electrodes(subject)
    params['CenterByClass'] = (variant == 'centered')
    return params


# Keys whose value is too long to print, and the unit to count them in.
_PARAM_SUMMARIZE = {'Electorde_Group': 'ch'}


def describe_params(params_dict=None):
    """
    Print every params key, grouped, with a ``*`` where it differs from the default.

    The notebooks spell the whole dict out in a parameters cell, so this is the
    readback: it names the keys the pipeline actually reads, says which of them the
    current ``pipeline_name`` ignores, and which are overwritten downstream - none of
    which is visible from the cell itself.
    """
    params_dict = params_dict if params_dict is not None else default_params()
    params_spec.describe(
        params_dict, PARAM_GROUPS, PARAM_DOCS,
        defaults=default_params(), titles=_PARAM_GROUP_TITLES,
        known=KNOWN_PARAM_KEYS, summarize=_PARAM_SUMMARIZE,
        pipeline=params_dict.get('pipeline_name'))
    print("  ('*' = differs from default_params())")


def _recorded_params(paths):
    """
    The params that produced the tree already at ``paths``, or ``{}`` if it is fresh.

    Two sources, because neither is complete on its own: the epoch cache's meta
    carries the _META_PARAM_KEYS for any run that ever built epochs, while an
    analysis summary carries the whole dict but only once one has been written.
    The summary wins where both have a key.
    """
    def _first(directory, pattern, extract):
        for path in sorted((directory or Path()).glob(pattern) if directory else []):
            try:
                return extract(json.loads(path.read_text())) or {}
            except (OSError, ValueError, AttributeError):
                continue
        return {}

    recorded = _first(getattr(paths, 'cache', None), '*_meta.json',
                      lambda blob: blob.get('params'))
    recorded.update(_first(getattr(paths, 'out', None), '*_summary.json',
                           lambda blob: (blob.get('run') or {}).get('params')))
    return recorded


def check_params(params_dict, label=None, paths=None):
    """
    Validate a notebook's params cell and return it, so the call can be chained.

    Raises on the mistakes that are otherwise silent or expensive: a misspelled key
    (which today is simply ignored), and combinations that either fail deep inside
    preprocessing or produce a nonsensical crop.

    Warns - rather than raises - when the params differ from the ones that produced
    the tree already on disk while ``label`` is None. That is the real hazard, since
    ``run_label`` names the tree from ``desired_events`` alone: the group JSONs are
    merged, not replaced (``_merge_group_results``), and ``run_windowed_batch``
    reuses per-subject entries whose only guard is the class list. The comparison is
    against what was actually recorded rather than against ``default_params()``, so a
    cell that deliberately departs from the library default does not nag every run.
    """
    unknown = params_spec.unknown_keys(params_dict, KNOWN_PARAM_KEYS)
    if unknown:
        raise ValueError(f"params keys read by nothing: {', '.join(unknown)}. "
                         f"A misspelled key is silently ignored, so this is an error.")

    if params_dict.get('AddRefChannel') and not params_dict.get('PerformAvgRef'):
        raise ValueError("AddRefChannel=True requires PerformAvgRef=True: without "
                         "average referencing the reconstructed FCz is all zeros.")
    if params_dict['HighPass'] <= params_dict['LowPass']:
        raise ValueError(f"HighPass ({params_dict['HighPass']}) must exceed LowPass "
                         f"({params_dict['LowPass']}).")
    if params_dict['epoch_tmin'] >= params_dict['epoch_tmax']:
        raise ValueError(f"epoch_tmin ({params_dict['epoch_tmin']}) must precede "
                         f"epoch_tmax ({params_dict['epoch_tmax']}).")
    win_s, win_e = params_dict['classifier_window_s'], params_dict['classifier_window_e']
    if win_s >= win_e:
        raise ValueError(f"classifier_window_s ({win_s}) must precede "
                         f"classifier_window_e ({win_e}).")
    if win_s < params_dict['epoch_tmin'] or win_e > params_dict['epoch_tmax']:
        raise ValueError(f"classifier window [{win_s}, {win_e}] falls outside the epoch "
                         f"[{params_dict['epoch_tmin']}, {params_dict['epoch_tmax']}].")
    if len(params_dict['desired_events']) < 2:
        raise ValueError(f"need at least 2 desired_events to classify, got "
                         f"{params_dict['desired_events']}.")
    if not params_dict['Electorde_Group']:
        raise ValueError("Electorde_Group is empty: no channels would be picked.")

    defaults = default_params()
    overwritten = [k for k in PARAM_GROUPS['auto']
                   if k in params_dict and k in defaults and defaults[k] != params_dict[k]]
    if overwritten:
        print(f"  !! {overwritten} are set by the pipeline per subject; the values in "
              f"the params cell are discarded.")

    # CenterByClass is ACTIVE (it is a real preprocessing switch and reaches
    # EEG_Preprocessing as written), but this stack builds BOTH epoch variants and
    # subject_params sets the flag from the variant - so a cell that turns it off here
    # would be honoured nowhere. Said out loud rather than left as a silent no-op.
    if params_dict.get('CenterByClass', True) is not True:
        print(f"  !! CenterByClass={params_dict['CenterByClass']!r} is overridden per epoch "
              f"variant here: {EPOCH_VARIANTS} are both built regardless, and the modes "
              f"pair them.\n"
              f"     It takes effect only for single-variant callers "
              f"(EEG_Preprocessing directly, e.g. the Main_* notebooks).")

    if label is None and paths is not None:
        recorded = _recorded_params(paths)
        # desired_events already names the tree, and the cache records the per-subject
        # RESOLVED class list, so a subject missing a class would always look changed.
        # CenterByClass is excluded for the same reason: the cache meta records the
        # PER-VARIANT value, so an 'uncentered' meta would always look changed.
        comparable = [k for k in PARAM_GROUPS['active'] + PARAM_GROUPS['pipeline']
                      if k not in ('desired_events', 'CenterByClass')]
        changed = params_spec.changed_against(params_dict, recorded, comparable)
        if changed:
            print(f"  !! {paths.label!r} was built with different params: {changed}.\n"
                  f"     Results merge by subject rather than replace, so this run would "
                  f"mix the two.\n"
                  f"     Set LABEL to a new name, or FORCE=True (and SKIP_DONE=False) to "
                  f"rebuild the tree.")
    return params_dict



#%%
# ============================================================
# Event discovery - what classes does this subject actually have?
# ============================================================

def probe_subject_events(subject, paths=None):
    """
    Marker labels present in each of a subject's recordings.

    Reads only the Markers stream, so this costs milliseconds instead of loading
    every EEG sample. Applies the same alias map as ``standardize_event_labels``
    (ClosePalm -> MiddleHand), so legacy recordings report their canonical labels.
    """
    paths = paths or project_paths()
    found = {}
    for xdf_file in subject_files(subject, paths):
        streams, _ = load_xdf(str(xdf_file), select_streams=[{'type': 'Markers'}])
        labels = set()
        for stream in streams:
            for sample in stream['time_series']:
                label = sample[0] if isinstance(sample, (list, tuple, np.ndarray)) else sample
                labels.add(EVENT_LABEL_ALIASES.get(str(label), str(label)))
        found[xdf_file.name] = labels
    return found


def resolve_subject_events(subject, params_dict, paths=None):
    """
    The desired events this subject actually has, in DESIRED_EVENTS order.

    Intersected across all of the subject's recordings, because ``EEG_Preprocessing``
    iterates ``desired_events`` and indexes ``epochs[event]`` per file - a class
    missing from one file raises there. Narrowing up front turns a mid-run KeyError
    into an explicit, recorded class list.
    """
    per_file = probe_subject_events(subject, paths)
    common = set.intersection(*per_file.values()) if per_file else set()
    resolved = [ev for ev in params_dict['desired_events'] if ev in common]
    if len(resolved) < 2:
        raise RuntimeError(
            f"[{subject}] only {len(resolved)} of the desired classes "
            f"{params_dict['desired_events']} are present in every recording "
            f"(found per file: { {k: sorted(v) for k, v in per_file.items()} }). "
            f"Need at least 2 to classify.")
    if len(resolved) < len(params_dict['desired_events']):
        missing = [ev for ev in params_dict['desired_events'] if ev not in resolved]
        print(f"  !! [{subject}] missing {missing} in at least one recording; "
              f"benchmarking on {resolved}")
    return resolved



#%%
# ============================================================
# Epoch cache (shared between the analyses)
# ============================================================

def _cache_paths(subject, variant, paths=None):
    """Cache file paths for one (subject, variant)."""
    paths = paths or project_paths()
    # The -epo suffix follows MNE's convention so save/read don't warn.
    return SimpleNamespace(
        epochs=paths.cache / f'{subject}_{variant}-epo.fif',
        meta=paths.cache / f'{subject}_{variant}_meta.json',
    )


def _serializable_params(params_dict):
    """
    The cache-invalidating params, JSON-safe (bad_electrodes is a set).

    Deliberately narrow: this is the epoch cache's KEY, compared field by field in
    ``_check_meta``. Widening it would mark every cached .fif in the repo stale.
    Use ``_all_serializable_params`` for provenance, which has no such constraint.
    """
    out = {}
    for key in _META_PARAM_KEYS:
        value = params_dict.get(key)
        out[key] = sorted(value) if isinstance(value, (set, frozenset)) else value
    return out


def _all_serializable_params(params_dict):
    """
    The WHOLE params dict, JSON-safe - what provenance records.

    The notebooks now set every key in a parameters cell, so a summary that recorded
    only the cache key would omit the pipeline, the classifier window and the
    prediction window: exactly the settings a reader would want to check a number
    against. Sorted by key so two summaries diff cleanly.
    """
    return {key: sorted(value) if isinstance(value, (set, frozenset)) else value
            for key, value in sorted(params_dict.items())}


def _build_meta(subject, variant, files, epochs, params_dict):
    """Provenance for a cached epochs file."""
    return {
        'subject': subject,
        'variant': variant,
        'files': [{'name': f.name, 'size': f.stat().st_size} for f in files],
        'n_epochs': len(epochs),
        'class_counts': _class_counts(epochs),
        'ch_names': list(epochs.ch_names),
        'params': _serializable_params(params_dict),
        'built': date.today().isoformat(),
    }


def _check_meta(subject, meta, files, params_dict):
    """
    True when the cache is stale. Prints what changed.

    Unlike tfr_batch, which only warns because a human's recorded ICA decision is
    pinned to its cache, nothing here is pinned to the cached epochs - so the caller
    rebuilds silently, keeping a re-run correct after any params edit without anyone
    having to remember force=True.
    """
    reasons = []
    current_files = [{'name': f.name, 'size': f.stat().st_size} for f in files]
    if meta.get('files') != current_files:
        reasons.append('recordings changed')
    cached, current = meta.get('params', {}), _serializable_params(params_dict)
    changed = [k for k in _META_PARAM_KEYS if cached.get(k) != current.get(k)]
    if changed:
        reasons.append(f"params changed: {changed}")
    if reasons:
        print(f"  [{subject}] cache is stale ({'; '.join(reasons)}) - rebuilding")
    return bool(reasons)


def load_or_build_epochs(subject, variant, params_dict, paths=None, force=False):
    """
    Cached preprocessed epochs for one (subject, variant), building them if needed.

    Narrows ``params_dict['desired_events']`` in place to the classes this subject
    actually has before any EEG is loaded. Returns ``(epochs, meta)``.
    """
    paths = paths or project_paths()
    cache = _cache_paths(subject, variant, paths)
    files = subject_files(subject, paths)

    if not force and cache.epochs.exists() and cache.meta.exists():
        meta = json.loads(cache.meta.read_text())
        # the cached run's class list is what the cached epochs contain
        cached_events = meta.get('params', {}).get('desired_events')
        probe = dict(params_dict, desired_events=cached_events or params_dict['desired_events'])
        if not _check_meta(subject, meta, files, probe):
            params_dict['desired_events'] = list(cached_events)
            epochs = mne.read_epochs(cache.epochs, preload=True, verbose='error')
            # Caches written before recording provenance existed have no
            # 'recording' column; leave-one-recording-out cannot work without it
            # and it is not recoverable after concatenation, so rebuild.
            if epochs.metadata is None or 'recording' not in epochs.metadata:
                print(f"  [{subject}/{variant}] cache predates recording provenance "
                      f"- rebuilding")
            else:
                print(f"  [{subject}/{variant}] cache hit: {len(epochs)} epochs, "
                      f"{len(epochs.ch_names)} channels, classes {list(epochs.event_id)}")
                return epochs, meta

    params_dict['desired_events'] = resolve_subject_events(subject, params_dict, paths)
    epochs = build_subject_epochs(subject, params_dict, paths)
    meta = _build_meta(subject, variant, files, epochs, params_dict)
    epochs.save(cache.epochs, overwrite=True, verbose='error')
    cache.meta.write_text(json.dumps(meta, indent=1))
    print(f"  [{subject}/{variant}] cached {cache.epochs.relative_to(paths.root)}")
    return epochs, meta



#%%
# ============================================================
# Figure helpers
# ============================================================

def _savefig(fig, path, paths=None):
    """Save one figure and report its root-relative path."""
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    paths = paths or project_paths()
    print(f"  saved {path.relative_to(paths.root)}")
    return path


@contextmanager
def _batch_backend(backend='Agg'):
    """
    Render headlessly for the duration, then restore the previous backend.

    Also what makes _capture_figures work: under a non-interactive backend
    plt.show() is a no-op, so the figures the evaluation plotters create survive
    the call and can still be saved.
    """
    previous = matplotlib.get_backend()
    plt.close('all')
    if backend is not None and previous.lower() != backend.lower():
        matplotlib.use(backend, force=True)
    try:
        yield
    finally:
        plt.close('all')
        if backend is not None and previous.lower() != backend.lower():
            matplotlib.use(previous, force=True)


def _capture_figures(plot_fn, out_path, paths=None, **kwargs):
    """
    Call a plotter that shows-and-discards, and save whatever figures it created.

    Every plotter in src/evaluation.py calls plt.show() internally and none returns
    a figure handle or takes a save path. Under a non-interactive backend plt.show()
    does nothing, so the new figures are still in plt.get_fignums() afterwards and
    can be saved here - which is why this module never needs to modify evaluation.py.
    """
    if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'ps', 'svg', 'template'):
        raise RuntimeError(
            f"_capture_figures needs a non-interactive backend (got "
            f"{matplotlib.get_backend()}); wrap the call in _batch_backend().")
    out_path = Path(out_path)
    before = set(plt.get_fignums())
    value = plot_fn(**kwargs)
    new = [n for n in plt.get_fignums() if n not in before]
    for i, num in enumerate(new):
        fig = plt.figure(num)
        path = out_path if len(new) == 1 else out_path.with_name(
            f'{out_path.stem}_{i + 1}{out_path.suffix}')
        _savefig(fig, path, paths)
        plt.close(fig)
    if not new:
        print(f"  !! {plot_fn.__name__} produced no figure for {out_path.name}")
    return value



#%%
# ============================================================
# Trials: counts, balancing, folds, alignment
# ============================================================

def _class_counts(epochs):
    """Trial count per class actually present."""
    return {label: int(len(epochs[label])) for label in epochs.event_id}


def _class_indices(epochs):
    """Positional epoch indices per class.

    Derived from ``events[:, 2]`` rather than ``epochs[label].selection``, which
    indexes the pre-drop data and only coincides with position while nothing has
    been dropped.
    """
    codes = epochs.events[:, 2]
    return {label: np.where(codes == code)[0] for label, code in epochs.event_id.items()}


def _balance_indices(epochs, trigger_class=BALANCE_TRIGGER_CLASS):
    """
    Which epochs to keep so the classes are balanced. Returns ``(keep, balanced)``.

    Balancing applies **only when ``trigger_class`` is one of the classes**, and it
    subsamples that class down to the smallest other one. That is the paradigm the
    original ``class_to_subsample='Rest'`` was written for: 'Rest' is the
    over-represented class there. Configs without it - such as the 4-class MI set -
    run unbalanced by design.

    ``keep`` is None when nothing is dropped. Returning indices rather than epochs
    is what lets the caller apply the *same* selection to both centering variants;
    balancing them independently would break the trial alignment the cross-mode
    evaluations depend on.
    """
    if not trigger_class:
        return None, False
    counts = _class_counts(epochs)
    if trigger_class not in counts:
        # The 4-class MI config (MiddleHand/LeftHand/RightHand/FixatedRest) lands
        # here: no 'Rest' class, so nothing is balanced and every trial is kept.
        # Raw accuracy then sits against a per-subject majority baseline rather
        # than 1/n_classes, which is why macro F1 is reported beside it.
        return None, False

    if len(counts) < 2:
        print(f"  !! only {len(counts)} class present; not balancing")
        return None, False

    others = min(v for k, v in counts.items() if k != trigger_class)
    if counts[trigger_class] <= others:
        print(f"  !! '{trigger_class}' has {counts[trigger_class]} trials, not more than the "
              f"smallest other class ({others}); nothing to subsample, running unbalanced")
        return None, False

    # Subsample the over-represented trigger class down to the smallest other class.
    per_class = _class_indices(epochs)
    keep = np.sort(np.concatenate(
        [np.random.choice(idx, size=others, replace=False) if label == trigger_class else idx
         for label, idx in per_class.items()]))
    return keep, True


def _make_cv_split(epochs_cropped, events):
    """
    A fresh CV generator. Never store one: it is consumed on first use.

    RepeatedStratifiedKFold rather than StratifiedShuffleSplit so every trial is
    tested exactly CV_N_REPEATS times and none is skipped.
    """
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS,
                                 random_state=CV_RANDOM_STATE)
    return cv.split(epochs_cropped.get_data(), events[:, 2])


def _assert_variants_aligned(subject, epochs_by_variant):
    """
    The two centering variants must be the same trials in the same order.

    Everything the cross modes claim rests on this: a c2u score is only meaningful
    if row i of the centered copy and row i of the uncentered copy are the same
    trial. run_windowed_classification_aug_cv checks counts and labels, but not
    onsets - two different orderings with the same label sequence would slip past
    it, so check onsets here.
    """
    reference = None
    for variant, epochs in epochs_by_variant.items():
        signature = (len(epochs), tuple(epochs.events[:, 0]), tuple(epochs.events[:, 2]),
                     tuple(epochs.ch_names))
        if reference is None:
            reference, ref_variant = signature, variant
            continue
        if signature != reference:
            raise RuntimeError(
                f"[{subject}] epoch variants '{ref_variant}' and '{variant}' are not "
                f"trial-aligned (counts/onsets/labels/channels differ); the cross-mode "
                f"evaluations would compare different trials.")


def assert_caveat_modes(caveats, known_modes, module):
    """
    Every ``<x>2<y>`` token in a module's CAVEATS must be one of its own modes.

    The caveats are copied verbatim into the summary JSON, so they travel with the
    numbers - which makes a caveat naming a mode that does not exist a wrong claim
    on file rather than a typo in a comment. This module stays mode-agnostic: the
    caller passes its own mode names in.

    Called at import time by each analysis module, so the check costs one regex pass
    over a handful of strings and fails the moment a module is imported.
    """
    named = {token for text in caveats for token in re.findall(r'\b[a-z]2[a-z]\b', text)}
    unknown = sorted(named - set(known_modes))
    if unknown:
        raise AssertionError(
            f"{module}.CAVEATS names mode(s) {unknown} that are not defined in that "
            f"module; known modes are {sorted(known_modes)}. The caveats are written "
            f"into the summary JSON, so this would ship a claim about a mode nobody "
            f"can run.")


#%%
# ============================================================
# Metrics derived from pooled confusion matrices
# ============================================================

def _pooled_confusion(entry, t_start, t_end):
    """
    Confusion matrix summed over every fold and every window in [t_start, t_end].

    Rows are true classes, columns predicted - the orientation sklearn produces and
    the one _f1_macro_from_cms / _balanced_accuracy_from_cms assume.
    """
    cms = entry.get('folds_confusion_matrices_per_window')
    w_times = entry.get('w_times')
    if not cms or w_times is None:
        return None
    w_times = np.asarray(w_times, dtype=float)
    idx = np.where((w_times >= t_start) & (w_times <= t_end))[0]
    if len(idx) == 0:
        return None
    total = None
    for fold in cms:
        for wi in idx:
            cm = np.asarray(fold[wi][0], dtype=float)
            total = cm if total is None else total + cm
    return total


def _balanced_accuracy_from_cms(entry, t_start, t_end):
    """
    Mean per-class recall. Survives class imbalance in a way raw accuracy does not.
    """
    total = _pooled_confusion(entry, t_start, t_end)
    if total is None:
        return float('nan')
    with np.errstate(invalid='ignore', divide='ignore'):
        recalls = np.diag(total) / total.sum(axis=1)
    return float(np.nanmean(recalls))


def _f1_macro_from_cms(entry, t_start, t_end):
    """
    Macro F1 - the unweighted mean of the per-class F1 scores.

    Computed from the pooled confusion matrix rather than from predictions, which
    is why it needs no extra plumbing: the matrices are already collected per fold
    per window. Equivalent to sklearn's f1_score(average='macro') on the pooled
    counts. Reported beside raw accuracy because the 4-class config runs unbalanced,
    so accuracy sits against a per-subject majority baseline rather than 1/n_classes.
    """
    total = _pooled_confusion(entry, t_start, t_end)
    if total is None:
        return float('nan')
    with np.errstate(invalid='ignore', divide='ignore'):
        precision = np.diag(total) / total.sum(axis=0)
        recall = np.diag(total) / total.sum(axis=1)
        f1 = 2 * precision * recall / (precision + recall)
    return float(np.nanmean(np.where(np.isnan(f1), 0.0, f1)))


#%%
# ============================================================
# Bookkeeping
# ============================================================

def _run_provenance(params_dict, subjects, paths, **extra):
    """
    Timestamp, git commit, versions and config - so a summary can be traced back.

    Only records what every analysis shares. Anything analysis-specific (the mode
    table, the evaluation window) is passed in as ``extra`` by the caller, so this
    function never needs to know which analyses exist.
    """
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=paths.root,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = None
    import sklearn
    params_dict = params_dict or default_params()
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'git_commit': commit or 'unknown',
        'versions': {'mne': mne.__version__, 'sklearn': sklearn.__version__,
                     'numpy': np.__version__},
        'params': _all_serializable_params(params_dict),
        'cv': {'scheme': 'RepeatedStratifiedKFold', 'n_splits': CV_N_SPLITS,
               'n_repeats': CV_N_REPEATS, 'random_state': CV_RANDOM_STATE},
        'seed': RANDOM_SEED,
        'balance_trigger_class': BALANCE_TRIGGER_CLASS,
        # From the run's own params, not the module default: a notebook is free to
        # pick a different electrode set, and reporting the default there would
        # contradict the n_channels column beside it.
        'electrodes_requested': list(params_dict['Electorde_Group']),
        'n_channels_expected': len(params_dict['Electorde_Group']),
        'subjects': list(subjects),
        **extra,
    }


def _meta_from_cache(subject, entry, paths):
    """
    Rebuild the summary metadata for a subject whose result was REUSED.

    A reused subject never runs the worker, so nothing populates subject_meta and
    its summary row would come out empty. The cached epochs' _meta.json carries
    everything the row needs; the class list comes from the stored entry.
    """
    classes = list(entry.get('desired_events') or [])
    meta = {'classes': classes, 'n_classes': len(classes) or None}
    for variant in EPOCH_VARIANTS:
        cache = _cache_paths(subject, variant, paths)
        if not cache.meta.exists():
            continue
        cached = json.loads(cache.meta.read_text())
        counts = cached.get('class_counts') or {}
        meta.update({'n_files': len(cached.get('files') or []),
                     'n_epochs': cached.get('n_epochs'),
                     'n_channels': len(cached.get('ch_names') or []) or None,
                     'counts_after': counts,
                     'majority_baseline': (max(counts.values()) / sum(counts.values())
                                           if counts else None)})
        break
    return meta



def _merge_group_results(entries, path, loader):
    """
    Union this run's entries with whatever is already on disk, by subject.

    Without this, running a subset (``run_..._batch(['SK'])``) would rewrite the
    group file with only SK and silently drop the other subjects - which is exactly
    what the batch's own "Rerun the failures with ..." message invites you to do.
    Entries from this run win for the subjects it covers.

    ``loader`` is the matching reader (``load_group_results`` for the windowed
    analysis, ``load_group_results_full_epoch`` for the other).
    """
    merged = {}
    if path.exists():
        try:
            for existing in loader(path):
                merged[existing.get('subject_name')] = existing
        except Exception as err:
            print(f"  !! could not read {path.name} ({type(err).__name__}); "
                  f"writing only this run's subjects")
    for entry in entries:
        merged[entry.get('subject_name')] = entry
    return [merged[k] for k in sorted(merged)]
