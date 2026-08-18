#%%
"""
Batch per-subject TFR / ERD-ERS pipeline.

Choosing which ICA components to remove is the only step that needs a human, so the
pipeline is split into three stages and the manual decision is written to disk once:

    Stage 1  prepare_subject('AEH')            interactive, once per subject
             -> Cache/AEH_pre_ica_epochs.fif, Cache/AEH_ica.fif, Cache/AEH_iclabel.json
             -> Figures/AEH/ICA/{components.png, iclabel.png}
             -> shows the component topographies, the sources browser and the
                ICLabel summary so the components can be picked

    Stage 2  record_exclusions('AEH', [0, 4, 7], note='blinks + left temporal EMG')
             -> configs/ica_exclusions.json

    Stage 3  run_tfr_batch()                   unattended, all reviewed subjects
             -> TFRs/<subject>_<event>_tfr.h5
             -> Figures/<subject>/TFR/*.png, Figures/<subject>/Contrasts/*.png

Stage 3 reads the cached epochs, so the recorded component indices always refer to
exactly the data ICA was fitted on. Deleting Cache/ is safe; stage 1 rebuilds it.

Driven from notebooks/TFR_Analysis.ipynb. This module imports src.preprocessing, which
runs `%matplotlib qt` at import time, so it needs an IPython kernel (notebook or
`ipython`), not a bare `python` process.
"""

import hashlib
import itertools
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA, read_ica
from mne.time_frequency import tfr_multitaper

from . import params_spec
from .preprocessing import (
    read_raw_xdf,
    fix_channel_names_for_subject,
    standardize_event_labels,
    EEG_Preprocessing,
    remap_epoch_events_to_standard,
    standard_event_id,
    epochs_to_continuous_raw,
    make_figure_scrollable,
    iclabel_suggested_exclusions,
    plot_iclabel_summary,
    apply_ica_to_epochs,
)

#%%
# ============================================================
# Configuration — single source of truth for the TFR pipeline
# ============================================================

ELECTRODE_GROUPS = {
    'FP': ['Fp1', 'Fp2'],
    'AF': ['AF7', 'AF3', 'AFz', 'AF4', 'AF8'],
    'F':  ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
    'FC': ['FT9','FT7','FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6','FT8', 'FT10'],
    'C':  ['T7','C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','T8'],
    'CP': ['TP9','TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6','TP8', 'TP10'],
    'P':  ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    'PO': ['PO7', 'PO3', 'POz', 'PO4', 'PO8'],
    'O':  ['Oz', 'O2', 'O1', 'Iz']
}

# Every group, i.e. the full 64-channel montage
ELECTRODE_GROUP_NAMES = 'F+AF+FP+PO+O+FC+C+CP+P'

DESIRED_EVENTS = ['MiddleHand', 'RightHand', 'LeftHand', 'FixatedRest']


def default_tfr_params():
    """
    Everything about the TFR itself, as one dict a notebook can edit.

    Split from ``default_params`` (the PREPROCESSING dict) because the two invalidate
    different things: a preprocessing change rebuilds the epoch cache and voids the
    recorded ICA components, while a change in here only recomputes ``TFRs/*.h5`` and
    the figures. Keeping them apart is what lets ``compute_subject_tfrs`` tell a
    harmless re-run from one that needs a fresh ICA review.

    Pure layout (figsizes, fontsizes, colorbar padding, the plot_joint timefreqs) stays
    as module constants below: it changes nothing about the numbers or how they read.
    """
    return {
        # --- spectral estimation ---
        # Frequency resolution in MNE's multitaper TFR is set by the analysis window
        # length in SECONDS:  T = n_cycles / freq,  full bandwidth = time_bandwidth / T.
        #
        # n_cycles = freqs holds T fixed at 1.0 s, so every frequency gets the same
        # +/-2 Hz smoothing: 10 Hz bin -> 8-12 Hz, 20 Hz bin -> 18-22 Hz. Mu and beta
        # stay separable. The old linspace(2.6, 8.4) let T collapse 1.30 s -> 0.25 s,
        # which smeared the 20 Hz bin across 13.2-26.8 Hz and pushed the 34 Hz bin past
        # the 40 Hz low-pass -- i.e. "Alpha" and "Beta" were not separable bands at all.
        'freqs': np.arange(4, 35),      # 4-34 Hz; below ~4 Hz is 1 Hz high-pass roll-off
        'n_cycles': None,               # None -> = freqs, i.e. T = 1.0 s at every freq
        'time_bandwidth': 4.0,          # MNE's default, stated rather than implied
        'decim': 10,                    # 500 -> 50 Hz (20 ms); T=1 s still ~50x oversampled
        'n_jobs': -1,
        'lowpass': 40,                  # applied to the epochs before the TFR

        # --- baseline and windows ---
        # Trial timing: Rest (2-5 s) -> fixation (2.5 s + 0.5-1.5 s jitter) -> cue at
        # t=0. So t in [-3.0, 0) is fixation on EVERY trial. The baseline sits fully
        # inside it with >= 0.5 s (= T/2) of guard at each end, so no wavelet reaches
        # past t=0 (which would pull post-cue power into the reference) or back into Rest.
        'mode': 'logratio',             # log10(power / baseline mean)
        'baseline': (-2, -0.5),
        'active_window': (0.5, 2.5),    # motor-imagery window used for every topomap
        'bands': {'Mu': (8, 13), 'Beta': (13, 30)},

        # --- figures ---
        # Colour limits for 'logratio': a 50% ERD is log10(0.5) = -0.30 and a 2x ERS is
        # +0.30. The old +/-1.5 was carried over from mode='percent' and means +/-31x on
        # a log10 scale, so every panel rendered as flat mid-colour.
        'vlim': (-0.5, 0.5),
        # Percentile of |value| the two AUTOSCALED figures (the subject grid and the
        # grand-average panel) scale to; 'vlim' does not reach either of them. 100 is the
        # max, i.e. nothing clips and no panel's peak can be hidden. Lower it to pull the
        # limits in when a handful of edge channels are setting a scale that leaves every
        # other cell mid-colour.
        'autoscale_pct': 100,
        'cmap': 'RdBu_r',
        'figure_dpi': 150,

        # --- ICA (stage 1) ---
        'ica_n_components': None,       # None -> len(ch_names) - 1, the rank after avg ref
        'ica_decim': 3,
        'ica_random_state': 97,
        'ica_method': 'fastica',
        # ICLabel classes treated as artifactual when SUGGESTING exclusions. Only ever a
        # suggestion - the decision recorded by record_exclusions is what stage 3 applies.
        'iclabel_exclude_labels': ['eye blink', 'eye movement', 'muscle artifact'],
    }


TFR_PARAM_GROUPS = {
    'spectral': ('freqs', 'n_cycles', 'time_bandwidth', 'decim', 'lowpass', 'n_jobs'),
    'baseline': ('mode', 'baseline', 'active_window', 'bands'),
    'figures': ('vlim', 'autoscale_pct', 'cmap', 'figure_dpi'),
    'ica': ('ica_n_components', 'ica_decim', 'ica_random_state', 'ica_method',
            'iclabel_exclude_labels'),
}

TFR_PARAM_DOCS = {
    'freqs': 'frequencies to estimate, Hz',
    'n_cycles': 'cycles per frequency; None -> = freqs (T = 1 s at every freq)',
    'time_bandwidth': 'multitaper product; full smoothing bandwidth = this / T',
    'decim': 'temporal decimation applied to the TFR',
    'lowpass': 'low-pass applied to the epochs before the TFR, Hz',
    'n_jobs': 'parallel jobs for tfr_multitaper (-1 = all cores)',
    'mode': "baseline correction ('logratio', 'percent', 'zscore', ...)",
    'baseline': 'baseline window, s relative to cue; must end at or before 0',
    'active_window': 'window every topomap and band map averages over, s',
    'bands': 'named frequency bands for the band maps, Hz',
    'vlim': 'colour limits for the fixed-scale figures',
    'autoscale_pct': 'percentile of |value| the autoscaled group figures scale to '
                     '(100 = max, nothing clips)',
    'cmap': 'colormap for the group band maps',
    'figure_dpi': 'resolution every figure is saved at',
    'ica_n_components': 'None -> len(ch_names) - 1 (one rank lost to the average ref)',
    'ica_decim': 'decimation while FITTING ICA',
    'ica_random_state': 'ICA seed; changing it reorders components',
    'ica_method': "ICA algorithm ('fastica', 'infomax', 'picard')",
    'iclabel_exclude_labels': 'ICLabel classes offered as suggested exclusions',
}

# What each apply_baseline mode is called on a colorbar, so a figure states the mode it
# was actually made in rather than a hard-coded one. Note 'percent' is MNE's name for a
# FRACTION - (power - baseline) / baseline, so a 50% ERD is -0.5 - which is why it is not
# labelled '%': that would misread every axis by 100x. Keep these short; they go into the
# per-row colorbar label in tfr_group.save_grand_average_panel, which is only as tall as
# one band row and clips a label much longer than these.
BASELINE_MODE_LABELS = {
    'logratio': 'log ratio',
    'percent': 'fractional change',
    'ratio': 'ratio to baseline',
    'zscore': 'z-score',
    'zlogratio': 'z-scored log ratio',
    'mean': 'power change',
}


def mode_label(tfr_params):
    """The baseline correction's name, for figure labels."""
    return BASELINE_MODE_LABELS.get(tfr_params['mode'], tfr_params['mode'])


KNOWN_TFR_KEYS = frozenset(k for group in TFR_PARAM_GROUPS.values() for k in group)

_TFR_GROUP_TITLES = {
    'spectral': 'TFR - SPECTRAL ESTIMATION',
    'baseline': 'TFR - BASELINE AND WINDOWS',
    'figures': 'TFR - FIGURES',
    'ica': 'TFR - ICA (stage 1; changing these needs a fresh review)',
}

_TFR_SUMMARIZE = {'freqs': 'Hz', 'n_cycles': 'values'}

# Backwards-compatible aliases. Nothing in this package reads these any more - every
# consumer takes the dict - but they keep an outside importer or an old notebook working.
_DEFAULT_TFR = default_tfr_params()
TFR_FREQS = _DEFAULT_TFR['freqs']
TFR_N_CYCLES = TFR_FREQS.astype(float)
TFR_DECIM = _DEFAULT_TFR['decim']
TFR_MODE = _DEFAULT_TFR['mode']
TFR_LOWPASS = _DEFAULT_TFR['lowpass']
TFR_BASELINE = _DEFAULT_TFR['baseline']
ACTIVE_WINDOW = _DEFAULT_TFR['active_window']
BANDS = _DEFAULT_TFR['bands']
TFR_VLIM = _DEFAULT_TFR['vlim']
ICLABEL_EXCLUDE_LABELS = _DEFAULT_TFR['iclabel_exclude_labels']
FIGURE_DPI = _DEFAULT_TFR['figure_dpi']


def resolve_n_cycles(tfr_params):
    """``n_cycles`` with the ``None`` sentinel expanded to ``freqs`` (T = 1 s)."""
    freqs = np.asarray(tfr_params['freqs'], dtype=float)
    return freqs if tfr_params.get('n_cycles') is None else np.asarray(
        tfr_params['n_cycles'], dtype=float)


# The preprocessing keys the TFR path actually consumes. The rest of default_params is
# inherited from the classifier stack and reaches nothing here - said out loud in the
# readback, because a cell that carefully sets classifier_window_s is setting nothing.
_TFR_LIVE_PREPROCESSING = ('desired_events', 'Electorde_Group', 'PerformAvgRef',
                           'AddRefChannel', 'CenterByClass', 'PerformCsd',
                           'filter_method', 'LowPass', 'HighPass',
                           'epoch_tmin', 'epoch_tmax', 'bad_electrodes')


def describe_tfr_params(params_dict=None, tfr_params=None):
    """Print both dicts, grouped, with a ``*`` where a value departs from the default."""
    params_dict = params_dict if params_dict is not None else default_params()
    tfr_params = tfr_params if tfr_params is not None else default_tfr_params()

    live = tuple(k for k in _TFR_LIVE_PREPROCESSING if k in params_dict)
    inert = tuple(k for k in params_dict if k not in _TFR_LIVE_PREPROCESSING)
    params_spec.describe(
        params_dict,
        {'live': live, 'inert': inert},
        PREPROCESSING_DOCS,
        defaults=default_params(),
        titles={'live': 'PREPROCESSING - rebuilds the epoch cache AND voids the ICA review',
                'inert': 'PREPROCESSING - inherited from the classifier stack, unused here'},
        known=set(params_dict), summarize={'Electorde_Group': 'ch'})
    params_spec.describe(
        tfr_params, TFR_PARAM_GROUPS, TFR_PARAM_DOCS,
        defaults=default_tfr_params(), titles=_TFR_GROUP_TITLES,
        known=KNOWN_TFR_KEYS, summarize=_TFR_SUMMARIZE)
    n_cycles = resolve_n_cycles(tfr_params)
    freqs = np.asarray(tfr_params['freqs'], dtype=float)
    windows = n_cycles / freqs
    print(f"  -> analysis window T = {windows.min():.2f}-{windows.max():.2f} s, "
          f"smoothing +/-{(tfr_params['time_bandwidth'] / windows.max() / 2):.1f} "
          f"to +/-{(tfr_params['time_bandwidth'] / windows.min() / 2):.1f} Hz")
    print("  ('*' = differs from the default)")


PREPROCESSING_DOCS = {
    'desired_events': 'conditions to compute a TFR for',
    'Electorde_Group': 'channels kept; all 9 groups keeps every subject stackable',
    'PerformAvgRef': 'average re-reference',
    'AddRefChannel': 'reconstruct FCz; off here, so the FC group has no FCz',
    'CenterByClass': 'per-class mean removal - this is what makes the TFRs INDUCED power',
    'PerformCsd': 'CSD inside EEG_Preprocessing; off, CSD is applied after ICA instead',
    'filter_method': "MNE filter method ('iir' / 'fir')",
    'LowPass': 'high-pass edge, Hz (1 Hz, as ICA needs)',
    'HighPass': 'low-pass edge, Hz; None = none here, the TFR lowpass does it',
    'epoch_tmin': 'epoch crop start, s relative to cue',
    'epoch_tmax': 'epoch crop end, s',
    'bad_electrodes': 'kept empty on purpose: ICA/ICLabel handles noisy channels, and '
                      'dropping per-subject bads would break group stacking',
}


def check_tfr_params(params_dict, tfr_params, label=None, paths=None):
    """
    Validate a TFR parameters cell and return the pair, so the call can be chained.

    Deliberately separate from ``analysis_common.check_params``: the rules are not the
    same pipeline's. That one requires ``HighPass > LowPass``; this path runs
    ``LowPass=1, HighPass=None`` and would raise comparing None to a number.
    """
    unknown = params_spec.unknown_keys(tfr_params, KNOWN_TFR_KEYS)
    if unknown:
        raise ValueError(f"TFR keys read by nothing: {', '.join(unknown)}. "
                         f"A misspelled key is silently ignored, so this is an error.")

    freqs = np.asarray(tfr_params['freqs'], dtype=float)
    if freqs.size == 0:
        raise ValueError("freqs is empty: there is nothing to estimate.")
    if tfr_params.get('n_cycles') is not None and \
            len(np.atleast_1d(tfr_params['n_cycles'])) not in (1, freqs.size):
        raise ValueError(
            f"n_cycles has {len(np.atleast_1d(tfr_params['n_cycles']))} entries but there "
            f"are {freqs.size} freqs; it must be a scalar, None, or one per frequency.")
    if tfr_params['decim'] < 1:
        raise ValueError(f"decim must be >= 1, got {tfr_params['decim']}.")
    if not tfr_params['bands']:
        raise ValueError("bands is empty: no band maps would be produced.")
    # Caught here rather than by MNE, which only sees it at apply_baseline - i.e. after
    # a subject's full ICA + epoching. run_tfr_batch swallows per-subject exceptions, so
    # a typo'd mode would otherwise burn the whole cohort one FAILED subject at a time.
    if tfr_params['mode'] not in BASELINE_MODE_LABELS:
        raise ValueError(f"mode {tfr_params['mode']!r} is not an apply_baseline mode; "
                         f"expected one of {sorted(BASELINE_MODE_LABELS)}.")
    if not 0 < tfr_params['autoscale_pct'] <= 100:
        raise ValueError(f"autoscale_pct = {tfr_params['autoscale_pct']} is not a "
                         f"percentile in (0, 100].")

    tmin, tmax = params_dict['epoch_tmin'], params_dict['epoch_tmax']
    for name in ('baseline', 'active_window'):
        start, end = tfr_params[name]
        if start >= end:
            raise ValueError(f"{name} = {tfr_params[name]} does not run forwards.")
        if start < tmin or end > tmax:
            raise ValueError(f"{name} = {tfr_params[name]} falls outside the epoch "
                             f"[{tmin}, {tmax}].")
    if tfr_params['baseline'][1] > 0:
        raise ValueError(
            f"baseline = {tfr_params['baseline']} extends past the cue at t=0, which "
            f"pulls task power into the reference every TFR is divided by.")
    for band, (fmin, fmax) in tfr_params['bands'].items():
        if fmin < freqs.min() or fmax > freqs.max():
            raise ValueError(f"band {band!r} = ({fmin}, {fmax}) Hz falls outside the "
                             f"estimated range {freqs.min():g}-{freqs.max():g} Hz.")

    # --- warnings: defensible choices, but ones worth seeing stated -------------
    if freqs.max() >= tfr_params['lowpass']:
        print(f"  !! freqs reach {freqs.max():g} Hz but the epochs are low-passed at "
              f"{tfr_params['lowpass']} Hz; the top bins sit in the filter roll-off.")
    longest = (resolve_n_cycles(tfr_params) / freqs).max()
    guard = -tfr_params['baseline'][1]
    if guard < longest / 2:
        print(f"  !! only {guard:g} s between the baseline and the cue, but the longest "
              f"analysis window is {longest:.2f} s: wavelets at the baseline edge reach "
              f"past t=0. Allow >= {longest / 2:.2f} s.")

    if label is None and paths is not None:
        run_path = paths.tfrs / 'tfr_run.json'
        if run_path.exists():
            try:
                recorded = (json.loads(run_path.read_text()).get('tfr_params') or {})
            except (OSError, ValueError):
                recorded = {}
            changed = params_spec.changed_against(
                jsonable(tfr_params), recorded, sorted(KNOWN_TFR_KEYS))
            if changed:
                print(f"  !! {paths.tfrs.name}/ was written with different TFR params: "
                      f"{changed}.\n"
                      f"     Re-running overwrites those TFRs in place. Set TFR_LABEL to "
                      f"keep both.")
    return params_dict, tfr_params

#%%
# ============================================================
# Paths and subject discovery
# ============================================================


def project_paths(root=None, label=None):
    """
    Resolve the project's directories, creating the output ones if needed.

    The root is derived from this file's location rather than the working directory,
    so it is identical whether the caller runs from ``notebooks/`` or the repo root.

    ``label`` nests the two OUTPUT trees so a run cannot overwrite a previous one -
    ``tfr.save`` and every figure write are unconditional overwrites, which is why
    ``TFRs_morlet/`` and ``TFRs_percent/`` exist as hand-made copies::

        label=None      TFRs/                Figures/<subject>/       Figures/Group/
        label='morlet'  TFRs/morlet/         Figures/morlet/<subject>/  Figures/morlet/Group/

    ``cache`` and ``configs`` stay at the root either way. The ICA review is per
    subject, not per TFR setting: nesting them would mean re-reviewing every subject
    for each parameter sweep, and the fingerprint check in ``compute_subject_tfrs``
    already catches the case where a review really has been invalidated.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    figures = root / 'Figures' / label if label else root / 'Figures'
    tfrs = root / 'TFRs' / label if label else root / 'TFRs'
    paths = SimpleNamespace(
        root=root,
        label=label,
        recordings=root / 'Recordings',
        figures=figures,
        tfrs=tfrs,
        cache=root / 'Cache',
        configs=root / 'configs',
    )
    for out_dir in (paths.figures, paths.tfrs, paths.cache, paths.configs):
        out_dir.mkdir(parents=True, exist_ok=True)
    return paths


def subject_figure_dir(subject, kind, paths=None):
    """``Figures/<subject>/<kind>/``, created on demand. kind: 'ICA'|'TFR'|'Contrasts'."""
    paths = paths or project_paths()
    out_dir = paths.figures / subject / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def list_subjects(paths=None):
    """
    Subject codes found in ``Recordings/``, sorted and deduplicated.

    The code is everything before the first underscore, e.g. ``AEH_MI2.xdf`` -> ``AEH``.
    """
    paths = paths or project_paths()
    return sorted({f.name.split('_')[0] for f in paths.recordings.glob('*.xdf')})


def subject_files(subject, paths=None):
    """
    This subject's recordings, sorted by name.

    Matches the code exactly rather than as a substring, so subjects whose codes are
    prefixes of one another (or of another subject's code) cannot pull in each other's
    files the way ``subject in f.name`` could.
    """
    paths = paths or project_paths()
    files = [f for f in paths.recordings.glob('*.xdf') if f.name.split('_')[0] == subject]
    if not files:
        raise FileNotFoundError(
            f"No recordings for subject '{subject}' in {paths.recordings}. "
            f"Available subjects: {list_subjects(paths)}")
    return sorted(files)


def default_params(electrode_group_names=ELECTRODE_GROUP_NAMES,
                   desired_events=None):
    """
    The preprocessing parameters used for every subject's TFR.

    ``bad_electrodes`` is deliberately empty: keeping the full 64-channel montage for
    every subject is what makes the per-subject TFRs stackable for group analysis
    (dropping each subject's own bads gives every subject a different channel set).
    Noisy channels are handled downstream by ICA / ICLabel instead, which is why
    ``get_subject_bad_electrodes`` is not consulted here.
    """
    params_dict = {}
    params_dict['PerformCsd'] = False           # CSD is applied after ICA, not here
    params_dict['PerformAvgRef'] = True
    params_dict['CenterByClass'] = True         # per-class mean removal; this is what
                                                # makes the TFRs induced power

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
    params_dict['LowPass'] = 1
    params_dict['HighPass'] = None
    params_dict['filters_bands'] = [[7, 12], [12, 20], [20, 28], [28, 35]]
    params_dict['augmentation_params'] = {'win_len': 0, 'win_step': 0.25}
    params_dict['classifier_window_s'] = 1
    params_dict['classifier_window_e'] = 4
    params_dict['windowed_prediction_params'] = {'win_len': 2, 'win_step': 0.25}
    params_dict['pipeline_name'] = 'ts+FGDA'
    params_dict['n_components_fbcsp'] = 8
    params_dict['desired_events'] = list(desired_events or DESIRED_EVENTS)
    return params_dict


#%%
# ============================================================
# Cache
# ============================================================
# Preprocessing 3-5 xdf files per subject is the slow part, and the recorded component
# indices are only meaningful for the exact epochs ICA saw, so both are cached.

def _cache_paths(subject, paths=None):
    paths = paths or project_paths()
    # The .fif names follow MNE's -epo/-ica conventions so save/read don't warn
    return SimpleNamespace(
        epochs=paths.cache / f'{subject}_pre_ica-epo.fif',
        ica=paths.cache / f'{subject}-ica.fif',
        iclabel=paths.cache / f'{subject}_iclabel.json',
        meta=paths.cache / f'{subject}_meta.json',
    )


# Parameters that change the epochs, and so invalidate a cache. Electorde_Group and
# bad_electrodes are in here because they change the CHANNEL SET, and ICA is fitted per
# channel set - so they change which component is which, which is the whole point of
# the fingerprint below. Adding them cannot disturb existing caches: _check_meta only
# compares keys a meta actually recorded.
_META_PARAM_KEYS = ('PerformCsd', 'PerformAvgRef', 'CenterByClass', 'filter_method',
                    'epoch_tmin', 'epoch_tmax', 'LowPass', 'HighPass', 'desired_events',
                    'Electorde_Group', 'bad_electrodes')


def _meta_params(params_dict):
    """The cache-relevant params, JSON-safe. ``.get`` so an incomplete cell diffs."""
    out = {}
    for key in _META_PARAM_KEYS:
        value = params_dict.get(key)
        out[key] = sorted(value) if isinstance(value, (set, frozenset)) else value
    return out


def _params_fingerprint(meta_params):
    """
    Short stable digest of the cache-relevant params.

    Stamped into a recorded ICA decision so stage 3 can tell whether the components a
    human picked still refer to the ICA those params produce. Sorted, separator-fixed
    JSON so the digest depends on the values and not on dict ordering.
    """
    blob = json.dumps(meta_params, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _build_meta(subject, files, epochs, params_dict):
    meta_params = _meta_params(params_dict)
    return {
        'subject': subject,
        'files': [{'name': f.name, 'size': f.stat().st_size} for f in files],
        'n_epochs': len(epochs),
        'ch_names': epochs.ch_names,
        'event_id': {k: int(v) for k, v in epochs.event_id.items()},
        'params': meta_params,
        'params_fingerprint': _params_fingerprint(meta_params),
        'built': date.today().isoformat(),
    }


def _cached_fingerprint(subject, paths=None):
    """This subject's cache fingerprint, or None when there is no cache to speak of."""
    cache = _cache_paths(subject, paths)
    if not cache.meta.exists():
        return None
    meta = json.loads(cache.meta.read_text())
    # Metas written before the fingerprint existed still carry the params it is
    # computed from, so derive it rather than treating them as unknowable.
    return meta.get('params_fingerprint') or (
        _params_fingerprint(meta['params']) if meta.get('params') else None)


def _check_meta(subject, meta, files, params_dict):
    """Warn when a cache no longer matches the recordings or parameters that made it."""
    current_files = [{'name': f.name, 'size': f.stat().st_size} for f in files]
    if meta.get('files') != current_files:
        print(f"  !! WARNING: {subject}'s recordings changed since the cache was built "
              f"({meta.get('built')}). Any recorded ICA components may no longer refer "
              f"to the same data — rerun prepare_subject('{subject}', force=True).")
    cached_params = meta.get('params', {})
    current_params = _meta_params(params_dict)
    # Only keys the cache actually recorded are comparable: a meta written before a
    # key joined _META_PARAM_KEYS simply has nothing to say about it, and treating
    # that silence as a difference would tell every existing cache to rebuild.
    differing = [key for key in _META_PARAM_KEYS
                 if key in cached_params and cached_params[key] != current_params[key]]
    if differing:
        print(f"  !! WARNING: {subject}'s cache was built with different preprocessing "
              f"parameters ({', '.join(differing)}). "
              f"Rerun prepare_subject('{subject}', force=True).")


#%%
# ============================================================
# Stage 1 — preprocess, fit ICA, show the plots to review
# ============================================================


def build_subject_epochs(subject, params_dict=None, paths=None):
    """
    Load, preprocess and concatenate every recording of one subject.

    Returns the pre-ICA epochs: montage set, average-referenced, 1-100 Hz + 50 Hz
    notch, epoched -5 to 6 s around each cue, restricted to the desired events and
    recoded to ``standard_event_id``.

    Note that ``EEG_Preprocessing`` subtracts each class's mean across epochs, so the
    class-average evoked response is gone and the TFRs downstream are induced power.
    """
    paths = paths or project_paths()
    params_dict = params_dict or default_params()
    files = subject_files(subject, paths)
    print(f"[{subject}] preprocessing {len(files)} recording(s): "
          f"{', '.join(f.name for f in files)}")

    epochs_list = []
    for xdf_file in files:
        print(f"\n--- {xdf_file.name} ---")
        raw = read_raw_xdf(xdf_file)
        raw = fix_channel_names_for_subject(raw, subject)
        raw = standardize_event_labels(raw)
        _, epoch, _, _, _ = EEG_Preprocessing(paths.root, raw, params_dict,
                                              pick_channels=True)
        epoch = remap_epoch_events_to_standard(epoch, standard_event_id,
                                               params_dict['desired_events'])
        if len(epoch) == 0:
            raise RuntimeError(
                f"{xdf_file.name} contributed no epochs for any of "
                f"{params_dict['desired_events']} — check its event labels.")
        # Record which recording every epoch came from. This is what makes
        # leave-one-recording-out possible downstream (see src/session_batch.py);
        # file boundaries are NOT recoverable from the concatenated epochs
        # afterwards, because event gaps reflect within-run block breaks, not files.
        #
        # Attached HERE, last, on purpose: EEG_Preprocessing's centering step
        # rebuilds an EpochsArray and re-sorts by onset, which would scramble a
        # per-trial column set earlier. Every epoch in this object came from this
        # one file, so a uniform assignment after the fact is correct regardless.
        epoch.metadata = pd.DataFrame({'recording': [xdf_file.name] * len(epoch)})
        epochs_list.append(epoch)

    print(f"\n[{subject}] concatenating {len(epochs_list)} file(s) of epochs...")
    epochs = mne.concatenate_epochs(epochs_list, on_mismatch='warn')

    expected = params_dict['Electorde_Group']
    if len(epochs.ch_names) != len(expected):
        print(f"  !! WARNING: expected {len(expected)} channels, got "
              f"{len(epochs.ch_names)}. Missing: "
              f"{sorted(set(expected) - set(epochs.ch_names))}. TFRs for this subject "
              f"will not stack with the others for group analysis.")
    print(f"[{subject}] {len(epochs)} epochs, {len(epochs.ch_names)} channels, "
          f"counts: { {ev: len(epochs[ev]) for ev in epochs.event_id} }")
    return epochs


def fit_subject_ica(epochs, tfr_params=None):
    """
    Fit ICA on the epoch data and classify the components with ICLabel.

    The epochs are flattened into a continuous Raw (annotated with one class-labelled
    span per epoch plus a cue marker, so trials stay visible in the sources browser),
    which is also what ICLabel needs as its ``inst``.

    Takes its hyperparameters from the ``ica_*`` keys of the TFR params dict. They used
    to be keyword arguments here, which no caller ever passed - so they were settings
    in name only.

    Returns
    -------
    ica : mne.preprocessing.ICA
    label_dict : dict
        ``mne_icalabel.label_components`` output; also populates ``ica.labels_scores_``.
    raw_from_epochs : mne.io.RawArray
        Keep it: ``ica.plot_sources`` and ``plot_properties`` need the same instance.
    """
    from mne_icalabel import label_components

    tfr_params = tfr_params or default_tfr_params()
    raw_from_epochs = epochs_to_continuous_raw(epochs)
    # One rank is lost to the average reference, so that is the ceiling by default.
    n_components = tfr_params.get('ica_n_components') or len(epochs.ch_names) - 1

    print(f"Fitting ICA ({n_components} components, {tfr_params['ica_method']}, "
          f"seed {tfr_params['ica_random_state']}) on "
          f"{raw_from_epochs.n_times / raw_from_epochs.info['sfreq']:.0f} s of epoch data...")
    ica = ICA(n_components=n_components, method=tfr_params['ica_method'],
              random_state=tfr_params['ica_random_state'])
    ica.fit(raw_from_epochs, decim=tfr_params['ica_decim'])
    print(f"ICA fitted ({ica.n_components_} components).")

    print("Running ICLabel classification...")
    label_dict = label_components(raw_from_epochs, ica, method='iclabel')
    return ica, label_dict, raw_from_epochs


def prepare_subject(subject, force=False, show=True, params_dict=None, tfr_params=None,
                    paths=None):
    """
    Stage 1: get this subject's epochs and fitted ICA (from cache or from scratch),
    save the ICA diagnostic figures, and show the plots needed to pick components.

    Parameters
    ----------
    subject : str
        Subject code, as returned by `list_subjects`.
    force : bool
        Ignore any cached epochs/ICA and rebuild them.
    show : bool
        Open the interactive component, sources and ICLabel plots. Needs a Qt backend
        (``%matplotlib qt``); the figures are saved either way.

    Returns
    -------
    dict with keys ``epochs``, ``ica``, ``label_dict``, ``raw_from_epochs``, ``suggested``.
    """
    paths = paths or project_paths()
    params_dict = params_dict or default_params()
    tfr_params = tfr_params or default_tfr_params()
    cache = _cache_paths(subject, paths)
    files = subject_files(subject, paths)

    cached = (not force and cache.epochs.exists() and cache.ica.exists()
              and cache.iclabel.exists())
    if cached:
        print(f"[{subject}] loading cached epochs and ICA from {paths.cache.name}/ "
              f"(pass force=True to rebuild)")
        epochs = mne.read_epochs(cache.epochs, preload=True)
        ica = read_ica(cache.ica)
        stored = json.loads(cache.iclabel.read_text())
        label_dict = {'labels': stored['labels'],
                      'y_pred_proba': np.asarray(stored['y_pred_proba'])}
        # read_ica does not restore the ICLabel probability matrix that
        # plot_iclabel_summary reads off the ICA object
        ica.labels_scores_ = np.asarray(stored['scores'])
        raw_from_epochs = epochs_to_continuous_raw(epochs)
        if cache.meta.exists():
            _check_meta(subject, json.loads(cache.meta.read_text()), files, params_dict)
    else:
        epochs = build_subject_epochs(subject, params_dict, paths)
        ica, label_dict, raw_from_epochs = fit_subject_ica(epochs, tfr_params)

        print(f"[{subject}] caching epochs and ICA to {paths.cache.name}/ ...")
        epochs.save(cache.epochs, overwrite=True)
        ica.save(cache.ica, overwrite=True)
        cache.iclabel.write_text(json.dumps({
            'labels': list(label_dict['labels']),
            'y_pred_proba': np.asarray(label_dict['y_pred_proba']).tolist(),
            'scores': np.asarray(ica.labels_scores_).tolist(),
        }, indent=1))
        cache.meta.write_text(json.dumps(
            _build_meta(subject, files, epochs, params_dict), indent=1))

    suggested = iclabel_suggested_exclusions(label_dict['labels'],
                                             tfr_params['iclabel_exclude_labels'])
    ica_dir = subject_figure_dir(subject, 'ICA', paths)

    # Component topographies: render once, save, then (optionally) re-host the same
    # figure in a scroll area for review — plot_components would otherwise split 59
    # components across separate windows.
    fig = ica.plot_components(picks=range(ica.n_components_), inst=raw_from_epochs,
                              show=False)
    _savefig(fig, ica_dir / 'components.png', paths, tfr_params)
    if show:
        make_figure_scrollable(fig)
    else:
        plt.close(fig)

    fig = plot_iclabel_summary(ica, label_dict, suggest=suggested, show=False,
                               title=f'ICLabel — {subject}')
    _savefig(fig, ica_dir / 'iclabel.png', paths, tfr_params)
    if not show:
        plt.close(fig)

    if show:
        ica.plot_sources(raw_from_epochs, start=0, stop=64)
        plt.show()

    recorded = load_ica_exclusions(paths).get(subject)
    print(f"\n[{subject}] ICLabel suggests excluding: {suggested}")
    if recorded is not None:
        print(f"[{subject}] already recorded: {recorded['exclude']}  "
              f"({recorded.get('note') or 'no note'}, reviewed {recorded.get('reviewed')})")
    print("\nReview the plots, then record your decision with e.g.:\n"
          f"    record_exclusions('{subject}', {suggested}, note='')")

    return {'epochs': epochs, 'ica': ica, 'label_dict': label_dict,
            'raw_from_epochs': raw_from_epochs, 'suggested': suggested}


#%%
# ============================================================
# Stage 2 — the manual decision, recorded to configs/ica_exclusions.json
# ============================================================


def _exclusions_path(paths=None):
    paths = paths or project_paths()
    return paths.configs / 'ica_exclusions.json'


def load_ica_exclusions(paths=None):
    """The recorded per-subject ICA exclusions, ``{subject: {exclude, note, reviewed}}``."""
    path = _exclusions_path(paths)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def record_exclusions(subject, exclude, note='', paths=None):
    """
    Record which ICA components to remove for one subject (stage 2).

    Merges into ``configs/ica_exclusions.json`` and stamps today's date. Call it again
    to revise a subject; stage 3 always reads the file, never a notebook variable.

    Also stamps the fingerprint of the params the reviewed cache was built with. A
    component index only means anything for the ICA it was picked from: change a
    preprocessing param, rebuild, and ICA refits with a different component order, so
    the recorded indices would silently point at different sources. ``compute_subject_tfrs``
    compares the two and refuses rather than applying them.
    """
    exclude = sorted({int(i) for i in exclude})
    path = _exclusions_path(paths)
    registry = load_ica_exclusions(paths)
    previous = registry.get(subject)
    entry = {'exclude': exclude, 'note': note, 'reviewed': date.today().isoformat()}
    fingerprint = _cached_fingerprint(subject, paths)
    if fingerprint is not None:
        entry['params_fingerprint'] = fingerprint
    registry[subject] = entry
    path.write_text(json.dumps(dict(sorted(registry.items())), indent=1))

    if previous is not None and previous['exclude'] != exclude:
        print(f"[{subject}] exclusions changed {previous['exclude']} -> {exclude}; "
              f"rerun run_tfr_batch(['{subject}']) to refresh its TFRs and figures.")
    print(f"[{subject}] recorded {len(exclude)} component(s) to remove: {exclude}"
          f"{f'  ({note})' if note else ''}")
    return registry[subject]


def existing_tfr_files(subject, paths=None, events=None):
    """
    This pipeline's TFR files for one subject.

    Matched per event name rather than by wildcard, so the legacy files in ``TFRs/``
    (older subjects, ``ClosePalm`` labels, the finer ``decim=2`` grid) are not counted
    as if this pipeline had produced them.
    """
    paths = paths or project_paths()
    events = events or DESIRED_EVENTS
    return [path for path in (paths.tfrs / f'{subject}_{event}_tfr.h5'
                              for event in events) if path.exists()]


def review_status(paths=None, verbose=True):
    """
    Per-subject table of pipeline progress: cached epochs/ICA, recorded exclusions,
    and how many TFR files exist. Use it to see what still needs stage 1 or stage 2.
    """
    paths = paths or project_paths()
    registry = load_ica_exclusions(paths)
    rows = []
    for subject in list_subjects(paths):
        cache = _cache_paths(subject, paths)
        entry = registry.get(subject)
        reviewed_fp = (entry or {}).get('params_fingerprint')
        current_fp = _cached_fingerprint(subject, paths)
        rows.append({
            'subject': subject,
            'n_files': len(subject_files(subject, paths)),
            'cached': cache.epochs.exists() and cache.ica.exists(),
            'reviewed': entry is not None,
            'n_excluded': len(entry['exclude']) if entry else None,
            # None = nothing to compare (an old entry, or no cache); True = the review
            # was made against params the cache no longer holds, so stage 3 will refuse.
            'stale_review': (None if entry is None or reviewed_fp is None
                             or current_fp is None else reviewed_fp != current_fp),
            'n_tfrs': len(existing_tfr_files(subject, paths)),
            'note': (entry or {}).get('note', ''),
        })

    if verbose:
        if paths.label:
            print(f"TFR label: {paths.label!r}  ->  {paths.tfrs.relative_to(paths.root)}")
        print(f"{'subject':<9}{'files':>6}{'cached':>8}{'reviewed':>10}"
              f"{'removed':>9}{'stale':>7}{'tfrs':>6}   note")
        for row in rows:
            stale = {None: '?', True: 'YES', False: '-'}[row['stale_review']]
            print(f"{row['subject']:<9}{row['n_files']:>6}"
                  f"{'yes' if row['cached'] else '-':>8}"
                  f"{'yes' if row['reviewed'] else '-':>10}"
                  f"{'-' if row['n_excluded'] is None else row['n_excluded']:>9}"
                  f"{stale:>7}{row['n_tfrs']:>6}   {row['note']}")
        todo_stage1 = [r['subject'] for r in rows if not r['cached']]
        todo_stage2 = [r['subject'] for r in rows if r['cached'] and not r['reviewed']]
        stale = [r['subject'] for r in rows if r['stale_review']]
        if todo_stage1:
            print(f"\nneed prepare_subject(): {todo_stage1}")
        if todo_stage2:
            print(f"need record_exclusions(): {todo_stage2}")
        if stale:
            print(f"!! reviewed against different params, stage 3 will refuse: {stale}")
        if not todo_stage1 and not todo_stage2 and not stale:
            print(f"\nall {len(rows)} subject(s) reviewed — run_tfr_batch() is ready")
    return rows


#%%
# ============================================================
# Stage 3 — apply ICA, compute the TFRs, save figures
# ============================================================


def _savefig(fig, path, paths=None, tfr_params=None):
    """Save one figure, keeping its own background (plot_topo sets its own facecolor)."""
    dpi = (tfr_params or default_tfr_params())['figure_dpi']
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    paths = paths or project_paths()
    print(f"  saved {path.relative_to(paths.root)}")
    return path


# Where the joint plots drop their topomaps. Layout, not a result: it picks which
# slices are drawn beside the time-frequency image, not what the image contains.
JOINT_TIMEFREQS = [(1.0, 10), (1.0, 20)]


def save_event_figures(tfr, subject, event, out_dir, paths=None, tfr_params=None):
    """
    The three per-condition views of one baselined TFR: the all-channel topographic
    TFR, the joint time-frequency + topomap plot, and one topomap per band averaged
    over the active window. Shared colour limits keep bands and conditions comparable.
    """
    tfr_params = tfr_params or default_tfr_params()
    vlim, bands = tfr_params['vlim'], tfr_params['bands']
    active = tfr_params['active_window']

    fig = tfr.plot_topo(
        baseline=None,      # already applied
        mode=None,
        title=f'TFR of All Channels ({subject} — {event})',
        vmin=vlim[0], vmax=vlim[1],
        fig_facecolor='w',
        font_color='k',
        show=False,
    )
    _savefig(fig, out_dir / f'{event}_topo.png', paths, tfr_params)
    plt.close(fig)

    fig = tfr.plot_joint(
        baseline=None,
        mode=None,
        title=f'Joint TFR ({subject} — {event})',
        timefreqs=JOINT_TIMEFREQS,
        topomap_args=dict(contours=0),
        show=False,
    )
    _savefig(fig, out_dir / f'{event}_joint.png', paths, tfr_params)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(bands), figsize=(6 * len(bands), 4))
    for ax, (band_name, (fmin, fmax)) in zip(np.atleast_1d(axes), bands.items()):
        tfr.plot_topomap(
            tmin=active[0], tmax=active[1],
            fmin=fmin, fmax=fmax,
            baseline=None,
            mode=None,
            vlim=tuple(vlim),
            axes=ax,
            show=False,
        )
        ax.set_title(f'{band_name} ({fmin}-{fmax} Hz) - {event}')
    fig.tight_layout()
    _savefig(fig, out_dir / f'{event}_bands.png', paths, tfr_params)
    plt.close(fig)


def save_contrast_figures(tfrs, subject, out_dir, paths=None, tfr_params=None):
    """
    Joint plot of every pairwise condition difference.

    Both TFRs are already baselined against the same window, so the difference needs no
    further correction — the same subtraction used for the group contrasts in
    Main2.ipynb. How to READ it depends on ``mode``: under 'logratio' the difference is
    the log ratio between the conditions, independent of the baseline; under 'percent'
    it is the difference of two fractional changes, still expressed relative to that
    baseline.
    """
    tfr_params = tfr_params or default_tfr_params()
    vlim = tfr_params['vlim']
    for cond1, cond2 in itertools.combinations(list(tfrs), 2):
        diff = tfrs[cond1].copy()
        diff.data = tfrs[cond1].data - tfrs[cond2].data
        fig = diff.plot_joint(
            baseline=None,
            mode=None,
            title=f'{subject}: {cond1} - {cond2}',
            timefreqs=JOINT_TIMEFREQS,
            vmin=vlim[0], vmax=vlim[1],
            topomap_args=dict(contours=0),
            show=False,
        )
        _savefig(fig, out_dir / f'{cond1}_minus_{cond2}_joint.png', paths, tfr_params)
        plt.close(fig)


def _assert_review_current(subject, entry, paths):
    """
    Refuse to apply ICA components picked against a different set of params.

    A component INDEX is only meaningful for the decomposition it was chosen from.
    Change a preprocessing param, rebuild the cache, and ICA refits with a different
    component ordering - so the recorded indices still apply cleanly and silently
    remove the wrong sources. This is the check that turns that into an error.

    Entries recorded before the fingerprint existed carry none, and are let through
    with a notice rather than invalidated: their indices are almost certainly fine,
    and there is no evidence either way to act on.
    """
    reviewed = entry.get('params_fingerprint')
    current = _cached_fingerprint(subject, paths)
    if reviewed is None:
        print(f"  note: [{subject}]'s ICA review predates params fingerprinting; "
              f"assuming it still matches. Re-record it to pin it down.")
        return
    if current is not None and reviewed != current:
        raise RuntimeError(
            f"[{subject}] the recorded ICA components were picked against different "
            f"preprocessing params (reviewed {reviewed}, cache now {current}). Component "
            f"indices are only meaningful for the ICA they came from, so applying them "
            f"now would remove the wrong sources. Re-review with "
            f"prepare_subject('{subject}') and record_exclusions('{subject}', [...]).")


def compute_subject_tfrs(subject, save_h5=True, save_figs=True, save_contrasts=True,
                         params_dict=None, tfr_params=None, paths=None):
    """
    Stage 3 for one subject: apply the recorded ICA exclusions to the cached epochs,
    convert to CSD, and compute + save one baselined TFR per condition.

    Returns ``{event: AverageTFR}``.
    """
    paths = paths or project_paths()
    params_dict = params_dict or default_params()
    tfr_params = tfr_params or default_tfr_params()
    cache = _cache_paths(subject, paths)

    if not (cache.epochs.exists() and cache.ica.exists()):
        raise FileNotFoundError(
            f"No cached epochs/ICA for '{subject}'. Run prepare_subject('{subject}') first.")
    entry = load_ica_exclusions(paths).get(subject)
    if entry is None:
        raise KeyError(
            f"No ICA exclusions recorded for '{subject}'. Review it with "
            f"prepare_subject('{subject}') and then call "
            f"record_exclusions('{subject}', [...]).")
    _assert_review_current(subject, entry, paths)

    if cache.meta.exists():
        _check_meta(subject, json.loads(cache.meta.read_text()),
                    subject_files(subject, paths), params_dict)

    epochs = mne.read_epochs(cache.epochs, preload=True)
    ica = read_ica(cache.ica)
    ica.exclude = list(entry['exclude'])
    print(f"[{subject}] removing {len(ica.exclude)} component(s): {ica.exclude}"
          f"  ({entry.get('note') or 'no note'}, reviewed {entry.get('reviewed')})")
    epochs = apply_ica_to_epochs(ica, epochs)

    picks = [elec for elec in params_dict['Electorde_Group'] if elec in epochs.ch_names]
    epochs.pick(picks)

    print(f"[{subject}] applying current source density...")
    epochs = mne.preprocessing.compute_current_source_density(epochs)
    epochs.filter(None, tfr_params['lowpass'], method=params_dict['filter_method'])

    # Document what was actually removed, alongside the pre-decision figure
    if save_figs:
        stored = json.loads(cache.iclabel.read_text()) if cache.iclabel.exists() else None
        if stored is not None:
            ica.labels_scores_ = np.asarray(stored['scores'])
            fig = plot_iclabel_summary(
                ica, {'labels': stored['labels']},
                suggest=iclabel_suggested_exclusions(
                    stored['labels'], tfr_params['iclabel_exclude_labels']),
                show=False,
                title=f'ICLabel — {subject} (removed: {ica.exclude})')
            _savefig(fig, subject_figure_dir(subject, 'ICA', paths) / 'iclabel_final.png',
                     paths, tfr_params)
            plt.close(fig)

    tfr_dir = subject_figure_dir(subject, 'TFR', paths)
    n_cycles = resolve_n_cycles(tfr_params)
    tfrs = {}
    for event in params_dict['desired_events']:
        if event not in epochs.event_id:
            print(f"  !! WARNING: [{subject}] has no '{event}' epochs — skipping it.")
            continue
        print(f"[{subject}] computing TFR for {event} ({len(epochs[event])} epochs)...")
        tfr = tfr_multitaper(
            epochs[event],
            freqs=tfr_params['freqs'],
            n_cycles=n_cycles,
            time_bandwidth=tfr_params['time_bandwidth'],
            use_fft=True,
            return_itc=False,
            average=True,
            decim=tfr_params['decim'],
            n_jobs=tfr_params['n_jobs'],
        )
        tfr.apply_baseline(tfr_params['baseline'], mode=tfr_params['mode'])
        tfrs[event] = tfr

        if save_h5:
            h5_path = paths.tfrs / f'{subject}_{event}_tfr.h5'
            tfr.save(h5_path, overwrite=True)
            print(f"  saved {h5_path.relative_to(paths.root)}")
        if save_figs:
            save_event_figures(tfr, subject, event, tfr_dir, paths, tfr_params)

    if save_figs and save_contrasts and len(tfrs) > 1:
        save_contrast_figures(tfrs, subject,
                              subject_figure_dir(subject, 'Contrasts', paths), paths,
                              tfr_params)
    return tfrs


def jsonable(value):
    """
    Anything in a params dict, as something ``json.dumps`` accepts.

    The TFR dict holds numpy arrays and tuples, which is why provenance cannot just
    dump it: an array raises, and a tuple round-trips as a list anyway.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def _tfr_provenance(params_dict, tfr_params, subjects, registry, paths):
    """
    What produced this run, written beside its outputs.

    ``analysis_common._run_provenance`` does this for the classifier analyses, but it
    cannot be imported here - analysis_common imports FROM this module - so this is
    the local equivalent. Records BOTH dicts in full: with the parameters cell in
    place they are the run, and until now nothing on disk recorded the TFR half of
    them at all.
    """
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=paths.root,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = None
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'git_commit': commit or 'unknown',
        'versions': {'mne': mne.__version__, 'numpy': np.__version__},
        'label': paths.label,
        'params': jsonable(params_dict),
        'tfr_params': jsonable(tfr_params),
        'subjects': list(subjects),
        'ica_exclusions': {s: jsonable(registry[s]) for s in subjects if s in registry},
    }


def run_tfr_batch(subjects=None, save_h5=True, save_figs=True, save_contrasts=True,
                  batch_backend='Agg', params_dict=None, tfr_params=None, paths=None):
    """
    Stage 3 for every reviewed subject.

    A failing subject is reported and skipped rather than aborting the batch. Figures
    are rendered on a non-interactive backend by default — otherwise a full run opens
    ~180 Qt windows — and the previous backend is restored afterwards.

    Writes ``tfr_run.json`` beside the h5 files recording both params dicts, so a TFR
    can be traced back to the settings that made it.

    Parameters
    ----------
    subjects : list of str | None
        Defaults to every subject with recorded exclusions, in ``list_subjects`` order.
    batch_backend : str | None
        Matplotlib backend to render into. Pass None to keep the current one.

    Returns
    -------
    dict
        ``{subject: {event: AverageTFR}}`` for the subjects that succeeded.
    """
    paths = paths or project_paths()
    params_dict = params_dict or default_params()
    tfr_params = tfr_params or default_tfr_params()
    registry = load_ica_exclusions(paths)
    if subjects is None:
        subjects = [s for s in list_subjects(paths) if s in registry]
        if not subjects:
            raise RuntimeError(
                "No subject has recorded ICA exclusions yet. Run prepare_subject(<subject>) "
                "and record_exclusions(<subject>, [...]) first; review_status() shows "
                "what is outstanding.")
    subjects = [subjects] if isinstance(subjects, str) else list(subjects)

    previous_backend = matplotlib.get_backend()
    if batch_backend is not None and previous_backend.lower() != batch_backend.lower():
        plt.close('all')
        matplotlib.use(batch_backend, force=True)
        print(f"rendering figures on '{batch_backend}' "
              f"(restoring '{previous_backend}' afterwards)")

    results, failures = {}, {}
    try:
        for i, subject in enumerate(subjects, 1):
            print(f"\n{'=' * 70}\n[{i}/{len(subjects)}] {subject}\n{'=' * 70}")
            try:
                results[subject] = compute_subject_tfrs(
                    subject, save_h5=save_h5, save_figs=save_figs,
                    save_contrasts=save_contrasts, params_dict=params_dict,
                    tfr_params=tfr_params, paths=paths)
            except Exception as err:
                failures[subject] = f'{type(err).__name__}: {err}'
                print(f"  !! FAILED [{subject}]: {failures[subject]}")
            finally:
                plt.close('all')
    finally:
        if batch_backend is not None and previous_backend.lower() != batch_backend.lower():
            matplotlib.use(previous_backend, force=True)

    print(f"\n{'=' * 70}\nBatch summary: {len(results)}/{len(subjects)} subject(s) done"
          f"\n{'=' * 70}")
    print(f"{'subject':<9}{'events':>8}{'tfr h5':>8}{'figures':>9}   conditions")
    for subject in subjects:
        if subject in failures:
            print(f"{subject:<9}{'FAILED':>8}{'':>8}{'':>9}   {failures[subject]}")
            continue
        tfrs = results[subject]
        n_figs = sum(len(list(subject_figure_dir(subject, kind, paths).glob('*.png')))
                     for kind in ('ICA', 'TFR', 'Contrasts'))
        print(f"{subject:<9}{len(tfrs):>8}"
              f"{len(existing_tfr_files(subject, paths)):>8}{n_figs:>9}"
              f"   {', '.join(tfrs)}")
    if results:
        run_path = paths.tfrs / 'tfr_run.json'
        run_path.write_text(json.dumps(
            _tfr_provenance(params_dict, tfr_params, sorted(results), registry, paths),
            indent=1))
        print(f"\n  saved {run_path.relative_to(paths.root)}")
    if failures:
        print(f"\nRerun the failures with run_tfr_batch({list(failures)}).")
    return results
# %%
