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

import itertools
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import ICA, read_ica
from mne.time_frequency import tfr_multitaper

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
    'F':  ['FT9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT10'],
    'FC': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6'],
    'C':  ['T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8'],
    'CP': ['TP9', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP10'],
    'P':  ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    'PO': ['PO7', 'PO3', 'POz', 'PO4', 'PO8'],
    'O':  ['Oz', 'O2', 'O1', 'Iz'],
}

# Every group, i.e. the full 60-channel montage
ELECTRODE_GROUP_NAMES = 'F+AF+FP+PO+O+FC+C+CP+P'

DESIRED_EVENTS = ['MiddleHand', 'RightHand', 'LeftHand', 'FixatedRest']

# Frequency resolution in MNE's multitaper TFR is set by the analysis window length
# in SECONDS:  T = n_cycles / freq,  and  full bandwidth = time_bandwidth / T
# (time_bandwidth is left at its 4.0 default in the tfr_multitaper call below).
#
# n_cycles = freqs holds T fixed at 1.0 s, so every frequency gets the same +/-2 Hz
# smoothing:  10 Hz bin -> 8-12 Hz,  20 Hz bin -> 18-22 Hz.  Mu and beta stay separable.
# The previous linspace(2.6, 8.4) let T collapse 1.30 s -> 0.25 s, which smeared the
# 20 Hz bin across 13.2-26.8 Hz and pushed the 34 Hz bin past the 40 Hz low-pass --
# i.e. the "Alpha" and "Beta" maps were not actually separable bands.
TFR_FREQS    = np.arange(4, 35)          # 4-34 Hz; below ~4 Hz is 1 Hz high-pass roll-off
TFR_N_CYCLES = TFR_FREQS.astype(float)   # T = 1.0 s at every frequency
TFR_DECIM    = 10                        # 500 -> 50 Hz (20 ms); T=1 s is still ~50x oversampled
TFR_MODE     = 'logratio'                # log10(power / baseline mean)
TFR_LOWPASS  = 40                        # applied to the epochs before the TFR

# Trial timing: Rest (2-5 s) -> fixation (2.5 s + 0.5-1.5 s jitter) -> cue at t=0.
# So t in [-3.0, 0) is fixation on EVERY trial.  The window below sits fully inside it
# with >= 0.5 s (= T/2) of guard at each end, so no wavelet reaches past t=0 (which would
# pull post-cue power into the reference) or back into Rest.
TFR_BASELINE  = (-2, -0.5)
ACTIVE_WINDOW = (0.5, 2.5)               # motor-imagery window used for every topomap
BANDS = {'Mu': (8, 13), 'Beta': (13, 30)}

# Colour limits for 'logratio': a 50% ERD is log10(0.5) = -0.30 and a 2x ERS is +0.30.
# The old +/-1.5 was carried over from mode='percent' and means +/-31x on a log10 scale,
# so every panel rendered as flat mid-colour.
TFR_VLIM = (-0.5, 0.5)

# ICLabel classes treated as artifactual when suggesting exclusions
ICLABEL_EXCLUDE_LABELS = ['eye blink', 'eye movement', 'muscle artifact']

FIGURE_DPI = 150

#%%
# ============================================================
# Paths and subject discovery
# ============================================================


def project_paths(root=None):
    """
    Resolve the project's directories, creating the output ones if needed.

    The root is derived from this file's location rather than the working directory,
    so it is identical whether the caller runs from ``notebooks/`` or the repo root.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    paths = SimpleNamespace(
        root=root,
        recordings=root / 'Recordings',
        figures=root / 'Figures',
        tfrs=root / 'TFRs',
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

    ``bad_electrodes`` is deliberately empty: keeping the full 60-channel montage for
    every subject is what makes the per-subject TFRs stackable for group analysis
    (dropping each subject's own bads gives every subject a different channel set).
    Noisy channels are handled downstream by ICA / ICLabel instead, which is why
    ``get_subject_bad_electrodes`` is not consulted here.
    """
    params_dict = {}
    params_dict['PerformCsd'] = False           # CSD is applied after ICA, not here
    params_dict['PerformAvgRef'] = True
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


# Parameters that change the epochs, and so invalidate a cache
_META_PARAM_KEYS = ('PerformCsd', 'PerformAvgRef', 'filter_method', 'epoch_tmin',
                    'epoch_tmax', 'LowPass', 'HighPass', 'desired_events')


def _build_meta(subject, files, epochs, params_dict):
    return {
        'subject': subject,
        'files': [{'name': f.name, 'size': f.stat().st_size} for f in files],
        'n_epochs': len(epochs),
        'ch_names': epochs.ch_names,
        'event_id': {k: int(v) for k, v in epochs.event_id.items()},
        'params': {key: params_dict[key] for key in _META_PARAM_KEYS},
        'built': date.today().isoformat(),
    }


def _check_meta(subject, meta, files, params_dict):
    """Warn when a cache no longer matches the recordings or parameters that made it."""
    current_files = [{'name': f.name, 'size': f.stat().st_size} for f in files]
    if meta.get('files') != current_files:
        print(f"  !! WARNING: {subject}'s recordings changed since the cache was built "
              f"({meta.get('built')}). Any recorded ICA components may no longer refer "
              f"to the same data — rerun prepare_subject('{subject}', force=True).")
    cached_params = meta.get('params', {})
    current_params = {key: params_dict[key] for key in _META_PARAM_KEYS}
    if cached_params != current_params:
        differing = [key for key in _META_PARAM_KEYS
                     if cached_params.get(key) != current_params[key]]
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


def fit_subject_ica(epochs, decim=3, random_state=97, method='fastica'):
    """
    Fit ICA on the epoch data and classify the components with ICLabel.

    The epochs are flattened into a continuous Raw (annotated with one class-labelled
    span per epoch plus a cue marker, so trials stay visible in the sources browser),
    which is also what ICLabel needs as its ``inst``.

    Returns
    -------
    ica : mne.preprocessing.ICA
    label_dict : dict
        ``mne_icalabel.label_components`` output; also populates ``ica.labels_scores_``.
    raw_from_epochs : mne.io.RawArray
        Keep it: ``ica.plot_sources`` and ``plot_properties`` need the same instance.
    """
    from mne_icalabel import label_components

    raw_from_epochs = epochs_to_continuous_raw(epochs)
    n_components = len(epochs.ch_names) - 1     # one rank lost to the average reference

    print(f"Fitting ICA ({n_components} components) on "
          f"{raw_from_epochs.n_times / raw_from_epochs.info['sfreq']:.0f} s of epoch data...")
    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    ica.fit(raw_from_epochs, decim=decim)
    print(f"ICA fitted ({ica.n_components_} components).")

    print("Running ICLabel classification...")
    label_dict = label_components(raw_from_epochs, ica, method='iclabel')
    return ica, label_dict, raw_from_epochs


def prepare_subject(subject, force=False, show=True, params_dict=None, paths=None):
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
        ica, label_dict, raw_from_epochs = fit_subject_ica(epochs)

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
                                             ICLABEL_EXCLUDE_LABELS)
    ica_dir = subject_figure_dir(subject, 'ICA', paths)

    # Component topographies: render once, save, then (optionally) re-host the same
    # figure in a scroll area for review — plot_components would otherwise split 59
    # components across separate windows.
    fig = ica.plot_components(picks=range(ica.n_components_), inst=raw_from_epochs,
                              show=False)
    _savefig(fig, ica_dir / 'components.png', paths)
    if show:
        make_figure_scrollable(fig)
    else:
        plt.close(fig)

    fig = plot_iclabel_summary(ica, label_dict, suggest=suggested, show=False,
                               title=f'ICLabel — {subject}')
    _savefig(fig, ica_dir / 'iclabel.png', paths)
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
    """
    exclude = sorted({int(i) for i in exclude})
    path = _exclusions_path(paths)
    registry = load_ica_exclusions(paths)
    previous = registry.get(subject)
    registry[subject] = {'exclude': exclude, 'note': note,
                         'reviewed': date.today().isoformat()}
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
        rows.append({
            'subject': subject,
            'n_files': len(subject_files(subject, paths)),
            'cached': cache.epochs.exists() and cache.ica.exists(),
            'reviewed': entry is not None,
            'n_excluded': len(entry['exclude']) if entry else None,
            'n_tfrs': len(existing_tfr_files(subject, paths)),
            'note': (entry or {}).get('note', ''),
        })

    if verbose:
        print(f"{'subject':<9}{'files':>6}{'cached':>8}{'reviewed':>10}"
              f"{'removed':>9}{'tfrs':>6}   note")
        for row in rows:
            print(f"{row['subject']:<9}{row['n_files']:>6}"
                  f"{'yes' if row['cached'] else '-':>8}"
                  f"{'yes' if row['reviewed'] else '-':>10}"
                  f"{'-' if row['n_excluded'] is None else row['n_excluded']:>9}"
                  f"{row['n_tfrs']:>6}   {row['note']}")
        todo_stage1 = [r['subject'] for r in rows if not r['cached']]
        todo_stage2 = [r['subject'] for r in rows if r['cached'] and not r['reviewed']]
        if todo_stage1:
            print(f"\nneed prepare_subject(): {todo_stage1}")
        if todo_stage2:
            print(f"need record_exclusions(): {todo_stage2}")
        if not todo_stage1 and not todo_stage2:
            print(f"\nall {len(rows)} subject(s) reviewed — run_tfr_batch() is ready")
    return rows


#%%
# ============================================================
# Stage 3 — apply ICA, compute the TFRs, save figures
# ============================================================


def _savefig(fig, path, paths=None):
    """Save one figure, keeping its own background (plot_topo sets its own facecolor)."""
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    paths = paths or project_paths()
    print(f"  saved {path.relative_to(paths.root)}")
    return path


def save_event_figures(tfr, subject, event, out_dir, paths=None):
    """
    The three per-condition views of one baselined TFR: the all-channel topographic
    TFR, the joint time-frequency + topomap plot, and one topomap per band averaged
    over ACTIVE_WINDOW. Shared colour limits keep bands and conditions comparable.
    """
    fig = tfr.plot_topo(
        baseline=None,      # already applied
        mode=None,
        title=f'TFR of All Channels ({subject} — {event})',
        vmin=TFR_VLIM[0], vmax=TFR_VLIM[1],
        fig_facecolor='w',
        font_color='k',
        show=False,
    )
    _savefig(fig, out_dir / f'{event}_topo.png', paths)
    plt.close(fig)

    fig = tfr.plot_joint(
        baseline=None,
        mode=None,
        title=f'Joint TFR ({subject} — {event})',
        timefreqs=[(1.0, 10), (1.0, 20)],
        topomap_args=dict(contours=0),
        show=False,
    )
    _savefig(fig, out_dir / f'{event}_joint.png', paths)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(BANDS), figsize=(6 * len(BANDS), 4))
    for ax, (band_name, (fmin, fmax)) in zip(np.atleast_1d(axes), BANDS.items()):
        tfr.plot_topomap(
            tmin=ACTIVE_WINDOW[0], tmax=ACTIVE_WINDOW[1],
            fmin=fmin, fmax=fmax,
            baseline=None,
            mode=None,
            vlim=TFR_VLIM,
            axes=ax,
            show=False,
        )
        ax.set_title(f'{band_name} ({fmin}-{fmax} Hz) - {event}')
    fig.tight_layout()
    _savefig(fig, out_dir / f'{event}_bands.png', paths)
    plt.close(fig)


def save_contrast_figures(tfrs, subject, out_dir, paths=None):
    """
    Joint plot of every pairwise condition difference.

    Both TFRs are already log10 ratios against the same baseline window, so the
    difference is the log ratio between conditions — the same subtraction used for the
    group contrasts in Main2.ipynb.
    """
    for cond1, cond2 in itertools.combinations(list(tfrs), 2):
        diff = tfrs[cond1].copy()
        diff.data = tfrs[cond1].data - tfrs[cond2].data
        fig = diff.plot_joint(
            baseline=None,
            mode=None,
            title=f'{subject}: {cond1} - {cond2}',
            timefreqs=[(1.0, 10), (1.0, 20)],
            vmin=TFR_VLIM[0], vmax=TFR_VLIM[1],
            topomap_args=dict(contours=0),
            show=False,
        )
        _savefig(fig, out_dir / f'{cond1}_minus_{cond2}_joint.png', paths)
        plt.close(fig)


def compute_subject_tfrs(subject, save_h5=True, save_figs=True, save_contrasts=True,
                         params_dict=None, paths=None):
    """
    Stage 3 for one subject: apply the recorded ICA exclusions to the cached epochs,
    convert to CSD, and compute + save one baselined TFR per condition.

    Returns ``{event: AverageTFR}``.
    """
    paths = paths or project_paths()
    params_dict = params_dict or default_params()
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
    epochs.filter(None, TFR_LOWPASS, method=params_dict['filter_method'])

    # Document what was actually removed, alongside the pre-decision figure
    if save_figs:
        stored = json.loads(cache.iclabel.read_text()) if cache.iclabel.exists() else None
        if stored is not None:
            ica.labels_scores_ = np.asarray(stored['scores'])
            fig = plot_iclabel_summary(
                ica, {'labels': stored['labels']},
                suggest=iclabel_suggested_exclusions(stored['labels'],
                                                     ICLABEL_EXCLUDE_LABELS),
                show=False,
                title=f'ICLabel — {subject} (removed: {ica.exclude})')
            _savefig(fig, subject_figure_dir(subject, 'ICA', paths) / 'iclabel_final.png',
                     paths)
            plt.close(fig)

    tfr_dir = subject_figure_dir(subject, 'TFR', paths)
    tfrs = {}
    for event in params_dict['desired_events']:
        if event not in epochs.event_id:
            print(f"  !! WARNING: [{subject}] has no '{event}' epochs — skipping it.")
            continue
        print(f"[{subject}] computing TFR for {event} ({len(epochs[event])} epochs)...")
        tfr = tfr_multitaper(
            epochs[event],
            freqs=TFR_FREQS,
            n_cycles=TFR_N_CYCLES,
            use_fft=True,
            return_itc=False,
            average=True,
            decim=TFR_DECIM,
            n_jobs=-1,
        )
        tfr.apply_baseline(TFR_BASELINE, mode=TFR_MODE)
        tfrs[event] = tfr

        if save_h5:
            h5_path = paths.tfrs / f'{subject}_{event}_tfr.h5'
            tfr.save(h5_path, overwrite=True)
            print(f"  saved {h5_path.relative_to(paths.root)}")
        if save_figs:
            save_event_figures(tfr, subject, event, tfr_dir, paths)

    if save_figs and save_contrasts and len(tfrs) > 1:
        save_contrast_figures(tfrs, subject,
                              subject_figure_dir(subject, 'Contrasts', paths), paths)
    return tfrs


def run_tfr_batch(subjects=None, save_h5=True, save_figs=True, save_contrasts=True,
                  batch_backend='Agg', params_dict=None, paths=None):
    """
    Stage 3 for every reviewed subject.

    A failing subject is reported and skipped rather than aborting the batch. Figures
    are rendered on a non-interactive backend by default — otherwise a full run opens
    ~180 Qt windows — and the previous backend is restored afterwards.

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
                    save_contrasts=save_contrasts, params_dict=params_dict, paths=paths)
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
    if failures:
        print(f"\nRerun the failures with run_tfr_batch({list(failures)}).")
    return results
# %%
