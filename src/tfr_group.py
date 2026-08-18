#%%
"""
Group-level (stage 4) aggregate TFR / ERD-ERS figures.

Reads the per-subject ``TFRs/<subject>_<event>_tfr.h5`` files written by stage 3 of
src/tfr_batch.py and produces the across-subject descriptive figures:

    run_tfr_group()
        -> Figures/Group/Subjects/<band>_subject_grid.png
           rows = subjects, columns = conditions, plus a grand-average GROUP row
        -> Figures/Group/GrandAverage/grand_average_panel.png
           rows = bands, columns = conditions, grand average only
        -> Figures/Group/GrandAverage/<event>_{topo,joint,bands}.png
        -> Figures/Group/Contrasts/<cond1>_minus_<cond2>_joint.png
        -> Figures/Group/tfr_group_summary.json   (cohort + config provenance)

The cohort is whoever has been ICA-reviewed and has a full set of current-pipeline TFRs,
so the figures grow as more subjects come through stages 1-2. Group *statistics* are
deliberately not implemented here: see the note above run_cluster_test_tfr's absence in
the module docstring of load_group_tfrs.

Driven from notebooks/TFR_Analysis.ipynb. Imports src.tfr_batch, which imports
src.preprocessing, which runs `%matplotlib qt` at import time, so this needs an IPython
kernel (notebook or `ipython`), not a bare `python` process.
"""

import itertools
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mne.time_frequency import read_tfrs

# Settings arrive as the tfr_params dict, never as imported constants. `from ... import
# BANDS` would bind a SECOND module global here, so a notebook that rebound
# tfr_batch.BANDS would change the per-subject figures and not these - the two halves of
# the pipeline would silently disagree about what a band is.
from .tfr_batch import (  # noqa: F401  (default_params/default_tfr_params re-exported)
    DESIRED_EVENTS,
    _savefig,
    default_params,
    default_tfr_params,
    existing_tfr_files,
    jsonable,
    list_subjects,
    load_ica_exclusions,
    mode_label,
    project_paths,
)

#%%
# ============================================================
# Paths
# ============================================================


def group_figure_dir(kind, paths=None):
    """``Figures/Group/<kind>/``, created on demand. kind: 'Subjects'|'GrandAverage'|'Contrasts'."""
    paths = paths or project_paths()
    out_dir = paths.figures / 'Group' / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


#%%
# ============================================================
# Cohort loading
# ============================================================


def group_status(paths=None, verbose=True, events=None):
    """
    Per-subject view of who can enter the group figures, and why the others cannot.

    The read-only counterpart of ``tfr_batch.review_status``: for every subject in
    ``Recordings/`` it reports whether the ICA exclusions have been recorded, which of
    the conditions have a current-pipeline TFR on disk, and the shape of those files.
    ``admitted`` is what ``load_group_tfrs`` would take.
    """
    paths = paths or project_paths()
    events = list(events) if events is not None else list(DESIRED_EVENTS)
    registry = load_ica_exclusions(paths)
    rows = []
    for subject in list_subjects(paths):
        found = [event for event in events
                 if (paths.tfrs / f'{subject}_{event}_tfr.h5').exists()]
        reviewed = subject in registry
        if not reviewed:
            reason = 'not reviewed — run prepare_subject() + record_exclusions()'
        elif len(found) < len(events):
            missing = [e for e in events if e not in found]
            reason = f"missing TFRs for {', '.join(missing)} — rerun run_tfr_batch(['{subject}'])"
        else:
            reason = ''
        rows.append({
            'subject': subject,
            'reviewed': reviewed,
            'events': found,
            'n_events': len(found),
            'shape': _tfr_shape(paths.tfrs / f'{subject}_{found[0]}_tfr.h5') if found else None,
            'admitted': reason == '',
            'reason': reason,
        })

    if verbose:
        if paths.label:
            print(f"TFR label: {paths.label!r}  ->  {paths.tfrs.relative_to(paths.root)}")
        print(f"{'subject':<9}{'reviewed':>10}{'events':>8}{'shape':>20}{'admitted':>10}   why not")
        for row in rows:
            events_col = '{}/{}'.format(row['n_events'], len(events))
            print(f"{row['subject']:<9}"
                  f"{'yes' if row['reviewed'] else '-':>10}"
                  f"{events_col:>8}"
                  f"{str(row['shape'] or '-'):>20}"
                  f"{'yes' if row['admitted'] else '-':>10}   {row['reason']}")
        admitted = [r['subject'] for r in rows if r['admitted']]
        print(f"\ncohort: N = {len(admitted)}  {admitted}")
        if len(admitted) < 2:
            print("at least 2 subjects are needed for a group average")
    return rows


def _tfr_shape(path):
    """``(n_channels, n_freqs, n_times)`` read from the file's header, without the data."""
    try:
        tfr = read_tfrs(path, verbose='ERROR')
    except Exception:
        return None
    tfr = tfr[0] if isinstance(tfr, list) else tfr
    return tuple(tfr.data.shape)


def load_group_tfrs(subjects=None, events=None, paths=None, verbose=True):
    """
    Load every admitted subject's TFRs into one cohort, aligned and checked.

    Selection is by subject code and exact ``<subject>_<event>_tfr.h5`` name, never by
    globbing ``TFRs/*.h5``. That matters because ``TFRs/`` also holds legacy files from
    an earlier pipeline (54 channels, 2-34 Hz, ``decim=2``, ``ClosePalm``/``Rest``
    labels) for subjects that no longer have recordings at all. A glob mixes the two
    vintages and only fails later, deep inside the stacking, after minutes of reading.

    Every admitted subject is then checked against the first one for an identical
    frequency grid, time grid and *ordered* channel list. Order matters as much as
    membership: the cluster permutation test this cohort is meant to feed builds its
    adjacency matrix from the channel order, so a silently permuted list would scramble
    the neighbour graph rather than raise.

    ``tfrs[event]`` is a list in ``subjects`` order, so a paired across-condition
    difference is matched by subject rather than by list position.

    Returns
    -------
    SimpleNamespace
        ``subjects, events, tfrs, by_subject, freqs, times, ch_names, info, skipped``
    """
    paths = paths or project_paths()
    events = list(events) if events is not None else list(DESIRED_EVENTS)

    if subjects is None:
        registry = load_ica_exclusions(paths)
        candidates = [s for s in list_subjects(paths) if s in registry]
        if not candidates:
            raise RuntimeError(
                "No subject has recorded ICA exclusions yet, so there is nothing to "
                "aggregate. group_status() shows what is outstanding.")
    else:
        candidates = [subjects] if isinstance(subjects, str) else list(subjects)

    admitted, skipped = [], {}
    for subject in candidates:
        found = existing_tfr_files(subject, paths, events)
        if len(found) < len(events):
            have = {p.name.replace(f'{subject}_', '').replace('_tfr.h5', '') for p in found}
            skipped[subject] = f"missing TFRs for {sorted(set(events) - have)}"
            continue
        admitted.append(subject)

    if not admitted:
        raise RuntimeError(
            f"No subject has a full set of TFRs for {events}. Skipped: {skipped}. "
            f"group_status() shows what is outstanding.")

    by_subject = {}
    for subject in admitted:
        by_subject[subject] = {}
        for event in events:
            tfr = read_tfrs(paths.tfrs / f'{subject}_{event}_tfr.h5', verbose='ERROR')
            by_subject[subject][event] = tfr[0] if isinstance(tfr, list) else tfr

    reference = by_subject[admitted[0]][events[0]]
    for subject in admitted:
        for event in events:
            _assert_same_grid(by_subject[subject][event], reference, subject, event,
                              admitted[0], events[0])

    group = SimpleNamespace(
        subjects=admitted,
        events=events,
        tfrs={event: [by_subject[s][event] for s in admitted] for event in events},
        by_subject=by_subject,
        freqs=reference.freqs,
        times=reference.times,
        ch_names=list(reference.ch_names),
        info=reference.info,
        skipped=skipped,
    )

    if verbose:
        print(f"cohort: N = {len(admitted)}  {admitted}")
        print(f"  conditions : {', '.join(events)}")
        print(f"  grid       : {len(group.ch_names)} ch x "
              f"{len(group.freqs)} freqs ({group.freqs[0]:.0f}-{group.freqs[-1]:.0f} Hz) x "
              f"{len(group.times)} times ({group.times[0]:.2f}..{group.times[-1]:.2f} s)")
        for subject, reason in skipped.items():
            print(f"  skipped {subject}: {reason}")
    return group


def _assert_same_grid(tfr, reference, subject, event, ref_subject, ref_event):
    """Raise with the concrete difference named if one subject sits on a different grid."""
    where = f"{subject}/{event} vs {ref_subject}/{ref_event}"
    if len(tfr.freqs) != len(reference.freqs) or not np.allclose(tfr.freqs, reference.freqs):
        raise ValueError(
            f"Frequency grid mismatch, {where}: "
            f"{len(tfr.freqs)} bins {tfr.freqs[0]:g}-{tfr.freqs[-1]:g} Hz vs "
            f"{len(reference.freqs)} bins {reference.freqs[0]:g}-{reference.freqs[-1]:g} Hz. "
            f"This is what a legacy TFR file looks like — rerun run_tfr_batch(['{subject}']) "
            f"to rebuild it on the current grid.")
    if len(tfr.times) != len(reference.times) or not np.allclose(tfr.times, reference.times):
        raise ValueError(
            f"Time grid mismatch, {where}: {len(tfr.times)} samples vs "
            f"{len(reference.times)} (different decim). "
            f"Rerun run_tfr_batch(['{subject}']) to rebuild it on the current grid.")
    if list(tfr.ch_names) != list(reference.ch_names):
        missing = sorted(set(reference.ch_names) - set(tfr.ch_names))
        extra = sorted(set(tfr.ch_names) - set(reference.ch_names))
        detail = (f"missing {missing}, extra {extra}" if (missing or extra)
                  else "same channels in a different order")
        raise ValueError(f"Channel mismatch, {where}: {detail}.")


#%%
# ============================================================
# Aggregation
# ============================================================


def band_map(tfr, fmin, fmax, tmin, tmax):
    """Mean baselined power per channel over a frequency band and time window."""
    f = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    t = (tfr.times >= tmin) & (tfr.times <= tmax)
    return tfr.data[:, f, :][:, :, t].mean(axis=(1, 2))


def grand_average_tfrs(group):
    """
    One equally-weighted average TFR per condition, ``{event: AverageTFR}``.

    Every subject counts once regardless of trial count, so a subject with 61 trials
    does not outweigh one with 45. ``mne.grand_average`` is deliberately not used: it
    silently intersects channels across inputs, whereas ``load_group_tfrs`` has already
    asserted they are identical and we would rather see that assert fail than get a
    quietly narrower montage.
    """
    grand = {}
    for event, tfr_list in group.tfrs.items():
        averaged = tfr_list[0].copy()
        averaged.data = np.stack([tfr.data for tfr in tfr_list], axis=0).mean(axis=0)
        averaged.nave = len(tfr_list)
        grand[event] = averaged
    return grand


#%%
# ============================================================
# Figures
# ============================================================


def _shared_vlim(values, pct=100, floor=0.1):
    """
    Symmetric colour limits covering every panel, so cells are comparable.

    ``pct`` is the percentile of ``|value|`` to scale to. At 100 (the default) the limits
    are the max, so nothing clips and no panel's peak is hidden. Below it the top
    ``100 - pct`` percent saturates, which is what pulls the scale in when a few edge
    channels would otherwise leave every other cell mid-colour. It is a percentile of the
    pooled values rather than a fixed number so each band still scales to itself.
    """
    vmax = float(np.percentile(np.abs(values), pct)) if len(values) else 0.0
    if vmax == 0.0:
        vmax = floor
    return (-vmax, vmax)


def save_subject_band_grid(group, grand, band_name, out_dir, paths=None,
                           tfr_params=None):
    """
    The headline figure for one band: rows = subjects, columns = conditions, with the
    grand average as a final GROUP row.

    One symmetric colour scale is shared by the whole grid, including the group row.
    Letting every panel autoscale (the obvious default, and what the prototype did
    before ``colorbar=False`` was paired with an explicit ``vlim``) makes a subject at
    +/-0.02 render identically to one at +/-0.5, so the grid cannot be read across cells
    at all — which is the only reason to lay it out as a grid.

    That scale covers the whole grid up to ``tfr_params['autoscale_pct']``. At the default
    100 it is the max, so no subject's peak is hidden; lowering it saturates the extremes
    — usually a few edge channels in one or two subjects — and makes the mid-range
    subjects legible at the cost of no longer being able to rank the strongest ones.
    """
    tfr_params = tfr_params or default_tfr_params()
    fmin, fmax = tfr_params['bands'][band_name]
    tmin, tmax = tfr_params['active_window']
    subjects, events = group.subjects, group.events
    rows = subjects + ['GROUP']

    maps = {(s, e): band_map(group.by_subject[s][e], fmin, fmax, tmin, tmax)
            for s in subjects for e in events}
    maps.update({('GROUP', e): band_map(grand[e], fmin, fmax, tmin, tmax) for e in events})
    vlim = _shared_vlim(np.concatenate(list(maps.values())),
                        pct=tfr_params['autoscale_pct'])

    fig, axes = plt.subplots(len(rows), len(events),
                             figsize=(3.2 * len(events), 3.0 * len(rows)))
    axes = np.atleast_2d(axes)
    if len(rows) == 1:
        axes = axes.reshape(1, -1)

    for row, subject in enumerate(rows):
        is_group = subject == 'GROUP'
        for col, event in enumerate(events):
            ax = axes[row, col]
            tfr = grand[event] if is_group else group.by_subject[subject][event]
            tfr.plot_topomap(
                tmin=tmin, tmax=tmax,
                fmin=fmin, fmax=fmax,
                baseline=None, mode=None,
                vlim=vlim, cmap='RdBu_r',
                axes=ax, show=False, colorbar=False,
            )
            ax.set_title(event if row == 0 else '', fontsize=13, fontweight='bold')

        label = f'GROUP\n(N={len(subjects)})' if is_group else subject
        axes[row, 0].annotate(
            label, xy=(-0.3, 0.5), xycoords='axes fraction',
            fontsize=12, fontweight='bold', ha='right', va='center')

    fig.suptitle(f'{band_name} band ({fmin}-{fmax} Hz), {tmin}-{tmax} s — per subject '
                 f'and grand average (N={len(subjects)})',
                 fontsize=15, fontweight='bold')
    sm = ScalarMappable(norm=Normalize(vmin=vlim[0], vmax=vlim[1]), cmap='RdBu_r')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f'ERD / ERS  ({mode_label(tfr_params)})')

    # Rule separating the per-subject rows from the group row. Placed after the colorbar,
    # which reflows every axes position when it is attached to all of them.
    if len(subjects):
        y = (axes[-1, 0].get_position().y1 + axes[-2, 0].get_position().y0) / 2
        fig.add_artist(plt.Line2D([0.08, 0.88], [y, y], color='0.5', linewidth=1,
                                  transform=fig.transFigure))

    path = _savefig(fig, out_dir / f'{band_name}_subject_grid.png', paths, tfr_params)
    plt.close(fig)
    return path


def save_grand_average_panel(group, grand, out_dir, paths=None, tfr_params=None):
    """
    Compact summary: rows = bands, columns = conditions, grand average only.

    Scaled per band row to the grand-average data itself (up to
    ``tfr_params['autoscale_pct']``), with the limits written into each row's colorbar
    label, rather than to the fixed ``tfr_params['vlim']`` the per-subject figures use —
    ``vlim`` reaches nothing here. Averaging across subjects roughly halves the amplitude — at
    N=4 the mu grand average peaks near 0.25 and beta near 0.14 against a fixed +/-0.5 —
    so a fixed scale renders the beta row as flat mid-colour, which is the exact failure
    ``vlim`` was tightened from +/-1.5 to avoid.

    Conditions therefore stay comparable *within* a band, which is the axis this figure
    exists for (lateralisation across MiddleHand / RightHand / LeftHand). Mu against beta
    is not comparable here; for that, read the subject grids or the ``<event>_bands.png``
    views, both of which keep a single scale.
    """
    tfr_params = tfr_params or default_tfr_params()
    bands = tfr_params['bands']
    tmin, tmax = tfr_params['active_window']
    events = group.events
    fig, axes = plt.subplots(len(bands), len(events),
                             figsize=(3.4 * len(events), 3.2 * len(bands)))
    axes = np.atleast_2d(axes)

    for row, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        vlim = _shared_vlim(
            np.concatenate(
                [band_map(grand[event], fmin, fmax, tmin, tmax) for event in events]),
            pct=tfr_params['autoscale_pct'])
        for col, event in enumerate(events):
            ax = axes[row, col]
            grand[event].plot_topomap(
                tmin=tmin, tmax=tmax,
                fmin=fmin, fmax=fmax,
                baseline=None, mode=None,
                vlim=vlim, cmap='RdBu_r',
                axes=ax, show=False, colorbar=False,
            )
            ax.set_title(event if row == 0 else '', fontsize=13, fontweight='bold')
        axes[row, 0].annotate(
            f'{band_name}\n({fmin}-{fmax} Hz)', xy=(-0.3, 0.5), xycoords='axes fraction',
            fontsize=12, fontweight='bold', ha='right', va='center')

        sm = ScalarMappable(norm=Normalize(vmin=vlim[0], vmax=vlim[1]), cmap='RdBu_r')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[row].tolist(), shrink=0.85, pad=0.02)
        # fontsize: this colorbar is only as tall as one band row, and the label carries
        # three things (band, mode, limits). At the default size a longer mode name than
        # 'log ratio' overruns the figure and bbox_inches='tight' clips the first letter.
        cbar.set_label(f'{band_name}  ({mode_label(tfr_params)}, +/-{vlim[1]:.2f})',
                       fontsize=9)

    fig.suptitle(f'Grand average ERD / ERS (N={len(group.subjects)}), {tmin}-{tmax} s',
                 fontsize=15, fontweight='bold')

    path = _savefig(fig, out_dir / 'grand_average_panel.png', paths, tfr_params)
    plt.close(fig)
    return path


def save_grand_average_figures(group, grand, out_dir, paths=None, tfr_params=None):
    """
    The three per-condition views of each grand-average TFR, mirroring
    ``tfr_batch.save_event_figures`` so a group figure can be read side by side with the
    per-subject one it summarises.
    """
    tfr_params = tfr_params or default_tfr_params()
    vlim, bands = tfr_params['vlim'], tfr_params['bands']
    active = tfr_params['active_window']
    n_subj = len(group.subjects)
    for event, tfr in grand.items():
        fig = tfr.plot_topo(
            baseline=None, mode=None,
            title=f'GROUP TFR of All Channels ({event} — N={n_subj})',
            vmin=vlim[0], vmax=vlim[1],
            fig_facecolor='w', font_color='k', show=False,
        )
        _savefig(fig, out_dir / f'{event}_topo.png', paths, tfr_params)
        plt.close(fig)

        fig = tfr.plot_joint(
            baseline=None, mode=None,
            title=f'GROUP Joint TFR ({event} — N={n_subj})',
            timefreqs=[(1.0, 10), (1.0, 20)],
            topomap_args=dict(contours=0), show=False,
        )
        _savefig(fig, out_dir / f'{event}_joint.png', paths, tfr_params)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(bands), figsize=(6 * len(bands), 4))
        for ax, (band_name, (fmin, fmax)) in zip(np.atleast_1d(axes), bands.items()):
            tfr.plot_topomap(
                tmin=active[0], tmax=active[1],
                fmin=fmin, fmax=fmax,
                baseline=None, mode=None,
                vlim=tuple(vlim), axes=ax, show=False,
            )
            ax.set_title(f'{band_name} ({fmin}-{fmax} Hz) - {event} (N={n_subj})')
        fig.tight_layout()
        _savefig(fig, out_dir / f'{event}_bands.png', paths, tfr_params)
        plt.close(fig)


def save_group_contrast_figures(group, grand, out_dir, paths=None, tfr_params=None):
    """
    Joint plot of every pairwise grand-average difference, mirroring
    ``tfr_batch.save_contrast_figures``.

    Both TFRs are already baselined against the same fixation window, so the difference
    needs no further correction; how to read it depends on ``mode`` (see
    ``tfr_batch.save_contrast_figures``). Either way a ``X - FixatedRest`` contrast is
    not raw ERD: it is what is left after removing whatever response the cue evokes in
    every condition alike. The hand-vs-hand contrasts are the lateralisation ones.
    """
    tfr_params = tfr_params or default_tfr_params()
    vlim = tfr_params['vlim']
    n_subj = len(group.subjects)
    for cond1, cond2 in itertools.combinations(group.events, 2):
        diff = grand[cond1].copy()
        diff.data = grand[cond1].data - grand[cond2].data
        fig = diff.plot_joint(
            baseline=None, mode=None,
            title=f'GROUP contrast: {cond1} - {cond2} (N={n_subj})',
            timefreqs=[(1.0, 10), (1.0, 20)],
            vmin=vlim[0], vmax=vlim[1],
            topomap_args=dict(contours=0), show=False,
        )
        _savefig(fig, out_dir / f'{cond1}_minus_{cond2}_joint.png', paths, tfr_params)
        plt.close(fig)


#%%
# ============================================================
# Driver
# ============================================================


def run_tfr_group(subjects=None, events=None, bands=None, save_figs=True,
                  batch_backend='Agg', tfr_params=None, paths=None):
    """
    Build every aggregate figure for the current cohort.

    A failing figure is reported and skipped rather than aborting the run, and figures
    render on a non-interactive backend by default with the previous one restored
    afterwards — the same contract as ``tfr_batch.run_tfr_batch``, so this can be called
    from the same kernel session that runs the interactive stage 1.

    Group statistics are not computed here. The cluster permutation test prototyped in
    notebooks/Main2.ipynb is deferred until data collection ends; with the handful of
    subjects available now it is powerless by construction, since
    ``permutation_cluster_1samp_test`` runs an exact test over 2**N sign-flips and the
    smallest attainable p-value is 1/2**N (0.0625 at N=4, i.e. never below 0.05).

    Returns
    -------
    dict
        ``{'group': <cohort namespace>, 'grand': {event: AverageTFR},
           'figures': [Path], 'failures': {step: message}, 'summary': Path|None}``
    """
    paths = paths or project_paths()
    tfr_params = dict(tfr_params or default_tfr_params())
    # `bands=` stays as a convenience for a quicker pass over one band. It is a view of
    # the same setting, so fold it in rather than letting the two disagree downstream.
    if bands is not None:
        tfr_params['bands'] = bands
    group = load_group_tfrs(subjects=subjects, events=events, paths=paths)
    grand = grand_average_tfrs(group)

    if not save_figs:
        return {'group': group, 'grand': grand, 'figures': [], 'failures': {},
                'summary': None}

    previous_backend = matplotlib.get_backend()
    if batch_backend is not None and previous_backend.lower() != batch_backend.lower():
        plt.close('all')
        matplotlib.use(batch_backend, force=True)
        print(f"rendering figures on '{batch_backend}' "
              f"(restoring '{previous_backend}' afterwards)")

    subject_dir = group_figure_dir('Subjects', paths)
    average_dir = group_figure_dir('GrandAverage', paths)
    contrast_dir = group_figure_dir('Contrasts', paths)

    steps = [(f'{band} subject grid',
              lambda band=band: save_subject_band_grid(group, grand, band, subject_dir,
                                                       paths, tfr_params))
             for band in tfr_params['bands']]
    steps += [
        ('grand-average panel',
         lambda: save_grand_average_panel(group, grand, average_dir, paths, tfr_params)),
        ('grand-average views',
         lambda: save_grand_average_figures(group, grand, average_dir, paths, tfr_params)),
        ('contrasts',
         lambda: save_group_contrast_figures(group, grand, contrast_dir, paths, tfr_params)),
    ]

    failures = {}
    try:
        for name, step in steps:
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
            try:
                step()
            except Exception as err:
                failures[name] = f'{type(err).__name__}: {err}'
                print(f"  !! FAILED [{name}]: {failures[name]}")
            finally:
                plt.close('all')
    finally:
        if batch_backend is not None and previous_backend.lower() != batch_backend.lower():
            matplotlib.use(previous_backend, force=True)

    summary_path = _write_summary(group, paths, tfr_params)
    figures = sorted((paths.figures / 'Group').rglob('*.png'))

    print(f"\n{'=' * 70}\nGroup summary: N = {len(group.subjects)}  {group.subjects}"
          f"\n{'=' * 70}")
    print(f"{'directory':<26}{'figures':>9}")
    for kind, out_dir in (('Subjects', subject_dir), ('GrandAverage', average_dir),
                          ('Contrasts', contrast_dir)):
        print(f"{f'Figures/Group/{kind}':<26}{len(list(out_dir.glob('*.png'))):>9}")
    for subject, reason in group.skipped.items():
        print(f"skipped {subject}: {reason}")
    if failures:
        print(f"\n{len(failures)} step(s) failed: {list(failures)}")

    return {'group': group, 'grand': grand, 'figures': figures,
            'failures': failures, 'summary': summary_path}


def _write_summary(group, paths=None, tfr_params=None):
    """
    Cohort and config provenance, so a figure can be traced back to what produced it.

    ``config`` is now the WHOLE tfr_params dict rather than eight hand-picked keys. The
    old block recorded ``freqs`` as its two endpoints and omitted ``n_cycles`` entirely
    - which is the setting that decides whether the bands are separable at all, so a
    summary without it could not distinguish the current pipeline from the one whose
    "Alpha" and "Beta" maps overlapped.
    """
    paths = paths or project_paths()
    summary = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'label': paths.label,
        'n_subjects': len(group.subjects),
        'subjects': group.subjects,
        'events': group.events,
        'skipped': group.skipped,
        'grid': {
            'n_channels': len(group.ch_names),
            'ch_names': group.ch_names,
            'freqs': [float(f) for f in group.freqs],
            'n_times': len(group.times),
            'tmin': float(group.times[0]),
            'tmax': float(group.times[-1]),
        },
        'config': jsonable(tfr_params or default_tfr_params()),
    }
    path = paths.figures / 'Group' / 'tfr_group_summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {path.relative_to(paths.root)}")
    return path
# %%
