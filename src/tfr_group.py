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
        -> Figures/Group/TimeBins/<band>_time_grid.png
           rows = conditions, columns = consecutive time bins, on a true PERCENT scale;
           the arithmetic mean of the subjects' percent change
        -> Figures/Group/TimeBins/<band>_time_grid_geomean.png
           the same grid, geometric mean of the subjects' power ratios
        -> Figures/Group/TimeBins/<band>_time_animation[_geomean].html
           the same sweep as a playable animation (play / pause / scrub), group only
        -> Figures/<subject>/TimeBins/<band>_time_grid.png  (the same, per subject)
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
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mne.time_frequency import read_tfrs
from mne.utils import use_log_level

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
    resolve_n_cycles,
    subject_figure_dir,
)

#%%
# ============================================================
# Paths
# ============================================================


def group_figure_dir(kind, paths=None):
    """``Figures/Group/<kind>/``, created on demand. kind: 'Subjects'|'GrandAverage'|'Contrasts'|'TimeBins'."""
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
# Time-binned topomap grid (stage 4b)
# ============================================================

# How to get from each apply_baseline mode back to the POWER RATIO, P / P_baseline.
#
# Everything the binned grid reports is derived from this ratio rather than from the
# stored numbers directly, which is what makes the figure independent of which tree it is
# run on: logratio, percent and ratio are three encodings of the same quantity, so all
# three must produce an identical grid. They did not always - the percent scale used to be
# reached by a per-mode formula, and the group mean it produced depended on which tree was
# canonical. Going through the ratio removes that coupling.
#
# MNE's mode='percent' is a FRACTION - (power - baseline) / baseline - which is why
# tfr_batch.BASELINE_MODE_LABELS calls it 'fractional change' and why it needs the +1 here
# and a 100x later. The modes that are absent are absent on purpose: 'mean' is an
# unnormalised power difference and the z-scores are in standard deviations, so neither is
# a ratio to the baseline and neither has a percent change to be recovered from it.
_RATIO_FROM_MODE = {
    'logratio': lambda data: 10.0 ** data,
    'percent': lambda data: 1.0 + data,
    'ratio': lambda data: data,
}

# Power is non-negative, so a percent-mode ratio is >= 0 and can be exactly 0 (a channel
# whose band power vanished). log(0) is -inf, which would propagate through the geometric
# mean and paint a whole topomap blank. Clip to something far below any real ERD - a
# ratio of 1e-12 is a 100% decrease to twelve decimal places - so the floor can never be
# mistaken for a measurement.
_RATIO_FLOOR = 1e-12


def to_ratio(tfr, tfr_params):
    """
    A copy of ``tfr`` with its data as the power ratio ``P / P_baseline``.

    The common currency for every percent-scale figure here: 1.0 is no change, 0.5 is a
    50% ERD, 2.0 is a doubling.
    """
    mode = tfr_params['mode']
    if mode not in _RATIO_FROM_MODE:
        raise ValueError(
            f"mode {mode!r} is not a ratio to the baseline, so no percent change can be "
            f"recovered from it. Re-run the TFRs with one of "
            f"{sorted(_RATIO_FROM_MODE)}, or read the binned grid in native units.")
    converted = tfr.copy()
    converted.data = _RATIO_FROM_MODE[mode](tfr.data)
    return converted


def to_percent_change(tfr, tfr_params):
    """
    A copy of ``tfr`` with its data rescaled to percent change from baseline.

    Negative is ERD (a power decrease) and positive is ERS, which is the sign convention
    the ``RdBu_r`` colormap already carries everywhere else here: blue is a decrease.
    """
    converted = to_ratio(tfr, tfr_params)
    converted.data = (converted.data - 1.0) * 100.0
    return converted


def percent_group(group, tfr_params):
    """
    The cohort on a percent scale, averaged ARITHMETICALLY: ``(pct_by_subject, pct_grand)``.

    Every subject is converted to percent first and the grand average is then built from
    the converted data, so ``pct_grand`` is the mean percent change - which is what a
    colorbar reading '%' is taken to mean, and the Pfurtscheller convention.

    This is deliberately not ``grand_average_tfrs`` followed by a conversion. That would
    average in whatever scale the tree happens to be stored in, so the same cohort would
    give a different answer on the logratio tree (a geometric mean) than on the percent
    tree (this one). See ``geometric_percent_grand`` for the other mean, computed
    explicitly rather than fallen into.

    The subject average stays equally weighted, for the reason in ``grand_average_tfrs``.
    """
    pct_by_subject = {
        subject: {event: to_percent_change(tfr, tfr_params)
                  for event, tfr in by_event.items()}
        for subject, by_event in group.by_subject.items()
    }
    pct_grand = {}
    for event in group.events:
        stack = [pct_by_subject[s][event] for s in group.subjects]
        averaged = stack[0].copy()
        averaged.data = np.stack([tfr.data for tfr in stack], axis=0).mean(axis=0)
        averaged.nave = len(stack)
        pct_grand[event] = averaged
    return pct_by_subject, pct_grand


def geometric_percent_grand(group, tfr_params):
    """
    The cohort on a percent scale, averaged GEOMETRICALLY: ``{event: AverageTFR}``.

    ``exp(mean_s(log(P_s / P_base_s))) - 1``, i.e. the geometric mean of the subjects'
    power ratios expressed as a percent change. Power ratios are multiplicative and
    right-skewed, so this damps a single extreme subject in a way the arithmetic mean does
    not; it is also how every other stage-4 figure averages, since they average the stored
    logratio data.

    Note what this takes: ``group``, not ``grand``. Converting ``grand`` is the obvious
    implementation and is right on exactly one of the two trees. ``grand_average_tfrs``
    averages whatever is stored, so on the logratio tree that is already a geometric mean
    and converting it works - but on the percent tree it is an ARITHMETIC mean, and
    converting it would return a second copy of ``percent_group``'s answer under a label
    saying 'geometric'. Recomputing from the per-subject ratios is correct on both.
    """
    geo = {}
    for event in group.events:
        ratios = np.stack(
            [to_ratio(group.by_subject[s][event], tfr_params).data
             for s in group.subjects], axis=0)
        mean_log = np.log(np.clip(ratios, _RATIO_FLOOR, None)).mean(axis=0)
        averaged = group.by_subject[group.subjects[0]][event].copy()
        averaged.data = (np.exp(mean_log) - 1.0) * 100.0
        averaged.nave = len(group.subjects)
        geo[event] = averaged
    return geo


def time_bin_edges(tfr_params, step=None):
    """
    ``[(t0, t1), ...]`` tiling ``time_grid_window`` exactly, at ``step`` seconds.

    ``step`` defaults to ``time_grid_step`` (the static grid's columns); the animation
    passes ``time_anim_step`` to get its finer frames off the same arithmetic, so the two
    cannot disagree about where a bin starts.

    Built by multiplying out the step rather than by ``np.arange``, whose accumulated
    float error at 0.2 s puts the last edge a hair past the window and can add a
    twenty-sixth, empty column.
    """
    start, end = tfr_params['time_grid_window']
    step = tfr_params['time_grid_step'] if step is None else step
    n_bins = int(round((end - start) / step))
    return [(start + i * step, start + (i + 1) * step) for i in range(n_bins)]


def _half_open(bins, tfr):
    """
    ``[(t0, t1), ...]`` shifted so each bin claims exactly the samples in ``[t0, t1)``.

    Both ``band_map`` and MNE's own time selection take ``tmin <= t <= tmax``, so
    contiguous bins would each claim the sample they share: every column would average 11
    samples instead of 10, with each boundary sample counted twice across the figure.

    Both edges move back by a QUARTER of a sample, not the right edge back by half. Half
    a sample is the arithmetically obvious shift and it silently loses samples, because
    it leaves each bin's left edge sitting exactly on one: the time vector carries float
    error - 0.2 s comes back as 0.19999999999999998 - so ``t >= 0.2`` rejects the very
    sample the bin was defined to start at, and the previous bin has already ended below
    it. That drops it from both, which showed up as columns averaging 9 samples instead
    of 10. A quarter-sample guard puts every edge a safe distance from any sample, so
    which bin a sample falls in is decided by the bins and not by the last bit of a float.
    """
    guard = float(np.diff(tfr.times).mean()) / 4
    return [(t0 - guard, t1 - guard) for t0, t1 in bins]


def time_grid_values(tfrs, band_name, tfr_params, step=None):
    """Every cell of one binned grid, ``{event: [per-bin channel map]}``."""
    fmin, fmax = tfr_params['bands'][band_name]
    bins = _half_open(time_bin_edges(tfr_params, step), next(iter(tfrs.values())))
    return {event: [band_map(tfr, fmin, fmax, t0, t1) for t0, t1 in bins]
            for event, tfr in tfrs.items()}


def time_scale_for_band(band_name, group_tfrs, tfr_params, by_subject=None):
    """
    The one symmetric colour scale every TimeBins artefact for a band shares.

    Pooled over BOTH bin schemes - the static grid's ``time_grid_step`` cells and the
    animation's finer ``time_anim_step`` frames - and over the per-subject grids when they
    are being written. Pooling both schemes is not fussiness: a narrower frame averages
    fewer samples and so reaches further into the tails, and a limit taken from the 0.2 s
    grid alone would clip the animation at exactly the moments it exists to show.

    ``group_tfrs`` is the list of group-level ``{event: AverageTFR}`` dicts to cover
    (arithmetic and geometric).
    """
    steps = [tfr_params['time_grid_step']]
    if tfr_params['time_anim_enabled']:
        steps.append(tfr_params['time_anim_step'])

    pooled = []
    for tfrs in group_tfrs:
        for step in steps:
            pooled += [v for maps in time_grid_values(tfrs, band_name, tfr_params,
                                                      step).values() for v in maps]
    for tfrs in (by_subject or {}).values():
        pooled += [v for maps in time_grid_values(tfrs, band_name,
                                                  tfr_params).values() for v in maps]
    return _shared_vlim(np.concatenate(pooled), pct=tfr_params['autoscale_pct'])


def save_time_binned_topomap_grid(tfrs, band_name, out_dir, filename, title,
                                  vlim=None, paths=None, tfr_params=None):
    """
    Rows = conditions, columns = consecutive time bins: where the ERD is, and when.

    The counterpart of ``save_grand_average_panel``, which collapses the whole
    ``active_window`` into one topomap per condition and so cannot show a time course at
    all. ``tfrs`` must already be on a percent scale (see ``to_percent_change``).

    The bins wrap into stacked blocks of ``tfr_params['time_grid_cols']`` columns, since
    twenty topomaps in a row is a figure nothing can be read off on screen. Every block
    carries its own time axis, and all of them share ONE symmetric colour scale - passed
    in as ``vlim`` when a set of figures has to be comparable, otherwise autoscaled to
    this figure at ``autoscale_pct``. Per-cell autoscaling would make a bin at 2% render
    identically to one at 40%, which would destroy the only thing the layout is for.

    Note the columns are not independent measurements: the analysis window is
    ``n_cycles / freq`` seconds long, typically several times the bin width, so adjacent
    columns are built from largely the same data. The suptitle states the overlap.
    """
    tfr_params = tfr_params or default_tfr_params()
    fmin, fmax = tfr_params['bands'][band_name]
    events = list(tfrs)
    bins = time_bin_edges(tfr_params)
    drawn = _half_open(bins, next(iter(tfrs.values())))
    step = tfr_params['time_grid_step']
    cols = min(tfr_params['time_grid_cols'], len(bins))
    n_blocks = int(np.ceil(len(bins) / cols))

    if vlim is None:
        values = time_grid_values(tfrs, band_name, tfr_params)
        vlim = _shared_vlim(np.concatenate([v for maps in values.values() for v in maps]),
                            pct=tfr_params['autoscale_pct'])

    # A hand-placed gridspec plus a hand-placed colorbar axes, rather than
    # fig.colorbar(ax=[...]) as the other group figures use: that call reflows every axes
    # it is given, which would slide the topomap columns out from under the time axes
    # drawn to line up with them.
    # Nested gridspecs, not one flat grid of every row: the time axis hangs its tick
    # labels and its 'Time (s)' below its own cell, which in a flat grid is the next
    # block's first row of topomaps and collides with them. An outer grid of blocks with
    # its own hspace puts real space between the blocks for those labels to live in.
    row_h, cell_w, axis_h = 1.55, 1.5, 0.5
    fig = plt.figure(figsize=(cell_w * cols + 2.6,
                              n_blocks * (row_h * len(events) + axis_h) + 1.4))
    outer = fig.add_gridspec(n_blocks, 1, left=0.13, right=0.86, top=0.90, bottom=0.05,
                             hspace=0.22)

    # plot_topomap logs "No baseline correction applied" once per call because it is
    # passed baseline=None - one line per cell, so 80 per figure and a couple of
    # thousand for a full run. It takes no `verbose` of its own in MNE 1.6, hence the
    # log level rather than a keyword. The TFRs were baselined in stage 3; there is
    # nothing to report.
    with use_log_level('ERROR'):
        for block in range(n_blocks):
            block_bins = list(range(block * cols, min((block + 1) * cols, len(bins))))
            gs = outer[block].subgridspec(
                len(events) + 1, cols, hspace=0.05, wspace=0.02,
                height_ratios=[row_h] * len(events) + [axis_h])
            for row, event in enumerate(events):
                for col, index in enumerate(block_bins):
                    ax = fig.add_subplot(gs[row, col])
                    t0, t1 = drawn[index]
                    tfrs[event].plot_topomap(
                        tmin=t0, tmax=t1,
                        fmin=fmin, fmax=fmax,
                        baseline=None, mode=None,
                        vlim=vlim, cmap=tfr_params['cmap'],
                        sensors=False,
                        axes=ax, show=False, colorbar=False,
                    )
                    if col == 0:
                        ax.annotate(event, xy=(-0.15, 0.5), xycoords='axes fraction',
                                    fontsize=11, fontweight='bold', ha='right', va='center')

            # Time axis under the block, spanning the same columns the topomaps sit in, so a
            # tick falls exactly on each bin boundary.
            axis = fig.add_subplot(gs[len(events), :len(block_bins)])
            start = bins[block_bins[0]][0]
            axis.set_xlim(start, bins[block_bins[-1]][1])
            ticks = [start + i * step for i in range(len(block_bins) + 1)]
            axis.set_xticks(ticks)
            axis.set_xticklabels([f'{t:g}' for t in ticks], fontsize=9)
            axis.set_yticks([])
            for side in ('top', 'left', 'right'):
                axis.spines[side].set_visible(False)
            axis.set_xlabel('Time (s)', fontsize=11, fontweight='bold', labelpad=2)

    cax = fig.add_axes([0.885, 0.30, 0.016, 0.40])
    sm = ScalarMappable(norm=Normalize(vmin=vlim[0], vmax=vlim[1]),
                        cmap=tfr_params['cmap'])
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('ERD / ERS  (%)', fontsize=11)

    longest = float((resolve_n_cycles(tfr_params) / np.asarray(
        tfr_params['freqs'], dtype=float)).max())
    overlap = max(0.0, 100 * (1 - step / longest))
    fig.suptitle(
        f'{title}\n{band_name} band ({fmin}-{fmax} Hz), {step:g} s bins over '
        f'{bins[0][0]:g}-{bins[-1][1]:g} s  -  {longest:.2f} s analysis window, so '
        f'adjacent columns overlap ~{overlap:.0f}%',
        fontsize=13, fontweight='bold')

    path = _savefig(fig, out_dir / filename, paths, tfr_params)
    plt.close(fig)
    return path


def save_group_time_grids(group, grand, out_dir, paths=None, tfr_params=None):
    """
    The binned grid for every band: both group means, and optionally each subject.

    Two group figures per band, because the two ways of averaging percent change across
    subjects are different quantities and the difference is worth seeing rather than
    assuming:

        <band>_time_grid.png          arithmetic mean of the subjects' percent change
        <band>_time_grid_geomean.png  geometric mean of the subjects' power ratios

    They are pooled into ONE colour scale, together with the per-subject grids. Letting
    each scale to itself is the obvious default and would defeat the point: the arithmetic
    mean's sensitivity to an extreme subject is visible only when both are drawn against
    the same limits. It is the same argument ``save_subject_band_grid`` makes for keeping
    subjects and the group on one scale.

    ``grand`` is accepted and deliberately not used. Both means are computed from
    ``group``, because ``grand`` averages whatever scale the tree is stored in and so
    means different things on the logratio and percent trees - see
    ``geometric_percent_grand``. It stays in the signature so this reads like every other
    figure function here and drops straight into ``run_tfr_group``'s step list.
    """
    tfr_params = tfr_params or default_tfr_params()
    pct_by_subject, pct_grand = percent_group(group, tfr_params)
    geo_grand = geometric_percent_grand(group, tfr_params)
    per_subject = tfr_params['time_grid_per_subject']
    n_subj = len(group.subjects)
    written = []

    for band_name in tfr_params['bands']:
        vlim = time_scale_for_band(
            band_name, [pct_grand, geo_grand], tfr_params,
            by_subject=pct_by_subject if per_subject else None)

        for tfrs, filename, mean_name in (
                (pct_grand, f'{band_name}_time_grid.png',
                 "arithmetic mean of the subjects' ERD%"),
                (geo_grand, f'{band_name}_time_grid_geomean.png',
                 "geometric mean of the subjects' power ratios")):
            written.append(save_time_binned_topomap_grid(
                tfrs, band_name, out_dir, filename,
                f'Grand average ERD / ERS over time (N={n_subj}) - {mean_name}',
                vlim=vlim, paths=paths, tfr_params=tfr_params))

        if per_subject:
            for subject in group.subjects:
                written.append(save_time_binned_topomap_grid(
                    pct_by_subject[subject], band_name,
                    subject_figure_dir(subject, 'TimeBins', paths),
                    f'{band_name}_time_grid.png',
                    f'{subject} - ERD / ERS over time '
                    f'(shared scale with the N={n_subj} group grids)',
                    vlim=vlim, paths=paths, tfr_params=tfr_params))

    return written


#%%
# ============================================================
# Time animation (stage 4b, playable)
# ============================================================


def build_time_animation(tfrs, band_name, title, vlim, tfr_params=None):
    """
    The binned grid as a playable sweep: ``(fig, anim)``.

    One topomap per condition, redrawn as the window steps through
    ``time_grid_window``, with a cursor tracking the position on a time axis underneath.
    The static grid shows every moment at once and is the one to read side by side; this
    is the one for watching a pattern arrive and go, which a grid of 25 small circles
    makes surprisingly hard.

    Each frame is drawn by the same ``plot_topomap`` call the grid uses, at the same
    ``vlim``, off bins from the same ``time_bin_edges`` arithmetic - so a frame and the
    grid cell covering it cannot drift apart in appearance or in which samples they average.

    ``tfrs`` must already be on a percent scale. Returns the animation rather than saving
    it, so a notebook can hand it straight to ``IPython.display.HTML(anim.to_jshtml())``
    without a file round trip.
    """
    tfr_params = tfr_params or default_tfr_params()
    fmin, fmax = tfr_params['bands'][band_name]
    events = list(tfrs)
    step = tfr_params['time_anim_step']
    bins = time_bin_edges(tfr_params, step)
    drawn = _half_open(bins, next(iter(tfrs.values())))
    start, end = tfr_params['time_grid_window']

    fig = plt.figure(figsize=(3.1 * len(events) + 1.8, 4.4),
                     dpi=tfr_params['time_anim_dpi'])
    gs = fig.add_gridspec(2, len(events), left=0.03, right=0.88, top=0.80, bottom=0.13,
                          hspace=0.10, wspace=0.02, height_ratios=[1.0, 0.16])
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(events))]

    # The cursor axis spans the whole window, so the marker's position is the fraction of
    # the trial elapsed - the thing a viewer needs while the topomaps are changing.
    cursor_ax = fig.add_subplot(gs[1, :])
    cursor_ax.set_xlim(start, end)
    cursor_ax.set_ylim(0, 1)
    cursor_ax.set_yticks([])
    cursor_ax.set_xticks(np.arange(start, end + 1e-9, 0.5))
    cursor_ax.set_xticklabels([f'{t:g}' for t in np.arange(start, end + 1e-9, 0.5)],
                              fontsize=8)
    for side in ('top', 'left', 'right'):
        cursor_ax.spines[side].set_visible(False)
    cursor_ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold', labelpad=1)
    elapsed = cursor_ax.axvspan(start, start, color='0.75', lw=0)
    cursor = cursor_ax.axvline(start, color='crimson', lw=2)

    cax = fig.add_axes([0.905, 0.30, 0.018, 0.48])
    sm = ScalarMappable(norm=Normalize(vmin=vlim[0], vmax=vlim[1]),
                        cmap=tfr_params['cmap'])
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label('ERD / ERS  (%)', fontsize=10)

    longest = float((resolve_n_cycles(tfr_params) / np.asarray(
        tfr_params['freqs'], dtype=float)).max())
    header = fig.suptitle('', fontsize=12, fontweight='bold')

    def draw(frame):
        t0, t1 = drawn[frame]
        for ax, event in zip(axes, events):
            ax.clear()
            tfrs[event].plot_topomap(
                tmin=t0, tmax=t1, fmin=fmin, fmax=fmax,
                baseline=None, mode=None,
                vlim=vlim, cmap=tfr_params['cmap'], sensors=False,
                axes=ax, show=False, colorbar=False,
            )
            ax.set_title(event, fontsize=11, fontweight='bold')
        label_t0, label_t1 = bins[frame]
        cursor.set_xdata([label_t1, label_t1])
        elapsed.set_width(label_t1 - start)
        header.set_text(
            f'{title}\n{band_name} band ({fmin}-{fmax} Hz)   '
            f't = {label_t0:.2f} - {label_t1:.2f} s   '
            f'(frame {frame + 1}/{len(bins)}; {longest:.2f} s analysis window)')
        return axes

    # blit=False because every frame clears and rebuilds its axes - there is no stable set
    # of artists for blitting to swap, and MNE redraws the head outline each call anyway.
    anim = FuncAnimation(fig, draw, frames=len(bins), interval=1000 / tfr_params['time_anim_fps'],
                         blit=False, repeat=True)
    return fig, anim


def save_time_animation(tfrs, band_name, out_dir, filename, title, vlim,
                        paths=None, tfr_params=None):
    """
    Write one animation as a self-contained ``.html`` with play / pause / scrub controls.

    ``to_jshtml`` embeds every frame as a base64 PNG, which is what makes the file portable
    - it needs no ffmpeg, no server and no sibling files - and also what makes it large.
    ``animation.embed_limit`` defaults to 20 MB and ``to_jshtml`` gives up with only a
    warning above it, producing an html that silently holds no frames, so the limit is
    raised for the duration of the call rather than left to chance. ``time_anim_dpi`` is the
    dial to turn if these get too big.
    """
    tfr_params = tfr_params or default_tfr_params()
    paths = paths or project_paths()
    fig, anim = build_time_animation(tfrs, band_name, title, vlim, tfr_params)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with plt.rc_context({'animation.embed_limit': 256}):
        path.write_text(anim.to_jshtml(fps=tfr_params['time_anim_fps'],
                                       default_mode='loop'), encoding='utf-8')
    plt.close(fig)
    print(f"  saved {path.relative_to(paths.root)}  "
          f"({path.stat().st_size / 1e6:.1f} MB, {len(time_bin_edges(tfr_params, tfr_params['time_anim_step']))} frames)")
    return path


def save_group_time_animations(group, grand, out_dir, paths=None, tfr_params=None):
    """
    The playable counterpart of ``save_group_time_grids``: both means, every band.

    Group only. A per-subject set would be 15 subjects x 2 bands of embedded-PNG html, which
    is hundreds of megabytes for something no one watches; the per-subject question is
    answered by the static grids, which stay comparable because they share this scale.
    """
    tfr_params = tfr_params or default_tfr_params()
    pct_by_subject, pct_grand = percent_group(group, tfr_params)
    geo_grand = geometric_percent_grand(group, tfr_params)
    n_subj = len(group.subjects)
    written = []

    for band_name in tfr_params['bands']:
        vlim = time_scale_for_band(
            band_name, [pct_grand, geo_grand], tfr_params,
            by_subject=pct_by_subject if tfr_params['time_grid_per_subject'] else None)

        for tfrs, filename, mean_name in (
                (pct_grand, f'{band_name}_time_animation.html',
                 "arithmetic mean of the subjects' ERD%"),
                (geo_grand, f'{band_name}_time_animation_geomean.html',
                 "geometric mean of the subjects' power ratios")):
            written.append(save_time_animation(
                tfrs, band_name, out_dir, filename,
                f'Grand average ERD / ERS (N={n_subj}) - {mean_name}',
                vlim, paths=paths, tfr_params=tfr_params))

    return written


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
    timebin_dir = group_figure_dir('TimeBins', paths)

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
        ('time-binned grids',
         lambda: save_group_time_grids(group, grand, timebin_dir, paths, tfr_params)),
    ]
    if tfr_params['time_anim_enabled']:
        steps.append(
            ('time animations',
             lambda: save_group_time_animations(group, grand, timebin_dir, paths,
                                                tfr_params)))

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
                          ('Contrasts', contrast_dir), ('TimeBins', timebin_dir)):
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
