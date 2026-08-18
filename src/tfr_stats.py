#%%
"""
Group-level (stage 5) inferential statistics on the per-subject TFRs.

Stage 4 (src/tfr_group.py) says what the cohort looks like. This says which of it
survives a test. Two tiers, because they answer different questions and have very
different power at N=10:

    run_roi_tests()       TIER 1, confirmatory. One number per subject/condition from
                          an a-priori sensorimotor ROI x band x active window, tested
                          with an EXACT paired sign-flip permutation t. This is where
                          the statistical power is.

    run_cluster_tests()   TIER 2, exploratory. Cluster-based permutation over
                          channels x freqs x times with real 3-D adjacency
                          (Maris & Oostenveld 2007), so effects that were not
                          pre-specified can still be found with FWER control across
                          the whole ~112k-point space.

    run_tfr_stats()       Both, plus figures and Figures/<label>/Group/Stats/
                          tfr_stats_summary.json.

WHY THIS EXISTS NOW. The stage-4 docs used to say group statistics had to wait,
because the sign-flip test is exact over ``2**N`` flips and at N=4 no cluster could
ever come out significant. At N=10 the test finally has room to reject.

The exact floor is ``2 / 2**N``, not ``1 / 2**N``: flipping EVERY subject's sign
negates t and so leaves |t| - and the two-tailed cluster statistic - untouched, so the
observed labelling always ties with its own mirror image and no two-tailed p can come
in below two counts. There are really only ``2**(N-1)`` distinct sign patterns, which
is what MNE enumerates. At N=4 that put the floor at 1/8 = 0.125; at N=10 it is
1/512 = 0.00195, which is the number that made this stage worth writing.

WHICH TREE TO RUN ON. Prefer the logratio tree (``project_paths(label=None)``) over
``TFRs/percent/``. Power ratios are multiplicative and right-skewed; the log makes the
per-subject differences much closer to symmetric, which is what both the sign-flip null
and the t-statistic want. The percent tree is for reporting effect magnitudes, not for
inference. Whichever tree is used is recorded in the summary JSON.

HOW TO READ A CLUSTER RESULT. A significant cluster licenses the claim "the conditions
differ". It does NOT establish that they differ *at* those channels, times or
frequencies - cluster extent is not a confidence interval on the effect's location
(Sassenhagen & Draschkow 2019). Every printed summary here repeats that, because the
per-channel figures below are exactly the kind that invite the forbidden reading.

Driven from notebooks/TFR_Analysis.ipynb. Imports src.tfr_group -> src.tfr_batch ->
src.preprocessing, which runs `%matplotlib qt` at import time, so this needs an IPython
kernel (notebook or `ipython`), not a bare `python` process.
"""

import itertools
import json
from datetime import datetime
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.channels import find_ch_adjacency
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy.stats import t as t_dist

from . import params_spec
from .tfr_batch import ELECTRODE_GROUPS, _savefig, jsonable, project_paths
from .tfr_group import load_group_tfrs

#%%
# ============================================================
# Configuration
# ============================================================

# A-priori sensorimotor ROIs, taken from the montage's own electrode groups
# (tfr_batch.ELECTRODE_GROUPS) rather than invented here. Left/right are the hand-knob
# neighbourhoods whose mu ERD should lateralise with the imagined hand; 'mid' is the
# medial strip that a midline / third-limb representation would be expected to load on.
#
# FCz is deliberately absent: this pipeline runs with AddRefChannel=False, so the online
# reference is never reconstructed and there is no FCz in the data to average in.
SENSORIMOTOR_ROIS = {
    'left':  ['FC5', 'FC3', 'C5', 'C3', 'CP5', 'CP3'],
    'right': ['FC4', 'FC6', 'C4', 'C6', 'CP4', 'CP6'],
    'mid':   ['FC1', 'FC2', 'C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2'],
}


def default_stats_params():
    """
    The statistical design, spelled out so it is pre-specified rather than chosen after
    looking at the maps.

    Everything that decides WHAT is tested lives here, which is the whole point: a
    contrast added after seeing the result is a different test with a different error
    rate, and this dict is what the summary JSON records.
    """
    return {
        # --- design: what is tested, and over what ---------------------------------
        'rois': {name: list(chs) for name, chs in SENSORIMOTOR_ROIS.items()},
        'bands': {'Mu': (8, 13), 'Beta': (13, 30)},
        'active_window': (0.5, 3.5),
        # Tier 1 only. "Is there any ERD at all", per condition, against its own
        # baseline. Valid because active_window does not overlap the baseline window.
        'zero_conditions': ['MiddleHand', 'RightHand', 'LeftHand', 'FixatedRest'],
        # The pre-specified contrast family. FWER is controlled WITHIN each contrast
        # (tmax in tier 1, cluster in tier 2) and never across them, so Holm is applied
        # across this list - which means adding a contrast here weakens every other one.
        'contrasts': [
            ('RightHand', 'LeftHand'),       # lateralisation; the sanity check
            ('MiddleHand', 'RightHand'),     # third arm vs a natural hand
            ('MiddleHand', 'LeftHand'),
            ('MiddleHand', 'FixatedRest'),   # third arm vs rest
        ],
        'alpha': 0.05,

        # --- tier 1: a-priori ROI test ----------------------------------------------
        # 'all' -> every one of the 2**N sign flips, i.e. the exact test. At N=10 that
        # is 1024 and costs nothing; above ~20 subjects it is sampled instead.
        'roi_n_permutations': 'all',

        # --- tier 2: cluster-based permutation --------------------------------------
        # Which contrasts get the expensive test. The 'vs zero' tests are deliberately
        # NOT here: with an effect that large the test returns one enormous cluster,
        # which costs a fortune and says nothing tier 1 has not already said better.
        'cluster_contrasts': [
            ('RightHand', 'LeftHand'),
            ('MiddleHand', 'RightHand'),
            ('MiddleHand', 'LeftHand'),
            ('MiddleHand', 'FixatedRest'),
        ],
        'cluster_freq_range': (8, 30),
        # The TFR is smooth on the scale of its analysis window (T = 1 s when
        # n_cycles = freqs), so testing every 20 ms sample is pure oversampling: it
        # inflates cluster extent in time and costs a fortune for no added information.
        'cluster_time_step': 0.05,
        # Cluster-FORMING threshold as a two-tailed p, converted to t at df = N-1. This
        # is not the significance level - it only decides which bins are eligible to
        # join a cluster, and so trades focal-strong effects against broad-weak ones.
        'cluster_p_threshold': 0.05,
        'cluster_n_permutations': 1024,
        'cluster_tail': 0,
        'n_jobs': -1,

        # --- figures ------------------------------------------------------------------
        'tf_channels': ['C3', 'Cz', 'C4'],
        'figure_dpi': 150,
    }


STATS_PARAM_GROUPS = {
    'design': ('rois', 'bands', 'active_window', 'zero_conditions', 'contrasts', 'alpha'),
    'roi': ('roi_n_permutations',),
    'cluster': ('cluster_contrasts', 'cluster_freq_range', 'cluster_time_step',
                'cluster_p_threshold', 'cluster_n_permutations', 'cluster_tail',
                'n_jobs'),
    'figures': ('tf_channels', 'figure_dpi'),
}

STATS_PARAM_DOCS = {
    'rois': 'a-priori channel groups each tier-1 value is averaged over',
    'bands': 'named frequency bands, Hz',
    'active_window': 'window every tier-1 value averages over, s; must miss the baseline',
    'zero_conditions': 'conditions tested against 0 (is there any ERD), tier 1 only',
    'contrasts': 'the pre-specified contrast family; Holm is applied across it',
    'alpha': 'significance level for the reported verdicts',
    'roi_n_permutations': "sign flips for tier 1; 'all' = the exact test",
    'cluster_contrasts': 'which contrasts get the expensive cluster test',
    'cluster_freq_range': 'frequencies the cluster test runs over, Hz',
    'cluster_time_step': 'temporal resolution the cluster test subsamples to, s',
    'cluster_p_threshold': 'cluster-FORMING threshold as a two-tailed p, not the alpha',
    'cluster_n_permutations': 'sign flips for tier 2; capped at 2**N, which is exact',
    'cluster_tail': '0 = two-tailed, 1 = greater, -1 = less',
    'n_jobs': 'parallel jobs for the cluster test (-1 = all cores)',
    'tf_channels': 'channels the per-channel time-frequency figures are drawn for',
    'figure_dpi': 'resolution every figure is saved at',
}

KNOWN_STATS_KEYS = frozenset(k for group in STATS_PARAM_GROUPS.values() for k in group)

_STATS_GROUP_TITLES = {
    'design': 'STATS - DESIGN (pre-specify this, do not tune it on the result)',
    'roi': 'STATS - TIER 1, A-PRIORI ROI TEST',
    'cluster': 'STATS - TIER 2, CLUSTER PERMUTATION TEST',
    'figures': 'STATS - FIGURES',
}

_STATS_SUMMARIZE = {'zero_conditions': 'conditions', 'tf_channels': 'ch'}


def check_stats_params(stats_params, tfr_params=None, group=None):
    """
    Raise on a stats dict that would test something other than what it says.

    Mirrors ``tfr_batch.check_tfr_params``: a misspelled key is a silent no-op, which
    for a statistical design means quietly running a different test than the one
    written down, so it is an error rather than a warning.
    """
    unknown = params_spec.unknown_keys(stats_params, KNOWN_STATS_KEYS)
    if unknown:
        raise ValueError(f"STATS keys read by nothing: {', '.join(unknown)}. "
                         f"A misspelled key is silently ignored, so this is an error.")

    if not stats_params['rois']:
        raise ValueError("rois is empty: tier 1 would have nothing to average over.")
    if not stats_params['bands']:
        raise ValueError("bands is empty: tier 1 would have nothing to average over.")

    start, end = stats_params['active_window']
    if start >= end:
        raise ValueError(f"active_window = {stats_params['active_window']} does not "
                         f"run forwards.")
    # The tier-1 'vs zero' tests ask whether the post-cue change differs from the
    # baseline. If the window overlapped the baseline the answer would be partly
    # circular - the baseline was subtracted to make exactly that region zero.
    if tfr_params is not None and start < tfr_params['baseline'][1]:
        raise ValueError(
            f"active_window starts at {start} s but the baseline runs to "
            f"{tfr_params['baseline'][1]} s. Testing inside the baseline is circular: "
            f"that region was subtracted to be zero by construction.")

    if not 0 < stats_params['alpha'] < 1:
        raise ValueError(f"alpha = {stats_params['alpha']} is not in (0, 1).")
    if not 0 < stats_params['cluster_p_threshold'] < 1:
        raise ValueError(f"cluster_p_threshold = {stats_params['cluster_p_threshold']} "
                         f"is not in (0, 1).")
    if stats_params['cluster_tail'] not in (-1, 0, 1):
        raise ValueError(f"cluster_tail = {stats_params['cluster_tail']}; "
                         f"expected -1, 0 or 1.")
    if stats_params['cluster_time_step'] <= 0:
        raise ValueError("cluster_time_step must be > 0 s.")

    fmin, fmax = stats_params['cluster_freq_range']
    if fmin >= fmax:
        raise ValueError(f"cluster_freq_range = {stats_params['cluster_freq_range']} "
                         f"does not run forwards.")

    # Every contrast must name conditions that exist, and every ROI channel must be in
    # the montage. Both fail late and confusingly otherwise - a missing ROI channel
    # would silently shrink the average rather than raise.
    if group is not None:
        known = set(group.events)
        named = {c for c in stats_params['zero_conditions']}
        named |= {c for pair in stats_params['contrasts'] for c in pair}
        named |= {c for pair in stats_params['cluster_contrasts'] for c in pair}
        missing = sorted(named - known)
        if missing:
            raise ValueError(f"STATS names conditions the cohort does not have: "
                             f"{missing}. Available: {sorted(known)}.")
        for roi, channels in stats_params['rois'].items():
            absent = [ch for ch in channels if ch not in group.ch_names]
            if absent:
                raise ValueError(
                    f"ROI {roi!r} names channels not in the montage: {absent}. "
                    f"(FCz is absent by design - this pipeline runs "
                    f"AddRefChannel=False, so the online reference is never rebuilt.)")

    if group is not None:
        n_subjects = len(group.subjects)
        floor = two_tailed_p_floor(n_subjects)
        if floor > stats_params['alpha']:
            raise ValueError(
                f"N = {n_subjects} gives only 2**({n_subjects}-1) = "
                f"{2 ** (n_subjects - 1)} distinct sign patterns, so the smallest "
                f"attainable two-tailed p is {floor:.4f} - already above alpha = "
                f"{stats_params['alpha']}. No result could ever be significant; this "
                f"is the reason stage 5 waited for the cohort to grow.")
    return stats_params


def two_tailed_p_floor(n_subjects):
    """
    The smallest p an exact two-tailed sign-flip test can return at this N.

    ``2 / 2**N``, not ``1 / 2**N``. Flipping every subject's sign negates t and leaves
    |t| unchanged, so the observed labelling always ties with its own mirror and the
    count can never be 1. Equivalently there are only ``2**(N-1)`` distinct patterns.
    """
    return 2.0 / 2 ** n_subjects


def describe_stats_params(stats_params):
    """Print the design back, marking anything that departs from the library default."""
    params_spec.describe(
        stats_params, groups=STATS_PARAM_GROUPS, docs=STATS_PARAM_DOCS,
        defaults=default_stats_params(), titles=_STATS_GROUP_TITLES,
        known=KNOWN_STATS_KEYS, summarize=_STATS_SUMMARIZE)
    n_tests = len(stats_params['zero_conditions']) + len(stats_params['contrasts'])
    print(f"  -> tier 1 family: {n_tests} tests, Holm-corrected across all of them")
    print(f"  -> tier 2: {len(stats_params['cluster_contrasts'])} cluster test(s)")
    print("  ('*' = differs from the default)")


#%%
# ============================================================
# The exact sign-flip test
# ============================================================


def _t_stat(X):
    """
    One-sample t along the first axis, with a zero-variance cell mapped to t = 0.

    A cell where every subject has the identical value carries no evidence either way;
    letting it come out as inf would hand it the largest statistic in the family and
    let it dominate the tmax null.
    """
    n = X.shape[0]
    sd = X.std(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = X.mean(axis=0) / (sd / np.sqrt(n))
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def sign_flip_orders(n_samples, n_permutations='all', max_exact=20, seed=0):
    """
    The sign vectors to test, and whether they are the complete set.

    Under the paired null the sign of each subject's difference is arbitrary, so the
    null distribution is generated by flipping them. With ``n_samples`` subjects there
    are only ``2**n_samples`` distinct flips: below ``max_exact`` we enumerate all of
    them and the p-value is exact, above it we sample.

    The observed labelling (all +1) is always included, which is what stops a p-value
    of exactly 0. For a two-tailed statistic the floor is 2/len(orders), not
    1/len(orders): the all-negative vector is in here too and mirrors the observed one
    (see ``two_tailed_p_floor``).
    """
    n_exact = 2 ** n_samples
    wanted = n_exact if n_permutations == 'all' else int(n_permutations)
    if n_samples <= max_exact and n_exact <= wanted:
        bits = ((np.arange(n_exact)[:, None] >> np.arange(n_samples)) & 1)
        return 1.0 - 2.0 * bits.astype(float), True
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(wanted, n_samples))
    signs[0] = 1.0
    return signs, False


def exact_sign_flip_test(X, n_permutations='all', seed=0):
    """
    Paired permutation t-test over ``X`` of shape ``(n_subjects, n_cells)``.

    Returns the observed t per cell, the uncorrected p, and the p corrected across
    cells by the **tmax** method: compare each cell's |t| against the distribution of
    the LARGEST |t| anywhere in the family. That controls the family-wise error rate
    like Bonferroni but is strictly more powerful when the cells are correlated
    (Nichols & Holmes 2002) - and ROI x band cells drawn from overlapping electrodes
    and adjacent frequencies certainly are.

    Returns
    -------
    SimpleNamespace
        ``t, p_raw, p_tmax, n_permutations, exact, dz``
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_subjects = X.shape[0]

    signs, exact = sign_flip_orders(n_subjects, n_permutations, seed=seed)
    t_obs = _t_stat(X)
    # (n_perm, n_subjects, n_cells) -> t along the subject axis. At 1024 x 10 x ~6 this
    # is a few hundred KB, so the whole null is built in one go rather than looped.
    t_null = _t_stat(np.swapaxes(signs[:, :, None] * X[None, :, :], 0, 1))

    abs_null, abs_obs = np.abs(t_null), np.abs(t_obs)
    p_raw = (abs_null >= abs_obs[None, :]).mean(axis=0)
    p_tmax = (abs_null.max(axis=1)[:, None] >= abs_obs[None, :]).mean(axis=0)

    # Cohen's dz - the paired effect size. At N=10 this matters more than the p-value:
    # a test this small can only reject on a large effect, so reporting the size is
    # what makes a null result interpretable.
    sd = X.std(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dz = np.nan_to_num(X.mean(axis=0) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    return SimpleNamespace(t=t_obs, p_raw=p_raw, p_tmax=p_tmax, dz=dz,
                           n_permutations=len(signs), exact=exact)


def holm_adjusted(pvalues):
    """
    Holm-Bonferroni adjusted p-values, in the input's order.

    Used across the CONTRAST family, one level above the within-contrast correction:
    tmax and the cluster test each control the error rate inside one contrast, and
    nothing in either of them knows that seven other contrasts were also run.
    """
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted, running, m = np.empty(len(p)), 0.0, len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


#%%
# ============================================================
# Tier 1 - a-priori ROI x band
# ============================================================


def roi_band_values(group, stats_params):
    """
    One number per subject, condition, ROI and band.

    ``{(condition, roi, band): array of shape (n_subjects,)}``, in ``group.subjects``
    order so a paired difference is matched by subject. Averaging over the ROI, the
    band and the active window collapses ~112k tested points down to a handful, which
    is the entire reason tier 1 has power that tier 2 does not.
    """
    tmin, tmax = stats_params['active_window']
    ch_index = {ch: i for i, ch in enumerate(group.ch_names)}
    time_mask = (group.times >= tmin) & (group.times <= tmax)

    values = {}
    for condition in group.events:
        for roi, channels in stats_params['rois'].items():
            picks = [ch_index[ch] for ch in channels]
            for band, (fmin, fmax) in stats_params['bands'].items():
                freq_mask = (group.freqs >= fmin) & (group.freqs <= fmax)
                values[(condition, roi, band)] = np.array([
                    group.by_subject[s][condition]
                    .data[picks][:, freq_mask][:, :, time_mask].mean()
                    for s in group.subjects])
    return values


def _tier1_cells(stats_params):
    """The (roi, band) cells of one tier-1 test, in a fixed reported order."""
    return [(roi, band) for roi in stats_params['rois']
            for band in stats_params['bands']]


def run_roi_tests(group, stats_params=None, verbose=True):
    """
    Tier 1: the confirmatory tests, on the pre-specified ROI x band cells.

    Each test is one exact paired sign-flip permutation t over the ROI x band cells,
    tmax-corrected within the test. A test against ``None`` is the one-sample "is there
    any ERD" question; a pair is the paired difference ``cond1 - cond2``.

    Holm is then applied across the whole family, using each test's smallest
    tmax-corrected p as that test's p. Both the within-test and the across-family
    numbers are reported, because they answer different questions: "does this contrast
    show anything" versus "does anything in this study show anything".
    """
    stats_params = stats_params or default_stats_params()
    values = roi_band_values(group, stats_params)
    cells = _tier1_cells(stats_params)

    tests = [(c, None) for c in stats_params['zero_conditions']]
    tests += [tuple(pair) for pair in stats_params['contrasts']]

    results = []
    for cond1, cond2 in tests:
        X = np.column_stack([
            values[(cond1, roi, band)] if cond2 is None
            else values[(cond1, roi, band)] - values[(cond2, roi, band)]
            for roi, band in cells])
        out = exact_sign_flip_test(X, stats_params['roi_n_permutations'])
        results.append({
            'contrast': cond1 if cond2 is None else f'{cond1} - {cond2}',
            'cond1': cond1, 'cond2': cond2,
            'kind': 'vs zero' if cond2 is None else 'paired difference',
            'n_permutations': int(out.n_permutations), 'exact': bool(out.exact),
            'cells': [{'roi': roi, 'band': band,
                       'mean': float(X[:, i].mean()), 'dz': float(out.dz[i]),
                       't': float(out.t[i]), 'p_raw': float(out.p_raw[i]),
                       'p_tmax': float(out.p_tmax[i])}
                      for i, (roi, band) in enumerate(cells)],
            'p_test': float(out.p_tmax.min()),
        })

    for row, p_holm in zip(results, holm_adjusted([r['p_test'] for r in results])):
        row['p_holm'] = float(p_holm)

    if verbose:
        _print_roi_tests(results, group, stats_params)
    return results


def _print_roi_tests(results, group, stats_params):
    """The tier-1 table, plus the caveats that belong beside the numbers."""
    alpha = stats_params['alpha']
    n_subjects = len(group.subjects)
    exact = all(r['exact'] for r in results)
    n_perm = results[0]['n_permutations'] if results else 0

    print(f"\n{'=' * 78}")
    print(f"TIER 1 - a-priori ROI x band, paired sign-flip permutation t  (N = {n_subjects})")
    print(f"{'=' * 78}")
    print(f"  {'EXACT' if exact else 'sampled'} test over {n_perm} sign flips; "
          f"smallest attainable two-tailed p = "
          f"{two_tailed_p_floor(n_subjects) if exact else 2 / n_perm:.4f}")
    print(f"  within a test: tmax across {len(_tier1_cells(stats_params))} ROI x band "
          f"cells.  across the {len(results)} tests: Holm.")
    bands = ', '.join(f'{name} {lo}-{hi} Hz'
                      for name, (lo, hi) in stats_params['bands'].items())
    print(f"  window {stats_params['active_window'][0]}-"
          f"{stats_params['active_window'][1]} s;  bands: {bands}")

    for row in results:
        verdict = 'SIGNIFICANT' if row['p_holm'] < alpha else 'n.s.'
        print(f"\n  {row['contrast']}   [{row['kind']}]")
        print(f"    test p = {row['p_test']:.4f} (tmax)   "
              f"Holm p = {row['p_holm']:.4f}   -> {verdict} at alpha={alpha}")
        print(f"    {'roi':<7}{'band':<7}{'mean':>9}{'dz':>8}{'t':>8}"
              f"{'p_raw':>9}{'p_tmax':>9}")
        for cell in row['cells']:
            star = ' *' if cell['p_tmax'] < alpha else '  '
            print(f"    {cell['roi']:<7}{cell['band']:<7}{cell['mean']:>9.3f}"
                  f"{cell['dz']:>8.2f}{cell['t']:>8.2f}"
                  f"{cell['p_raw']:>9.4f}{cell['p_tmax']:>9.4f}{star}")
        if row['cond2'] == 'FixatedRest':
            print("    NOTE: both sides are already baselined to fixation, so this is "
                  "NOT raw ERD -\n"
                  "          it is what is left after removing whatever the cue evokes "
                  "in every\n          condition alike.")
    print(f"\n  ('*' = p_tmax < {alpha} within its own test; the Holm column is what "
          f"survives the whole family)")


#%%
# ============================================================
# Tier 2 - cluster-based permutation
# ============================================================


def _eeg_info_for_adjacency(info):
    """
    A plain EEG ``Info`` carrying the same sensor positions.

    ``find_ch_adjacency`` only understands ``ch_type='eeg'``, but these TFRs are
    CSD-transformed (channel type 'csd', unit V/m^2) and ``set_channel_types`` refuses
    to convert them ("Channel ... has unknown unit (117)"). Rebuilding the Info is the
    way to get the neighbour graph - and the same trick is what lets ``plot_topomap``
    draw these channels.
    """
    eeg_info = mne.create_info(list(info['ch_names']), info['sfreq'], ch_types='eeg')
    for src, dst in zip(info['chs'], eeg_info['chs']):
        dst['loc'] = src['loc'].copy()
    return eeg_info


def run_cluster_test(group, cond1, cond2=None, stats_params=None, verbose=True):
    """
    Tier 2: one cluster-based permutation test over channels x freqs x times.

    ONE test over the whole space with an explicit 3-D adjacency, so the family-wise
    error rate is controlled across all three dimensions at once and clusters are free
    to extend across neighbouring electrodes. This is the shape that matters: an
    earlier version of this test (a) ran an independent test per channel and reported
    every p < 0.05 with no correction across channels, and (b) flattened freq x time
    into a 1-D vector, which with ``adjacency=None`` makes MNE infer a 1-D lattice -
    destroying frequency adjacency and making the last time bin of one frequency row a
    neighbour of the first bin of the next. A single broad ERD blob was therefore
    shredded into one thin sliver per frequency row, each reported as its own cluster.

    ``cond2=None`` tests ``cond1`` against zero.
    """
    stats_params = stats_params or default_stats_params()
    n_subjects = len(group.subjects)
    fmin, fmax = stats_params['cluster_freq_range']
    tmin, tmax = stats_params['active_window']

    freq_mask = (group.freqs >= fmin) & (group.freqs <= fmax)
    in_window = np.where((group.times >= tmin) & (group.times <= tmax))[0]
    dt = float(np.median(np.diff(group.times)))
    stride = max(1, int(round(stats_params['cluster_time_step'] / dt)))
    time_idx = in_window[::stride]

    freqs_sel, times_sel = group.freqs[freq_mask], group.times[time_idx]
    n_freqs, n_times = len(freqs_sel), len(times_sel)
    n_channels = len(group.ch_names)
    label = cond1 if cond2 is None else f'{cond1} - {cond2}'

    # Paired within-subject difference -> (n_subj, n_freqs, n_times, n_channels).
    # by_subject keys the lookup by subject code, so the pairing cannot silently drift
    # the way indexing two parallel lists by position could.
    data = np.zeros((n_subjects, n_freqs, n_times, n_channels))
    for i, subject in enumerate(group.subjects):
        d1 = group.by_subject[subject][cond1].data[:, freq_mask, :][:, :, time_idx]
        if cond2 is None:
            diff = d1
        else:
            d2 = group.by_subject[subject][cond2].data[:, freq_mask, :][:, :, time_idx]
            diff = d1 - d2
        data[i] = diff.transpose(1, 2, 0)

    df = n_subjects - 1
    threshold = t_dist.ppf(1 - stats_params['cluster_p_threshold'] / 2, df)
    n_perm = stats_params['cluster_n_permutations']
    n_exact = 2 ** n_subjects

    if verbose:
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(f"  {n_freqs} freqs ({freqs_sel[0]:.0f}-{freqs_sel[-1]:.0f} Hz) x "
              f"{n_times} times ({times_sel[0]:.2f}-{times_sel[-1]:.2f} s @ "
              f"{stride * dt * 1000:.0f} ms) x {n_channels} ch"
              f"  ->  {n_freqs * n_times * n_channels} points in one family")
        print(f"  cluster-forming threshold t = {threshold:.3f} "
              f"(two-tailed p<{stats_params['cluster_p_threshold']}, df={df})")
        if n_exact <= n_perm:
            print(f"  only 2^{n_subjects} = {n_exact} sign flips exist -> EXACT test "
                  f"(n_permutations={n_perm} is never reached); smallest attainable "
                  f"two-tailed p = {two_tailed_p_floor(n_subjects):.4f}")

    T_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
        data, adjacency=_tfr_adjacency(group, n_freqs, n_times),
        threshold=threshold, tail=stats_params['cluster_tail'],
        n_permutations=n_perm, out_type='mask', n_jobs=stats_params['n_jobs'],
        verbose=False)

    found = []
    for mask, p_value in zip(clusters, cluster_p):
        f_i, t_i, c_i = np.where(mask)
        found.append({
            'mask': mask, 'p_value': float(p_value), 'size': int(mask.sum()),
            'freq_range': (float(freqs_sel[f_i.min()]), float(freqs_sel[f_i.max()])),
            'time_range': (float(times_sel[t_i.min()]), float(times_sel[t_i.max()])),
            'channels': sorted({group.ch_names[k] for k in c_i}),
            'mean_T': float(T_obs[mask].mean()),
        })
    significant = sorted([c for c in found if c['p_value'] < stats_params['alpha']],
                         key=lambda c: c['p_value'])

    result = {
        'contrast': label, 'cond1': cond1, 'cond2': cond2,
        'T_obs': T_obs, 'clusters': found, 'sig_clusters': significant,
        'freqs': freqs_sel, 'times': times_sel, 'ch_names': list(group.ch_names),
        'info': group.info, 'threshold': float(threshold),
        'n_permutations': int(min(n_perm, n_exact)), 'exact': n_exact <= n_perm,
        'p_test': float(min(cluster_p)) if len(cluster_p) else 1.0,
    }
    if verbose:
        _print_cluster_result(result, stats_params)
    return result


def _tfr_adjacency(group, n_freqs, n_times):
    """
    Neighbour graph over freq x time x channel, in the order the data's axes are in.

    The channel order is asserted against the data rather than trusted: a permuted
    list would scramble the neighbour graph silently instead of raising, and every
    cluster after that would be an artefact of the wrong topography.
    """
    ch_adjacency, names = find_ch_adjacency(_eeg_info_for_adjacency(group.info),
                                            ch_type='eeg')
    if list(names) != list(group.ch_names):
        raise ValueError("adjacency channel order does not match the data; the "
                         "neighbour graph would be built on the wrong topography.")
    return combine_adjacency(n_freqs, n_times, ch_adjacency)


def _print_cluster_result(result, stats_params):
    """The cluster table, and the inference the table does NOT license."""
    alpha = stats_params['alpha']
    print(f"  clusters found: {len(result['clusters'])}   "
          f"significant (p<{alpha}): {len(result['sig_clusters'])}")
    for cluster in result['sig_clusters']:
        print(f"    p={cluster['p_value']:.4f}  {cluster['size']:6d} bins  "
              f"{cluster['freq_range'][0]:.0f}-{cluster['freq_range'][1]:.0f} Hz  "
              f"{cluster['time_range'][0]:.2f}-{cluster['time_range'][1]:.2f} s  "
              f"mean T={cluster['mean_T']:+.2f}")
        print(f"      {len(cluster['channels'])} channels: "
              f"{', '.join(cluster['channels'])}")
    if result['sig_clusters']:
        print("  READ THIS AS: the conditions differ. NOT that they differ AT these\n"
              "  channels/times/frequencies - cluster extent is not a confidence\n"
              "  interval on the effect's location (Sassenhagen & Draschkow 2019).")
    if result['cond2'] == 'FixatedRest':
        print("  NOTE: both sides are already baselined to fixation, so this is not raw\n"
              "  ERD - it is what is left after removing the cue response common to all.")


def run_cluster_tests(group, stats_params=None, verbose=True):
    """Every contrast in ``cluster_contrasts``, Holm-corrected across the family."""
    stats_params = stats_params or default_stats_params()
    results = []
    for cond1, cond2 in stats_params['cluster_contrasts']:
        results.append(run_cluster_test(group, cond1, cond2, stats_params, verbose))
    for row, p_holm in zip(results, holm_adjusted([r['p_test'] for r in results])):
        row['p_holm'] = float(p_holm)
    return results


#%%
# ============================================================
# Figures
# ============================================================


def save_roi_figure(group, stats_params, out_dir, paths=None):
    """
    Tier-1 values per condition: one panel per band, ROI on x, one line per subject.

    Individual subjects are drawn rather than only the mean, because at N=10 whether an
    effect is carried by the whole cohort or by two subjects is the single most useful
    thing a reader can see, and no summary statistic shows it.
    """
    values = roi_band_values(group, stats_params)
    rois = list(stats_params['rois'])
    bands = list(stats_params['bands'])
    conditions = group.events

    fig, axes = plt.subplots(len(bands), len(conditions),
                             figsize=(3.1 * len(conditions), 3.2 * len(bands)),
                             sharey='row')
    axes = np.atleast_2d(axes)
    x = np.arange(len(rois))
    for row, band in enumerate(bands):
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            block = np.column_stack([values[(condition, roi, band)] for roi in rois])
            ax.axhline(0, color='0.6', linewidth=0.8, zorder=1)
            ax.plot(x, block.T, color='0.7', linewidth=0.8, marker='o',
                    markersize=3, zorder=2)
            ax.plot(x, block.mean(axis=0), color='crimson', linewidth=2.2,
                    marker='o', markersize=6, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(rois)
            if row == 0:
                ax.set_title(condition, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{band}\n{stats_params["bands"][band]} Hz',
                              fontsize=11, fontweight='bold')
    tmin, tmax = stats_params['active_window']
    fig.suptitle(f'Tier 1 values per subject (N={len(group.subjects)}), '
                 f'{tmin}-{tmax} s — grey = subject, red = mean',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = _savefig(fig, out_dir / 'roi_values.png', paths, stats_params)
    plt.close(fig)
    return path


def save_cluster_figures(result, out_dir, stats_params, paths=None):
    """
    Two views of one cluster test, both explicitly descriptive.

    An earlier version had a 'mean T per channel' panel that averaged t over every
    freq x time bin, the large majority of which carry no effect, so it diluted a real
    effect toward zero. It is not a valid summary statistic and is not here. What
    remains describes where the significant clusters fell and how strong the effect is
    inside them - useful for reading the result, never a claim about localisation.
    """
    info = _eeg_info_for_adjacency(result['info'])
    T_obs, ch_names = result['T_obs'], result['ch_names']
    n_bins = T_obs.shape[0] * T_obs.shape[1]
    slug = result['contrast'].replace(' - ', '_minus_').replace(' ', '')

    combined = np.zeros(T_obs.shape, dtype=bool)
    for cluster in result['sig_clusters']:
        combined |= cluster['mask']

    extent_frac = combined.sum(axis=(0, 1)) / n_bins
    mean_T = np.array([T_obs[:, :, c][combined[:, :, c]].mean()
                       if combined[:, :, c].any() else 0.0
                       for c in range(len(ch_names))])

    paths_out = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im, _ = mne.viz.plot_topomap(extent_frac, info, axes=axes[0], show=False,
                                 cmap='hot', vlim=(0, max(float(extent_frac.max()), 0.01)))
    axes[0].set_title('Where the significant bins fell\n(fraction of this channel\'s '
                      'TF bins)', fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label='fraction')

    vmax = max(float(np.abs(mean_T).max()), 1.0)
    im, _ = mne.viz.plot_topomap(mean_T, info, axes=axes[1], show=False,
                                 cmap='RdBu_r', vlim=(-vmax, vmax))
    axes[1].set_title('Mean T inside significant clusters', fontsize=10)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label='T')

    fig.suptitle(f"{result['contrast']} — {len(result['sig_clusters'])} significant "
                 f"cluster(s)\nDESCRIPTIVE ONLY: a cluster shows the conditions differ, "
                 f"not that they differ HERE",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    paths_out.append(_savefig(fig, out_dir / f'{slug}_topo.png', paths, stats_params))
    plt.close(fig)

    channels = [ch for ch in stats_params['tf_channels'] if ch in ch_names]
    if channels:
        freqs, times = result['freqs'], result['times']
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        fig, axes = plt.subplots(1, len(channels), figsize=(4.6 * len(channels), 4),
                                 squeeze=False)
        for ax, ch in zip(axes[0], channels):
            idx = ch_names.index(ch)
            T_map = T_obs[:, :, idx]
            vmax = max(float(np.abs(T_map).max()), 1.0)
            ax.imshow(T_map, aspect='auto', origin='lower', extent=extent,
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.35)
            mask = combined[:, :, idx]
            if mask.any():
                ax.imshow(np.where(mask, T_map, np.nan), aspect='auto', origin='lower',
                          extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax.set_title(f'{ch} — faded = all T, solid = inside a cluster')
            else:
                ax.set_title(f'{ch} — in no significant cluster')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
        fig.suptitle(f"{result['contrast']} — T-statistic per channel",
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        paths_out.append(_savefig(fig, out_dir / f'{slug}_tf.png', paths, stats_params))
        plt.close(fig)
    return paths_out


#%%
# ============================================================
# Driver
# ============================================================


def stats_dir(paths):
    """``Figures/<label>/Group/Stats/``, created on demand."""
    out_dir = paths.figures / 'Group' / 'Stats'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def tested_tree_params(paths):
    """
    The TFR params the tested tree was actually built with, read from its own
    ``tfr_run.json``.

    Not the notebook's ``TFR`` dict: stage 5 deliberately reads a different tree from
    the one the figures were last rendered into, so the in-memory dict can describe
    'percent' while the tests run on logratio. Taking the mode from the tree itself is
    the only way the summary cannot lie about what was tested.
    """
    run_path = paths.tfrs / 'tfr_run.json'
    if not run_path.exists():
        return None
    try:
        return json.loads(run_path.read_text()).get('tfr_params')
    except (OSError, ValueError):
        return None


def _stats_provenance(group, stats_params, tfr_params, roi_results, cluster_results,
                      paths):
    """
    What produced these numbers, written beside them.

    Records which TREE the tests ran on, because that is the one thing a reader cannot
    recover from the p-values and the one most likely to differ between runs: the
    logratio and percent trees hold the same subjects on the same grid and produce
    different statistics.
    """
    tested = tested_tree_params(paths)
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'tfr_tree': str(paths.tfrs.relative_to(paths.root)),
        'label': paths.label,
        # From the tree on disk, NOT from the caller's dict - see tested_tree_params.
        'baseline_mode': (tested or {}).get('mode'),
        'tested_tree_params': jsonable(tested) if tested else None,
        'n_subjects': len(group.subjects),
        'subjects': list(group.subjects),
        'conditions': list(group.events),
        'stats_params': jsonable(stats_params),
        # The caller's dict, kept separately and never conflated with the tested tree's:
        # it describes whichever tree the FIGURES were last rendered into.
        'notebook_tfr_params': jsonable(tfr_params) if tfr_params else None,
        'tier1': [{k: v for k, v in row.items()} for row in roi_results],
        'tier2': [{k: v for k, v in row.items()
                   if k not in ('T_obs', 'clusters', 'sig_clusters', 'freqs', 'times',
                                'info', 'ch_names')}
                  | {'sig_clusters': [{k: v for k, v in c.items() if k != 'mask'}
                                      for c in row['sig_clusters']],
                     'n_clusters': len(row['clusters'])}
                  for row in cluster_results],
    }


def run_tfr_stats(subjects=None, events=None, stats_params=None, tfr_params=None,
                  run_cluster=True, save_figs=True, batch_backend='Agg', paths=None):
    """
    Stage 5 end to end: tier 1, then tier 2, then figures and provenance.

    Tier 1 runs first on purpose. It costs seconds and it is the sanity check: if
    lateralisation does not come out there, something upstream is wrong and the
    expensive cluster pass is not worth starting.

    A failing contrast is reported and skipped rather than aborting the run, and
    figures render on a non-interactive backend by default with the previous one
    restored - the same contract as ``tfr_group.run_tfr_group``.
    """
    paths = paths or project_paths()
    stats_params = stats_params or default_stats_params()
    group = load_group_tfrs(subjects, events=events, paths=paths)
    check_stats_params(stats_params, tfr_params=tfr_params, group=group)

    # State the baseline mode of the tree actually being tested, read from the tree
    # rather than from tfr_params - those describe whichever tree the FIGURES were last
    # rendered into, and stage 5 is normally pointed somewhere else on purpose.
    tested = tested_tree_params(paths) or {}
    print(f"\ntests run on : {paths.tfrs.relative_to(paths.root)}   "
          f"baseline mode: {tested.get('mode', 'unrecorded')!r}")
    if tfr_params and tested.get('mode') and tested['mode'] != tfr_params.get('mode'):
        print(f"  (the notebook's TFR dict says {tfr_params['mode']!r}; that is the "
              f"figure tree, not this one)")

    previous_backend = matplotlib.get_backend()
    switching = batch_backend is not None and \
        previous_backend.lower() != batch_backend.lower()
    if switching:
        plt.close('all')
        matplotlib.use(batch_backend, force=True)
        print(f"rendering figures on '{batch_backend}' "
              f"(restoring '{previous_backend}' afterwards)")

    out_dir = stats_dir(paths)
    cluster_results, failures = [], {}
    try:
        roi_results = run_roi_tests(group, stats_params)
        if save_figs:
            save_roi_figure(group, stats_params, out_dir, paths)

        if run_cluster:
            for cond1, cond2 in stats_params['cluster_contrasts']:
                try:
                    result = run_cluster_test(group, cond1, cond2, stats_params)
                    cluster_results.append(result)
                    if save_figs:
                        save_cluster_figures(result, out_dir, stats_params, paths)
                except Exception as err:
                    label = f'{cond1} - {cond2}'
                    failures[label] = f'{type(err).__name__}: {err}'
                    print(f"  !! FAILED [{label}]: {failures[label]}")
                finally:
                    plt.close('all')
            if cluster_results:
                holm = holm_adjusted([r['p_test'] for r in cluster_results])
                for row, p_holm in zip(cluster_results, holm):
                    row['p_holm'] = float(p_holm)
    finally:
        if switching:
            matplotlib.use(previous_backend, force=True)

    summary = _stats_provenance(group, stats_params, tfr_params, roi_results,
                                cluster_results, paths)
    summary['failures'] = failures
    summary_path = out_dir / 'tfr_stats_summary.json'
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"\n  saved {summary_path.relative_to(paths.root)}")

    _print_final_summary(roi_results, cluster_results, stats_params)
    return {'group': group, 'tier1': roi_results, 'tier2': cluster_results,
            'summary': summary}


def _print_final_summary(roi_results, cluster_results, stats_params):
    """One table for the whole study, Holm-corrected within each tier."""
    alpha = stats_params['alpha']
    print(f"\n{'=' * 78}\nSTAGE 5 SUMMARY\n{'=' * 78}")
    print(f"{'tier':<7}{'contrast':<32}{'p (within)':>12}{'p (Holm)':>11}{'verdict':>12}")
    for row in roi_results:
        verdict = 'significant' if row['p_holm'] < alpha else 'n.s.'
        print(f"{'1':<7}{row['contrast']:<32}{row['p_test']:>12.4f}"
              f"{row['p_holm']:>11.4f}{verdict:>12}")
    for row in cluster_results:
        verdict = 'significant' if row.get('p_holm', 1.0) < alpha else 'n.s.'
        print(f"{'2':<7}{row['contrast']:<32}{row['p_test']:>12.4f}"
              f"{row.get('p_holm', float('nan')):>11.4f}{verdict:>12}")
    print("\nTier 1 is confirmatory; tier 2 is exploratory and its clusters describe "
          "THAT the\nconditions differ, never WHERE.")
