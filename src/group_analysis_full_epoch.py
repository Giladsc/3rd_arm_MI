"""
Group analysis utilities for full-epoch (majority-vote) classification results.

These mirror the windowed equivalents in group_analysis.py but are designed for
results produced by run_full_epoch_classification_cv, where a single label is
assigned per trial via majority vote across all windows in [tmin, tmax].
"""
import json
import numpy as np
from datetime import datetime

from .group_analysis import _get_classes_initials


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_fold_confusion_matrices(fold_confusion_matrices):
    """
    Serialize a flat list of (cm, labels) — one entry per fold — to
    JSON-safe dicts.
    """
    result = []
    for cm, labels in fold_confusion_matrices:
        result.append({
            'cm': cm.tolist() if isinstance(cm, np.ndarray) else cm,
            'labels': labels.tolist() if isinstance(labels, np.ndarray) else list(labels),
        })
    return result


def _deserialize_fold_confusion_matrices(data):
    """Reconstruct a list of (cm, labels) numpy tuples from the serialized format."""
    if data is None:
        return None
    return [(np.array(item['cm']), np.array(item['labels'])) for item in data]


# ---------------------------------------------------------------------------
# Add / save individual subject
# ---------------------------------------------------------------------------

def add_subject_to_group_full_epoch(group_results_full_epoch, subject_name,
                                    fold_accuracies, fold_confusion_matrices,
                                    tmin=0.0, tmax=5.0,
                                    params_dict=None, save_dir=None):
    """
    Add a subject's full-epoch majority-vote results to the group list.
    Optionally saves that subject's metrics to disk immediately.

    Parameters
    ----------
    group_results_full_epoch : list
        Mutable list to accumulate subjects (pass [] on first call).
    subject_name : str
        Subject identifier (e.g. 'AN', 'SK').
    fold_accuracies : list of float
        Trial-level accuracy for each CV fold, as returned by
        run_full_epoch_classification_cv.
    fold_confusion_matrices : list of (cm, labels)
        One (confusion_matrix, class_labels) tuple per CV fold.
    tmin, tmax : float
        Epoch window (s) used for evaluation — stored for reference.
    params_dict : dict, optional
        If provided, pipeline_name and desired_events are stored with the entry.
    save_dir : pathlib.Path or str, optional
        If provided, saves individual subject metrics to:
        <save_dir>/<subject>_<class_initials>_<pipeline>_full_epoch.json

    Returns
    -------
    group_results_full_epoch : list
        The same list, with the new entry appended.
    """
    entry = {
        'subject_name': subject_name,
        'timestamp': datetime.now().isoformat(),
        'tmin': tmin,
        'tmax': tmax,
        'fold_accuracies': list(fold_accuracies),
        'fold_confusion_matrices': fold_confusion_matrices,
    }

    if params_dict is not None:
        entry['pipeline_name'] = params_dict.get('pipeline_name', '')
        entry['desired_events'] = params_dict.get('desired_events', [])

    group_results_full_epoch.append(entry)

    n_folds = len(fold_accuracies)
    mean_acc = float(np.mean(fold_accuracies)) if n_folds > 0 else float('nan')
    std_acc  = float(np.std(fold_accuracies))  if n_folds > 0 else float('nan')
    print(f"Added {subject_name}: {n_folds} folds | "
          f"acc = {mean_acc:.3f} ± {std_acc:.3f} (epoch {tmin}–{tmax}s)")

    if save_dir is not None:
        _save_subject_metrics_full_epoch(entry, save_dir)

    return group_results_full_epoch


def _save_subject_metrics_full_epoch(entry, save_dir):
    """Save a single subject's full-epoch metrics to a JSON file."""
    import pathlib
    save_dir = pathlib.Path(save_dir)

    subject      = entry['subject_name']
    classes_str  = _get_classes_initials(entry.get('desired_events', []))
    pipeline_str = entry.get('pipeline_name', '')

    parts = [subject]
    if classes_str:
        parts.append(classes_str)
    if pipeline_str:
        parts.append(pipeline_str)
    parts.append('full_epoch')
    filename = '_'.join(parts) + '.json'

    filepath = save_dir / filename

    e = entry.copy()
    e['fold_confusion_matrices'] = _serialize_fold_confusion_matrices(
        e.get('fold_confusion_matrices', [])
    )

    with open(filepath, 'w') as f:
        json.dump(e, f, indent=2)
    print(f"  Saved full-epoch metrics to {filepath}")


# ---------------------------------------------------------------------------
# Save / load the full group
# ---------------------------------------------------------------------------

def save_group_results_full_epoch(group_results_full_epoch,
                                  filepath_or_dir=None, filename=None):
    """
    Save the full-epoch group results list to a JSON file.

    Parameters
    ----------
    group_results_full_epoch : list
        Built with add_subject_to_group_full_epoch.
    filepath_or_dir : pathlib.Path or str, optional
        If it ends in .json, used as the full path (backward compatible).
        Otherwise treated as a directory and the filename is auto-generated.
    filename : str, optional
        Override the auto-generated filename when filepath_or_dir is a directory.

    Returns
    -------
    filepath : pathlib.Path
        Where the file was written.
    """
    import pathlib
    path = pathlib.Path(filepath_or_dir) if filepath_or_dir is not None else pathlib.Path('.')

    if path.suffix == '.json':
        filepath = path
    else:
        save_dir = path
        if filename is None:
            subjects = '_'.join(e['subject_name'] for e in group_results_full_epoch)
            if len(subjects) > 40:
                subjects = subjects[:37] + '...'

            classes_str  = ''
            pipeline_str = ''
            for e in group_results_full_epoch:
                if e.get('desired_events') and not classes_str:
                    classes_str = _get_classes_initials(e['desired_events'])
                if e.get('pipeline_name') and not pipeline_str:
                    pipeline_str = e['pipeline_name']

            parts = ['group', subjects]
            if classes_str:
                parts.append(classes_str)
            if pipeline_str:
                parts.append(pipeline_str)
            parts.append('full_epoch')
            filename = '_'.join(parts) + '.json'

        filepath = save_dir / filename

    serializable = []
    for entry in group_results_full_epoch:
        e = entry.copy()
        e['fold_confusion_matrices'] = _serialize_fold_confusion_matrices(
            e.get('fold_confusion_matrices', [])
        )
        serializable.append(e)

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {len(group_results_full_epoch)} subjects (full-epoch) to {filepath}")
    return filepath


def descriptive_stats_full_epoch(group_results_full_epoch, n_classes=None, chance_level=None):
    """
    Compute and print descriptive statistics for full-epoch group results.

    Columns
    -------
    W-Std  : within-subject std (fold-to-fold variability) — same scale for all rows.
    B-Std  : between-subject std (std of subject means) — GROUP row only; used for
             the group-level SEM and CI.

    Parameters
    ----------
    group_results_full_epoch : list
        Built with add_subject_to_group_full_epoch.
    n_classes : int, optional
        Number of classes — used to compute chance level (1/n_classes).
        Ignored if chance_level is provided.
    chance_level : float, optional
        Override for chance level (e.g. 0.25 for 4 classes).

    Returns
    -------
    stats : dict
        'per_subject' : list of dicts with subject-level stats
        'group'       : dict with grand mean, between_std, within_std_mean,
                        sem, ci95, median, min, max, above_chance
        'chance'      : chance level used (or None)
    """
    if not group_results_full_epoch:
        raise ValueError("group_results_full_epoch is empty.")

    if chance_level is None and n_classes is not None:
        chance_level = 1.0 / n_classes

    per_subject = []
    subject_means = []
    subject_within_stds = []

    for entry in group_results_full_epoch:
        accs = np.array(entry['fold_accuracies'], dtype=float)
        n = len(accs)
        mean       = float(np.mean(accs))
        within_std = float(np.std(accs, ddof=1)) if n > 1 else float('nan')
        se         = within_std / np.sqrt(n)      if n > 1 else float('nan')
        ci95       = 1.96 * se                    if n > 1 else float('nan')
        above      = float(mean - chance_level)   if chance_level is not None else float('nan')

        per_subject.append({
            'subject':     entry['subject_name'],
            'n_folds':     n,
            'mean':        mean,
            'within_std':  within_std,
            'sem':         se,
            'ci95':        ci95,
            'min':         float(np.min(accs)),
            'max':         float(np.max(accs)),
            'above_chance': above,
        })
        subject_means.append(mean)
        subject_within_stds.append(within_std)

    means = np.array(subject_means)
    n_subs = len(means)

    between_std  = float(np.std(means, ddof=1))                if n_subs > 1 else float('nan')
    group_sem    = float(between_std / np.sqrt(n_subs))        if n_subs > 1 else float('nan')
    within_mean  = float(np.nanmean(subject_within_stds))

    group = {
        'n_subjects':      n_subs,
        'grand_mean':      float(np.mean(means)),
        'between_std':     between_std,
        'within_std_mean': within_mean,
        'sem':             group_sem,
        'ci95':            1.96 * group_sem if n_subs > 1 else float('nan'),
        'median':          float(np.median(means)),
        'min':             float(np.min(means)),
        'max':             float(np.max(means)),
        'above_chance':    float(np.mean(means) - chance_level) if chance_level is not None else float('nan'),
    }

    # --- Print table ---
    header = (f"{'Subject':<10} {'N folds':>7} {'Mean':>7} {'W-Std':>7} "
              f"{'B-Std':>7} {'SEM':>7} {'95% CI':>8} {'Min':>7} {'Max':>7}")
    if chance_level is not None:
        header += f" {'vs Chance':>10}"
    print(header)
    print('-' * len(header))

    for s in per_subject:
        row = (f"{s['subject']:<10} {s['n_folds']:>7d} {s['mean']:>7.3f} "
               f"{s['within_std']:>7.3f} {'---':>7} "
               f"{s['sem']:>7.3f} {s['ci95']:>8.3f} "
               f"{s['min']:>7.3f} {s['max']:>7.3f}")
        if chance_level is not None:
            row += f" {s['above_chance']:>+10.3f}"
        print(row)

    print('-' * len(header))
    g = group
    row = (f"{'GROUP':<10} {g['n_subjects']:>7d} {g['grand_mean']:>7.3f} "
           f"{g['within_std_mean']:>7.3f} {g['between_std']:>7.3f} "
           f"{g['sem']:>7.3f} {g['ci95']:>8.3f} "
           f"{g['min']:>7.3f} {g['max']:>7.3f}")
    if chance_level is not None:
        row += f" {g['above_chance']:>+10.3f}"
    print(row)

    if chance_level is not None:
        print(f"\nChance level: {chance_level:.3f} ({'' if n_classes is None else str(n_classes) + ' classes'})")
    print("W-Std = within-subject fold std  |  B-Std = between-subject std of means (used for SEM/CI)")

    return {'per_subject': per_subject, 'group': group, 'chance': chance_level}


def load_group_results_full_epoch(filepath):
    """
    Load full-epoch group results from JSON, restoring numpy arrays.

    Returns
    -------
    list of dict, each with keys:
        subject_name, timestamp, tmin, tmax,
        fold_accuracies (list of float),
        fold_confusion_matrices (list of (cm, labels) tuples),
        pipeline_name, desired_events  (if present)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    for entry in data:
        entry['fold_accuracies'] = list(entry.get('fold_accuracies', []))
        entry['fold_confusion_matrices'] = _deserialize_fold_confusion_matrices(
            entry.get('fold_confusion_matrices')
        )

    print(f"Loaded {len(data)} subjects (full-epoch) from {filepath}")
    return data
