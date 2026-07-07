"""
Group analysis utilities for tracking per-subject classification results
across manual experiment runs.
"""
import json
import numpy as np
from datetime import datetime


def add_subject_to_group(group_results, subject_name, scores_windows,
                         folds_confusion_matrices_per_window, w_times=None,
                         params_dict=None, save_dir=None):
    """
    Add a subject's windowed classification results to the group array.
    Optionally saves individual subject metrics to disk.

    Parameters
    ----------
    group_results : list
        Mutable list to store group results.
    subject_name : str
        Subject identifier (e.g., 'DD', 'Tomer').
    scores_windows : list or array
        Per-fold accuracy scores across time windows. Shape: (n_folds, n_windows).
    folds_confusion_matrices_per_window : list
        Nested list: [fold_0, fold_1, ...], each a list of (cm, labels) per window.
    w_times : array, optional
        Time points (s) corresponding to each window.
    params_dict : dict, optional
        If provided, pipeline_name and desired_events are stored with the entry.
    save_dir : pathlib.Path or str, optional
        If provided, saves individual subject metrics to:
        <save_dir>/<subject>_<class_initials>_<pipeline>.json
    """
    entry = {
        'subject_name': subject_name,
        'timestamp': datetime.now().isoformat(),
        'scores_windows': scores_windows,
        'folds_confusion_matrices_per_window': folds_confusion_matrices_per_window,
        'w_times': w_times,
    }

    if params_dict is not None:
        entry['pipeline_name'] = params_dict.get('pipeline_name', '')
        entry['desired_events'] = params_dict.get('desired_events', [])

    group_results.append(entry)

    n_folds = len(scores_windows) if scores_windows else 0
    n_windows = len(scores_windows[0]) if n_folds > 0 else 0
    print(f"Added {subject_name}: {n_folds} folds x {n_windows} windows")

    # Auto-save individual subject metrics
    if save_dir is not None:
        _save_subject_metrics(entry, save_dir)

    return group_results


def _save_subject_metrics(entry, save_dir):
    """Save a single subject's metrics to a JSON file."""
    import pathlib
    save_dir = pathlib.Path(save_dir)

    subject = entry['subject_name']
    classes_str = _get_classes_initials(entry.get('desired_events', []))
    pipeline_str = entry.get('pipeline_name', '')

    parts = [subject]
    if classes_str:
        parts.append(classes_str)
    if pipeline_str:
        parts.append(pipeline_str)
    filename = '_'.join(parts) + '.json'

    filepath = save_dir / filename

    # Serialize
    e = entry.copy()
    if isinstance(e.get('scores_windows'), np.ndarray):
        e['scores_windows'] = e['scores_windows'].tolist()
    elif isinstance(e.get('scores_windows'), list):
        e['scores_windows'] = [
            row.tolist() if isinstance(row, np.ndarray) else row
            for row in e['scores_windows']
        ]
    if isinstance(e.get('w_times'), np.ndarray):
        e['w_times'] = e['w_times'].tolist()
    e['folds_confusion_matrices_per_window'] = _serialize_confusion_matrices(
        e.get('folds_confusion_matrices_per_window')
    )

    with open(filepath, 'w') as f:
        json.dump(e, f, indent=2)
    print(f"  Saved metrics to {filepath}")


def _serialize_confusion_matrices(folds_confusion_matrices_per_window):
    """Convert nested (cm, labels) structure to a JSON-serializable list of lists of dicts."""
    if folds_confusion_matrices_per_window is None:
        return None
    result = []
    for fold in folds_confusion_matrices_per_window:
        fold_result = []
        for cm, labels in fold:
            fold_result.append({
                'cm': cm.tolist() if isinstance(cm, np.ndarray) else cm,
                'labels': labels.tolist() if isinstance(labels, np.ndarray) else list(labels),
            })
        result.append(fold_result)
    return result


def _deserialize_confusion_matrices(data):
    """Reconstruct (cm, labels) tuples with numpy arrays from the serialized format."""
    if data is None:
        return None
    result = []
    for fold in data:
        fold_result = []
        for item in fold:
            cm = np.array(item['cm'])
            labels = np.array(item['labels'])
            fold_result.append((cm, labels))
        result.append(fold_result)
    return result


def _get_classes_initials(desired_events):
    """Generate short class initials from event names, e.g. ['RightHand','LeftHand','Rest'] -> 'RH_LH_R'."""
    initials = []
    for event in desired_events:
        # Split camelCase or take first letters of each word
        parts = []
        current = []
        for ch in event:
            if ch.isupper() and current:
                parts.append(''.join(current))
                current = [ch]
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        initials.append(''.join(p[0].upper() for p in parts))
    return '_'.join(initials)


def _merge_cm_classes(cm, labels, merge_map):
    """
    Merge rows/columns of a confusion matrix according to merge_map.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
    labels : array-like of str
    merge_map : dict  e.g. {'Idle': ['FixatedRest', 'Rest']}
        Keys are the new class names; values are lists of existing class names
        to collapse into that key.  Classes not mentioned are kept unchanged.

    Returns
    -------
    merged_cm : np.ndarray
    new_labels : list of str  (order preserved from original label sequence)
    """
    labels = list(labels)
    label_remap = {lbl: lbl for lbl in labels}
    for new_name, old_names in merge_map.items():
        for old in old_names:
            if old in label_remap:
                label_remap[old] = new_name

    new_labels = []
    seen = set()
    for lbl in labels:
        mapped = label_remap[lbl]
        if mapped not in seen:
            new_labels.append(mapped)
            seen.add(mapped)

    new_idx = {lbl: i for i, lbl in enumerate(new_labels)}
    merged_cm = np.zeros((len(new_labels), len(new_labels)), dtype=float)
    for i, true_lbl in enumerate(labels):
        for j, pred_lbl in enumerate(labels):
            merged_cm[new_idx[label_remap[true_lbl]],
                      new_idx[label_remap[pred_lbl]]] += cm[i, j]
    return merged_cm, new_labels


def descriptive_stats_group(group_results, n_classes=None, chance_level=None,
                            tmin=None, tmax=None, class_merge=None):
    """
    Compute and print descriptive statistics for windowed group results.

    Each subject's scalar summary is the mean accuracy across the selected time
    windows, averaged first within each fold, then across folds.  Peak accuracy
    (max of the fold-mean time-series) is reported separately.

    Columns
    -------
    W-Std  : within-subject std of fold time-averages (fold-to-fold variability).
    B-Std  : between-subject std of subject means — GROUP row only; used for SEM/CI.

    Parameters
    ----------
    group_results : list
        Built with add_subject_to_group.
    n_classes : int, optional
        Number of classes — used to derive chance level (1/n_classes).
    chance_level : float, optional
        Override for chance level.
    tmin, tmax : float, optional
        Restrict analysis to windows whose centre time satisfies tmin <= t <= tmax.
        Requires w_times to be stored in the entries.
    class_merge : dict, optional
        Merge classes before computing accuracy.  Requires confusion matrices to
        be present.  Example: ``{'Idle': ['FixatedRest', 'Rest']}``
        Accuracy is recomputed from the merged confusion matrices; the original
        scores_windows values are ignored when this is provided.

    Returns
    -------
    stats : dict
        'per_subject' : list of per-subject stat dicts
        'group'       : grand-mean stats dict
        'chance'      : chance level used (or None)
    """
    if not group_results:
        raise ValueError("group_results is empty.")

    if chance_level is None and n_classes is not None:
        chance_level = 1.0 / n_classes

    per_subject = []
    subject_means = []
    subject_within_stds = []

    for entry in group_results:
        w_times = entry.get('w_times')
        if w_times is not None:
            w_times = np.array(w_times, dtype=float)

        # --- Time window mask ---
        if w_times is not None and (tmin is not None or tmax is not None):
            mask = np.ones(len(w_times), dtype=bool)
            if tmin is not None:
                mask &= w_times >= tmin
            if tmax is not None:
                mask &= w_times <= tmax
            win_indices = np.where(mask)[0]
            w_times_sel = w_times[mask]
        else:
            sw_full = np.array(entry['scores_windows'], dtype=float)
            win_indices = np.arange(sw_full.shape[1])
            w_times_sel = w_times

        # --- Build scores_windows for selected windows ---
        if class_merge is not None:
            cms = entry.get('folds_confusion_matrices_per_window')
            if cms is None:
                raise ValueError(
                    f"class_merge requested but confusion matrices are missing "
                    f"for subject '{entry['subject_name']}'."
                )
            n_folds = len(cms)
            n_windows_sel = len(win_indices)
            sw = np.zeros((n_folds, n_windows_sel), dtype=float)
            for fi in range(n_folds):
                for wi_out, wi_in in enumerate(win_indices):
                    cm, labels = cms[fi][wi_in]
                    merged_cm, _ = _merge_cm_classes(cm, labels, class_merge)
                    total = merged_cm.sum()
                    sw[fi, wi_out] = np.trace(merged_cm) / total if total > 0 else float('nan')
        else:
            sw_full = np.array(entry['scores_windows'], dtype=float)
            sw = sw_full[:, win_indices]

        n_folds, n_windows_sel = sw.shape

        fold_time_means = sw.mean(axis=1)           # (n_folds,)
        mean       = float(fold_time_means.mean())
        within_std = float(fold_time_means.std(ddof=1)) if n_folds > 1 else float('nan')
        se         = within_std / np.sqrt(n_folds)   if n_folds > 1 else float('nan')
        ci95       = 1.96 * se                        if n_folds > 1 else float('nan')
        above      = float(mean - chance_level)       if chance_level is not None else float('nan')

        mean_over_folds = sw.mean(axis=0)            # (n_windows_sel,)
        peak_acc  = float(mean_over_folds.max())
        peak_idx  = int(mean_over_folds.argmax())
        peak_time = float(w_times_sel[peak_idx]) if w_times_sel is not None else float('nan')

        per_subject.append({
            'subject':     entry['subject_name'],
            'n_folds':     n_folds,
            'n_windows':   n_windows_sel,
            'mean':        mean,
            'within_std':  within_std,
            'sem':         se,
            'ci95':        ci95,
            'min':         float(fold_time_means.min()),
            'max':         float(fold_time_means.max()),
            'peak_acc':    peak_acc,
            'peak_time':   peak_time,
            'above_chance': above,
        })
        subject_means.append(mean)
        subject_within_stds.append(within_std)

    means = np.array(subject_means)
    n_subs = len(means)

    between_std = float(np.std(means, ddof=1))           if n_subs > 1 else float('nan')
    group_sem   = float(between_std / np.sqrt(n_subs))   if n_subs > 1 else float('nan')
    within_mean = float(np.nanmean(subject_within_stds))

    group_peak_accs  = [s['peak_acc']  for s in per_subject]
    group_peak_times = [s['peak_time'] for s in per_subject]

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
        'mean_peak_acc':   float(np.mean(group_peak_accs)),
        'mean_peak_time':  float(np.nanmean(group_peak_times)),
        'above_chance':    float(np.mean(means) - chance_level) if chance_level is not None else float('nan'),
    }

    # --- Print header info ---
    if tmin is not None or tmax is not None:
        t_lo = f"{tmin:.2f}s" if tmin is not None else "-inf"
        t_hi = f"{tmax:.2f}s" if tmax is not None else "+inf"
        print(f"Time range: [{t_lo}, {t_hi}]  ({per_subject[0]['n_windows']} windows)")
    if class_merge is not None:
        for new_name, old_names in class_merge.items():
            print(f"Merged: {old_names} -> '{new_name}'")

    # --- Print table ---
    header = (f"{'Subject':<10} {'Folds':>5} {'Windows':>7} {'Mean':>7} {'W-Std':>7} "
              f"{'B-Std':>7} {'SEM':>7} {'95% CI':>8} {'Peak Acc':>9} {'Peak t(s)':>10}")
    if chance_level is not None:
        header += f" {'vs Chance':>10}"
    print(header)
    print('-' * len(header))

    for s in per_subject:
        peak_t_str = f"{s['peak_time']:>10.3f}" if not np.isnan(s['peak_time']) else f"{'N/A':>10}"
        row = (f"{s['subject']:<10} {s['n_folds']:>5d} {s['n_windows']:>7d} "
               f"{s['mean']:>7.3f} {s['within_std']:>7.3f} {'---':>7} "
               f"{s['sem']:>7.3f} {s['ci95']:>8.3f} "
               f"{s['peak_acc']:>9.3f} {peak_t_str}")
        if chance_level is not None:
            row += f" {s['above_chance']:>+10.3f}"
        print(row)

    print('-' * len(header))
    g = group
    peak_t_str = f"{g['mean_peak_time']:>10.3f}" if not np.isnan(g['mean_peak_time']) else f"{'N/A':>10}"
    row = (f"{'GROUP':<10} {g['n_subjects']:>5d} {'---':>7} "
           f"{g['grand_mean']:>7.3f} {g['within_std_mean']:>7.3f} {g['between_std']:>7.3f} "
           f"{g['sem']:>7.3f} {g['ci95']:>8.3f} "
           f"{g['mean_peak_acc']:>9.3f} {peak_t_str}")
    if chance_level is not None:
        row += f" {g['above_chance']:>+10.3f}"
    print(row)

    if chance_level is not None:
        print(f"\nChance level: {chance_level:.3f} ({'' if n_classes is None else str(n_classes) + ' classes'})")
    print("Mean/Std are over time-averaged fold scores  |  "
          "W-Std = within-subject (fold-to-fold)  |  B-Std = between-subject (GROUP only)")

    return {'per_subject': per_subject, 'group': group, 'chance': chance_level}


def save_group_results(group_results, filepath_or_dir=None, filename=None):
    """
    Save group_results to JSON.

    Parameters
    ----------
    group_results : list
        Group results list from add_subject_to_group.
    filepath_or_dir : pathlib.Path or str, optional
        If it ends in .json, treated as a full filepath (backward compatible).
        Otherwise treated as a directory, and filename is auto-generated.
    filename : str, optional
        Override filename when filepath_or_dir is a directory.
    """
    import pathlib
    path = pathlib.Path(filepath_or_dir) if filepath_or_dir is not None else pathlib.Path('.')

    # If path looks like a .json file, use it directly (backward compatible)
    if path.suffix == '.json':
        filepath = path
    else:
        # It's a directory — build or use filename
        save_dir = path
        if filename is None:
            subjects = '_'.join(e['subject_name'] for e in group_results)
            if len(subjects) > 40:
                subjects = subjects[:37] + '...'

            classes_str = ''
            pipeline_str = ''
            for e in group_results:
                if e.get('desired_events') and not classes_str:
                    classes_str = _get_classes_initials(e['desired_events'])
                if e.get('pipeline_name') and not pipeline_str:
                    pipeline_str = e['pipeline_name']

            parts = ['group', subjects]
            if classes_str:
                parts.append(classes_str)
            if pipeline_str:
                parts.append(pipeline_str)

            filename = '_'.join(parts) + '.json'

        filepath = save_dir / filename

    serializable = []
    for entry in group_results:
        e = entry.copy()
        if isinstance(e.get('scores_windows'), np.ndarray):
            e['scores_windows'] = e['scores_windows'].tolist()
        elif isinstance(e.get('scores_windows'), list):
            e['scores_windows'] = [
                row.tolist() if isinstance(row, np.ndarray) else row
                for row in e['scores_windows']
            ]
        if isinstance(e.get('w_times'), np.ndarray):
            e['w_times'] = e['w_times'].tolist()
        e['folds_confusion_matrices_per_window'] = _serialize_confusion_matrices(
            e.get('folds_confusion_matrices_per_window')
        )
        serializable.append(e)

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {len(group_results)} subjects to {filepath}")
    return filepath


def load_group_results(filepath):
    """Load group_results from JSON (returns list without confusion matrices)."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    for entry in data:
        if entry.get('scores_windows') is not None:
            entry['scores_windows'] = np.array(entry['scores_windows'])
        if entry.get('w_times') is not None:
            entry['w_times'] = np.array(entry['w_times'])
        entry['folds_confusion_matrices_per_window'] = _deserialize_confusion_matrices(
            entry.get('folds_confusion_matrices_per_window')
        )

    print(f"Loaded {len(data)} subjects from {filepath}")
    return data
