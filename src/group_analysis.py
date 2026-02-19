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
    e['folds_confusion_matrices_per_window'] = None

    with open(filepath, 'w') as f:
        json.dump(e, f, indent=2)
    print(f"  Saved metrics to {filepath}")


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
        # Confusion matrices can't be serialized to JSON
        e['folds_confusion_matrices_per_window'] = None
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

    print(f"Loaded {len(data)} subjects from {filepath}")
    return data
