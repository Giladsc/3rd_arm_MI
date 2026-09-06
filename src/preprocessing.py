#%%

# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import logging
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt
import itertools
from tqdm import tqdm
import json
import random
# MNE library for EEG data analysis
import mne
from mne import Epochs,find_events
from mne.decoding import Vectorizer
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

# XDF file format support in MNE
import pyxdf
from .mne_import_xdf import *

# Scikit-learn and Pyriemann for feature extraction and machine learning functionalities
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn, Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

#import moab to get the filterbank implementation: 
from moabb.pipelines.utils import FilterBank

# For  GUI elements
from easygui import *

# For path manipulation
import pathlib
from os import listdir
from os.path import isfile, join

# For interactive plots
import PyQt5
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

#imports for precision_recall_curve related plot: 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, precision_recall_curve,PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import pickle

import copy


from braindecode.models import ShallowFBCSPNet
from braindecode.training import CroppedLoss
from braindecode.training.scoring import trial_preds_from_window_preds
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch



#%%
standard_event_id = {'FixatedRest': 1,'ActiveRest': 11, 'OpenPalm': 2,'ClosePalm': 3, 'MiddleHand':33, 'Rating': 4,'Rest': 55,'Long Break': 6,'RightHand' : 7, 'LeftHand' : 8, 'Idle': 0, 'Right': 77,'Left': 88}

# Old recordings label this event 'ClosePalm'; newer recordings (e.g. subject LD)
# label it 'MiddleHand'. 'MiddleHand' is the canonical name going forward.
EVENT_LABEL_ALIASES = {'ClosePalm': 'MiddleHand'}

def standardize_event_labels(raw, label_aliases=None):
    """Rename raw annotation descriptions per label_aliases (old -> new), in place.
    Only renames aliases actually present in this recording's annotations, so it
    is a guaranteed no-op for recordings that already use the canonical label
    (mne.Annotations.rename raises if given a key that isn't present)."""
    if label_aliases is None:
        label_aliases = EVENT_LABEL_ALIASES
    present_aliases = {old: new for old, new in label_aliases.items()
                        if old in raw.annotations.description}
    if present_aliases:
        raw.annotations.rename(present_aliases)
    return raw

def remap_epoch_events_to_standard(epochs, standard_event_id, desired_events):
    """
    Remap event codes to standard_event_id, keep only desired events,
    and strictly preserve the original epochs.event_id order.
    """
    original_order = [key for key in epochs.event_id if key in desired_events]
    val_to_label = {val: label for label, val in epochs.event_id.items()}

    new_events = epochs.events.copy()
    for i, code in enumerate(new_events[:, 2]):
        label = val_to_label[code]
        new_events[i, 2] = standard_event_id[label]

    epochs.events = new_events
    epochs.event_id = OrderedDict((label, standard_event_id[label]) for label in original_order)

    return epochs


def expand_triggers_to_events(ann, raw=None, recording_end=None, use_raw_timebase=True):
    """
    Turn trigger annotations (point markers) into event intervals:
    each trigger's label lasts until the next trigger onset.

    Parameters
    ----------
    ann : mne.Annotations
        Trigger-like annotations (often duration=0).
    raw : mne.io.BaseRaw or None
        If given, we'll use raw.times[-1] as the end of recording and (optionally)
        align orig_time to raw.info['meas_date'].
    recording_end : float or None
        End time in seconds (from recording start). If None, inferred from `raw`.
    use_raw_timebase : bool
        If True and `raw` is provided, set orig_time to raw.info['meas_date'] so
        plotting aligns perfectly with `raw`.

    Returns
    -------
    events_ann : mne.Annotations
        Annotations whose durations now extend to the next trigger (last to end).
    """
    if recording_end is None:
        if raw is None:
            raise ValueError("Provide either `raw` or `recording_end` (seconds).")
        recording_end = float(raw.times[-1])

    on = np.asarray(ann.onset, float)
    desc = np.asarray(ann.description, dtype=object)

    order = np.argsort(on, kind="mergesort")
    on = on[order]
    desc = desc[order]

    if len(on) == 0:
        base_time = raw.info['meas_date'] if (use_raw_timebase and raw is not None) else ann.orig_time
        return mne.Annotations([], [], [], orig_time=base_time)

    next_on = np.r_[on[1:], recording_end]
    dur = next_on - on

    keep = dur > 0
    on, dur, desc = on[keep], dur[keep], desc[keep]

    base_time = raw.info['meas_date'] if (use_raw_timebase and raw is not None) else ann.orig_time
    return mne.Annotations(on, dur, desc.tolist(), orig_time=base_time)


def balance_epochs_by_subsampling(epochs, class_to_subsample='Rating'):
    """
    Subsamples the specified class to match the smallest number of epochs in other classes.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object with labeled events.
    class_to_subsample : str
        Class label to downsample.

    Returns
    -------
    balanced_epochs : mne.Epochs
        New Epochs object with balanced classes.
    """
    event_id = epochs.event_id
    all_classes = list(event_id.keys())

    class_counts = {label: len(epochs[label]) for label in all_classes if label != class_to_subsample}
    min_count = min(class_counts.values())

    selected_indices = []
    for label in all_classes:
        picks = epochs[label].selection
        if label == class_to_subsample:
            picked = np.random.choice(picks, size=min_count, replace=False)
        else:
            picked = picks
        selected_indices.extend(picked)

    selected_indices = np.sort(selected_indices)
    balanced_epochs = epochs[selected_indices]

    return balanced_epochs


def epochs_to_continuous_raw(epochs, mark_cue=True, cue_label='cue'):
    """
    Flatten Epochs into a continuous Raw, annotated with the epoch structure.

    Concatenating epochs creates artificial edges that look like real events, so the
    returned Raw carries one annotation span per epoch, labelled with its condition
    (the MNE browser then colours trials by class), plus an optional zero-duration
    marker at each epoch's t=0 (cue onset).

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to flatten. Their data are concatenated along time in epoch order.
    mark_cue : bool
        Add a zero-duration marker at t=0 of every epoch. Only applies when
        ``epochs.tmin < 0`` (otherwise t=0 coincides with the epoch onset).
    cue_label : str
        Description used for the cue markers.

    Returns
    -------
    raw : mne.io.RawArray
        Continuous Raw of shape (n_channels, n_epochs * n_times) with annotations.
    """
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    n_epochs, n_ch, n_times = data.shape
    data_concat = data.transpose(1, 0, 2).reshape(n_ch, -1)
    raw = mne.io.RawArray(data_concat, epochs.info.copy())

    epoch_duration = n_times / epochs.info['sfreq']
    onsets = np.arange(n_epochs) * epoch_duration

    # One span per epoch, described by its condition name (one colour per class)
    id_to_label = {code: label for label, code in epochs.event_id.items()}
    labels = [id_to_label.get(code, str(code)) for code in epochs.events[:, 2]]
    annotations = mne.Annotations(onset=onsets,
                                  duration=[epoch_duration] * n_epochs,
                                  description=labels)

    if mark_cue and epochs.tmin < 0:
        annotations += mne.Annotations(onset=onsets - epochs.tmin,
                                       duration=0.0,
                                       description=cue_label)

    raw.set_annotations(annotations)
    return raw


def make_figure_scrollable(fig, max_window=(1400, 900)):
    """
    Re-host a matplotlib figure's canvas inside a Qt scroll area.

    Tall trellis figures (e.g. all ICA topographies at once) get squashed into the
    window by the Qt backend. This keeps the canvas at its natural pixel size and
    lets the window scroll over it instead.

    No-op (returns ``fig`` unchanged) on non-Qt backends such as ``%matplotlib
    inline``, where a tall figure already scrolls in the notebook output area.
    """
    try:
        from matplotlib.backends.qt_compat import QtWidgets
    except Exception:
        return fig

    try:
        canvas = fig.canvas
        window = getattr(getattr(canvas, 'manager', None), 'window', None)
        if not isinstance(window, QtWidgets.QMainWindow):
            return fig

        width = int(fig.get_figwidth() * fig.dpi)
        height = int(fig.get_figheight() * fig.dpi)

        # Detach first: setCentralWidget() deletes the widget it replaces
        canvas.setParent(None)
        scroll = QtWidgets.QScrollArea(window)
        scroll.setWidgetResizable(False)
        scroll.setWidget(canvas)
        window.setCentralWidget(scroll)
        canvas.setFixedSize(width, height)
        window.resize(min(width + 40, max_window[0]), min(height + 40, max_window[1]))
    except Exception as err:
        print(f"Could not make figure scrollable ({err}); showing it as-is.")
    return fig


def plot_ica_components_scrollable(ica, inst=None, show=True, **kwargs):
    """
    Plot all ICA topographies in a single scrollable figure.

    ``ICA.plot_components()`` splits components into separate figures of 20; passing
    an explicit ``picks`` keeps them in one figure, which is then made scrollable.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    inst : mne.io.Raw | mne.Epochs | None
        Passed through to ``plot_components`` so that clicking a topography opens
        ``ica.plot_properties`` for that component.
    show : bool
        Show the figure once it has been made scrollable.
    **kwargs
        Forwarded to ``ICA.plot_components``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The single figure holding every component.
    """
    from mne.viz.utils import plt_show

    # Explicit picks bypasses MNE's "one figure per 20 components" behaviour
    fig = ica.plot_components(picks=range(ica.n_components_), inst=inst,
                              show=False, **kwargs)
    make_figure_scrollable(fig)
    plt_show(show)
    return fig


# ICLabel output classes, in the column order of ``ica.labels_scores_``
# (see mne_icalabel.iclabel.iclabel_label_components)
ICLABEL_CLASSES = ['brain', 'muscle artifact', 'eye blink', 'heart beat',
                   'line noise', 'channel noise', 'other']


def iclabel_suggested_exclusions(labels, exclude_labels):
    """
    Component indices whose ICLabel class is one of `exclude_labels`.

    Parameters
    ----------
    labels : list of str
        Per-component ICLabel class, i.e. ``label_dict['labels']``.
    exclude_labels : list of str
        ICLabel classes to flag, e.g. ``['eye blink', 'muscle artifact']``.

    Returns
    -------
    list of int
        Indices of components whose label is in `exclude_labels`.
    """
    unknown = [name for name in exclude_labels if name not in ICLABEL_CLASSES]
    if unknown:
        warnings.warn(f"{unknown} are not ICLabel classes and will never match. "
                      f"Valid classes: {ICLABEL_CLASSES}")
    return [idx for idx, label in enumerate(labels) if label in exclude_labels]


def plot_iclabel_summary(ica, label_dict=None, proba=None, exclude=None, suggest=None,
                         figsize=None, title=None, show=True):
    """
    Stacked bar of the full ICLabel probability distribution per component.

    Shows the complete 7-class distribution rather than only the winning label, so a
    component that is 51% brain / 49% muscle is distinguishable from one that is 99%
    brain. Components are outlined in crimson when excluded and hatched when ICLabel
    suggests excluding them, making disagreements between the two visible at a glance.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        ICA that has been through ``mne_icalabel.label_components``, which populates
        ``ica.labels_scores_`` with the (n_components, 7) probability matrix.
    label_dict : dict | None
        Return value of ``label_components``; only ``'labels'`` is used. Derived from
        `proba` when not given.
    proba : array, shape (n_components, 7) | None
        Probability matrix. Defaults to ``ica.labels_scores_``.
    exclude : list of int | None
        Components to outline in crimson. Defaults to ``ica.exclude``.
    suggest : list of int | None
        Components to hatch, e.g. the output of `iclabel_suggested_exclusions`.
    figsize : tuple | None
        Defaults to a width that scales with the number of components.
    title : str | None
        Overrides the auto-generated title.
    show : bool
        Show the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib.patches import Patch, Rectangle
    from mne.viz.utils import plt_show

    if proba is None:
        proba = getattr(ica, 'labels_scores_', None)
    if proba is None:
        raise RuntimeError(
            "No ICLabel probabilities available. Run "
            "`label_components(inst, ica, method='iclabel')` first (it populates "
            "`ica.labels_scores_`), or pass `proba` explicitly.")
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != len(ICLABEL_CLASSES):
        raise ValueError(f"Expected proba of shape (n_components, {len(ICLABEL_CLASSES)}), "
                         f"got {proba.shape}.")

    n_components = proba.shape[0]
    if label_dict is not None:
        labels = list(label_dict['labels'])
    else:
        labels = [ICLABEL_CLASSES[i] for i in proba.argmax(axis=1)]

    exclude = sorted(set(ica.exclude if exclude is None else exclude))
    suggest = sorted(set([] if suggest is None else suggest))

    if figsize is None:
        figsize = (max(10, 0.5 * n_components), 5.5)
    fig, ax = plt.subplots(figsize=figsize)

    # Stacked probability bars, one segment per ICLabel class
    x = np.arange(n_components)
    colors = plt.get_cmap('tab10').colors
    bottom = np.zeros(n_components)
    for cls_idx, cls_name in enumerate(ICLABEL_CLASSES):
        ax.bar(x, proba[:, cls_idx], bottom=bottom, width=0.8,
               color=colors[cls_idx], label=cls_name)
        bottom += proba[:, cls_idx]

    # Highlights drawn as overlays so a component can carry both styles at once
    for idx in suggest:
        ax.add_patch(Rectangle((idx - 0.4, 0), 0.8, 1.0, fill=False,
                               edgecolor='0.3', linewidth=0.8, hatch='//', zorder=4))
    for idx in exclude:
        ax.add_patch(Rectangle((idx - 0.4, 0), 0.8, 1.0, fill=False,
                               edgecolor='crimson', linewidth=2, zorder=5))

    ax.set_xticks(x)
    ax.set_xticklabels([f"x{i}" if i in exclude else str(i) for i in x],
                       fontsize=8 if n_components > 40 else 10)
    ax.set_xlim(-0.7, n_components - 0.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel('component  (x = excluded, hatched = ICLabel suggestion)')
    ax.set_ylabel('ICLabel probability')
    ax.set_title(title if title is not None
                 else _iclabel_title(n_components, labels, proba, exclude, suggest))

    handles = [Patch(facecolor=colors[i], label=name)
               for i, name in enumerate(ICLABEL_CLASSES)]
    handles.append(Patch(facecolor='white', edgecolor='crimson', linewidth=2,
                         label='excluded'))
    handles.append(Patch(facecolor='white', edgecolor='0.3', hatch='//',
                         label='ICLabel suggestion'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=5, frameon=True)
    fig.tight_layout()

    # Very wide figures get squashed by the Qt backend; scroll them instead
    if figsize[0] * fig.dpi > 1400:
        make_figure_scrollable(fig)
    plt_show(show)
    return fig


def _iclabel_title(n_components, labels, proba, exclude, suggest, max_listed=6):
    """Build the summary title for `plot_iclabel_summary`."""
    header = (f"ICLabel classification — {len(exclude)}/{n_components} components "
              f"excluded, {n_components - len(exclude)} kept")
    if exclude:
        listed = [f"IC{i} {labels[i]} ({proba[i].max():.2f})" for i in exclude[:max_listed]]
        header += " — " + ", ".join(listed)
        if len(exclude) > max_listed:
            header += f" (+{len(exclude) - max_listed} more)"

    kept_but_suggested = [i for i in suggest if i not in exclude]
    excluded_not_suggested = [i for i in exclude if i not in suggest]
    disagreement = []
    if kept_but_suggested:
        disagreement.append("suggested but kept: "
                            + ", ".join(f"IC{i}" for i in kept_but_suggested))
    if suggest and excluded_not_suggested:
        disagreement.append("excluded but not suggested: "
                            + ", ".join(f"IC{i}" for i in excluded_not_suggested))
    if disagreement:
        header += "\n" + " · ".join(disagreement)
    return header

#%%
def Load_and_concatenate_xdf(xdf_files, scale_to_mv=True):
    """
    Load and concatenate multiple XDF files into a single MNE Raw object, preserving annotations.

    Parameters:
    - xdf_files (list): List of paths to the XDF files.
    - scale_to_mv (bool): If True, scales the data to millivolts (mV).

    Returns:
    - raw_combined (mne.io.Raw): Concatenated Raw object with annotations preserved.
    """
    raws = []
    combined_annotations = mne.Annotations(onset=[], duration=[], description=[])
    cumulative_time_offset = 0  # To align annotations across files
    
    for file in xdf_files:
        streams, _ = pyxdf.load_xdf(file)

        # Scale to mV if requested
        if scale_to_mv:
            scale = 1e-6  # Assuming data is in Volts (V) initially
        # Assuming EEG stream is the first stream
        eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
        
        # Extract EEG data
        data = np.array(eeg_stream['time_series'] * scale).T
        sfreq = float(eeg_stream['info']['nominal_srate'][0])
        ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
        ch_types = ['eeg'] * len(ch_names)

    
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        # Extract annotations (marker streams)
        marker_streams = [s for s in streams if s['info']['type'][0] in ['Markers', 'Events']]
        for marker_stream in marker_streams:
            timestamps = np.array(marker_stream['time_stamps'])
            descriptions = np.array(marker_stream['time_series']).astype(str).flatten()

            # Adjust annotation onsets with cumulative time offset
            adjusted_onsets = timestamps - timestamps[0] + cumulative_time_offset
            durations = np.zeros_like(adjusted_onsets)  # Events typically have zero duration

            # Add annotations
            combined_annotations += mne.Annotations(
                onset=adjusted_onsets,
                duration=durations,
                description=descriptions
            )

        # Update cumulative offset for the next file
        cumulative_time_offset += raw.times[-1] + (1 / sfreq)  # Add one sample to avoid overlap

        raws.append(raw)
    
    # Concatenate Raw objects
    raw_combined = mne.concatenate_raws(raws)

    # Attach merged annotations
    raw_combined.set_annotations(combined_annotations)

    return raw_combined
#%%
def filter_events_by_rating(raw, movement_events, rating_prefix="Rating-", rating_threshold=5):
    """
    Keeps only movement events that are followed by a rating >= threshold.
    - Keeps original event labels (e.g., 'Right', 'Left').
    - Removes all rating annotations.
    - Discards movement events with no rating or with a rating < threshold.
    """
    annotations = raw.annotations
    new_annotations = []

    i = 0
    while i < len(annotations):
        desc = annotations.description[i]
        onset = annotations.onset[i]
        duration = annotations.duration[i]

        if desc in movement_events:
            # Check if the next annotation is a rating
            if i + 1 < len(annotations):
                next_desc = annotations.description[i + 1]
                if next_desc.startswith(rating_prefix):
                    rating = int(next_desc.replace(rating_prefix, ""))
                    rating_onset = annotations.onset[i]  # <-- ⭐️ NEW LINE
                    if rating >= rating_threshold:
                        # Keep the original movement event
                        new_annotations.append((onset, duration, desc))
                    new_annotations.append((rating_onset + 5, 0.0, 'Rating'))  # <-- ⭐️ ADD 'Rating' MARKER
                    # Skip the rating annotation either way
                    i += 2
                    continue
                else:
                    # No rating following → discard this event
                    i += 1
                    continue
            else:
                # Last annotation is a movement with no rating → discard
                i += 1
                continue

        elif not desc.startswith(rating_prefix):
            # Keep non-movement, non-rating events (like Beep, Rest, etc.)
            new_annotations.append((onset, duration, desc))

        # Skip rating annotations entirely
        i += 1

    # Apply new annotations
    if new_annotations:
        onsets, durations, descriptions = zip(*new_annotations)
        raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
    else:
        raw.set_annotations(mne.Annotations([], [], []))  # If empty

    return raw
#%%
def fix_channel_names(raw):
    """Fix swapped/misassigned channel labels in the EEG montage."""
    
    # Step 1: move all affected channels to temporary names
    raw.rename_channels({
        'F8':   '_tmp_F8',   'F4':   '_tmp_F4',
        'FC2':  '_tmp_FC2',  'FT10': '_tmp_FT10',
        'Cz':   '_tmp_Cz',   'T8':   '_tmp_T8',
        'CP2':  '_tmp_CP2',  'CP6':  '_tmp_CP6',
        'P4':   '_tmp_P4',   'TP10': '_tmp_TP10',
        'P7':   '_tmp_P7',   'P3':   '_tmp_P3',
        'Pz':   '_tmp_Pz',   'CP1':  '_tmp_CP1',
        'CP5':  '_tmp_CP5',  'TP9':  '_tmp_TP9',
    })

    # Step 2: rename temps to their final destinations
    raw.rename_channels({
        # swaps
        '_tmp_F8':   'F4',   '_tmp_F4':   'F8',
        '_tmp_FC2':  'FT10', '_tmp_FT10': 'FC2',
        '_tmp_Cz':   'T8',   '_tmp_T8':   'Cz',
        '_tmp_CP2':  'CP6',  '_tmp_CP6':  'CP2',
        '_tmp_P4':   'TP10', '_tmp_TP10': 'P4',
        # cycle: P7→P3→Pz→CP1→CP5→TP9→P7
        '_tmp_P7':  'P3',
        '_tmp_P3':  'Pz',
        '_tmp_Pz':  'CP1',
        '_tmp_CP1': 'CP5',
        '_tmp_CP5': 'TP9',
        '_tmp_TP9': 'P7',
    })

    return raw
#%%
SUBJECTS_REQUIRING_CHANNEL_RENAME = {'EA', 'AN', 'NZ', 'NS', 'SK', 'OT'}

def fix_channel_names_for_subject(raw, subject):
    """Apply fix_channel_names(raw) only for subjects whose recordings need the
    swapped/misassigned channel-label correction. No-op for all other subjects."""
    if subject in SUBJECTS_REQUIRING_CHANNEL_RENAME:
        raw = fix_channel_names(raw)
    return raw
#%%
def get_subject_bad_electrodes(subject):
    elecs_to_drop={}
    #define here the subject specific electdodes to make sure are removed from the data: 
    bad_elecs_dict={'Dekel':{'FT10', 'TP10', 'FT9'},
                    'Gilad':{'FT10', 'TP10', 'FT9', 'TP9'},
                    'Neta':{'TP9'},
                    'Ron-Block':{'PO7'},
                    'sub-Roei': {'TP9'},
                    'Or': {'FT9','T7','FC2','FT7','Iz'},
                    'Roei-MI': {'FT10', 'TP10','P2','AF8','AF7','AF4'},
                    'Fudge':{'Iz','FT10', 'TP10', 'FT9', 'TP9','F1'},
                    'g': {'T7','CP1','TP9','P7','PO7','O1'},
                    'Ron': {'Iz','Cz'},                   
                    'GiladRSL' : {'C5','FC4','CP5','T7','FT9','FT10','TP9','TP10','T8'},
                    'NoamV' : {'Iz', 'T7','O1','O2','Oz'},
                    'DD' : {'Cz','CP5','FC2','T7','P4','Iz','FT8','P5','FT10', 'TP10'},
                    'JE' : {'T7','TP9','Iz','TP7','FT7'},
                    'NC' : {'TP9', 'AF8','AF7','Iz'},
                    'EA' : {'Fp1', 'P4','T7','FT9','FT10','TP9','TP10','T8'},
                    'SK' : {'Iz','T7','O1','O2','Oz','FT10','TP9','TP10','T8'},
                    'Tomer' : {'FC5','CP1','F4','TP9','FT8','P5'},
                    'NS' : {'T7','TP10'},
                    'ID' : {'TP9','TP7'},
                    'LD' : {'TP10','FT8'},
                    'AEH' : {'T8','T7','TP9'},
                    'BA' : {'T7','T8','TP9','TP10','FT10','FT9'}
                }
    if subject in bad_elecs_dict.keys():
        subject_bad_electrodes=bad_elecs_dict[subject]
    else: 
        subject_bad_electrodes={}
        print('note that no bad electrodes were defined for the current subject:',subject)
    return subject_bad_electrodes
#%%
# ---------------------------------------------------------------------------
# Montage + online-reference reconstruction.
# Shared by the offline pipeline and the live-stream loop: both MUST run the
# identical sequence or the model sees a different channel set online than it
# was trained on. Do not inline a copy of this - call it.
# ---------------------------------------------------------------------------
REF_CHANNEL_NAME = 'FCz'   # the amplifier's online reference ('REF' in the .bvef)


def load_montage(current_path, ref_channel=REF_CHANNEL_NAME):
    """Read the CACS-64 montage, exposing the online reference as a real electrode.

    The .bvef ships 66 entries: 'GND', 'REF' and the 64 recorded electrodes.
    'REF' is the amplifier's online reference (Theta=23, Phi=90) i.e. FCz, and
    read_custom_montage exposes it as an ordinary montage name. Renaming it to
    ref_channel puts a usable position (0.0, 37.1, 87.4) mm - midway between Fz
    and Cz, same y as FC1/FC2 - into the montage, so a reconstructed FCz channel
    gets a real location from set_montage instead of NaN.

    The montage file itself is left untouched. A fresh DigMontage is returned on
    every call, so the rename can never mutate a montage held by a caller.
    """
    montage = mne.channels.read_custom_montage((f"{current_path}\Montages\CACS-64_REF.bvef"), head_size=0.095, coord_frame=None)
    if 'REF' in montage.ch_names and ref_channel not in montage.ch_names:
        montage.rename_channels({'REF': ref_channel})
    return montage


def apply_montage_and_reference(inst, montage, params_dict, ref_channel=REF_CHANNEL_NAME):
    """Set the montage, optionally rebuild the online reference, drop bads, average-reference.

    Enabled with params_dict['AddRefChannel'] = True (default False, i.e. the
    historical behaviour). When on, the online reference is re-added as a data
    channel: it starts as zeros, and the average reference then leaves it holding
    -(sum of every other channel) == -(64/65) * mean(recorded channels), the
    estimated potential at the reference site. Note this is the common-mode
    estimate, not an independent measurement - against the full 64-channel set it
    adds no information to a linear model. It adds real information only to a
    channel *subset* (Electorde_Group), where it carries the aggregate of the
    electrodes the subset excludes. Corollary: it is a good canary for an
    unlisted bad electrode - if std(FCz)/std(Cz) >> 1, something is leaking into
    the reference, and hence into every channel.

    THE ORDER BELOW IS LOAD-BEARING. Do not reshuffle it:

    1. add_reference_channels FIRST, while info['dig'] is still None. MNE
       positions the new channel from an EEG dig point with ident==0; a
       bvef-derived montage has none, so once set_montage has run the new
       channel's loc is left NaN (silently - MNE 1.6 emits no catchable
       warning), and compute_current_source_density later dies with
       "Zero or infinite position found in chs". It also refuses to run at all
       once an average-reference projection is active.
    2. set_montage, which gives the new channel its real position.
    3. Drop bad electrodes, so they cannot contaminate the average.
    4. Average reference. This is per-time-sample and purely spatial, so the
       result is bit-identical whether computed over a whole recording or a
       55-sample live chunk - which is what lets the live loop match training
       exactly, provided the channel SET here is the same.

    Steps 2-4 are exactly the pre-existing block, so with 'AddRefChannel' absent
    or False this function is a pure refactor.

    Parameters
    ----------
    inst : instance of Raw
        Preloaded, with non-EEG channels already dropped. Modified in place.
    montage : DigMontage
        As returned by load_montage (i.e. with ref_channel present).
    params_dict : dict
        Reads 'AddRefChannel' (default False), 'bad_electrodes', 'PerformAvgRef'.
    ref_channel : str
        Name of the online reference to reconstruct.

    Returns
    -------
    inst : instance of Raw
        The same (mutated) object, with ref_channel appended last if enabled.
    elecs_to_drop : set of str
        Bad electrodes that were actually present and dropped. Callers need this
        to build the pick list - see select_electrodes.
    """
    bad_electrodes = set(params_dict.get('bad_electrodes') or [])
    add_ref = params_dict.get('AddRefChannel', False)

    if add_ref and not params_dict['PerformAvgRef']:
        raise ValueError(
            "AddRefChannel=True requires PerformAvgRef=True: without average "
            f"referencing the reconstructed {ref_channel} stays all-zeros.")

    if add_ref:
        if ref_channel in bad_electrodes:
            print(f'AddRefChannel is on but {ref_channel} is listed in bad_electrodes '
                  f'- not reconstructing it.')
        elif ref_channel in inst.ch_names:
            #keep this idempotent: EEG_Preprocessing mutates the caller's raw in
            #place, so re-running a cell must not raise.
            print(f'{ref_channel} is already present - not adding it again.')
        else:
            inst = mne.add_reference_channels(inst, [ref_channel], copy=False)

    inst.set_montage(montage, match_case=True, match_alias=False, on_missing='raise', verbose=None)

    print('\n###########################################################')
    print('removing subject specific bad electrodes from the raw data')
    elecs_to_drop = set(inst.info['ch_names']).intersection(bad_electrodes)
    if len(elecs_to_drop) > 0:
        inst.drop_channels(list(elecs_to_drop))
    inst.drop_channels(inst.info['bads'])

    if (params_dict['PerformAvgRef']):
        inst.set_eeg_reference(ref_channels="average")
    return inst, elecs_to_drop


def select_electrodes(inst, params_dict, elecs_to_drop, ref_channel=REF_CHANNEL_NAME):
    """Build the pick list: Electorde_Group minus the dropped bad electrodes, in group order.

    Also removes ref_channel when it is not in the data - the one legitimate
    mismatch, e.g. a notebook that keeps 'FCz' in its electrode group while
    running with AddRefChannel off, or a subject with FCz marked bad. Every
    *other* missing name is deliberately left in, so that pick() still raises:
    silently skipping absent electrodes is exactly how the offline and live
    channel sets could drift apart unnoticed.

    pick(names) honours the requested order, so the returned list defines the
    channel order the classifier sees. It must be identical offline and live.
    """
    selected_elecs = [elec for elec in params_dict['Electorde_Group'] if elec not in elecs_to_drop]
    if ref_channel in selected_elecs and ref_channel not in inst.ch_names:
        print(f'{ref_channel} is in Electorde_Group but not in the data '
              f'(AddRefChannel off?) - excluding it from the picks.')
        selected_elecs = [elec for elec in selected_elecs if elec != ref_channel]
    return selected_elecs
#%%
# ---------------------------------------------------------------------------
# Artifact Subspace Reconstruction, the optional step before ICA.
# ---------------------------------------------------------------------------
# asrpy 0.0.8 documents a 'riemann' variant but does not implement it: ASR.__init__
# overwrites self.method with "euclid" whatever you pass, and no code downstream of
# that reads the attribute, so the two settings produce bit-identical output with no
# warning. Accepting 'riemann' here would therefore be exactly the silent no-op this
# repo turns into an error everywhere else - so only the variant that actually runs
# is allowed, and asking for the other one raises.
#
# What it would mean: ASR thresholds come from an average of block covariances and a
# decomposition of each window's covariance. Euclidean ASR treats those as flat
# vectors (geometric_median of the flattened matrices, then linalg.eigh). Riemannian
# ASR (Blum et al. 2019) treats them as SPD matrices on a curved manifold, replacing
# the mean with pyriemann's mean_covariance(metric='riemann') and the eigenvectors
# with principal geodesic analysis; asrpy leaves both as comments at the exact call
# sites. It is reported to be more robust when the CALIBRATION data is itself
# contaminated - the regime these recordings are in, where clean_windows keeps only
# ~a third of the samples - so it is worth wanting. meegkit.asr implements it if that
# becomes the reason to switch backends.
ASR_BACKENDS = ('asrpy', 'meegkit')
ASR_METHODS = ('euclid', 'riemann')

# Which backend can actually run which method. asrpy documents a 'riemann' mode but
# does not implement it - its ASR.__init__ overwrites self.method with "euclid"
# whatever you pass, nothing downstream reads the attribute, and the two settings
# produce bit-identical output with no warning. So riemann is routed to meegkit,
# and asking asrpy for it raises instead of silently doing euclid.
ASR_METHOD_BACKENDS = {'euclid': ('asrpy', 'meegkit'), 'riemann': ('meegkit',)}

# Covariance estimator for the meegkit backend, passed through to its block
# covariance step. 'lwf' (Ledoit-Wolf shrinkage) is the default rather than
# meegkit's own 'scm' because RIEMANN CANNOT RUN ON 'scm' HERE: the average
# reference costs one rank, so the block covariances are singular - measured on
# BA, 30 of 50 blocks have a non-positive eigenvalue (min -1.06e-23, condition
# number 6e16) - and the riemannian mean needs a matrix logarithm, which is only
# defined for strictly positive definite input. meegkit stops with "Matrices must
# be positive definite. Add regularization to avoid this error." Shrinkage
# regularises them back to SPD. Euclidean ASR is unaffected either way.
ASR_ESTIMATORS = ('scm', 'lwf', 'oas', 'mcd')

# Budget for the largest intermediate array inside asrpy's asr_process, which holds
# a moving covariance of shape (samples per split, n_channels ** 2) in float64. Its
# own mem_splits default of 3 sizes that from nothing at all: on a 64-channel,
# half-hour recording it asks for ~10 GB and the call dies with MemoryError before
# it has cleaned anything. Splitting is not an approximation - asr_process carries
# its filter state, covariance and mixing matrix across the split boundaries - so
# this trades only a little speed. 512 MB leaves room for the two or three copies
# of that array numpy makes en route. meegkit chunks internally and needs none of it.
ASR_SPLIT_BYTES = 512 * 1024 ** 2

# A channel this many times louder than the median one is called out before ASR
# runs. Not a rejection threshold - nothing is dropped - just the point past which
# a channel is loud enough to drive ASR's calibration for the whole montage.
ASR_LOUD_CHANNEL_RATIO = 5


def _import_asr(backend):
    """Import an ASR backend on demand, with an actionable message when it is missing.

    Imported here rather than at module scope so that this module - and hence the
    whole package - still imports in an environment without either backend, which
    is every environment while 'PerformAsr' is off (the default).

    A note on versions, because it is not a free choice: the meegkit backend must be
    **0.1.7**. From 0.1.9 meegkit requires pyriemann >= 0.7, and 0.2.0 requires
    >= 0.12 - and pyriemann 0.12 calls ndarray.mT, which exists only in numpy >= 2.0.
    This environment is pinned to numpy 1.26 for mne 1.6.1, so installing that chain
    breaks the classifier stack outright (Covariances(estimator='oas') dies with
    "'numpy.ndarray' object has no attribute 'mT'"). 0.1.7 is the last release that
    talks to pyriemann 0.3, which is what the TS+FGDA pipeline is built on:

        python -m pip install "meegkit==0.1.7" pymanopt
    """
    if backend == 'asrpy':
        try:
            import asrpy
        except ImportError as err:
            raise ImportError(
                "PerformAsr is on but asrpy is not installed. Install it into the "
                "kernel's environment:\n"
                "    python -m pip install asrpy") from err
        return asrpy
    if backend == 'meegkit':
        try:
            from meegkit.asr import ASR as MeegkitASR
        except ImportError as err:
            raise ImportError(
                "asr_backend='meegkit' needs meegkit and pymanopt. Pin the version - "
                "newer ones force a pyriemann upgrade this environment cannot take "
                "(see _import_asr):\n"
                '    python -m pip install "meegkit==0.1.7" pymanopt') from err
        return MeegkitASR
    raise ValueError(f"asr_backend must be one of {ASR_BACKENDS}, got {backend!r}.")


def _asr_with_asrpy(raw, params_dict, eeg_picks, n_times):
    """Run asrpy's euclidean ASR. Returns (cleaned raw, calibration sample mask)."""
    asrpy = _import_asr('asrpy')
    n_channels = len(eeg_picks)
    samples_per_split = max(1, int(ASR_SPLIT_BYTES / (n_channels ** 2 * 8)))
    mem_splits = max(1, int(np.ceil(n_times / samples_per_split)))

    asr = asrpy.ASR(sfreq=raw.info['sfreq'],
                    cutoff=params_dict.get('asr_cutoff', 20),
                    max_bad_chans=params_dict.get('asr_max_bad_chans', 0.1),
                    method='euclid')
    # return_clean_window hands back which samples the calibration considered clean.
    # It costs nothing extra - fit computes the mask either way - and it is what makes
    # the report below meaningful.
    _, sample_mask = asr.fit(raw, return_clean_window=True)
    # returns a copy (apply_function on raw.copy()), so the input is left alone
    return asr.transform(raw, mem_splits=mem_splits), sample_mask


def _asr_with_meegkit(raw, params_dict, eeg_picks, before):
    """Run meegkit's ASR, which is the one that implements the riemannian variant.

    meegkit works on plain arrays, so the cleaned data is written back through
    ``apply_function`` on a copy - the same route asrpy takes internally, which
    keeps info and annotations intact.
    """
    MeegkitASR = _import_asr('meegkit')
    method = params_dict.get('asr_method', 'euclid')
    asr = MeegkitASR(sfreq=raw.info['sfreq'],
                     cutoff=params_dict.get('asr_cutoff', 20),
                     method=method,
                     estimator=params_dict.get('asr_estimator', 'lwf'))
    # max_bad_chans is hard-coded to 0.3 in meegkit 0.1.7's __init__ and read off the
    # instance by fit(), so honouring the parameter means setting it here. Without
    # this the key would quietly mean nothing on this backend.
    asr.max_bad_chans = params_dict.get('asr_max_bad_chans', 0.1)
    # meegkit downgrades riemann -> euclid with a logging.warning (not a Python
    # warning, so it is easy to miss) when pyriemann is absent. Catch that here
    # rather than let a run be labelled riemann when it was not.
    if asr.method != method:
        raise RuntimeError(
            f"meegkit silently changed the ASR method from {method!r} to "
            f"{asr.method!r} - pyriemann is probably not importable.")

    clean, sample_mask = asr.fit(before)
    cleaned_data = asr.transform(before)
    cleaned = raw.copy()
    cleaned.apply_function(lambda _: cleaned_data, picks=eeg_picks,
                           channel_wise=False)
    return cleaned, sample_mask


def apply_asr(raw, params_dict):
    """Optionally clean transient artifacts out of continuous data with ASR.

    Does nothing unless params_dict['PerformAsr'] is True. While it is off - the
    default - this is a pure no-op returning the caller's own object, so every
    result predating this function is reproduced exactly.

    ASR calibrates on the quiet stretches of THIS recording (the backend's own
    clean-window search finds them), builds a covariance of what clean data looks
    like, then slides over the data reconstructing any subspace whose variance
    exceeds 'asr_cutoff' standard deviations of that reference from the remaining
    channels. Unlike ICA it is local in time: it edits the seconds that are
    contaminated and leaves the rest alone. Measured on BA_MI1 at the default
    cutoff of 20: 0.1% of the variance removed from the stretches its calibration
    called clean, which correlate at 0.999 with the input, against 57% from the
    rest. That contrast is the thing to check when tuning the cutoff, and it is
    what the report printed at the end of this function shows.

    BACKENDS AND METHODS
    --------------------
    'euclid' is classic ASR (Mullen et al.), available on both backends.

    'riemann' is the Blum et al. 2019 modification and needs asr_backend='meegkit'.
    Covariance matrices are SPD matrices on a curved manifold, and averaging them as
    if the space were flat is biased; the riemannian variant replaces the euclidean
    mean with the geometric one - in BOTH the calibration and the per-window
    processing step - which is reported to be more robust when the calibration data
    is itself contaminated. That is this cohort's regime: clean_windows keeps only
    about a third of these recordings.

    BUT MEASURE BEFORE YOU TRUST IT. meegkit marks its riemannian eigenstep
    (nonlinear_eigenspace, a pymanopt trust-region solve on a Grassmann manifold)
    with a TODO upstream, and it shows. On BA_MI1, 5 minutes, cutoff 20 vs 40 vs 100
    produce identical output: 100% of samples rewritten, 25% of variance removed
    overall and 34% removed from the samples its own calibration called clean. A
    cutoff of 100 SD should be nearly a no-op. The threshold is reaching the backend
    (meegkit's euclid does differ between cutoff 5 and 20), so this is the riemannian
    path itself, not the plumbing. For contrast, asrpy euclid at cutoff 100 alters
    8.3% of samples and removes 0.0% on the clean stretches - which is what a
    correctly thresholded ASR looks like. Riemann is available here so it can be
    tried and measured; it is not a drop-in improvement.

    'riemann' also requires a regularising asr_estimator ('lwf' by default) - see
    ASR_ESTIMATORS for why 'scm' cannot work on average-referenced data.

    WHERE THIS SITS IN THE CHAIN, and why it is here and nowhere else:

    * AFTER the 1-100 Hz high-pass and the 50 Hz notch. ASR's calibration is a
      variance estimate, so on unfiltered data the drift dominates it and the
      thresholds describe the drift rather than the artifacts.
    * BEFORE the analysis band-pass, so the statistics are computed on broadband
      data whatever band the caller ends up analysing. This is what lets the same
      'asr_cutoff' mean the same thing in the TFR stack (1-None Hz) and the
      classifier stack (8-32 Hz).
    * BEFORE compute_current_source_density, which mixes channels and changes the
      units the thresholds are in.
    * BEFORE ICA, which is the point of the exercise: removing the large
      transients first stops them from dominating the decomposition, so the
      components left to review are the stationary artifacts ICA is good at.

    Calibration is per call, i.e. per recording, because that is how the callers
    are structured - EEG_Preprocessing runs once per xdf file. That is the
    desirable grain anyway: impedance and electrode contact differ between
    sessions, and a threshold from one session describes the next one poorly.

    COST: asrpy runs at about 0.22 s per second of 64-channel 500 Hz data, measured
    linearly over 2 and 5 minute segments - ~11 min for one subject's three
    recordings, roughly 3 hours to rebuild the whole cohort's epoch cache. meegkit
    is faster for euclid and somewhat slower for riemann.

    Parameters
    ----------
    raw : instance of Raw
        Preloaded, high-pass filtered, EEG channels only. NOT modified.
    params_dict : dict
        Reads 'PerformAsr' (default False), 'asr_backend', 'asr_cutoff',
        'asr_max_bad_chans', 'asr_method', 'asr_estimator'.

    Returns
    -------
    raw : instance of Raw
        The caller's own object when ASR is off; a cleaned copy when it is on.
    """
    if not params_dict.get('PerformAsr', False):
        return raw

    backend = params_dict.get('asr_backend', 'asrpy')
    method = params_dict.get('asr_method', 'euclid')
    estimator = params_dict.get('asr_estimator', 'lwf')
    cutoff = params_dict.get('asr_cutoff', 20)
    max_bad_chans = params_dict.get('asr_max_bad_chans', 0.1)

    if backend not in ASR_BACKENDS:
        raise ValueError(f"asr_backend must be one of {ASR_BACKENDS}, got {backend!r}.")
    if method not in ASR_METHODS:
        raise ValueError(f"asr_method must be one of {ASR_METHODS}, got {method!r}.")
    if backend not in ASR_METHOD_BACKENDS[method]:
        raise ValueError(
            f"asr_method={method!r} is not available on asr_backend={backend!r}; "
            f"it needs one of {ASR_METHOD_BACKENDS[method]}. asrpy documents a "
            f"riemannian mode but hard-codes itself back to euclidean, so allowing "
            f"this would silently give you euclidean ASR labelled as riemannian.")
    if estimator not in ASR_ESTIMATORS:
        raise ValueError(f"asr_estimator must be one of {ASR_ESTIMATORS}, got "
                         f"{estimator!r}.")

    detail = f", estimator {estimator!r}" if backend == 'meegkit' else ''
    print('\n###########################################################')
    print(f'running ASR before ICA ({backend}, method {method!r}{detail}, '
          f'cutoff {cutoff} SD, max_bad_chans {max_bad_chans})')
    if method == 'riemann':
        print("  !! riemannian ASR here does NOT respond to asr_cutoff. Measured on "
              "BA_MI1, 5 min:")
        print("     cutoff 20 / 40 / 100 all give the same output - 100% of samples "
              "rewritten and 34%")
        print("     of the variance removed from the stretches its own calibration "
              "called clean (asrpy")
        print("     euclid removes 0.0-0.1% there). meegkit marks its riemannian "
              "eigenstep TODO upstream,")
        print("     which fits. Treat this as an experiment, not a tuned method.")
    #picked by index, so `before`, `after` and `cleaned_names` are the same
    #channels in the same order - raw.ch_names is not necessarily eeg-only
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    cleaned_names = [raw.ch_names[i] for i in eeg_picks]
    before = raw.get_data(picks=eeg_picks)

    # A single loud electrode is the one thing that derails this step, and these
    # pipelines drop no bad channels by design (the TFR/benchmark params set
    # bad_electrodes empty to keep every subject's montage stackable). After
    # average referencing, that electrode is present in EVERY channel, so ASR
    # chases it across the whole montage and clean_windows rejects most of the
    # recording as uncalibratable. Say so rather than letting it show up as an
    # unexplained "ASR removed most of the variance".
    channel_std = before.std(axis=1)
    loud = [(name, std / np.median(channel_std))
            for name, std in zip(cleaned_names, channel_std)
            if std > ASR_LOUD_CHANNEL_RATIO * np.median(channel_std)]
    if loud:
        print('  !! ' + ', '.join(f'{name} is {ratio:.0f}x the median channel'
                                  for name, ratio in loud))
        print('     ASR calibrates on the montage as given and nothing here drops bad '
              'electrodes, so')
        print('     after the average reference this sits in every channel and ASR will '
              'chase it in all')
        print('     of them. Consider listing it in bad_electrodes, or expect heavy '
              'cleaning.')

    if backend == 'asrpy':
        cleaned, sample_mask = _asr_with_asrpy(raw, params_dict, eeg_picks, raw.n_times)
    else:
        cleaned, sample_mask = _asr_with_meegkit(raw, params_dict, eeg_picks, before)

    #both backends rebuild the Raw through apply_function, so info and annotations
    #come through untouched - but the epoching downstream reads the annotations, and
    #a silent loss of them would show up as "0 epochs" several steps later rather
    #than here. Cheap to assert, so assert it.
    if len(cleaned.annotations) != len(raw.annotations):
        raise RuntimeError(
            f"ASR changed the annotation count "
            f"({len(raw.annotations)} -> {len(cleaned.annotations)}); the events "
            f"downstream are derived from these.")

    # What ASR actually did. Reported as three numbers rather than one, because on a
    # recording with a genuinely bad channel the single "variance removed" figure
    # reads like ASR ate the signal when it did nothing of the sort:
    #
    #   overall     dominated by the artifacts, so it tracks how dirty the recording
    #               is more than how aggressive ASR was
    #   on clean    the number that says whether ASR stayed off the data it was
    #               supposed to leave alone. Near zero is correct and expected;
    #               anything substantial means the cutoff is eating real signal
    #   altered     what fraction of the recording ASR actually rewrote
    after = cleaned.get_data(picks=eeg_picks)
    mask = np.asarray(sample_mask).ravel().astype(bool)
    variance_removed = 1 - after.var(axis=1) / np.where(
        before.var(axis=1) == 0, np.nan, before.var(axis=1))
    scale = np.median(channel_std)
    altered = (np.abs(before - after).max(axis=0) > 0.1 * scale).mean()
    report = (f'ASR altered {100 * altered:.1f}% of samples; variance removed '
              f'{100 * np.nanmean(variance_removed):.1f}% overall')
    if mask.any() and mask.size == before.shape[1]:
        on_clean = 1 - after[:, mask].var(axis=1) / np.where(
            before[:, mask].var(axis=1) == 0, np.nan, before[:, mask].var(axis=1))
        report += (f', {100 * np.nanmean(on_clean):.1f}% on the '
                   f'{100 * mask.mean():.0f}% of samples its calibration called clean')
    print(report)
    return cleaned


#%%
def raw_EEG_Preprocessing (current_path,raw, params_dict):

    #extract the current run paramaters: 
    PerformCsd=params_dict['PerformCsd']
    LowPass, HighPass, filter_method = params_dict['LowPass'],params_dict['HighPass'],params_dict['filter_method']
    tmin=params_dict['epoch_tmin']
    tmax=params_dict['epoch_tmax']
    filter_bank_epochs = None
    #read the file:
    Raw=raw
    #remove non existent channels: 
    if 'ACC_X' in Raw.ch_names:
        Raw.drop_channels(['ACC_X','ACC_Y','ACC_Z']) ## Drop non eeg channels
    #rename channels for consistency (no longer required for future recordings):
    #mne.rename_channels(Raw.info, {'F9' : 'FT9','P9' : 'TP9','P10' : 'TP10','F10' : 'FT10','AF1' : 'AF7' }, allow_duplicates=False, verbose=None)
    #set the montage, optionally rebuild the online reference (FCz), drop bad
    #electrodes and average-reference - see apply_montage_and_reference, whose
    #step order must not be reshuffled:
    montage = load_montage(current_path)
    Raw, elecs_to_drop = apply_montage_and_reference(Raw, montage, params_dict)
    print('\n###########################################################')
    print('filtering the data')
    unfiltered_Raw=Raw.copy()
    if (filter_method == 'iir'):
        notched_Raw = unfiltered_Raw.filter(1,100, method=filter_method, phase='forward', pad=0)
        notched_Raw.notch_filter(50, method=filter_method, phase='forward') 
        notched_Raw = apply_asr(notched_Raw, params_dict)
        if PerformCsd:
            notched_Raw = mne.preprocessing.compute_current_source_density(notched_Raw) # Perform current source density
        Raw_Filtered = notched_Raw.filter(LowPass, 100, method=filter_method, iir_params = dict(order=4, ftype='butter'),phase='forward',pad=0)
    if (filter_method == 'fir'):
        notched_Raw = unfiltered_Raw.filter(1,None, method=filter_method)  
        notched_Raw.notch_filter(50, method=filter_method) 
        notched_Raw = apply_asr(notched_Raw, params_dict)
        if PerformCsd:
            notched_Raw = mne.preprocessing.compute_current_source_density(notched_Raw) # Perform current source density
        Raw_Filtered = notched_Raw
    return Raw_Filtered

def Post_ICA_EEG_Preprocessing (current_path,raw, params_dict):
    #NO apply_asr call in here, deliberately. ASR belongs before ICA, and this
    #function is the second half of a raw_EEG_Preprocessing -> ICA -> here
    #sequence whose first half already ran it - calling it again would clean
    #already-cleaned data against a fresh set of thresholds.
    #extract the current run paramaters: 
    PerformCsd=params_dict['PerformCsd']
    LowPass, HighPass, filter_method = params_dict['LowPass'],params_dict['HighPass'],params_dict['filter_method']
    tmin=params_dict['epoch_tmin']
    tmax=params_dict['epoch_tmax']
    filter_bank_epochs = None
    Raw = raw
    print('\n###########################################################')
    print('filtering the data')  
    unfiltered_Raw=Raw.copy()
    if (filter_method == 'iir'):
        notched_Raw = unfiltered_Raw.filter(1, 100, method=filter_method, phase='forward', pad=0)  
        notched_Raw.notch_filter(50, method=filter_method, phase='forward') 
        if PerformCsd:
            notched_Raw = mne.preprocessing.compute_current_source_density(notched_Raw) # Perform current source density
        Raw_Filtered = notched_Raw.filter(LowPass, HighPass, method=filter_method, iir_params = dict(order=4, ftype='butter'),phase='forward',pad=0)
    if (filter_method == 'fir'):
        notched_Raw = unfiltered_Raw.filter(1, 100, method=filter_method)  
        notched_Raw.notch_filter(50, method=filter_method) 
        if PerformCsd:
            notched_Raw = mne.preprocessing.compute_current_source_density(notched_Raw) # Perform current source density
        Raw_Filtered = notched_Raw.filter(LowPass, HighPass, method=filter_method)
    if params_dict['pipeline_name']=='fbcsp+lda':
        #extract filterbank feequencies:
        filters_bands=tuple(params_dict['filters_bands'])
        filtered_data_band_passed=[]
        for i,(LowPass,HighPass) in enumerate(filters_bands):
            unfiltered_Raw=Raw.copy()
            Raw_Filtered_band= mne.filter.filter_data(unfiltered_Raw.get_data(),sfreq=500, l_freq=LowPass, h_freq=HighPass, method='fir',copy = True)
            filtered_data_band_passed.append(Raw_Filtered_band)
    
    events_from_annot,event_dict = mne.events_from_annotations(Raw_Filtered)
    events_trigger_dict = {key: event_dict[key] for key in event_dict.keys() if key in params_dict['desired_events']}

    # if events_from_annot is None or len(events_from_annot) == 0:
    #     print("No events found in this recording. Returning filtered_raw only.")
    #     filter_bank_epochs = []
    #     mean_across_epochs = None
    #     epochs = None
    #     return Raw_Filtered, epochs, filter_bank_epochs, mean_across_epochs, events_trigger_dict

    print('\n###########################################################')
    print('extracting event info:',event_dict)
    
    #filtered_electrodes  = [elec for elec in params_dict['Electorde_Group'] if elec not in elecs_to_drop]
    #selected_elecs=filtered_electrodes
    
    # Handle the case where there are NO events (e.g., pure idle file)
    if events_trigger_dict is None or len(events_trigger_dict) == 0:
        print("No events found in this recording. Returning filtered_raw only.")
        filter_bank_epochs = []
        mean_across_epochs = None
        events_idle = mne.make_fixed_length_events(
            Raw_Filtered,
            id=0,
            start=8.0,     # first event at 4 s -> epoch spans 0..10 s
            duration=10.0  # events spaced every 10 s
        )

        # 2) Epoch around these events with the same window as your MI epochs
        idle_event_id = {'Idle': 0}

        idle_epochs = mne.Epochs(
            Raw_Filtered,
            events_idle,
            event_id=idle_event_id,
            tmin=-8.0,
            tmax=6.0,
            baseline=None,   # same baseline as MI
            detrend=0,
            preload=True
        )
        idle_epochs.pick(selected_elecs)
        return Raw_Filtered, idle_epochs, filter_bank_epochs, mean_across_epochs, events_trigger_dict

    if params_dict['pipeline_name']=='fbcsp+lda':
        #filter bank related: 
        filter_bank_epochs=[]
        for filtered_data_band in filtered_data_band_passed:
            filtered_data_band_raw = mne.io.RawArray(filtered_data_band,unfiltered_Raw.info)
            epochs = mne.Epochs(filtered_data_band_raw, events_from_annot, preload = True,baseline= None, tmin=tmin, tmax=tmax, event_id=events_trigger_dict,detrend=0)
            # Calculate the mean across epochs for the current event
            mean_across_epochs = epochs.get_data().mean(axis=0)
            event_data = epochs.get_data()         
            # Subtract the mean from each epoch of the current event
            centered_event_data = event_data - mean_across_epochs
            event_epochs= epochs.events
            epochs = mne.EpochsArray(centered_event_data, epochs.info, events=event_epochs, event_id=epochs.event_id, tmin=epochs.tmin)
            filter_bank_epochs.append(epochs)
        
    
    epochs = mne.Epochs(Raw_Filtered, events_from_annot, preload = True,baseline= None, tmin=tmin, tmax=tmax, event_id=events_trigger_dict,detrend=0)
    
    # If we want to perform auto rejection of epochs (time expensive)
    #ar = AutoReject()
    #epochs = ar.fit_transform(epochs)  

    
    #epochs.pick(selected_elecs)
    ## Centering the data
    #the grand mean is returned to the caller either way:
    mean_across_epochs = epochs.get_data().mean(axis=0)

    #Per-class centering, controlled by params_dict['CenterByClass'] - read exactly as
    #in EEG_Preprocessing (default True, the historical behaviour), so the two
    #functions cannot drift apart. See that function for what the flag means.
    if params_dict.get('CenterByClass', True):
        centered_data_list = []
        events_list = []
        # Loop through each event ID
        for idx,event_id in enumerate(params_dict['desired_events']):
            print (event_id)
            # Extract epochs for the current event
            event_epochs = epochs[event_id]
            event_data = event_epochs.get_data()

            # Calculate the mean across epochs for the current event
            mean_across_event_epochs = event_data.mean(axis=0)

            # Subtract the mean from each epoch of the current event
            centered_event_data = event_data - mean_across_event_epochs

            # Store the centered data
            centered_data_list.append(centered_event_data)

            # Prepare the events list and event_id_map for the combined EpochsArray
            events_list.append(event_epochs.events)

        # Concatenate all centered data and events
        centered_data = np.concatenate(centered_data_list, axis=0)
        combined_events = np.concatenate(events_list, axis=0)

        # Sort the combined events based on their original occurrence time to preserve the temporal sequence
        sort_indices = np.argsort(combined_events[:, 0])
        combined_events = combined_events[sort_indices]
        centered_data = centered_data[sort_indices]

        # Create a new EpochsArray with the centered data
        centered_epochs = mne.EpochsArray(centered_data, epochs.info, events=combined_events, event_id=epochs.event_id, tmin=epochs.tmin)
        epochs = centered_epochs
    else:
        print('CenterByClass is off - leaving the per-class evoked response in the data.')

    #this section drops electrodes after epoching: but currently we drop all bad electrodes from the raw data
    print('\n###########################################################')
    print('removing bad channels from epochs:')
    curr_elecs_in_epochs_set=set(epochs.info['ch_names'])
    elecs_to_remove=params_dict['bad_electrodes']
    elecs_to_drop=curr_elecs_in_epochs_set.intersection(elecs_to_remove)

    if len(elecs_to_drop)>0:
        epochs.info['bads']=elecs_to_drop
        epochs.drop_channels(epochs.info['bads'])
        print('\n###########################################################')
        print(f'Removed: {elecs_to_drop} from the current selected electrodes: {curr_elecs_in_epochs_set} from the overall set of bad electrodes {elecs_to_remove}')
        print('#############################################################')
    
        #filter bank related: 
        filter_bank_epochs_after_elec_drops=[]
        for curr_epochs in filter_bank_epochs:
            curr_epochs.info['bads']=elecs_to_drop
            curr_epochs.drop_channels(epochs.info['bads'])
            filter_bank_epochs_after_elec_drops.append(curr_epochs)    
    elif params_dict['pipeline_name']=='fbcsp+lda': 
        #filter bank related: 
        filter_bank_epochs_after_elec_drops=[]
        for curr_epochs in filter_bank_epochs:
            filter_bank_epochs_after_elec_drops.append(curr_epochs)  

        print('\n###########################################################')
        print(f'the current selected electrodes: {curr_elecs_in_epochs_set} allready exclude the requested electrodes to remove {elecs_to_remove}')
        print('#############################################################')

    return Raw_Filtered,epochs,filter_bank_epochs,mean_across_epochs, events_trigger_dict


def EEG_Preprocessing (current_path,raw, params_dict, pick_channels=True):

    #extract the current run paramaters: 
    PerformCsd=params_dict['PerformCsd']
    LowPass, HighPass, filter_method = params_dict['LowPass'],params_dict['HighPass'],params_dict['filter_method']
    tmin=params_dict['epoch_tmin']
    tmax=params_dict['epoch_tmax']
    filter_bank_epochs = None
    #read the file:
    Raw=raw
    #remove non existent channels: 
    if 'ACC_X' in Raw.ch_names:
        Raw.drop_channels(['ACC_X','ACC_Y','ACC_Z']) ## Drop non eeg channels
    #rename channels for consistency (no longer required for future recordings):
    #mne.rename_channels(Raw.info, {'F9' : 'FT9','P9' : 'TP9','P10' : 'TP10','F10' : 'FT10','AF1' : 'AF7' }, allow_duplicates=False, verbose=None)
    #set the montage, optionally rebuild the online reference (FCz), drop bad
    #electrodes and average-reference - see apply_montage_and_reference, whose
    #step order must not be reshuffled:
    montage = load_montage(current_path)
    Raw, elecs_to_drop = apply_montage_and_reference(Raw, montage, params_dict)
    selected_elecs = select_electrodes(Raw, params_dict, elecs_to_drop)
    print('\n###########################################################')
    print('filtering the data')
    unfiltered_Raw=Raw.copy()
    if (filter_method == 'iir'):
        notched_Raw = unfiltered_Raw.filter(1, 100, method=filter_method, phase='forward', pad=0)
        notched_Raw.notch_filter(50, method=filter_method, phase='forward') 
        notched_Raw = apply_asr(notched_Raw, params_dict)
        if PerformCsd:
            notched_Raw = mne.preprocessing.compute_current_source_density(notched_Raw) # Perform current source density
        Raw_Filtered = notched_Raw.filter(LowPass, HighPass, method=filter_method, iir_params = dict(order=4, ftype='butter'),phase='forward',pad=0)
    if (filter_method == 'fir'):
        notched_Raw = unfiltered_Raw.filter(1, 100, method=filter_method)  
        notched_Raw.notch_filter(50, method=filter_method) 
        notched_Raw = apply_asr(notched_Raw, params_dict)
        if PerformCsd:
            notched_Raw = mne.preprocessing.compute_current_source_density(notched_Raw) # Perform current source density
        Raw_Filtered = notched_Raw.filter(LowPass, HighPass, method=filter_method)
    if params_dict['pipeline_name']=='fbcsp+lda':
        #extract filterbank feequencies:
        filters_bands=tuple(params_dict['filters_bands'])
        filtered_data_band_passed=[]
        for i,(LowPass,HighPass) in enumerate(filters_bands):
            unfiltered_Raw=Raw.copy()
            Raw_Filtered_band= mne.filter.filter_data(unfiltered_Raw.get_data(),sfreq=500, l_freq=LowPass, h_freq=HighPass, method='fir',copy = True)
            filtered_data_band_passed.append(Raw_Filtered_band)
    
    events_from_annot,event_dict = mne.events_from_annotations(Raw_Filtered)
    events_trigger_dict = {key: event_dict[key] for key in event_dict.keys() if key in params_dict['desired_events']}

    # if events_from_annot is None or len(events_from_annot) == 0:
    #     print("No events found in this recording. Returning filtered_raw only.")
    #     filter_bank_epochs = []
    #     mean_across_epochs = None
    #     epochs = None
    #     return Raw_Filtered, epochs, filter_bank_epochs, mean_across_epochs, events_trigger_dict

    print('\n###########################################################')
    print('extracting event info:',event_dict)

    #selected_elecs was already derived from the post-drop channel set above, by
    #select_electrodes - do not recompute it here, or the offline pick list can
    #drift from the live one.

    # Handle the case where there are NO events (e.g., pure idle file)
    if events_trigger_dict is None or len(events_trigger_dict) == 0:
        print("No events found in this recording. Returning filtered_raw only.")
        filter_bank_epochs = []
        mean_across_epochs = None
        events_idle = mne.make_fixed_length_events(
            Raw_Filtered,
            id=0,
            start=8.0,     # first event at 4 s -> epoch spans 0..10 s
            duration=10.0  # events spaced every 10 s
        )

        # 2) Epoch around these events with the same window as your MI epochs
        idle_event_id = {'Idle': 0}

        idle_epochs = mne.Epochs(
            Raw_Filtered,
            events_idle,
            event_id=idle_event_id,
            tmin=-8.0,
            tmax=6.0,
            baseline=None,   # same baseline as MI
            detrend=0,
            preload=True
        )
        idle_epochs.pick(selected_elecs)
        return Raw_Filtered, idle_epochs, filter_bank_epochs, mean_across_epochs, events_trigger_dict

    if params_dict['pipeline_name']=='fbcsp+lda':
        #filter bank related: 
        filter_bank_epochs=[]
        for filtered_data_band in filtered_data_band_passed:
            filtered_data_band_raw = mne.io.RawArray(filtered_data_band,unfiltered_Raw.info)
            epochs = mne.Epochs(filtered_data_band_raw, events_from_annot, preload = True,baseline= None, tmin=tmin, tmax=tmax, event_id=events_trigger_dict,detrend=0)
            # Calculate the mean across epochs for the current event
            mean_across_epochs = epochs.get_data().mean(axis=0)
            event_data = epochs.get_data()         
            # Subtract the mean from each epoch of the current event
            centered_event_data = event_data - mean_across_epochs
            event_epochs= epochs.events
            epochs = mne.EpochsArray(centered_event_data, epochs.info, events=event_epochs, event_id=epochs.event_id, tmin=epochs.tmin)
            filter_bank_epochs.append(epochs)
        
    
    epochs = mne.Epochs(Raw_Filtered, events_from_annot, preload = True,baseline= None, tmin=tmin, tmax=tmax, event_id=events_trigger_dict,detrend=0)
    
    # If we want to perform auto rejection of epochs (time expensive)
    #ar = AutoReject()
    #epochs = ar.fit_transform(epochs)  

    
    if pick_channels:
        epochs.pick(selected_elecs)
    ## Centering the data
    #the grand mean is returned to the caller either way:
    mean_across_epochs = epochs.get_data().mean(axis=0)

    #Per-class centering, controlled by params_dict['CenterByClass'] (default True,
    #i.e. the historical behaviour - so every existing caller is unaffected).
    #
    #WHAT IT DOES: subtracts from each epoch the mean of all epochs of that epoch's
    #own class, computed per recording file (~27 trials/class/file).
    #
    #WHY YOU MAY WANT IT OFF: this is a LABEL-DEPENDENT transform - you must already
    #know a trial's class to centre it - so it cannot be applied to an unlabelled
    #trial. The live loop does not apply it (the mean subtraction in
    #Live_Stream.ipynb is commented out and the saved mean-* artifact is unused), so
    #a model trained with this on is served data online that was never centred. It
    #also removes the per-class evoked response, making everything downstream
    #induced power. src/windowed_batch.py, src/full_epoch_batch.py and
    #src/session_batch.py measure both settings: the centered/uncentered epoch
    #variants are EPOCH_VARIANTS in src/analysis_common.py, and each of those three
    #modules pairs them into its own <train>2<test> MODES (c2c, c2u, ...).
    if params_dict.get('CenterByClass', True):
        centered_data_list = []
        events_list = []
        # Loop through each event ID
        for idx,event_id in enumerate(params_dict['desired_events']):
            print (event_id)
            # Extract epochs for the current event
            event_epochs = epochs[event_id]
            event_data = event_epochs.get_data()

            # Calculate the mean across epochs for the current event
            mean_across_event_epochs = event_data.mean(axis=0)

            # Subtract the mean from each epoch of the current event
            centered_event_data = event_data - mean_across_event_epochs

            # Store the centered data
            centered_data_list.append(centered_event_data)

            # Prepare the events list and event_id_map for the combined EpochsArray
            events_list.append(event_epochs.events)

        # Concatenate all centered data and events
        centered_data = np.concatenate(centered_data_list, axis=0)
        combined_events = np.concatenate(events_list, axis=0)

        # Sort the combined events based on their original occurrence time to preserve the temporal sequence
        sort_indices = np.argsort(combined_events[:, 0])
        combined_events = combined_events[sort_indices]
        centered_data = centered_data[sort_indices]

        # Create a new EpochsArray with the centered data
        centered_epochs = mne.EpochsArray(centered_data, epochs.info, events=combined_events, event_id=epochs.event_id, tmin=epochs.tmin)
        epochs = centered_epochs
    else:
        print('CenterByClass is off - leaving the per-class evoked response in the data.')

    #this section drops electrodes after epoching: but currently we drop all bad electrodes from the raw data
    print('\n###########################################################')
    print('removing bad channels from epochs:')
    curr_elecs_in_epochs_set=set(epochs.info['ch_names'])
    elecs_to_remove=params_dict['bad_electrodes']
    elecs_to_drop=curr_elecs_in_epochs_set.intersection(elecs_to_remove)

    if len(elecs_to_drop)>0:
        epochs.info['bads']=elecs_to_drop
        epochs.drop_channels(epochs.info['bads'])
        print('\n###########################################################')
        print(f'Removed: {elecs_to_drop} from the current selected electrodes: {curr_elecs_in_epochs_set} from the overall set of bad electrodes {elecs_to_remove}')
        print('#############################################################')
    
        #filter bank related: 
        filter_bank_epochs_after_elec_drops=[]
        for curr_epochs in filter_bank_epochs:
            curr_epochs.info['bads']=elecs_to_drop
            curr_epochs.drop_channels(epochs.info['bads'])
            filter_bank_epochs_after_elec_drops.append(curr_epochs)    
    elif params_dict['pipeline_name']=='fbcsp+lda': 
        #filter bank related: 
        filter_bank_epochs_after_elec_drops=[]
        for curr_epochs in filter_bank_epochs:
            filter_bank_epochs_after_elec_drops.append(curr_epochs)  

        print('\n###########################################################')
        print(f'the current selected electrodes: {curr_elecs_in_epochs_set} allready exclude the requested electrodes to remove {elecs_to_remove}')
        print('#############################################################')

    return Raw_Filtered,epochs,filter_bank_epochs,mean_across_epochs, events_trigger_dict

def Split_training_validation (epochs,filter_bank_epochs, events_trigger_dict):

    data_df=pd.DataFrame(data=epochs.events[:, -1], columns=['label'] ,index=range(len(epochs.events[:, -1])))
    data_df['original_trial_ind']=range(len(epochs.events[:, -1]))
    train,validation=train_test_split(data_df,shuffle=True,random_state=random.randrange(1,80),stratify=data_df['label'],test_size=0.2)
 
    train_inds=train['original_trial_ind'].values
    validation_inds=validation['original_trial_ind'].values
    print(f'putting aside 20% of the data: trial numbers are:\n {validation_inds}\n')
    print(f'remaining 80% of the trials go into training for cv:\n {train_inds}\n')

    return_dict={'train_inds':train_inds,
                'validation_inds':validation,
                'epochs':epochs,
                'filter_bank_epochs':filter_bank_epochs,
                'events_triggers_dict':events_trigger_dict}
    return train_inds,validation_inds,return_dict

def crop_the_data(epochs,
                  train_inds,
                  validation_inds,
                  tmin,
                  tmax,
                  full_epoch_tmin=0,
                  full_epoch_tmax=5,
                  use_all_for_training=False):
    """
    Returns a dictionary containing cropped and uncropped versions of
    the training and validation epochs.

    If use_all_for_training=True:
        - all epochs are used as training data
        - validation sets are empty (but still present in the dict)
    """
    tmin = float(tmin)
    tmax = float(tmax)

    n_epochs = len(epochs)

    # ---- NEW: option to use *all* data for training (for final model) ----
    if use_all_for_training:
        train_inds = np.arange(n_epochs)
        # keep an empty validation set so downstream code doesn't crash
        validation_inds = np.array([], dtype=int)

    # Get full (uncropped) data and labels once
    data_uncropped = epochs.get_data()            # shape: (n_epochs, n_ch, n_times_full)
    labels_uncropped = epochs.events[:, -1]       # event IDs

    # Crop the epochs to the desired window
    epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    data_cropped = epochs_cropped.get_data()
    labels_cropped = epochs_cropped.events[:, -1]

    # ---- TRAIN SET ----
    train_set_data_uncropped = data_uncropped[train_inds]
    train_set_labels_uncropped = labels_uncropped[train_inds]

    train_set_data = data_cropped[train_inds]
    train_set_labels = labels_cropped[train_inds]

    # ---- VALIDATION SET (may be empty if use_all_for_training=True) ----
    validation_set_data_uncropped = data_uncropped[validation_inds]
    validation_Set_labels_uncropped = labels_uncropped[validation_inds]

    validation_set_data = data_cropped[validation_inds]
    validation_set_labels = labels_cropped[validation_inds]

    return_dict = {
        'train_set_data_uncropped': train_set_data_uncropped,
        'train_set_labels_uncroped': train_set_labels_uncropped,
        'validation_set_data_uncropped': validation_set_data_uncropped,
        'validation_Set_labels_uncropped': validation_Set_labels_uncropped,
        'epochs_cropped': epochs_cropped,
        'train_set_data': train_set_data,
        'train_set_labels': train_set_labels,
        'validation_set_data': validation_set_data,
        'validation_set_labels': validation_set_labels
    }
    return return_dict

def augment_data(augmentation_params, data_x_to_augment, y, sfreq):
    """
    Augments EEG data by sliding window segmentation.

    Parameters:
    - augmentation_params: dict with 'win_step' (s) and 'win_len' (s)
    - data_x_to_augment: EEG data (epochs, channels, samples) or (epochs, channels, samples, filters)
    - y: Labels for each epoch
    - sfreq: Sampling frequency in Hz

    Returns:
    - augmented_x: Augmented EEG data
    - augmented_y: Corresponding labels
    """
    win_step = augmentation_params.get('win_step', 0)
    win_len = augmentation_params.get('win_len', 0)
    
    # No augmentation requested
    if win_step == 0 or win_len == 0:
        return data_x_to_augment, y

    # Calculate window start and end indices
    num_samples = data_x_to_augment.shape[2]
    window_starts = np.arange(0, num_samples - win_len * sfreq + 1, win_step * sfreq).astype(int)
    window_ends = window_starts + int(win_len * sfreq)
    
    # Augmentation
    augmented_x = []
    augmented_y = []

    for start, end in zip(window_starts, window_ends):
        if data_x_to_augment.ndim == 3:
            window_data = data_x_to_augment[:, :, start:end]
        elif data_x_to_augment.ndim == 4:  # For filter banks
            window_data = data_x_to_augment[:, :, start:end, :]

        augmented_x.append(window_data)
        augmented_y.append(y)  # Replicate labels for each window

    augmented_x = np.concatenate(augmented_x, axis=0)
    augmented_y = np.concatenate(augmented_y)

    return augmented_x, augmented_y


# %%
from mne.preprocessing import ICA

def run_ica_on_raw(
    raw,
    l_freq_ica=1.0,
    h_freq_ica=None,
    n_components=25,
    method='fastica',
    random_state=97,
    eog_chs=None,
    decim=3
):
    """
    Fit ICA on (optionally) filtered copy of raw and open plots for inspection.
    Returns the fitted ICA object (with .exclude ready for you to edit).

    Parameters
    ----------
    raw : mne.io.Raw
        Original raw object (will NOT be modified).
    l_freq_ica, h_freq_ica : float | None
        Band-pass for ICA fitting (often 1–None Hz).
    n_components : int | None
        Number of ICA components.
    method : str
        ICA method for mne.preprocessing.ICA.
    eog_chs : list of str | None
        EOG channel names for automatic EOG component suggestion.
    decim : int
        Decimation factor during ICA fitting.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA instance (you decide .exclude afterwards).
    """
    print("⏳ Copying & filtering raw for ICA (this does not modify the original raw)...")
    raw_for_ica = raw.copy().filter(l_freq=l_freq_ica, h_freq=h_freq_ica)

    print("⏳ Fitting ICA...")
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state
    )
    ica.fit(raw_for_ica, decim=decim)
    print("✅ ICA fitted.")

    # Optional: try to automatically mark EOG-related components
    if eog_chs is not None and len(eog_chs) > 0:
        print(f"🔍 Searching for EOG-related components using channels: {eog_chs}")
        eog_inds, eog_scores = ica.find_bads_eog(raw_for_ica, ch_name=eog_chs)
        print(f"Suggested EOG components: {eog_inds}")
        ica.exclude = eog_inds  # you can change this later
        ica.plot_scores(eog_scores)

    # Visual inspection: you will interact with these in the notebook
    print("📈 Plotting ICA components (topographies)...")
    ica.plot_components()  # click components to inspect

    print("📈 Plotting IC time courses on a short segment...")
    ica.plot_sources(raw_for_ica, start=0, stop=60)  # adjust window as needed

    print(
        "\nNow inspect the plots and manually update:"
        "\n    ica.exclude = [comp_idx1, comp_idx2, ...]"
        "\nwhen you are satisfied. Then pass this `ica` to the next stage."
    )

    return ica

# %%
def run_ica_on_epochs(
    epochs,
    n_components=25,
    method='fastica',
    random_state=97,
    eog_chs=None,
    decim=3,
    iclabel_threshold=0.5,
    exclude_labels=('eye blink', 'eye movement', 'muscle artifact',
                    'heart beat', 'line noise', 'channel noise')
):
    """
    Fit ICA on a temporary Raw constructed from concatenated epoch data,
    auto-label every component with ICLabel, and pre-populate
    ``ica.exclude`` with non-brain artifacts.

    Assumes the epochs are already preprocessed (filtered, bad channels
    removed, etc.) — no additional filtering is applied.

    The workflow is interactive:
      1. A pseudo-Raw is built from the epoch data so ICA can be fitted.
      2. ICLabel classifies each component and those whose predicted
         artifact probability exceeds *iclabel_threshold* are added to
         ``ica.exclude``.
      3. Component topographies, time-courses, and the label bar-chart
         are plotted for manual review.
      4. The user can override ``ica.exclude`` in the notebook before
         calling ``apply_ica_to_epochs``.

    Parameters
    ----------
    epochs : mne.Epochs | mne.EpochsArray
        The (combined, already preprocessed) epochs to clean.  Must be preloaded.
    n_components : int | None
        Number of ICA components to estimate.
    method : str
        ICA algorithm (default ``'fastica'``).
    random_state : int
        Random seed for reproducibility.
    eog_chs : list of str | None
        Channel names treated as EOG for automatic component suggestion
        (used as fallback alongside ICLabel).
    decim : int
        Decimation factor during ICA fitting (trades speed for precision).
    iclabel_threshold : float
        Probability threshold (0–1) above which a component is considered
        an artifact and added to ``ica.exclude``.  Default 0.5.
    exclude_labels : tuple of str
        ICLabel class names to treat as artifacts.  Components whose
        highest-probability label is in this set *and* exceeds
        *iclabel_threshold* will be auto-excluded.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA instance with ``.exclude`` pre-populated.
        Inspect the plots, adjust if needed, then call
        ``apply_ica_to_epochs(ica, epochs)``.
    labels_df : pandas.DataFrame
        Per-component predicted probabilities for every ICLabel class.
    """
    from mne_icalabel import label_components

    # Build a continuous Raw from the epoch data so ICA.fit() works
    data = epochs.get_data()                        # (n_epochs, n_ch, n_times)
    data_concat = data.transpose(1, 0, 2).reshape(len(epochs.ch_names), -1)
    raw_from_epochs = mne.io.RawArray(data_concat, epochs.info.copy())

    print("Fitting ICA on epoch data (already preprocessed, no extra filtering)...")
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state
    )
    ica.fit(raw_from_epochs, decim=decim)
    print(f"ICA fitted  ({ica.n_components_} components).")

    # ---- ICLabel automatic labelling ----
    print("Running ICLabel auto-classification...")
    label_dict = label_components(raw_from_epochs, ica, method='iclabel')

    ic_labels = label_dict['labels']          # list of str per component
    ic_probs  = label_dict['y_pred_proba']    # (n_components, 7) array

    # Debug: Check what we actually got
    print(f"DEBUG: ic_probs type = {type(ic_probs)}")
    print(f"DEBUG: ic_probs shape = {ic_probs.shape}")
    print(f"DEBUG: ic_probs ndim = {ic_probs.ndim}")
    print(f"DEBUG: label_dict keys = {label_dict.keys()}")

    class_names = ['brain', 'muscle artifact', 'eye blink',
                   'heart beat', 'line noise', 'channel noise', 'other']

    # Handle different output formats
    # First check if it's 1D and reshape if needed
    if ic_probs.ndim == 1:
        print("DEBUG: Reshaping 1D array to 2D")
        ic_probs = ic_probs.reshape(-1, 1)

    if ic_probs.shape[1] == 7:
        labels_df = pd.DataFrame(ic_probs, columns=class_names)
    elif ic_probs.shape[1] == 1:
        # Only one probability column - likely just the max probability
        # Try to get the full probability matrix from another key
        if 'y_pred_proba_full' in label_dict:
            ic_probs = label_dict['y_pred_proba_full']
            labels_df = pd.DataFrame(ic_probs, columns=class_names)
        else:
            # Fallback: create a simple dataframe with just the labels
            print("WARNING: Full probability matrix not available, using labels only")
            labels_df = pd.DataFrame({'predicted_label': ic_labels})
            labels_df.index.name = 'IC'
            # Skip inserting predicted_label again below
            ic_probs = None
    else:
        raise ValueError(f"Unexpected ic_probs shape: {ic_probs.shape}")

    if ic_probs is not None and ic_probs.shape[1] == 7:
        labels_df.insert(0, 'predicted_label', ic_labels)
        labels_df.index.name = 'IC'

        # Auto-exclude components labelled as artifacts above threshold
        auto_exclude = []
        for idx, (label, prob_row) in enumerate(zip(ic_labels, ic_probs)):
            if label in exclude_labels and prob_row.max() >= iclabel_threshold:
                auto_exclude.append(idx)
    else:
        # Fallback: just exclude based on label names without probability threshold
        auto_exclude = []
        for idx, label in enumerate(ic_labels):
            if label in exclude_labels:
                auto_exclude.append(idx)

    ica.exclude = auto_exclude

    print("\n--- ICLabel results ---")
    print(labels_df.to_string())
    print(f"\nAuto-excluded (threshold={iclabel_threshold}): {auto_exclude}")

    # ---- Fallback / additional EOG detection ----
    if eog_chs is not None and len(eog_chs) > 0:
        present = [ch for ch in eog_chs if ch in epochs.ch_names]
        if present:
            print(f"Also checking EOG channels: {present}")
            eog_inds, eog_scores = ica.find_bads_eog(raw_from_epochs, ch_name=present)
            for ind in eog_inds:
                if ind not in ica.exclude:
                    ica.exclude.append(ind)
            print(f"EOG-based suggestions added: {eog_inds}")
            ica.plot_scores(eog_scores)

    # ---- Visual inspection ----
    print("Plotting ICA component topographies...")
    ica.plot_components()

    print("Plotting IC time-courses (first 60 s of pseudo-Raw)...")
    ica.plot_sources(raw_from_epochs, start=0, stop=60)

    # Bar chart of ICLabel probabilities (only if we have full probabilities)
    if ic_probs is not None and ic_probs.shape[1] == 7:
        fig, ax = plt.subplots(figsize=(12, 4))
        labels_df[class_names].plot.bar(stacked=True, ax=ax, colormap='Set2')
        ax.set_ylabel('Probability')
        ax.set_xlabel('IC component')
        ax.set_title('ICLabel classification')
        ax.legend(loc='upper right', fontsize=8)
        for idx in ica.exclude:
            ax.get_children()[idx].set_edgecolor('red')
            ax.get_children()[idx].set_linewidth(2)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNote: Probability bar chart skipped (full probabilities not available)")

    print(
        f"\nica.exclude = {ica.exclude}  (auto-set by ICLabel)\n"
        "Review the plots and adjust if needed, then call:\n"
        "    epochs = apply_ica_to_epochs(ica, epochs)"
    )
    return ica, labels_df


def apply_ica_to_epochs(ica, epochs):
    """
    Project out the excluded ICA components from *epochs* and return a
    new EpochsArray with cleaned data.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA with ``.exclude`` set.
    epochs : mne.Epochs | mne.EpochsArray
        Original (uncleaned) epochs.

    Returns
    -------
    cleaned_epochs : mne.EpochsArray
        New EpochsArray with the excluded components removed.
    """
    print(f"Applying ICA — removing components: {ica.exclude}")

    # Build temporary Raw, apply ICA, reshape back to epochs
    data = epochs.get_data()                        # (n_epochs, n_ch, n_times)
    n_epochs, n_ch, n_times = data.shape
    data_concat = data.transpose(1, 0, 2).reshape(n_ch, -1)
    raw_tmp = mne.io.RawArray(data_concat, epochs.info.copy())

    ica.apply(raw_tmp)

    cleaned_data = raw_tmp.get_data().reshape(n_ch, n_epochs, n_times).transpose(1, 0, 2)

    cleaned_epochs = mne.EpochsArray(
        cleaned_data,
        epochs.info.copy(),
        events=epochs.events.copy(),
        event_id=epochs.event_id,
        tmin=epochs.tmin
    )
    print(f"Done — returned {n_epochs} cleaned epochs.")
    return cleaned_epochs
# %%
