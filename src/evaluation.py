#%%

# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import logging
import os,numpy as np,pandas as pd
from collections import Counter, OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt
import itertools
from tqdm import tqdm
import json

# MNE library for EEG data analysis
import mne
from mne import Epochs,find_events
from mne.decoding import Vectorizer
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.preprocessing import ICA

from autoreject import AutoReject

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
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.base import clone
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn, Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.utils.tangentspace import unupper

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


def get_epoch_events(epochs):
    """Get the list of event names for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object with labeled events.

    Returns
    -------
    epoch_event_names : np.ndarray of str
        Event name for every epoch, in order.
    """
    events = epochs.events
    event_ids = epochs.event_id
    epoch_event_names = []
    for event in events:
        event_code = event[-1]
        event_name = [key for key, val in event_ids.items() if val == event_code][0]
        epoch_event_names.append(event_name)
    return np.array(epoch_event_names)


def compare_events(actual_events, predicted_events):
    """Compare actual event names with predicted event names and summarise mismatches.

    Parameters
    ----------
    actual_events : array-like of str
        Ground-truth event labels.
    predicted_events : array-like of str
        Predicted event labels (same length).

    Returns
    -------
    dict with keys 'accuracy', 'mismatch_details' (list of str), 'mismatch_counts' (Counter).
    """
    comparison = actual_events == predicted_events
    accuracy = np.mean(comparison)
    mismatches = np.where(comparison == False)[0]

    mismatch_details = []
    mismatch_counts = Counter()
    for idx in mismatches:
        actual = actual_events[idx]
        predicted = predicted_events[idx]
        mismatch_details.append(f"{idx + 1} - {actual} (actual), {predicted} (predicted)")
        mismatch_counts[f"{actual} -> {predicted}"] += 1

    return {
        'accuracy': accuracy,
        'mismatch_details': mismatch_details,
        'mismatch_counts': mismatch_counts,
    }


def compare_events_without_rest(actual_events, predicted_events, rest_label='FixatedRest'):
    """Compare actual vs predicted events, excluding *rest_label* epochs from accuracy.

    Parameters
    ----------
    actual_events : array-like of str
        Ground-truth event labels.
    predicted_events : array-like of str
        Predicted event labels (same length).
    rest_label : str
        Label to exclude from the accuracy calculation.

    Returns
    -------
    dict with keys 'accuracy', 'total_non_rest_epochs', 'mismatch_details', 'mismatch_counts'.
    """
    non_rest_mask = actual_events != rest_label
    non_rest_indices = np.where(non_rest_mask)[0]

    actual_non_rest = actual_events[non_rest_mask]
    predicted_non_rest = predicted_events[non_rest_mask]

    comparison = actual_non_rest == predicted_non_rest
    accuracy = np.mean(comparison)
    mismatches = np.where(comparison == False)[0]

    mismatch_details = []
    mismatch_counts = Counter()
    for mismatch_idx in mismatches:
        original_idx = non_rest_indices[mismatch_idx]
        actual = actual_non_rest[mismatch_idx]
        predicted = predicted_non_rest[mismatch_idx]
        mismatch_details.append(f"{original_idx + 1} - {actual} (actual), {predicted} (predicted)")
        mismatch_counts[f"{actual} -> {predicted}"] += 1

    return {
        'accuracy': accuracy,
        'total_non_rest_epochs': len(actual_non_rest),
        'mismatch_details': mismatch_details,
        'mismatch_counts': mismatch_counts,
    }


# %%
def plot_accuracy_over_time(scores_windows, w_times, params_dict=None, axes_handle=None):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Validate inputs
    if params_dict is None:
        params_dict = {}
    if axes_handle is None:
        _, axes_handle = plt.subplots()

    # Extract the number of classes from params_dict or default to 3
    num_classes = len(params_dict['desired_events'])
    chance_level = 1 / num_classes  # Calculate chance level dynamically

    # Convert scores_windows to long-form DataFrame
    # Adjust the time range (extend to 5 seconds)
    times_col_names = [np.round(w_times[s], 2) for s in range(len(w_times))]
    scores_windows_array = np.atleast_2d(np.array(scores_windows))
    if scores_windows_array.shape[1] != len(w_times):
        raise ValueError("Mismatch between scores_windows columns and w_times length.")
    
    scores_windows_df = pd.DataFrame(columns=times_col_names, data=scores_windows_array)
    scores_windows_df['fold_id'] = range(len(scores_windows_df))
    longform_scores_windows_df = pd.melt(scores_windows_df, id_vars='fold_id', 
                                         value_vars=scores_windows_df.columns)
    longform_scores_windows_df.rename(columns={'variable': 'Time', 'value': 'Accuracy'}, inplace=True)

    # Plot using seaborn
    sns.lineplot(data=longform_scores_windows_df, x='Time', y='Accuracy', ax=axes_handle)

    # Add onset line if applicable
    if any(w_times > 0):
        onset_location = np.round(w_times[w_times >= 0][0], 2)
        axes_handle.axvline(onset_location, linestyle='--', color='k', label='Onset')

    # Add chance level line as dotted
    axes_handle.axhline(chance_level, linestyle='-.', color='k', label=f'Chance')

    # Add shaded area for Cue
    axes_handle.axvspan(-1.25, -0, color='blue', alpha=0.3, label='Cue (Jittered)')

    # Customize the plot
    axes_handle.set_xlabel('Time (s)')
    axes_handle.set_ylabel('Classification Accuracy')
    axes_handle.set_title('Classification Score Over Time')
    axes_handle.set_ylim([0.2, 1])
    axes_handle.set_xlim([np.min(w_times), np.max(w_times)])  # Derive xlim from w_times
    axes_handle.legend()
    axes_handle.grid(True)

def plot_accuracy_over_time_multiple_subjects(subjects_scores_windows, w_times, params_dict=None, axes_handle=None):    
    from scipy.stats import sem

    # Initialize defaults
    if params_dict is None:
        params_dict = {}
    if axes_handle is None:
        _, axes_handle = plt.subplots()

    # Wrap single subject data
    if not isinstance(subjects_scores_windows, list):
        subjects_scores_windows = [subjects_scores_windows]

    # Number of classes and chance level
    num_classes = len(params_dict.get('desired_events'))
    chance_level = 1 / num_classes

    # Convert all subjects' data to 2D: (total_folds_across_subjects, n_times)
    all_folds = []
    for subj_data in subjects_scores_windows:
        subj_data = np.squeeze(np.array(subj_data))  # shape (n_folds, n_times)
        if subj_data.shape[1] != len(w_times):
            raise ValueError("Mismatch between scores_windows and w_times.")
        all_folds.append(subj_data)
    all_folds_array = np.concatenate(all_folds, axis=0)  # shape (total_folds, n_times)

    # Calculate mean and SEM across all folds (and subjects)
    mean_acc = np.mean(all_folds_array, axis=0)
    sem_acc = sem(all_folds_array, axis=0)

    # Plot with shaded SEM
    times = np.round(w_times, 2)
    axes_handle.plot(times, mean_acc, label='Mean Accuracy', color='#708090')
    axes_handle.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc,
                         alpha=0.3, color='#008080', label='±1 SEM')

    # Vertical line for MI onset (if relevant)
    if any(np.array(w_times) > 0):
        onset_time = np.round(np.array(w_times)[np.array(w_times) >= 0][0], 2)
        axes_handle.axvline(onset_time, linestyle='--', color='k', label='Onset')

    # Chance level line
    axes_handle.axhline(chance_level, linestyle='-.', color='gray', label=f'Chance ({chance_level:.2f})')

    # Cue shading
    axes_handle.axvspan(-1.25, 0, color='blue', alpha=0.2, label='Cue (Jittered)')

    # Styling
    axes_handle.set_xlabel('Time (s)')
    axes_handle.set_ylabel('Classification Accuracy')
    axes_handle.set_title('Mean Accuracy Over Time (±SEM)')
    axes_handle.set_ylim([0.2, 1])
    axes_handle.set_xlim([-2, 5])
    axes_handle.legend(loc='lower right')  # moved legend out of the plot area
    axes_handle.grid(True)
def plot_precision_recall_curves_from_trained_classifier(train_inds,validation_inds,params_dict,precision_recall_curve_timerange,trained_clf,epochs,filter_bank_epochs,predict_validation=True):
    #to learn on precision recall curves see :https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html 
    #the code is adapated for our usage: 
    #prepreocessing_dict - dictionary that contains the {original epoched data, and the training/validation indexes}
    #params_dict - dictionary that contains the prediction paramaters
    #precision_Revall_curve_timerange - list #in seconds relative to epoch (so if epoch is -3 to +4, 3-4 will take the last second in the epoch )
    #trained_clf - the classifier that was previously trained on all the data: (note that this means that the report here is biased (better than really is))
    #predict_validation - true - will use only validation indexes, false - will use only training indexes (much more biased ofcourse) 
    
    #define what time_range you want to extract the recall/precision for: 
    print(f'chosen window prediction range is {precision_recall_curve_timerange}\nnote that the prediction paramaters (that the classifier is trained on) are: {params_dict["windowed_prediction_params"]}\nconsider if you want the preciction range to match the prediction_param')
    
    #decide if we use the training or the validation set to plot: 
    if predict_validation: 
    #get the relevant data for the validation set: 
        inds=validation_inds
    else:
        inds=train_inds


    #extract the labels: 
    labels=epochs.events[inds, -1]
    #extract the decision function: 

    #fbcsp
    if params_dict['pipeline_name']=='fbcsp+lda':
        data_set_fb = []
        for filtered_data_band_epoch in filter_bank_epochs:
            temp_data = filtered_data_band_epoch.copy().crop(tmin=precision_recall_curve_timerange[0],tmax=precision_recall_curve_timerange[1]).get_data()[:]
            data_set_fb.append(temp_data)
        data_set_fb_4d_array= np.transpose(np.array(data_set_fb),(1,2,3,0))
        decision_function=trained_clf.decision_function((data_set_fb_4d_array)[inds,:])
    else:
            decision_function=trained_clf.decision_function(epochs.copy().crop(tmin=precision_recall_curve_timerange[0],tmax=precision_recall_curve_timerange[1]).get_data()[inds,:])
    y_score=decision_function
    # Use label_binarize to be multi-label like settings (basicly the current label position is 1 and rest are 0): 
    #so the label list of say, 0 2 4 4 will output = [1,0,0],[0,1,0],[0,0,1],[0,0,1]
    classes_numeric_list=list(params_dict['events_trigger_dict'].values())
    classes_names_list=list(params_dict['events_trigger_dict'].keys())
    #take the classes from the preprocessing dict:
    # Combine classes 3 and 5 into a single class, e.g., class 1
    binarized_labels = np.where(np.isin(labels, [3, 5]), 1, 0)
    Y = label_binarize(labels, classes=classes_numeric_list)
    n_classes = Y.shape[1]

    #calculate precision and recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    thresholds=dict()
    if n_classes==1: #in a binary setting where the score is only relates to being in group "1" (or maybe 0, worth checking)
        precision[0], recall[0], thresholds[0]  = precision_recall_curve(Y, y_score)
        average_precision[0] = average_precision_score(Y, y_score)
    else: 
        for i in range(n_classes):
            precision[i], recall[i], thresholds[i] = precision_recall_curve(Y[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(Y[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y, y_score, average="micro")

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {classes_names_list[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title(f"multi-class Precision-Recall curve\npredicted time range: {precision_recall_curve_timerange}")

    plt.show()
    #create a dataframe with all information relevant to the plot. 
    precision.pop('micro')
    precision_df=pd.DataFrame(precision)
    precision_df.columns=['precision_'+str(colname) for colname in precision_df.columns] 
    recall.pop('micro')
    recall_df=pd.DataFrame(recall)
    recall_df.columns=['recall_'+str(colname) for colname in recall_df.columns]
    thresholds_df=pd.DataFrame(thresholds)
    thresholds_df.columns=['thresholds'+str(colname) for colname in thresholds_df.columns]

    return_df=pd.concat([precision_df,recall_df,thresholds_df],axis=1)


    return return_df

from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np


def plot_binary_precision_recall_curve(train_inds, validation_inds, params_dict, precision_recall_curve_timerange, trained_clf, epochs, filter_bank_epochs, predict_validation=True):
    # Print chosen time window
    print(f'Chosen window prediction range: {precision_recall_curve_timerange}\n'
          f'Prediction parameters: {params_dict["windowed_prediction_params"]}')

    inds = validation_inds if predict_validation else train_inds

    labels = epochs.events[inds, -1]

    # Binary mapping (3,5)->Motor Imagery(1), rest->Idle(0)
    binary_labels = np.where(np.isin(labels, [3, 5]), 1, 0)

    # Extract decision function based on the pipeline
    if params_dict['pipeline_name'] == 'fbcsp+lda':
        data_set_fb = []
        for fb_epoch in filter_bank_epochs:
            cropped_data = fb_epoch.copy().crop(tmin=precision_recall_curve_timerange[0],
                                                tmax=precision_recall_curve_timerange[1]).get_data()[inds, :, :]
            data_set_fb.append(cropped_data)
        data_array = np.transpose(np.array(data_set_fb), (1, 2, 3, 0))
        y_score = trained_clf.decision_function(data_array)
    else:
        cropped_epochs = epochs.copy().crop(
            tmin=precision_recall_curve_timerange[0],
            tmax=precision_recall_curve_timerange[1]
        ).get_data()[inds, :, :]  # <-- select inds here
        y_score = trained_clf.decision_function(cropped_epochs)


    # Ensure y_score is 1-dimensional (binary)
    if y_score.ndim > 1:
        y_score = y_score.ravel()

    # Compute Precision-Recall
    precision, recall, thresholds = precision_recall_curve(binary_labels, y_score)
    average_precision = average_precision_score(binary_labels, y_score)

    # Plot
    plt.figure(figsize=(7, 7))
    display = PrecisionRecallDisplay(recall=recall, precision=precision, average_precision=average_precision)
    display.plot()
    plt.title(f"Binary Precision-Recall Curve\nPredicted Time Range: {precision_recall_curve_timerange}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.show()

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'precision': precision[:-1],
        'recall': recall[:-1],
        'thresholds': thresholds
    })

    return results_df

def plot_confusion_matrix(conf_mat, class_labels, title="Confusion Matrix"):
    conf_mat = np.array(conf_mat, dtype=float)
    np.nan_to_num(conf_mat, copy=False)
    is_float = np.any(conf_mat != conf_mat.astype(int))
    fmt = '.2f' if is_float else 'd'
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_labels)
    disp.plot(cmap='Blues', ax=ax, values_format=fmt)
    ax.set_title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_average_confusion_fixed_cv(folds_conf_matrices_per_window, w_times, t_start, t_end, normalize=True):
    """
    Plot average confusion matrix across CV folds and time windows within a time range.

    Parameters
    ----------
    folds_conf_matrices_per_window : list of lists of (cm, classes) tuples
        Output from run_windowed_classification_aug_cv or sanity_check_trained_clf.
        Outer list = folds, inner list = time windows.
    w_times : np.ndarray
        Time (s) for each window.
    t_start : float
        Start of time range (seconds).
    t_end : float
        End of time range (seconds).
    normalize : bool
        If True, normalize rows to sum to 1 (proportions). Default True.
    """
    # Convert time range to window indices
    mask = (w_times >= t_start) & (w_times <= t_end)
    window_indices = list(np.where(mask)[0])

    if len(window_indices) == 0:
        raise ValueError(f"No windows found between {t_start}s and {t_end}s. "
                         f"w_times range: [{w_times[0]:.2f}, {w_times[-1]:.2f}]")

    all_matrices = []
    for fold in folds_conf_matrices_per_window:
        for w_idx in window_indices:
            cm, _ = fold[w_idx]
            all_matrices.append(cm)

    avg_matrix = np.mean(all_matrices, axis=0)
    labels = folds_conf_matrices_per_window[0][window_indices[0]][1]

    if normalize:
        avg_matrix = avg_matrix / avg_matrix.sum(axis=1, keepdims=True)
        avg_matrix = np.nan_to_num(avg_matrix)

    disp = ConfusionMatrixDisplay(confusion_matrix=avg_matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format=".2f")

    time_start = np.round(w_times[window_indices[0]], 2)
    time_end = np.round(w_times[window_indices[-1]], 2)
    ax.set_title(f"Avg Confusion Matrix (Time {time_start}\u2013{time_end}s)")
    plt.grid(False)
    plt.show()

    return avg_matrix, labels


def plot_accuracy_over_time_group(group_results, n_classes=2, figsize=(12, 5)):
    """
    Plot average accuracy over time windows across all subjects in group_results.

    Parameters
    ----------
    group_results : list
        List of dicts, each containing 'subject_name', 'scores_windows', 'w_times'.
    n_classes : int
        Number of classes (for chance level line). Default 2.
    figsize : tuple
        Figure size.
    """
    from scipy.stats import sem as scipy_sem

    series_by_subject = []
    w_times_list = []
    subject_names = []

    for entry in group_results:
        scores = entry.get('scores_windows')
        w_times = entry.get('w_times')
        if scores is None or w_times is None:
            continue

        if isinstance(scores, list) and len(scores) > 0:
            scores_arr = np.array(scores)
            subject_mean_across_folds = np.mean(scores_arr, axis=0)
        else:
            subject_mean_across_folds = np.array(scores)

        series_by_subject.append(subject_mean_across_folds)
        w_times_list.append(w_times)
        subject_names.append(entry['subject_name'])

    if len(series_by_subject) == 0:
        print("No subjects with scores_windows available.")
        return

    all_scores = np.vstack(series_by_subject)
    w_times = w_times_list[0]

    group_mean = np.mean(all_scores, axis=0)
    group_sem = scipy_sem(all_scores, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Onset (t=0)')
    chance_level = 1.0 / n_classes
    ax.axhline(y=chance_level, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Chance ({chance_level:.2f})')

    colors = plt.cm.viridis(np.linspace(0, 1, len(subject_names)))
    for i, (subj_scores, subj_name) in enumerate(zip(all_scores, subject_names)):
        ax.plot(w_times, subj_scores, color=colors[i], alpha=0.3, linewidth=1.5, label=subj_name)

    ax.plot(w_times, group_mean, 'o-', color='black', linewidth=3, markersize=6, label='Group Mean')
    ax.fill_between(w_times, group_mean - group_sem, group_mean + group_sem,
                    color='gray', alpha=0.3, label='\u00b1SEM')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title(f'Group Windowed Accuracy Over Time (n={len(subject_names)} subjects)', fontsize=12)

    if len(subject_names) > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    else:
        ax.legend(loc='best')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_average_confusion_fixed_cv_group(group_results, w_times, t_start, t_end, normalize=True, figsize=(8, 7)):
    """
    Average confusion matrices across all subjects and CV folds for a time range and plot.

    Parameters
    ----------
    group_results : list
        List of dicts, each containing 'folds_confusion_matrices_per_window'.
    w_times : np.ndarray
        Time (s) for each window.
    t_start : float
        Start of time range (seconds).
    t_end : float
        End of time range (seconds).
    normalize : bool
        If True, normalize rows to sum to 1.
    figsize : tuple
        Figure size.
    """
    mask = (w_times >= t_start) & (w_times <= t_end)
    window_idxs = list(np.where(mask)[0])

    if len(window_idxs) == 0:
        raise ValueError(f"No windows found between {t_start}s and {t_end}s. "
                         f"w_times range: [{w_times[0]:.2f}, {w_times[-1]:.2f}]")

    all_matrices = []
    all_labels = None

    for entry in group_results:
        folds_conf_matrices = entry.get('folds_confusion_matrices_per_window')
        if not folds_conf_matrices:
            continue

        for fold_matrices in folds_conf_matrices:
            for w_idx in window_idxs:
                if w_idx < len(fold_matrices):
                    cm, labels = fold_matrices[w_idx]
                    all_matrices.append(np.array(cm, dtype=float))
                    if all_labels is None:
                        all_labels = labels

    if len(all_matrices) == 0:
        print("No confusion matrices found in group_results.")
        return None

    avg_matrix = np.mean(np.stack(all_matrices, axis=0), axis=0)

    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = avg_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            avg_matrix = avg_matrix / row_sums
            avg_matrix = np.nan_to_num(avg_matrix, nan=0.0)

    if all_labels is None:
        all_labels = [str(i) for i in range(avg_matrix.shape[0])]

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=avg_matrix, display_labels=all_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='.2f' if normalize else '.0f')
    ax.grid(False)

    time_start = np.round(w_times[window_idxs[0]], 2)
    time_end = np.round(w_times[window_idxs[-1]], 2)
    ax.set_title(f'Group Average Confusion Matrix ({time_start}\u2013{time_end}s)\n'
                 f'({len(group_results)} subjects, averaged across folds)', fontsize=12)
    plt.tight_layout()
    plt.show()

    return avg_matrix, all_labels


def plot_permutation_test_confusion(fold_confusion_matrices, normalize=True, axes_handle=None):
    """
    Plot the average confusion matrix from the true run of run_permutation_test.

    Works with confusion matrices returned by either scoring method:
    - majority_vote: fold_confusion_matrices is list of (cm, classes)
    - windowed_mean: fold_confusion_matrices is list of lists of (cm, classes) per window;
      the matrix is averaged across folds and all windows.

    Parameters
    ----------
    fold_confusion_matrices : list
        The 4th return value of run_permutation_test.
    normalize : bool
        Row-normalize the averaged matrix. Default True.
    axes_handle : Axes or None

    Returns
    -------
    avg_matrix : np.ndarray
    all_labels : list of str
    """
    # Detect format: flat list of (cm, classes) vs nested list of lists
    if isinstance(fold_confusion_matrices[0], (list, tuple)) and isinstance(fold_confusion_matrices[0][0], np.ndarray):
        # majority_vote format: [(cm, classes), ...]
        cms = [cm for cm, _ in fold_confusion_matrices]
        all_labels = list(fold_confusion_matrices[0][1])
    else:
        # windowed_mean format: list of lists of (cm, classes)
        cms = []
        all_labels = None
        for fold in fold_confusion_matrices:
            for cm, classes in fold:
                cms.append(cm)
                if all_labels is None:
                    all_labels = list(classes)

    avg_matrix = np.mean(np.stack(cms, axis=0), axis=0)

    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = avg_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            avg_matrix = avg_matrix / row_sums
            avg_matrix = np.nan_to_num(avg_matrix, nan=0.0)

    if axes_handle is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        ax = axes_handle
        fig = ax.get_figure()

    disp = ConfusionMatrixDisplay(confusion_matrix=avg_matrix, display_labels=all_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='.2f' if normalize else '.0f')
    ax.grid(False)
    ax.set_title('Permutation Test — True Classifier\nAverage Confusion Matrix', fontsize=11)
    plt.tight_layout()
    plt.show()
    return avg_matrix, all_labels


def plot_permutation_test(
    true_score,
    perm_scores,
    p_value,
    params_dict=None,
    axes_handle=None,
    score_tmin=None,
    score_tmax=None,
):
    """
    Plot the permutation test null distribution as a histogram.

    Parameters
    ----------
    true_score : float          Real classifier accuracy (from run_permutation_test).
    perm_scores : np.ndarray    Null distribution accuracies.
    p_value : float             Pre-computed p-value.
    params_dict : dict or None  If provided, reads 'desired_events' for chance level.
    axes_handle : Axes or None  Existing axes; creates new figure if None.
    score_tmin : float or None  For axis label only.
    score_tmax : float or None  For axis label only.

    Returns
    -------
    fig, ax
    """
    if axes_handle is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        ax = axes_handle
        fig = ax.get_figure()

    n_perm = len(perm_scores)
    ax.hist(perm_scores, bins=max(10, n_perm // 10), color='steelblue',
            edgecolor='white', alpha=0.8, label='Null distribution')
    ax.axvline(true_score, color='crimson', linewidth=2.5, linestyle='--',
               label=f'True score: {true_score:.3f}')

    if params_dict is not None:
        n_classes = len(params_dict.get('desired_events', []))
        if n_classes > 0:
            chance = 1.0 / n_classes
            ax.axvline(chance, color='gray', linewidth=1.5, linestyle=':',
                       label=f'Chance ({chance:.2f})')

    ax.text(0.97, 0.95,
            f'p = {p_value:.3f}  (n={n_perm})',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    time_label = ''
    if score_tmin is not None or score_tmax is not None:
        lo = f'{score_tmin:.1f}s' if score_tmin is not None else 'start'
        hi = f'{score_tmax:.1f}s' if score_tmax is not None else 'end'
        time_label = f' ({lo}–{hi})'

    ax.set_xlabel(f'Mean accuracy{time_label}', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Permutation Test: Null Distribution vs True Classifier Accuracy', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax

# %%

# ---------------------------------------------------------------------------
# Full-epoch (majority-vote) group plotting
# ---------------------------------------------------------------------------

def plot_accuracy_group_full_epoch(group_results_full_epoch, n_classes=None,
                                   figsize=(10, 5)):
    """
    Bar chart of per-subject trial-level accuracy for full-epoch majority-vote
    classification, with fold standard deviation as error bars and a group
    mean ± SEM band.

    Parameters
    ----------
    group_results_full_epoch : list
        Built with add_subject_to_group_full_epoch.
    n_classes : int or None
        Number of classes for the chance-level line.
        If None, inferred from the first entry's desired_events (if present),
        otherwise omitted.
    figsize : tuple
    """
    from scipy.stats import sem as scipy_sem

    subject_names = []
    subject_means = []
    subject_stds  = []

    for entry in group_results_full_epoch:
        accs = entry.get('fold_accuracies')
        if not accs:
            continue
        accs = np.array(accs)
        subject_names.append(entry['subject_name'])
        subject_means.append(float(np.mean(accs)))
        subject_stds.append(float(np.std(accs)))

    if len(subject_names) == 0:
        print("No subjects with fold_accuracies available.")
        return

    subject_means = np.array(subject_means)
    subject_stds  = np.array(subject_stds)
    group_mean    = float(np.mean(subject_means))
    group_sem     = float(scipy_sem(subject_means))

    # Infer n_classes if not provided
    if n_classes is None:
        events = group_results_full_epoch[0].get('desired_events', [])
        n_classes = len(events) if events else None

    x = np.arange(len(subject_names))
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x, subject_means, yerr=subject_stds, capsize=5,
           color='steelblue', alpha=0.75, edgecolor='white',
           error_kw=dict(elinewidth=1.5, ecolor='black', capthick=1.5),
           label='Mean ± SD (across folds)')

    # Group mean band
    ax.axhline(group_mean, color='black', linewidth=2.5, linestyle='-',
               label=f'Group mean ({group_mean:.3f})')
    ax.fill_between([-0.5, len(subject_names) - 0.5],
                    group_mean - group_sem, group_mean + group_sem,
                    color='gray', alpha=0.25, label='±SEM')

    if n_classes is not None:
        chance = 1.0 / n_classes
        ax.axhline(chance, color='red', linewidth=1.5, linestyle='--',
                   label=f'Chance ({chance:.2f})')

    ax.set_xticks(x)
    ax.set_xticklabels(subject_names, fontsize=11)
    ax.set_ylabel('Trial accuracy (majority vote)', fontsize=11)
    ax.set_ylim([0, 1])

    tmin = group_results_full_epoch[0].get('tmin', 0.0)
    tmax = group_results_full_epoch[0].get('tmax', '?')
    ax.set_title(f'Full-Epoch Accuracy — {tmin}–{tmax}s '
                 f'(n={len(subject_names)} subjects)', fontsize=12)

    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_average_confusion_full_epoch_group(group_results_full_epoch,
                                            normalize=True, figsize=(8, 7)):
    """
    Average confusion matrix across all subjects and CV folds for full-epoch
    majority-vote results, then plot.

    Parameters
    ----------
    group_results_full_epoch : list
        Built with add_subject_to_group_full_epoch.
    normalize : bool
        If True, row-normalise so each row sums to 1.
    figsize : tuple
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    all_matrices = []
    all_labels   = None

    for entry in group_results_full_epoch:
        fold_cms = entry.get('fold_confusion_matrices')
        if not fold_cms:
            continue
        for cm, labels in fold_cms:
            all_matrices.append(np.array(cm, dtype=float))
            if all_labels is None:
                all_labels = labels

    if len(all_matrices) == 0:
        print("No confusion matrices found in group_results_full_epoch.")
        return None

    avg_matrix = np.mean(np.stack(all_matrices, axis=0), axis=0)

    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = avg_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            avg_matrix = avg_matrix / row_sums
            avg_matrix = np.nan_to_num(avg_matrix, nan=0.0)

    if all_labels is None:
        all_labels = [str(i) for i in range(avg_matrix.shape[0])]

    tmin = group_results_full_epoch[0].get('tmin', 0.0)
    tmax = group_results_full_epoch[0].get('tmax', '?')
    n_subj = len(group_results_full_epoch)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=avg_matrix,
                                  display_labels=all_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='.2f' if normalize else '.0f')
    ax.grid(False)
    ax.set_title(f'Group Avg Confusion — Full Epoch ({tmin}–{tmax}s)\n'
                 f'({n_subj} subjects, averaged across folds)', fontsize=12)
    plt.tight_layout()
    plt.show()

    return avg_matrix, all_labels


def plot_trial_count_sweep(sweep_results, n_classes=None, title=None, save_path=None):
    """
    Visualise the output of run_trial_count_sweep_cv() as a two-panel figure.

    Panel 1 — Learning curve: mean accuracy ± SEM vs. number of training trials.
    Panel 2 — Marginal gain: accuracy improvement per additional 10 trials, useful
               for identifying the "knee" where more data gives diminishing returns.

    Parameters
    ----------
    sweep_results : dict
        Output of run_trial_count_sweep_cv().
        Keys are n_trials (int); values contain 'mean_acc', 'sem_acc', 'all_accs'.
    n_classes : int or None
        Number of classes, used to draw the chance level line (1/n_classes).
        If None, no chance line is drawn.
    title : str or None
        Optional suptitle for the figure.
    save_path : str or None
        If provided, saves the figure to this path before showing.
    """
    from scipy.stats import sem as scipy_sem
    import numpy as np
    import matplotlib.pyplot as plt

    # Sort by trial count; skip any n_trials entries with no valid data
    n_trials_sorted = sorted(k for k in sweep_results if not np.isnan(sweep_results[k]['mean_acc']))
    mean_accs = np.array([sweep_results[n]['mean_acc'] for n in n_trials_sorted])
    sem_accs = np.array([sweep_results[n]['sem_acc'] for n in n_trials_sorted])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: Learning curve ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(n_trials_sorted, mean_accs, color='#708090', marker='o', linewidth=2,
            label='Mean Accuracy')
    ax.fill_between(n_trials_sorted,
                    mean_accs - sem_accs,
                    mean_accs + sem_accs,
                    alpha=0.3, color='#008080', label='±1 SEM')

    if n_classes is not None:
        chance = 1.0 / n_classes
        ax.axhline(chance, linestyle='-.', color='gray',
                   label=f'Chance ({chance:.2f})')

    ax.set_xlabel('Number of Training Trials', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Learning Curve: Accuracy vs. Training Trials', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_xticks(n_trials_sorted)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.4)

    # ── Panel 2: Marginal gain ───────────────────────────────────────────────
    ax2 = axes[1]
    if len(n_trials_sorted) >= 2:
        delta_acc = np.diff(mean_accs)
        delta_trials = np.diff(n_trials_sorted)
        # Marginal gain per 10 additional trials
        marginal_gain = (delta_acc / delta_trials) * 10
        midpoints = [(n_trials_sorted[i] + n_trials_sorted[i + 1]) / 2
                     for i in range(len(n_trials_sorted) - 1)]

        bar_widths = [dt * 0.6 for dt in delta_trials]
        bar_colors = ['#2ca02c' if g > 0 else '#d62728' for g in marginal_gain]
        ax2.bar(midpoints, marginal_gain, width=bar_widths, color=bar_colors, alpha=0.7)
        ax2.axhline(0, color='k', linewidth=0.8)
        ax2.set_xlabel('Training Trials (midpoint of interval)', fontsize=12)
        ax2.set_ylabel('Accuracy Gain per 10 Trials', fontsize=12)
        ax2.set_title('Marginal Accuracy Gain (Diminishing Returns)', fontsize=12)
        ax2.set_xticks([int(m) for m in midpoints])
        ax2.grid(True, alpha=0.4, axis='y')
    else:
        ax2.text(0.5, 0.5, 'Need ≥ 2 trial counts for marginal gain plot',
                 ha='center', va='center', transform=ax2.transAxes)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()
    return fig, axes


def _extract_fgda_ts_features(clf, X, y, params_dict):
    """
    Run raw epoch/window data through a fitted ts+FGDA pipeline's cov/fgda/ts steps,
    and map integer trigger-code labels to their class-name strings.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Fitted pipeline with named steps 'cov', 'fgda', 'ts' (as built by
        classifier_training() for pipeline_name='ts+FGDA').
    X : np.ndarray, shape (n_samples, n_channels, n_times)
        Epoch or window data, e.g. train_set_data or augmented_x. Must already be
        aligned with y (one label per sample along axis 0) — this function does not
        expand/repeat labels itself.
    y : np.ndarray, shape (n_samples,)
        Integer trigger codes matching X, e.g. train_set_labels or augmented_y.
    params_dict : dict
        Must contain 'events_trigger_dict' ({class_name: trigger_code}).

    Returns
    -------
    ts_vectors : np.ndarray, shape (n_samples, n_channels * (n_channels + 1) / 2)
    labels_mapped : np.ndarray of str, shape (n_samples,)
    """
    covs = clf.named_steps['cov'].transform(X)
    fgda_out = clf.named_steps['fgda'].transform(covs)
    ts_vectors = clf.named_steps['ts'].transform(fgda_out)

    triggers_label_dict = {val: key for key, val in params_dict['events_trigger_dict'].items()}
    labels_mapped = np.array([triggers_label_dict[label] for label in y])

    return ts_vectors, labels_mapped


def plot_fgda_lda_scatter(clf, X, y, params_dict, n_components=2, palette='Set1', axes_handle=None):
    """
    2D LDA projection of ts+FGDA tangent-space features, colored by class.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Fitted ts+FGDA pipeline (see _extract_fgda_ts_features).
    X : np.ndarray, shape (n_samples, n_channels, n_times)
    y : np.ndarray, shape (n_samples,)
        Integer trigger codes aligned with X.
    params_dict : dict
    n_components : int, default=2
        Number of LDA components to fit; only the first two are plotted.
    palette : str, default='Set1'
    axes_handle : Axes or None

    Returns
    -------
    fig, ax
    lda : fitted LinearDiscriminantAnalysis
    X_transformed : np.ndarray, shape (n_samples, n_components)
    """
    ts_vectors, labels_mapped = _extract_fgda_ts_features(clf, X, y, params_dict)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_transformed = lda.fit_transform(ts_vectors, labels_mapped)

    plot_data = pd.DataFrame({
        'Component_1': X_transformed[:, 0],
        'Component_2': X_transformed[:, 1],
        'Movement': labels_mapped,
    })

    if axes_handle is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        ax = axes_handle
        fig = ax.get_figure()

    sns.scatterplot(data=plot_data, x='Component_1', y='Component_2', hue='Movement',
                     palette=palette, legend='full', alpha=0.7, s=40, ax=ax)
    ax.set_title('Hand Movements Separation in FGDA & LDA projection', fontsize=14)
    ax.set_xlabel('Discriminant Component 1', fontsize=12)
    ax.set_ylabel('Discriminant Component 2', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return fig, ax, lda, X_transformed


def plot_fgda_lda_scatter_cv(clf, X, y, params_dict, cv=None, n_components=2,
                             palette='Set1', axes_handle=None, random_state=42):
    """
    Cross-validated (held-out) 2D LDA projection of ts+FGDA features.

    Unlike plot_fgda_lda_scatter, no point is used to fit the projection it is
    drawn in: for every CV fold the cov/fgda/ts feature extractor AND the
    visualization LDA are fit on the training split only, then the held-out test
    split is projected through them. This yields realistic class overlap that is
    directly comparable to a cross-validated confusion matrix, rather than the
    over-optimistic separation of an in-sample supervised projection.

    Because an LDA fit independently per fold is defined only up to an orthogonal
    transform (sign flips / axis swaps / rotation), each fold's held-out
    projection is aligned into a shared reference frame via orthogonal Procrustes
    on the training-class centroids. The reference frame comes from one in-sample
    LDA fit on all data — it fixes only the plot's orientation and never
    contributes a plotted point or a decision boundary.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Fitted ts+FGDA pipeline; its 'cov'/'fgda'/'ts' steps are cloned (with their
        hyperparameters) and re-fit per fold, so the passed clf is not modified.
    X : np.ndarray, shape (n_samples, n_channels, n_times)
    y : np.ndarray, shape (n_samples,)
        Integer trigger codes aligned with X.
    params_dict : dict
        Must contain 'events_trigger_dict'.
    cv : cross-validation splitter, int, or None
        Splitter instance (e.g. the notebook's StratifiedShuffleSplit, for parity
        with the confusion-matrix CV), an int (-> StratifiedKFold with that many
        splits), or None (-> StratifiedKFold(5, shuffle=True)).
    n_components : int, default=2
        Number of LDA components; capped at n_classes - 1. Only two are plotted.
    palette : str, default='Set1'
    axes_handle : Axes or None
    random_state : int, default=42
        Used only when cv is None or an int.

    Returns
    -------
    fig, ax
    proj_all : np.ndarray, shape (n_heldout_points, 2)
        Aligned held-out projections (a sample may appear more than once if the
        splitter's test sets overlap, e.g. StratifiedShuffleSplit).
    labels_all : np.ndarray of str
        True class name for each held-out point.
    """
    from scipy.linalg import orthogonal_procrustes

    triggers_label_dict = {val: key for key, val in params_dict['events_trigger_dict'].items()}
    y_names = np.array([triggers_label_dict[label] for label in y])
    classes = np.unique(y_names)
    n_classes = len(classes)

    n_comp = min(n_components, n_classes - 1)
    if n_comp < 2:
        raise ValueError(
            f"A 2D CV projection needs >= 3 classes (LDA yields at most "
            f"n_classes - 1 = {n_classes - 1} components); got {n_classes} classes."
        )

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def _feature_extractor():
        return Pipeline([
            ('cov', clone(clf.named_steps['cov'])),
            ('fgda', clone(clf.named_steps['fgda'])),
            ('ts', clone(clf.named_steps['ts'])),
        ])

    def _centroids(proj, labels):
        return np.vstack([proj[labels == c].mean(axis=0) for c in classes])

    # Reference orientation only (in-sample): defines stable plot axes so per-fold
    # projections can be rotated into a common frame. Not plotted, no boundary.
    ref_ts = _feature_extractor().fit_transform(X, y_names)
    ref_proj = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(ref_ts, y_names)
    ref_centroids = _centroids(ref_proj, y_names)

    proj_parts, label_parts = [], []
    for train_idx, test_idx in cv.split(np.zeros(len(y_names)), y_names):
        feat = _feature_extractor()
        ts_train = feat.fit_transform(X[train_idx], y_names[train_idx])
        ts_test = feat.transform(X[test_idx])

        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        lda.fit(ts_train, y_names[train_idx])
        train_proj = lda.transform(ts_train)
        test_proj = lda.transform(ts_test)

        # Align this fold's arbitrary orientation to the shared reference frame.
        R, _ = orthogonal_procrustes(_centroids(train_proj, y_names[train_idx]), ref_centroids)
        proj_parts.append(test_proj @ R)
        label_parts.append(y_names[test_idx])

    proj_all = np.vstack(proj_parts)
    labels_all = np.concatenate(label_parts)

    plot_data = pd.DataFrame({
        'Component_1': proj_all[:, 0],
        'Component_2': proj_all[:, 1],
        'Movement': labels_all,
    })

    if axes_handle is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        ax = axes_handle
        fig = ax.get_figure()

    sns.scatterplot(data=plot_data, x='Component_1', y='Component_2', hue='Movement',
                     palette=palette, legend='full', alpha=0.7, s=40, ax=ax)
    ax.set_title('Held-out FGDA & LDA projection (cross-validated)', fontsize=14)
    ax.set_xlabel('Discriminant Component 1', fontsize=12)
    ax.set_ylabel('Discriminant Component 2', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return fig, ax, proj_all, labels_all


def plot_fgda_lda_pairplot(clf, X, y, params_dict, max_components=4, palette='Set1'):
    """
    Full-rank LDA fit on ts+FGDA features: prints explained variance ratio and
    pairplots the first max_components discriminant components.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Fitted ts+FGDA pipeline (see _extract_fgda_ts_features).
    X : np.ndarray, shape (n_samples, n_channels, n_times)
    y : np.ndarray, shape (n_samples,)
        Integer trigger codes aligned with X.
    params_dict : dict
    max_components : int, default=4
        Number of leading components to include in the pairplot (capped at the
        LDA's full rank, min(n_classes - 1, n_features)).
    palette : str, default='Set1'

    Returns
    -------
    pairgrid : seaborn.PairGrid
    lda : fitted LinearDiscriminantAnalysis (n_components=None, full rank)
    explained_variance_ratio_ : np.ndarray
    """
    ts_vectors, labels_mapped = _extract_fgda_ts_features(clf, X, y, params_dict)

    lda = LinearDiscriminantAnalysis(n_components=None)
    X_all_components = lda.fit_transform(ts_vectors, labels_mapped)

    print("Explained variance ratio for all components:", lda.explained_variance_ratio_)

    n_components_to_plot = min(max_components, X_all_components.shape[1])
    columns = [f'C{i + 1}' for i in range(n_components_to_plot)]
    df_components = pd.DataFrame(X_all_components[:, :n_components_to_plot], columns=columns)
    df_components['Movement'] = labels_mapped

    pairgrid = sns.pairplot(df_components, hue='Movement', palette=palette, diag_kind=None)
    plt.show()

    return pairgrid, lda, lda.explained_variance_ratio_


def plot_lda_channel_pair_loadings(clf, lda, ch_names, component=0, cmap='RdBu_r', axes_handle=None):
    """
    Map an LDA discriminant axis back onto electrode-pair covariance terms.

    ts+FGDA features are the weighted upper-triangular vectorization of the
    (FGDA-filtered) covariance matrix (pyriemann.utils.tangentspace.upper). Since
    lda.scalings_[:, component] is a loading vector in that same feature space,
    unupper() reconstructs it into an (n_channels, n_channels) symmetric matrix,
    showing which electrode-pair covariances drive that discriminant component.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Fitted ts+FGDA pipeline (used only to confirm channel count via 'ts' step).
    lda : fitted LinearDiscriminantAnalysis
        As returned by plot_fgda_lda_scatter or plot_fgda_lda_pairplot.
    ch_names : list of str, length n_channels
        Channel names in the same order as the data fed into clf, e.g.
        epochs_cropped.info['ch_names'].
    component : int, default=0
        Index of the discriminant axis (column of lda.scalings_) to visualize.
    cmap : str, default='RdBu_r'
    axes_handle : Axes or None

    Returns
    -------
    fig, ax
    loading_matrix : np.ndarray, shape (n_channels, n_channels)
    """
    n_channels = len(ch_names)
    loading_vec = lda.scalings_[:, component]
    loading_matrix = unupper(loading_vec[np.newaxis, :])[0]

    if loading_matrix.shape != (n_channels, n_channels):
        raise ValueError(
            f"ch_names has {n_channels} entries but the loading vector reconstructs to "
            f"{loading_matrix.shape} — check that ch_names matches the channels clf was fit on."
        )

    if axes_handle is None:
        fig, ax = plt.subplots(figsize=(10, 9))
    else:
        ax = axes_handle
        fig = ax.get_figure()

    vmax = np.abs(loading_matrix).max()
    sns.heatmap(loading_matrix, xticklabels=ch_names, yticklabels=ch_names,
                cmap=cmap, center=0, vmin=-vmax, vmax=vmax, square=True, ax=ax,
                cbar_kws={'label': 'Discriminant loading'})
    ax.set_title(f'LDA Component {component + 1} — Channel-Pair Loadings', fontsize=13)
    plt.tight_layout()
    plt.show()

    return fig, ax, loading_matrix
