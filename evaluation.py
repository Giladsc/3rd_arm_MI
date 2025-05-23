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
from mne_import_xdf import *

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
from itertools import cycle
import matplotlib.pyplot as plt
import pickle

import copy


from braindecode.models import ShallowFBCSPNet
from braindecode.datautil import create_from_mne_epochs
from braindecode.training import CroppedLoss
from braindecode.training.scoring import trial_preds_from_window_preds
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch




from training import *
#%%
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
    scores_windows_array = np.squeeze(np.array(scores_windows))
    if scores_windows_array.shape[1] != len(w_times):
        raise ValueError("Mismatch between scores_windows columns and w_times length.")
    
    scores_windows_df = pd.DataFrame(columns=times_col_names, data=scores_windows_array)
    scores_windows_df['fold_id'] = range(len(scores_windows_df))
    longform_scores_windows_df = pd.melt(scores_windows_df, id_vars='fold_id', 
                                         value_vars=scores_windows_df.columns)
    longform_scores_windows_df.rename(columns={'variable': 'Time', 'value': 'Accuracy'}, inplace=True)

    # Plot using seaborn
    sns.lineplot(data=longform_scores_windows_df, x='Time', y='Accuracy', ax=axes_handle)

    # Add onset and chance lines
    if any(w_times > 0):
        onset_location = np.round(w_times[w_times >= 0][0], 2)
        axes_handle.axvline(onset_location, linestyle='--', color='k', label='Onset') 
    axes_handle.axhline(chance_level, linestyle='-', color='k', label='Chance') 

    # Customize the plot
    axes_handle.set_xlabel('Time (s)')
    axes_handle.set_ylabel('Classification Accuracy')
    axes_handle.set_title('Classification Score Over Time')
    axes_handle.set_ylim([0.2, 1])
    axes_handle.set_xlim([-2, 5])  # Adjust the x-axis limits to extend to 5 seconds
    axes_handle.legend()
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
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()
