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
from sklearn.preprocessing import label_binarize
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



from preprocessing import *
from evaluation import *
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
                    'Ron': {'Iz','Cz'}                   }
    if subject in bad_elecs_dict.keys():
        subject_bad_electrodes=bad_elecs_dict[subject]
    else: 
        subject_bad_electrodes={}
        print('note that no bad electrodes were defined for the current subject:',subject)
    return subject_bad_electrodes 

def EEG_Preprocessing (current_path,raw, params_dict):

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
    #set the correct (Brainvision Montage) montage:
    montage = mne.channels.read_custom_montage((f"{current_path}\Montages\CACS-64_REF.bvef"), head_size=0.095, coord_frame=None) 
    #rename channels for consistency (no longer required for future recordings): 
    #mne.rename_channels(Raw.info, {'F9' : 'FT9','P9' : 'TP9','P10' : 'TP10','F10' : 'FT10','AF1' : 'AF7' }, allow_duplicates=False, verbose=None)
    Raw.set_montage(montage, match_case=True, match_alias=False, on_missing='raise', verbose=None)

    print('\n###########################################################')
    print('removing subject specific bad electrodes from the raw data')
    #drop bad electrodes according to the current subject name: 
    print('\n###########################################################')
    print('removing bad channels from epochs:')
    curr_elecs_in_epochs_set=set(Raw.info['ch_names'])
    elecs_to_remove=params_dict['bad_electrodes']
    elecs_to_drop=curr_elecs_in_epochs_set.intersection(elecs_to_remove)

    if len(elecs_to_drop)>0: 
        Raw.drop_channels(list(elecs_to_drop))
    Raw.drop_channels(Raw.info['bads'])
    Raw.set_eeg_reference(ref_channels="average")
    mne.set_eeg_reference(Raw, copy=False)
    print('\n###########################################################')
    print('filtering the data')  
    unfiltered_Raw=Raw.copy()
    Raw_Filtered = unfiltered_Raw.filter(LowPass, HighPass, method=filter_method, pad='reflect_limited')

    if params_dict['pipeline_name']=='fbcsp+lda':
        #extract filterbank feequencies:
        filters_bands=tuple(params_dict['filters_bands'])
        filtered_data_band_passed=[]
        for i,(LowPass,HighPass) in enumerate(filters_bands):
            unfiltered_Raw=Raw.copy()
            Raw_Filtered_band= mne.filter.filter_data(unfiltered_Raw.get_data(),sfreq=500, l_freq=LowPass, h_freq=HighPass, method='fir',copy = True)
            filtered_data_band_passed.append(Raw_Filtered_band)

    events_from_annot,event_dict = mne.events_from_annotations(Raw_Filtered)
    print('\n###########################################################')
    print('extracting event info:',event_dict)
    
    events_trigger_dict = {key: event_dict[key] for key in event_dict.keys() if key in params_dict['desired_events']}
    print('\n###########################################################')
    filtered_electrodes  = [elec for elec in params_dict['Electorde_Group'] if elec not in elecs_to_drop]
    selected_elecs=filtered_electrodes

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

    if PerformCsd:
        epochs = mne.preprocessing.compute_current_source_density(epochs) # Perform current source density
    epochs.pick(selected_elecs)
    ## Centering the data

    centered_data_list = []
    events_list = []
    mean_across_epochs = epochs.get_data().mean(axis=0)
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

    return epochs,filter_bank_epochs,mean_across_epochs, events_trigger_dict

def Split_training_validation (epochs,filter_bank_epochs, events_trigger_dict):

    data_df=pd.DataFrame(data=epochs.events[:, -1], columns=['label'] ,index=range(len(epochs.events[:, -1])))
    data_df['original_trial_ind']=range(len(epochs.events[:, -1]))
    train,validation=train_test_split(data_df,shuffle=True,random_state=42,stratify=data_df['label'],test_size=0.2)
 
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

def crop_the_data(epochs,train_inds,validation_inds,tmin,tmax,full_epoch_tmin=0,full_epoch_tmax=5):
    #returns a dictionary containing the cropped and uncropped versions of the validation and training epochs.
    tmin=float(tmin)
    tmax=float(tmax)
    #save uncropped versions of the data: 
    #save the training data:
    train_set_data_uncroped=epochs.get_data()[train_inds]
    train_set_labels_uncroped=epochs.events[train_inds,-1]

    #save the validation data: 
    validation_set_data_uncroped=epochs.get_data()[validation_inds]
    validation_Set_labels_uncroped=epochs.events[validation_inds,-1]

    #crop the epochs (use the epochs structure)
    epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)

    #from here on - we extract the data as matrices (not epoch object anymore):

    #save the training data:
    train_set_data=epochs_cropped.get_data()[train_inds]
    train_set_labels=epochs_cropped.events[train_inds,-1]

    #save the validation data: 
    validation_set_data=epochs_cropped.get_data()[validation_inds]
    validation_set_labels=epochs_cropped.events[validation_inds,-1]

    return_dict={'train_set_data_uncropped':train_set_data_uncroped,
                'train_set_labels_uncroped':train_set_labels_uncroped,
                'validation_set_data_uncropped':validation_set_data_uncroped,
                'validation_Set_labels_uncropped':validation_Set_labels_uncroped,
                'epochs_cropped':epochs_cropped,
                'train_set_data':train_set_data,
                'train_set_labels':train_set_labels,
                'validation_set_data':validation_set_data,
                'validation_set_labels':validation_set_labels}
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
