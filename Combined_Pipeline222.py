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
#%%

# Define paths for data storage and processing
current_path = pathlib.Path().absolute()  
recording_path = current_path / 'Recordings'
figure_outputs_path=current_path / 'Figures'
hyper_param_search_output=current_path / 'hyper_param_search_outputs'
#extract all recorded files and subject names
recording_files = [f for f in listdir(recording_path) if isfile(join(recording_path, f)) and ('.xdf' in f)]
if not(figure_outputs_path.exists()):
    print('the output folder does not exists:  ',figure_outputs_path)

if not(hyper_param_search_output.exists()):
    print('the output folder does not exists:  ',hyper_param_search_output)


print('all available recording files',recording_files)
subject_names=[r.split('_')[0] for r in recording_files]
print('only subjects IDS:',subject_names)

# Initial definitions: 

print('filenames:\n',recording_files)
print('names:\n',subject_names)

Use_test_grid=False #change to False when you want to use the real grid_search and not a toy one: 

#define the electrode groups: the key can be anything, the values should be a list of electrodes
Electorde_Groups = {'FP': ['Fp1', 'Fp2'],
                   'AF': ['AF7', 'AF3', 'AFz', 'AF4', 'AF8'],
                   'F' : ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
                   'FC': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6'],
                   'C' : ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4' ,'C6'],
                   'CP': ['CP5', 'CP3','CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
                   'P' : ['P7','P5','P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
                   'PO': ['PO7','PO3', 'POz', 'PO4', 'PO8'],
                   'O' : ['Oz', 'O2', 'O1', 'Iz']
                  } 

#define the grid search (dont go all at once because some params are not relevant to other params and might just increase running time: 
# i.e if using fbcsp, the n_components_grid paramater is not used, so if it has more than 1 value, it will run the fbcsp twice while changing a paramter that does not effect the calculation)
grid_search_dict=OrderedDict()
grid_search_dict={'filter_methods':['iir'], #['irr' or 'fir']
                'run_csd':[True, False],
                'pipeline_name':['csp+lda','ts+lda','fbcsp+lda'], #these classifiers pipelines are defined in "run_windowed_classification_on_fold"
                #things to do: filter bank csp + lda, csp+ts+lda
                'bandpass_borders_grid':[[7,32]], #each list defines the low and high cutoffs
                'Electorde_Groups_names_grid':['C','PO'], #each "name" refers to an elec group defined above
                'n_components_grid':[4], #the n component options for the csp classifier
                'n_components_fbcsp_grid':[2,3], # the n components options to use in the fbcsp classifier (n * filter_bank_bands)\
                'filters_bands':[[[8, 12], [12, 20], [20, 32]]],
                'epoch_tmins_and_maxes_grid':[[-3,5]], #times (sec: pre,post) for initial epoching (this should be the longest epoch as the windowed prediction will be tested on it)
                'classifier_training_windows_grid':[[1 , 2],[1,3]], #what times(sec: start,end) to use for the classifer training (data augmentation is also using this window)
                'augmentation_windows_grid':[[0,0],[0.5,0.5]], #referes to proportions (win_len,win_step) of sfreq, [1,1] means taking the classification epochs, and creating 1 second long epochs with 1 second long steps
                'windowed_prediction_params':[[0.5,0.1],[0.75,0.1],[1,0.1]]} #refers to prportions (win_len,win_step) of sfreq to try and predict i.e. 0.5 = half a second window, with a 100ms steps  

#here you can define a test grid (make it small so it wont take long, and use it to check that everything is working) 
test_grid_search_dict={'filter_methods':['fir'], #['irr' or 'fir']
                'run_csd':[True],
                'pipeline_name':['csp+lda'], #these classifiers pipelines are defined in "run_windowed_classification_on_fold
                #things to do: filter bank csp + lda, csp+ts+lda
                'bandpass_borders_grid':[[8,32]], #each list defines the low and high cutoffs
                'Electorde_Groups_names_grid':['C','PO'], #each "name" refers to an elec group defined above
                'n_components_grid':[4], #the n component options for the csp classifier
                'n_components_fbcsp_grid':[3], # the n components options to use in the fbcsp classifier (n * filter_bank_bands)
                'filters_bands':[[[8, 12], [12, 20], [20, 32]]],
                'epoch_tmins_and_maxes_grid':[[-3,5]], #times (sec: pre,post) for initial epoching (this should be the longest epoch as the windowed prediction will be tested on it)
                'classifier_training_windows_grid':[[0,2]], #what times(sec: start,end) to use for the classifer training (data augmentation is also using this window)
                'augmentation_windows_grid':[[1,0.1]], #referes to proportions (win_len,win_step) of sfreq, [1,1] means taking the classification epochs, and creating 1 second long epochs with 1 second long steps
                'windowed_prediction_params':[[1,0.1]]} #refers to prportions (win_len,win_step) of sfreq to try and predict i.e. 0.5 = half a second window, with a 100ms steps  

if Use_test_grid: 
   print('\n######\nusing a test grid search\n######\n')
   grid_search_dict=test_grid_search_dict

all_options=[list(range(len(val))) for key,val in grid_search_dict.items()]
print(f'grid options {all_options}')
# Get all possible grid_search combinations: 
all_grid_combinations = list(itertools.product(*all_options))
print(f'number of grid search iterations: {len(all_grid_combinations)}')
print('Grid info:',grid_search_dict)
#save the hyper_grid_search: 
with open(hyper_param_search_output/'grid_search_info.json', 'w') as file:
    json.dump(grid_search_dict, file)
    
# Define what electrodes should be excluded 
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
                    'Fudge':{'Iz'},
                    'g': {'T7','CP1','TP9','P7','PO7'},
                    'Ron': {'Iz'}                   }
    if subject in bad_elecs_dict.keys():
        subject_bad_electrodes=bad_elecs_dict[subject]
    else: 
        subject_bad_electrodes={}
        print('note that no bad electrodes were defined for the current subject:',subject)
    return subject_bad_electrodes   
    
def set_up_params_for_current_grid_iteration(all_grid_combinations,iteration_ind,grid_search_dict):
    curr_grid_comb=all_grid_combinations[iteration_ind]
    print(f'setting up current params: iteration {iteration_ind} - grid settings: {curr_grid_comb}')
    #create a dictionary from the current grid combination (note that the dictionary is ordered): 
    iteration_dictionary={key:val[inner_ind] for ((key,val),inner_ind) in zip(grid_search_dict.items(),curr_grid_comb)}
   
    #extract the paramaters for the current iteration: 
    LowPass=iteration_dictionary['bandpass_borders_grid'][0]
    HighPass=iteration_dictionary['bandpass_borders_grid'][1]
    PerformCsd=iteration_dictionary['run_csd']
    filter_method=iteration_dictionary['filter_methods']

    #extract current electrodes (allow for combination of electrode groups i.e 'C+AF+F')
    Electorde_Group_name=iteration_dictionary['Electorde_Groups_names_grid'] #['FP', 'AF', 'F', 'FC', 'C', 'CP', 'P', 'PO', 'O']
    Electorde_Group=[]
    for cur_elec_group_name in Electorde_Group_name.split('+'):
        Electorde_Group=Electorde_Group+Electorde_Groups[cur_elec_group_name]
    classifier_window_s=iteration_dictionary['classifier_training_windows_grid'][0]
    classifier_window_e=iteration_dictionary['classifier_training_windows_grid'][1]
    epoch_tmin=iteration_dictionary['epoch_tmins_and_maxes_grid'][0]
    epoch_tmax=iteration_dictionary['epoch_tmins_and_maxes_grid'][1]
    n_components=iteration_dictionary['n_components_grid']
    n_components_fbcsp=iteration_dictionary['n_components_fbcsp_grid']
    #define the current augmentation paramaters to test: note that they are defined by samples
    augmentation_params={'win_len':iteration_dictionary['augmentation_windows_grid'][0],
                        'win_step':iteration_dictionary['augmentation_windows_grid'][1]}
    #define the windowed prediction paramaters: #here they are defined as proportions of the sampling frequency
    windowed_prediction_params={'win_len':iteration_dictionary['windowed_prediction_params'][0],
                                'win_step':iteration_dictionary['windowed_prediction_params'][1]}
    #get the current pipeline name: 
    pipeline_name=iteration_dictionary['pipeline_name']
    filters_bands=iteration_dictionary['filters_bands']                          
    #set paramaters dict for current run: 
    params_dict={'LowPass': LowPass,
                'HighPass': HighPass,
                'PerformCsd':PerformCsd,
                'filter_method':filter_method,
                'n_components':n_components,
                'n_components_fbcsp':n_components_fbcsp,
                'filters_bands':filters_bands,
                'Electorde_Group':Electorde_Group,
                'Electorde_Group_name':Electorde_Group_name,
                'epoch_tmin':epoch_tmin,
                'epoch_tmax':epoch_tmax,
                'classifier_window_s':classifier_window_s,
                'classifier_window_e':classifier_window_e,
                'augmentation_params':augmentation_params,
                'windowed_prediction_params':windowed_prediction_params,
                'pipeline_name':pipeline_name}

    return params_dict

# Need to define a preprocessing function that can accept several participant files and add them to a single structure
def run_pre_processing_extract_validation_set(recording_path,current_path,params_dict):

    #extract the current run paramaters: 
    Subject=params_dict['subject']
    PerformCsd=params_dict['PerformCsd']
    LowPass, HighPass, filter_method = params_dict['LowPass'],params_dict['HighPass'],params_dict['filter_method']
    tmin=params_dict['epoch_tmin']
    tmax=params_dict['epoch_tmax']

    #read the file:
    Raw=read_raw_xdf(recording_path / params_dict['recording_file'])
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
    elecs_to_remove=get_subject_bad_electrodes(Subject)
    elecs_to_drop=curr_elecs_in_epochs_set.intersection(elecs_to_remove)

    if len(elecs_to_drop)>0: 
        Raw.drop_channels(list(elecs_to_drop))
    Raw.drop_channels(Raw.info['bads'])
    Raw.set_eeg_reference(ref_channels="average")
    # Do csd: 
    if (PerformCsd):
        print('\n###########################################################')
        print('running csd')
        Raw_CSD = mne.preprocessing.compute_current_source_density(Raw) ## Compute CSD
    else :
        print('\n###########################################################')
        print('not using csd')
        mne.set_eeg_reference(Raw, copy=False)
        Raw_CSD =Raw
    print('\n###########################################################')
    print('filtering the data')  
   # Raw_CSD.pick= (params_dict['Electorde_Group'])
    unfiltered_Raw_CSD=Raw_CSD.copy()
    Raw_CSD_Filtered = unfiltered_Raw_CSD.filter(LowPass, HighPass, method=filter_method)

    #extract filterbank feequencies:
    filters_bands=tuple(params_dict['filters_bands'])
    filtered_data_band_passed=[]
    for i,(LowPass,HighPass) in enumerate(filters_bands):
        unfiltered_Raw_CSD=Raw_CSD.copy()
        Raw_CSD_Filtered_band= mne.filter.filter_data(unfiltered_Raw_CSD.get_data(),sfreq=500, l_freq=LowPass, h_freq=HighPass, method='fir',copy = True)
        filtered_data_band_passed.append(Raw_CSD_Filtered_band)

    events_from_annot,event_dict = mne.events_from_annotations(Raw_CSD_Filtered)
    print('\n###########################################################')
    print('extracting event info:',event_dict)
    
    events_trigger_dict = {key: event_dict[key] for key in event_dict.keys() if key in desired_events}
    print('\n###########################################################')
    filtered_electrodes  = [elec for elec in params_dict['Electorde_Group'] if elec not in elecs_to_drop]
    selected_elecs=filtered_electrodes

    #filter bank related: 
    filter_bank_epochs=[]
    for filtered_data_band in filtered_data_band_passed:
        filtered_data_band_raw = mne.io.RawArray(filtered_data_band,unfiltered_Raw_CSD.info)
        epochs = mne.Epochs(filtered_data_band_raw, events_from_annot, picks = selected_elecs, preload = True,baseline= None, tmin=tmin, tmax=tmax, event_id=events_trigger_dict,detrend=0)
        # Calculate the mean across epochs for the current event
        mean_across_epochs = epochs.get_data().mean(axis=0)
        event_data = epochs.get_data()         
        # Subtract the mean from each epoch of the current event
        centered_event_data = event_data - mean_across_epochs
        event_epochs= epochs.events
        epochs = mne.EpochsArray(centered_event_data, epochs.info, events=event_epochs, event_id=epochs.event_id, tmin=epochs.tmin)
        filter_bank_epochs.append(epochs)
        

    print(f'epoching + selecting current electodes set for analysis:\n{selected_elecs}')
    epochs = mne.Epochs(Raw_CSD_Filtered, events_from_annot,picks = selected_elecs, preload = True,baseline= None, tmin=tmin, tmax=tmax, event_id=events_trigger_dict,detrend=0)
    
    ## Centering the data

    centered_data_list = []
    events_list = []
    mean_across_epochs = epochs.get_data().mean(axis=0)
    # Loop through each event ID
    for idx,event_id in enumerate(desired_events):
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
    elecs_to_remove=get_subject_bad_electrodes(Subject)
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
    else: 
        #filter bank related: 
        filter_bank_epochs_after_elec_drops=[]
        for curr_epochs in filter_bank_epochs:
            filter_bank_epochs_after_elec_drops.append(curr_epochs)  

        print('\n###########################################################')
        print(f'the current selected electrodes: {curr_elecs_in_epochs_set} allready exclude the requested electrodes to remove {elecs_to_remove}')
        print('#############################################################')

    #extract the validation set: we will use it only after selecting all hyper paramaters to get a better representation of out-of-sample performence: 
    #using a very small test size as we currently mostly look at the CV scores: 
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
                'filter_bank_epochs':filter_bank_epochs_after_elec_drops,
                'events_triggers_dict':events_trigger_dict}
    return train_inds,validation_inds,return_dict,mean_across_epochs

def crop_the_data(epochs,train_inds,validation_inds,full_epoch_tmin=0,full_epoch_tmax=5,tmin=1,tmax=2):
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
    epochs_cropped = epochs.copy().crop(tmin=full_epoch_tmin, tmax=full_epoch_tmax)

    #from here on - we extract the data as matrices (not epoch object anymore):

    #save the training data:
    train_set_data=epochs_cropped.get_data()[train_inds]
    train_set_labels=epochs_cropped.events[train_inds,-1]

    #save the validation data: 
    validation_set_data=epochs_cropped.get_data()[validation_inds]
    validation_set_labels=epochs_cropped.events[validation_inds,-1]

    return_dict={'train_set_data_uncroped':train_set_data_uncroped,
                'train_set_labels_uncroped':train_set_labels_uncroped,
                'validation_set_data_uncroped':validation_set_data_uncroped,
                'validation_Set_labels_uncroped':validation_Set_labels_uncroped,
                'epochs_cropped':epochs_cropped,
                'train_set_data':train_set_data,
                'train_set_labels':train_set_labels,
                'validation_set_data':validation_set_data,
                'validation_set_labels':validation_set_labels}
    return return_dict
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
    num_classes = len(desired_events)
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

class ShallowFBCSPNetWrapper:
    def __init__(self, n_channels, n_classes, sfreq, n_times, input_window_seconds=None, learning_rate=0.001, n_epochs=10, batch_size=32):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sfreq = sfreq
        self.n_times = n_times
        self.input_window_seconds = input_window_seconds
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # If n_times is not passed, calculate it from input_window_seconds
        if self.input_window_seconds is not None:
            n_times = int(self.input_window_seconds * self.sfreq)
        
        # Initialize the model with input_window_seconds
        self.model = ShallowFBCSPNet(
            in_chans=n_channels, n_classes=n_classes, input_window_seconds=self.input_window_seconds, final_conv_length="auto"
        ).cuda()  # Move to GPU if available
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(self, X, y):
        """Train the model."""
        # Get n_times from the data if not specified
        if self.n_times is None:
            self.n_times = X.get_data().shape[2]  # Get n_times from the data
            self.input_window_seconds = self.n_times / self.sfreq  # Recalculate input_window_seconds
        
        # Convert data to PyTorch-compatible format
        dataset = create_from_mne_epochs(X, y, input_window_seconds=self.input_window_seconds, sfreq=self.sfreq)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.n_epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        """Generate predictions."""
        dataset = create_from_mne_epochs(X, input_window_seconds=self.input_window_seconds, sfreq=self.sfreq)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.cuda()
                outputs = self.model(batch_x)
                preds.append(outputs.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def score(self, X, y):
        """Evaluate model accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)

def run_windowed_classification_on_fold(fold_train_data_x,fold_train_data_y,fold_test_data_x_uncroped,fold_test_data_y,params_dict,w_start,w_length, BinaryClassification = False):
    #note that this is currently the  function that really does the classification and extracts the performence measure (the previous calls to run_lda.... for example, are just tests)
    curr_classifier_name=params_dict['pipeline_name']
    if curr_classifier_name=='csp+lda':  
        #define the classifier components:  
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=params_dict['n_components'], reg=None, log=True, norm_trace=True)
        #define the pipeline: 
        clf = Pipeline([('csp',csp),('classifier_LDA',lda)])
    elif curr_classifier_name == 'shallowfbcspnet':
        # Define the number of channels and classes
        n_classes = len(np.unique(fold_train_data_y))
        n_channels = fold_train_data_x.shape[1]
        n_classes = len(np.unique(fold_train_data_y))
        n_channels = fold_train_data_x.shape[1]
        n_times = fold_train_data_x.shape[2]
        sfreq = params_dict.get('sfreq', 100)  # Use the sampling frequency from params_dict, or default to 100 Hz

        # Calculate input_window_seconds
        input_window_seconds = n_times / sfreq

        # Instantiate the wrapper
        clf = ShallowFBCSPNetWrapper(
            n_channels=n_channels,
            n_classes=n_classes,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
            learning_rate=0.001,
            n_epochs=10,
            batch_size=32,
        )

        # Instantiate the wrapper
        clf = ShallowFBCSPNetWrapper(n_classes, sfreq)

    elif curr_classifier_name=='csp+svm':
        #define the classifier components:  
        csp = CSP(n_components=params_dict['n_components'], reg=None, log=True, norm_trace=False)
        #define the pipeline: 
        clf = Pipeline([('csp',csp), ('ovo_svm', OneVsOneClassifier(SVC(kernel='linear', random_state=42)))])
    elif curr_classifier_name=='ts+lda':
        #define the classifier components:  
        covest = Covariances()
        ts = TangentSpace()
        lda = LinearDiscriminantAnalysis()
        #define the pipeline: 
        clf = Pipeline([('conv',covest),('ts', ts), ('LDA', lda)])
    elif curr_classifier_name=='fbcsp+lda':
        #define the classifier components: 
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=params_dict['n_components_fbcsp'], reg=None, log=True, norm_trace=False)
        fb=FilterBank(csp)
        #define the pipeline: 
        clf = Pipeline([('fbcsp',fb),('classifier_LDA',lda)])
    else: 
        raise Exception(f'the requested classifier is not defined in "run_windowed_classification_on_fold": {curr_classifier_name}')
    
    #get string labels instead of numeric (so the classifier will have an informative clf.classes_ )
    triggers_label_dict={val:key for key,val in params_dict['preprocessing_dict']['events_triggers_dict'].items()} 
    fold_train_data_y_labels=np.array([triggers_label_dict[cur_y] for cur_y in fold_train_data_y])  
    A, B  = 'OpenPalm', 'ClosedPalm'  # Replace with actual trigger names/values
    combined_labels_train = np.array(['motor_imagery' if label in [A, B] else label for label in fold_train_data_y_labels])
    
    
    # Define class weights based on class distribution
    class_weights = {'Rest': 2, 'ActiveRest': 6, 'OpenPalm': 6, 'ClosePalm': 6}

    # Assign a sample weight to each sample based on its class
    #sample_weights = np.array([class_weights[cls] for cls in combined_labels_train])
    if BinaryClassification:
        clf.fit(fold_train_data_x, combined_labels_train)
    else:
        clf.fit(fold_train_data_x, fold_train_data_y_labels)
    # running classifier: test classifier on sliding window

    #get string labels instead of numeric for the test
    fold_test_data_y_labels=np.array([triggers_label_dict[cur_y] for cur_y in fold_test_data_y])
    if BinaryClassification:
        combined_labels_test = np.array(['motor_imagery' if label in [A, B] else label for label in fold_test_data_y_labels])

        fold_windowed_scores,confusion_metrices_per_window=run_windowed_pretrained_classifier(clf,fold_test_data_x_uncroped,combined_labels_test,w_start,w_length)
    else:
        fold_windowed_scores,confusion_metrices_per_window=run_windowed_pretrained_classifier(clf,fold_test_data_x_uncroped,fold_test_data_y_labels,w_start,w_length)
    return fold_windowed_scores,confusion_metrices_per_window,clf

def run_windowed_pretrained_classifier(clf,x_uncropped,y,w_start,w_length):
    scores_per_time_window = []
    confusion_metrices_per_window=[]
    if len(x_uncropped.shape)==3: #reshape it as if it was a 4d matrix (assuming the 4th dimention is the filterbank)
        x_uncropped=x_uncropped.reshape(list(x_uncropped.shape)+[1])
    for n in w_start:
        fold_data=np.squeeze(x_uncropped[:, :, n:(n + w_length),:]) #using squeeze here so that if the 4th dimention size is 1 it will reduce it to a 3d vector
        #if the classifier uses a filterbank its input should be 4d (trials,channels,timesteps,filter_bands) and if it doesnt its 3d (trials,channels,timesteps)
        # Get predictions
        predictions = clf.predict(fold_data)
        fold_score_on_time_window=clf.score(fold_data, y)
        #fold_score_on_time_window=f1_score(y, predictions, average='micro', zero_division=0)
        #append the score for the LDA, using this csp to predict the relevant test scores: 
        scores_per_time_window.append(fold_score_on_time_window)
        confusion_mat=confusion_matrix(y,clf.predict(fold_data),labels=clf.classes_)
        confusion_metrices_per_window.append(confusion_mat)
    return scores_per_time_window,confusion_metrices_per_window

def augment_data(augmentation_params,data_x_to_augment,y,sfreq):
    #do augmentation: 
    if (augmentation_params['win_step']==0 or augmentation_params['win_len']==0): #check if augmentation is not requested/invalid:
        augmented_x=data_x_to_augment
        augmented_y=y
    else: #augmentation requested
        #set up the augmentation window boundaries based on the augmentation paramaters:                      
        aug_epochs_s=np.arange(0,data_x_to_augment.shape[2],augmentation_params['win_step']*sfreq)
        aug_epochs_e=np.array([a+augmentation_params['win_len']*sfreq for a in aug_epochs_s])
        #remove start and ends that exceeds the relevant epoch lengths: 
        aug_epochs_s=aug_epochs_s[aug_epochs_e<data_x_to_augment.shape[2]]
        aug_epochs_e=aug_epochs_e[aug_epochs_e<data_x_to_augment.shape[2]]

        #pile all augmented (sub windows) to have the regular structure of epochs (>original due to augmentation,channels,samples)
        data_fold_x_augmented=[]
        data_fold_y_augmented=[]
        for aug_s,aug_e in zip(aug_epochs_s,aug_epochs_e):
            if len(data_x_to_augment.shape)==3:
                data_x_in_cur_window=data_x_to_augment[:,:,int(aug_s):int(aug_e)]
            elif len(data_x_to_augment.shape)==4: #with filterbank: 
                data_x_in_cur_window=data_x_to_augment[:,:,int(aug_s):int(aug_e),:]
            data_y_in_cur_window=y
            data_fold_x_augmented.append(data_x_in_cur_window)
            data_fold_y_augmented.append(data_y_in_cur_window)

        augmented_x=np.concatenate(data_fold_x_augmented,axis=0)
        augmented_y=np.concatenate(data_fold_y_augmented)
    return augmented_x,augmented_y

def run_windowed_classification_aug_cv(epochs_cropped,cv_split,train_set_data,train_set_labels,train_set_data_uncroped,params_dict, BinaryClassification):
    augmentation_params=params_dict['augmentation_params']
    windowed_prediction_params=params_dict['windowed_prediction_params']
    win_len=windowed_prediction_params['win_len']
    win_step=windowed_prediction_params['win_step']
    
    sfreq = epochs_cropped.info['sfreq']
    w_length = int(sfreq * win_len)   # running classifier: window length
    w_step = int(sfreq * win_step)  # running classifier: window step size
    w_start = np.arange(0, train_set_data_uncroped.shape[2] + 500- w_length, w_step)
    print('uncroped train set length = ',train_set_data_uncroped.shape[2])

    scores_windows = []
    folds_confusion_metrices_per_window=[]
    #this section first extracts each CV fold, only then it augments it (to avoid data leakage)
    for train_idx, test_idx in cv_split:
        #seperate the cv fold for labels - train-test:
        y_train, y_test = train_set_labels[train_idx], train_set_labels[test_idx] 
        #seperate the cv fold for features information: 
        if len(train_set_data.shape)==3:
            data_fold_x_train_to_augment = train_set_data[train_idx,:,:]
        elif len(train_set_data.shape)==4: #there are filter bank info in the data: 
            data_fold_x_train_to_augment = train_set_data[train_idx,:,:,:] 
        #do augmentation: 
        augmented_x,augmented_y=augment_data(augmentation_params,data_fold_x_train_to_augment,y_train,sfreq)
        #run classifier on the data fold
        curr_scores_windows,confusion_metrices_per_window,_=run_windowed_classification_on_fold(augmented_x,augmented_y,train_set_data_uncroped[test_idx],y_test,params_dict,w_start,w_length, BinaryClassification)         
        scores_windows.append(curr_scores_windows)
        folds_confusion_metrices_per_window.append(confusion_metrices_per_window)
    w_times = (w_start + w_length / 2.) / sfreq + params_dict['epoch_tmin']
    return scores_windows,folds_confusion_metrices_per_window,w_times

def run_windowed_classification_aug(epochs_cropped,train_set_data,train_set_labels,train_set_data_uncroped,test_y,params_dict,BinaryClassification):
    augmentation_params=params_dict['augmentation_params']
    windowed_prediction_params=params_dict['windowed_prediction_params']
    win_len=windowed_prediction_params['win_len']
    win_step=windowed_prediction_params['win_step']
    sfreq = epochs_cropped.info['sfreq']
    w_length = int(sfreq * win_len)   # running classifier: window length
    w_step = int(sfreq * win_step)  # running classifier: window step size
    w_start = np.arange(0, train_set_data_uncroped.shape[2] - w_length, w_step)
    w_times = (w_start + w_length / 2.) / sfreq + params_dict['epoch_tmin']

    augmented_x,augmented_y=augment_data(augmentation_params,train_set_data,train_set_labels,sfreq)
    scores_windows,confusion_metrices_per_window,trained_clf=run_windowed_classification_on_fold(augmented_x,augmented_y,train_set_data_uncroped,test_y,params_dict,w_start,w_length, BinaryClassification)         

    return scores_windows,confusion_metrices_per_window,w_times,trained_clf

def run_training_and_classification_on_selected_params(params_dict,preprocessing_dict, BinaryClassification = False ,to_plot=False,figure_outputs_path='',fig_name='temp'):
    epochs_copy=preprocessing_dict['epochs']
    train_inds=preprocessing_dict['train_inds']
    validation_inds=preprocessing_dict['validation_inds']['original_trial_ind'].values

    #crop the data according to the training window: 
    returned_dict=crop_the_data(epochs_copy,train_inds,validation_inds,params_dict['classifier_window_s'],params_dict['classifier_window_e']) #two more paramters here are tmin and tmax which are not used apparently. 
    train_set_data_uncropped=returned_dict['train_set_data_uncroped']
    epochs_cropped=returned_dict['epochs_cropped']
    train_set_data=returned_dict['train_set_data']
    train_set_labels=returned_dict['train_set_labels']

    validation_set_labels=returned_dict['validation_set_labels']
    validation_set_data_uncropped=returned_dict['validation_set_data_uncroped']
    #define cv on the data: 
    cv = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(train_set_data,train_set_labels)

    #filter bank related:
    if params_dict['pipeline_name']=='fbcsp+lda': 
        train_set_data_fb=[]
        train_set_data_uncropped_fb=[]
        validation_set_data_fb=[]
        validation_set_data_uncropped_fb=[]
        for filtered_data_band_epoch in preprocessing_dict['filter_bank_epochs']:
            returned_dict_temp=crop_the_data(filtered_data_band_epoch,train_inds,validation_inds, params_dict['classifier_window_s'],params_dict['classifier_window_e'])
            #extract the train set data: 
            train_set_data_uncroped_temp=returned_dict_temp['train_set_data_uncroped']
            train_set_data_temp=returned_dict_temp['train_set_data']
            train_set_data_fb.append(train_set_data_temp)
            train_set_data_uncropped_fb.append(train_set_data_uncroped_temp)
            #extract the validation set data: 
            validation_set_data_uncroped_temp=returned_dict_temp['validation_set_data_uncroped']
            validation_set_data_temp=returned_dict_temp['validation_set_data']
            validation_set_data_fb.append(validation_set_data_temp)
            validation_set_data_uncropped_fb.append(validation_set_data_uncroped_temp)
        #create a 4d matrix of train data:     
        train_set_data_4d_array= np.transpose(np.array(train_set_data_fb),(1,2,3,0))
        train_set_data_uncropped_4d_array=np.transpose(np.array(train_set_data_uncropped_fb),(1,2,3,0)) 
        train_set_data=train_set_data_4d_array
        train_set_data_uncropped=train_set_data_uncropped_4d_array
        #create a 4d matrix of validation data: 
        validation_set_data_4d_array= np.transpose(np.array(validation_set_data_fb),(1,2,3,0))
        validation_set_data_uncropped_4d_array=np.transpose(np.array(validation_set_data_uncropped_fb),(1,2,3,0)) 
        validation_set_data_uncropped=validation_set_data_uncropped_4d_array

    #get scores over time using CV: 
    scores_windows,folds_confusion_metrices_per_window,w_times=run_windowed_classification_aug_cv(epochs_cropped,cv_split,train_set_data,train_set_labels,train_set_data_uncropped,params_dict, BinaryClassification)
    
    #train the classifier based on ALL training data, and test its prediction on the unseen validation set: 
    validaiton_scores,validation_confusion_metrices_per_window,_,trained_clf=run_windowed_classification_aug(epochs_cropped,train_set_data,train_set_labels,validation_set_data_uncropped,validation_set_labels,params_dict, BinaryClassification)
    if to_plot:
        fig,axes=plt.subplots(nrows=1,ncols=2)
        plot_accuracy_over_time(scores_windows,w_times,params_dict,axes_handle=axes[0])
        epochs_copy.plot_sensors(show_names=True,axes=axes[1])
        figname=fig_name + '.svg'
        fig.savefig(figure_outputs_path / figname)
    else:
        fig=[]
    return fig,w_times,scores_windows,folds_confusion_metrices_per_window,validaiton_scores,validation_confusion_metrices_per_window,trained_clf

def run_grid_search_on_single_participant(grid_search_dict,recording_file,Subject,save_every_n_iter,save_location_path,to_plot=True):
    all_options=[list(range(len(val))) for key,val in grid_search_dict.items()]
    #get all possible grid_search combinations: 
    all_grid_combinations = list(itertools.product(*all_options))
    print(f'number of grid search iterations: {len(all_grid_combinations)}')

    grid_search_data_frame_info=pd.DataFrame()
    print('running grid search on:',recording_file)
    #put all in a single params_dictionary for the current run: 
    #run all grid_search iterations: 
    for iteration_ind in tqdm(range(len(all_grid_combinations))):
        #extract current iteration paramaters:
        params_dict=set_up_params_for_current_grid_iteration(all_grid_combinations,iteration_ind,grid_search_dict)
        
        #add subject specific information: 
        params_dict['recording_file']=recording_file
        params_dict['subject']=Subject
        print(f'test iteration: dictionary paramaters: {params_dict}')
        #run preprocessing:
        train_inds,validation_inds,preprocessing_dict,mean_across_epochs=run_pre_processing_extract_validation_set(recording_path,current_path,params_dict)
        #run prediction on cross validation:
        fig,w_times,scores_windows,validation_scores=run_training_and_classification_on_selected_params(params_dict,preprocessing_dict,to_plot=to_plot,figure_outputs_path=figure_outputs_path,fig_name='test')
        #train the model on all the training data: 
        #   tbd
        #run prediction on the validation set(?)
        #   tbd
        #add scores related information: 
        curr_params_df=pd.DataFrame([params_dict],index=[iteration_ind])
        curr_params_df['mean_scores']=np.nan
        curr_params_df['std_scores']=np.nan
        curr_params_df['mean_scores']=curr_params_df['mean_scores'].astype(object)
        curr_params_df['std_scores']=curr_params_df['mean_scores'].astype(object)
        curr_params_df['mean_scores']=[np.mean(scores_windows,axis=1)]
        curr_params_df['std_scores']=[np.std(scores_windows,axis=1)]
        grid_search_data_frame_info=pd.concat([grid_search_data_frame_info,curr_params_df],axis=0)
        df_name='hypter_param_search_' + recording_file.split('.')[0] + '.csv'
        if np.mod(iteration_ind,save_every_n_iter)==0:
            print('saving')
            grid_search_data_frame_info.to_csv(hyper_param_search_output / df_name)

    #save all grid_search results: 
    grid_search_data_frame_info.to_csv(hyper_param_search_output / df_name)
    return grid_search_data_frame_info

def plot_precision_recall_curves_from_trained_classifier(preprocessing_dict,params_dict,precision_recall_curve_timerange,trained_clf,predict_validation=True):
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
        inds=preprocessing_dict['validation_inds']['original_trial_ind'].values
    else:
        inds=preprocessing_dict['train_inds']


    #extract the labels: 
    labels=preprocessing_dict['epochs'].events[inds, -1]
    #extract the decision function: 

    #fbcsp
    if params_dict['pipeline_name']=='fbcsp+lda':
        data_set_fb = []
        for filtered_data_band_epoch in preprocessing_dict['filter_bank_epochs']:
            temp_data = filtered_data_band_epoch.copy().crop(tmin=precision_recall_curve_timerange[0],tmax=precision_recall_curve_timerange[1]).get_data()[:]
            data_set_fb.append(temp_data)
        data_set_fb_4d_array= np.transpose(np.array(data_set_fb),(1,2,3,0))
        decision_function=trained_clf.decision_function((data_set_fb_4d_array)[inds,:])
    else:
            decision_function=trained_clf.decision_function(preprocessing_dict['epochs'].copy().crop(tmin=precision_recall_curve_timerange[0],tmax=precision_recall_curve_timerange[1]).get_data()[inds,:])
    y_score=decision_function
    # Use label_binarize to be multi-label like settings (basicly the current label position is 1 and rest are 0): 
    #so the label list of say, 0 2 4 4 will output = [1,0,0],[0,1,0],[0,0,1],[0,0,1]
    classes_numeric_list=list(preprocessing_dict['events_triggers_dict'].values())
    classes_names_list=list(preprocessing_dict['events_triggers_dict'].keys())
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


#%%
def TEST_plot_precision_recall_curves_from_trained_classifier(predict_test,epcohs,preprocessing_dict,params_dict,precision_recall_curve_timerange,trained_clf,predict_validation=True):
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
    inds=epochs[:]


    #extract the labels: 
    labels=epochs.events[inds, -1]
    #extract the decision function: 

    #fbcsp
    # if params_dict['pipeline_name']=='fbcsp+lda':
    #     data_set_fb = []
    #     for filtered_data_band_epoch in preprocessing_dict['filter_bank_epochs']:
    #         temp_data = filtered_data_band_epoch.copy().crop(tmin=precision_recall_curve_timerange[0],tmax=precision_recall_curve_timerange[1]).get_data()[:]
    #         data_set_fb.append(temp_data)
    #     data_set_fb_4d_array= np.transpose(np.array(data_set_fb),(1,2,3,0))
    #     decision_function=trained_clf.decision_function((data_set_fb_4d_array)[inds,:])

    decision_function=trained_clf.decision_function(epochs.copy().crop(tmin=precision_recall_curve_timerange[0],tmax=precision_recall_curve_timerange[1]).get_data()[inds,:])
    y_score=decision_function
    # Use label_binarize to be multi-label like settings (basicly the current label position is 1 and rest are 0): 
    #so the label list of say, 0 2 4 4 will output = [1,0,0],[0,1,0],[0,0,1],[0,0,1]
    classes_numeric_list=list(preprocessing_dict['events_triggers_dict'].values())
    classes_names_list=list(preprocessing_dict['events_triggers_dict'].keys())
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
#%%
# After everything is set up, train and test the classifier 
# We load the defined datafiles and run the same preprocessing pipeline, then concatinate it into a single "combined_preprocessing_dict"
# From here, everything works with the same functions that run on single participants

# Select relevant events for epoching
desired_events = ['ActiveRest','ClosePalm'] 
# Define which subject to currently check: 
for recording_file,Subject in zip(recording_files[5:9],subject_names[5:9]):
    print(recording_file,Subject)

#this code is custom to aggregate all 3 of roi recordings into a single processing_dict structure. 

#####################################################
#define manually paramaters that we wish to change: 
#get all possible grid_search combinations: 
iteration_ind=0 #select some grid search combination - you can manualy change the params after getting the "params_dict" below
grid_search_dict_copy=grid_search_dict.copy()
all_grid_combinations = list(itertools.product(*all_options))
#here i can change manually the current iteration params: 
grid_search_dict_copy['Electorde_Groups_names_grid']=['F+C+CP+P+PO']
grid_search_dict_copy['filters_bands']=[[[7, 12], [12, 20], [20, 28], [28, 35]]]#[[[8,12], [12, 16],[16,20],[20,24],[24,28],[28,32]]]
#this cell allow to test specific iterations
params_dict=set_up_params_for_current_grid_iteration(all_grid_combinations,iteration_ind,grid_search_dict_copy)
params_dict['subject']=Subject
params_dict['recording_file']=recording_file
params_dict['PerformCsd']=True
params_dict['filter_method']='fir'
params_dict['epoch_tmins_and_maxes_grid'] = [-3,5]
params_dict['n_components']= 4
params_dict['LowPass']=5
params_dict['HighPass']=35
params_dict['augmentation_params']={'win_len': 1, 'win_step': 0.1}
params_dict['classifier_window_s']=0
params_dict['classifier_window_e']=4
params_dict['windowed_prediction_params']={'win_len': 2, 'win_step': 0.1}
params_dict['pipeline_name']='csp+lda'
params_dict['n_components_fbcsp']=4

BinaryClassification = False

##########################preprocess each of the recording seperately#########################
preprocessing_dicts=[]

for recording_file,subject in zip(recording_files[5:9],subject_names[5:9]):
    print(recording_file,subject)
    params_dict['recording_file']=recording_file
    train_inds,validation_inds,preprocessing_dict,mean_across_epochs=run_pre_processing_extract_validation_set(recording_path,current_path,params_dict)
    preprocessing_dicts.append(preprocessing_dict)
##############################################################################################

#combine all preprocessing structures: 
#combine each file preprocessing_dict into a single dictionary:
#take the first roi file and modify its triggers to be consistent with the later two: 

combined_preprocessing_dict=copy.deepcopy(preprocessing_dicts[0]) #the deep copy is important here as we have a dictionary that contains lists/dictionaries

#change the epochs structures triggers information to be consistent with the next 2 files: (roi first file had triggers of 2,5 and 6) 
#combined_preprocessing_dict['epochs'].event_id={'left': 3, 'rest': 6, 'right': 7} #change the event ids to be the same as his other 2 files
#combined_preprocessing_dict['epochs'].events[:,2]=combined_preprocessing_dict['epochs'].events[:,2]+1 #change the event numeric data within the epoch structure
#for fb_num in range(len(combined_preprocessing_dict['filter_bank_epochs'])): #do the same on the filter bank epochs (should be able to handle arbitraty number of bands)
#    combined_preprocessing_dict['filter_bank_epochs'][fb_num].events[:,2]=combined_preprocessing_dict['filter_bank_epochs'][fb_num].events[:,2]+1
#    combined_preprocessing_dict['filter_bank_epochs'][fb_num].event_id={'left': 3, 'rest': 6, 'right': 7}

#now that the first recording file is set as standard - read the other two files and add their information to the combined dictionary: 
add_to_inds=combined_preprocessing_dict['epochs'].events.shape[0] 
for i,cur_preprocessing_dict in enumerate(preprocessing_dicts[1:]): 
    #aggregate all training_indexes: 
    combined_preprocessing_dict['train_inds'] = np.concatenate([combined_preprocessing_dict['train_inds'],cur_preprocessing_dict['train_inds']+add_to_inds])
    #aggregate all validation indexes
    validation_inds_df=cur_preprocessing_dict['validation_inds']
    validation_inds_df.index=validation_inds_df.index+add_to_inds
    validation_inds_df['original_trial_ind']=validation_inds_df['original_trial_ind']+add_to_inds
    combined_preprocessing_dict['validation_inds'] = pd.concat([combined_preprocessing_dict['validation_inds'],validation_inds_df],axis=0)
    #increase the number to add to keep indexes consistent (i.e. first was 0-60, next file should have them at the range of 60-120 and so on)
    add_to_inds+=cur_preprocessing_dict['epochs'].events.shape[0] #this should allow for files with different number of epochs. 

    for fb_filter_num in range(len(combined_preprocessing_dict['filter_bank_epochs'])):
        combined_preprocessing_dict['filter_bank_epochs'][fb_filter_num]=mne.concatenate_epochs([combined_preprocessing_dict['filter_bank_epochs'][fb_filter_num],cur_preprocessing_dict['filter_bank_epochs'][fb_filter_num]], on_mismatch='warn' , verbose=None)
    combined_preprocessing_dict['epochs']=mne.concatenate_epochs([combined_preprocessing_dict['epochs'],cur_preprocessing_dict['epochs']], on_mismatch='warn' , verbose=None)
    
combined_preprocessing_dict['events_triggers_dict']=preprocessing_dict['events_triggers_dict']
params_dict['preprocessing_dict']=preprocessing_dict


#test it: 
fig,w_times,scores_windows,folds_confusion_metrices_per_window,validation_scores,validation_confusion_metrices_per_window,trained_clf=run_training_and_classification_on_selected_params(params_dict,combined_preprocessing_dict,BinaryClassification,to_plot=True,figure_outputs_path=figure_outputs_path,fig_name='test')



#%%
# Plot the precision recall curve, and extract the relevant decision information into a dataframe

return_df=plot_precision_recall_curves_from_trained_classifier(combined_preprocessing_dict,params_dict,precision_recall_curve_timerange=[0,2],trained_clf=trained_clf,predict_validation=True)
print(return_df)

if True: 
    #get the relevant data for the validation set: 
    inds=combined_preprocessing_dict['validation_inds']['original_trial_ind'].values
else:
    inds=combined_preprocessing_dict['train_inds']
data_set_fb = []
for filtered_data_band_epoch in combined_preprocessing_dict['filter_bank_epochs']:
     temp_data = filtered_data_band_epoch.copy().crop(tmin=2.45,tmax=3.45).get_data()[:]
     data_set_fb.append(temp_data)
data_set_fb_4d_array= np.transpose(np.array(data_set_fb),(1,2,3,0))
#data_to_predict=(data_set_fb_4d_array[inds,:])
data_to_predict=combined_preprocessing_dict['epochs'].copy().crop(tmin=1,tmax=3).get_data()[inds,:]
thresholded_prediction=trained_clf.decision_function(data_to_predict)
prediction=trained_clf.predict(data_to_predict)

#note that here you can decide on which thresholds to use to better optimize your "real" usecase
#thresholded_prediction
yhat=trained_clf.predict(data_to_predict)
lr_probs = trained_clf.predict_proba(data_to_predict)
#%%


import matplotlib.pyplot as plt
# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
 
model = trained_clf
testy = combined_preprocessing_dict['validation_inds']['original_trial_ind'].values
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# %%

# Save the trained model 
Saved_Model = trained_clf
fname = 'NH_3rd_arm'+'_4CSP_model_1111'
path_fname = current_path /'Models'/ fname

#create a pickle file
picklefile = open(path_fname, 'wb')
#pickle the dictionary and write it to file
pickle.dump(Saved_Model, picklefile)
#close the file
picklefile.close()



picks=params_dict['Electorde_Group']
fname = 'electrode_picks-1111'
path_fname = current_path /'Models'/ fname

#create a pickle file
picklefile = open(path_fname, 'wb')
#pickle the dictionary and write it to file
pickle.dump(picks, picklefile)
#close the file
picklefile.close()


fname = 'mean'
path_fname = current_path /'Models'/ fname

#create a pickle file
picklefile = open(path_fname, 'wb')
#pickle the dictionary and write it to file
pickle.dump(mean_across_epochs, picklefile)
#close the file
picklefile.close()

# %%
# Sanity check on unhandled recording: 
current_path = pathlib.Path().absolute()  
recording_path = current_path / 'Recordings'

OriginalRaw=read_raw_xdf(recording_path / 'Roei-MI_3.xdf')
OriginalRaw.drop_channels(['ACC_X','ACC_Y','ACC_Z']) ## Drop non eeg channels
#mne.rename_channels(OriginalRaw.info, {'F9' : 'FT9','P9' : 'TP9','P10' : 'TP10','F10' : 'FT10','AF1' : 'AF7' }, allow_duplicates=False, verbose=None)
montage = mne.channels.read_custom_montage((f"{current_path}\Montages\CACS-64_REF.bvef"), head_size=0.095, coord_frame=None) 
unfiltered_OriginalRaw = OriginalRaw.copy()
unfiltered_OriginalRaw.set_montage(montage, match_case=True, match_alias=False, on_missing='raise', verbose=None)
Original_Raw_CSD = mne.preprocessing.compute_current_source_density(unfiltered_OriginalRaw) ## Compute CSD

filters_bands=tuple(params_dict['filters_bands'])
filtered_data_band_passed=[]
for i,(LowPass,HighPass) in enumerate(filters_bands):
    unfiltered_Raw_CSD=Original_Raw_CSD.copy()
    Raw_CSD_Filtered_band= unfiltered_Raw_CSD.filter(LowPass, HighPass, method='fir')
    filtered_data_band_passed.append(Raw_CSD_Filtered_band)

filtered_data_band_passed_data=[]
for i,(LowPass,HighPass) in enumerate(filters_bands):
    Raw_CSD_Filtered_band= mne.filter.filter_data(OriginalRaw.get_data(),sfreq = 500, l_freq = LowPass, h_freq=HighPass, method='fir',copy = True)
    filtered_data_band_passed_data.append(Raw_CSD_Filtered_band)
#filter bank related: 
filter_bank_epochs=[]
for filtered_data_band in filtered_data_band_passed:
    epochs = mne.Epochs(filtered_data_band, events_from_annot, picks=params_dict['Electorde_Group'],preload = True,baseline= None, tmin=-3, tmax=5, event_id=events_trigger_dict,detrend=0)
    filter_bank_epochs.append(epochs)
# %%
# Sanity check on unhandled recording: 
current_path = pathlib.Path().absolute()  
recording_path = current_path / 'Recordings'

OriginalRaw=read_raw_xdf(recording_path / 'Ron_3rd_MI_03.xdf')
OriginalRaw.drop_channels(['ACC_X','ACC_Y','ACC_Z']) ## Drop non eeg channels
#mne.rename_channels(OriginalRaw.info, {'F9' : 'FT9','P9' : 'TP9','P10' : 'TP10','F10' : 'FT10','AF1' : 'AF7' }, allow_duplicates=False, verbose=None)
unfiltered_OriginalRaw = OriginalRaw.copy()
montage = mne.channels.read_custom_montage((f"{current_path}\Montages\CACS-64_REF.bvef"), head_size=0.095, coord_frame=None) 
unfiltered_OriginalRaw.set_montage(montage, match_case=True, match_alias=False, on_missing='raise', verbose=None)
elecs_to_remove=get_subject_bad_electrodes(subject)
# if len(elecs_to_remove)>0: 
#     unfiltered_OriginalRaw.drop_channels(list(elecs_to_remove))
#     unfiltered_OriginalRaw.drop_channels(unfiltered_OriginalRaw.info['bads'])
#     unfiltered_OriginalRaw.set_eeg_reference(ref_channels="average")

filtered_electrodes  = [elec for elec in params_dict['Electorde_Group'] if elec not in elecs_to_remove]
selected_elecs=filtered_electrodes
Original_Raw_CSD = mne.preprocessing.compute_current_source_density(unfiltered_OriginalRaw) ## Compute CSD
OriginalRaw_Filtered = Original_Raw_CSD.filter(5, 35, method='fir')
events_from_annot,event_dict = mne.events_from_annotations(unfiltered_OriginalRaw)
desired_events = ['ActiveRest','ClosePalm'] 
events_trigger_dict = {key: event_dict[key] for key in event_dict.keys() if key in desired_events}

epochs = mne.Epochs(OriginalRaw_Filtered, events_from_annot,picks=selected_elecs, preload = True,baseline= None, tmin=-3, tmax=5, event_id=events_trigger_dict,detrend=0)
# %%

from collections import Counter

# Function to get the list of event names for each epoch
def get_epoch_events(epochs):
    events = epochs.events  # Get the array of events
    event_ids = epochs.event_id  # Get the mapping from event name to event code
    epoch_event_names = []

    for event in events:
        event_code = event[-1]  # Extract the event ID
        # Get the event name using the event code
        event_name = [key for key, val in event_ids.items() if val == event_code][0]
        epoch_event_names.append(event_name)

    return np.array(epoch_event_names)

# Function to compare actual event names with predicted event names and summarize mismatches
def compare_events(actual_events, predicted_events):
    comparison = actual_events == predicted_events  # Element-wise comparison
    accuracy = np.mean(comparison)  # Calculate accuracy
    mismatches = np.where(comparison == False)[0]  # Get indices where predictions don't match
    
    # Collect mismatched epochs and count occurrences of each mismatch case
    mismatch_details = []
    mismatch_counts = Counter()
    
    for idx in mismatches:
        actual = actual_events[idx]
        predicted = predicted_events[idx]
        mismatch_details.append(f"{idx + 1} - {actual} (actual), {predicted} (predicted)")
        mismatch_counts[f"{actual} -> {predicted}"] += 1  # Count each specific mismatch case

    return {
        'accuracy': accuracy,
        'mismatch_details': mismatch_details,
        'mismatch_counts': mismatch_counts
    }

# Example usage:
# Assume `epochs` is your mne.Epochs object, and `predictions` is the model output

actual_events = get_epoch_events(epochs)  # Actual events from the epochs
predicted_events = np.array(prediction)  # Replace with your model predictions

# Compare actual events to predicted events
result = compare_events(actual_events, predicted_events)

# Output the comparison result
print(f"Accuracy: {result['accuracy']:.2f}")
if result['mismatch_details']:
    print("Mismatched epochs:")
    for mismatch in result['mismatch_details']:
        print(mismatch)
    
    print("\nMismatch Summary:")
    for mismatch_case, count in result['mismatch_counts'].items():
        print(f"{mismatch_case}: {count} occurrence(s)")
else:
    print("No mismatches!")

#%%


# Assuming 'epochs' is your MNE Epochs object and it contains events named 'Left', 'Resting', 'Right'

centered_data_list = []
events_list = []

# Loop through each event ID
for idx, event_id in enumerate(desired_events):
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
centered_epochs.plot()
# Note: This code assumes event_ids are correctly listed and exist within the original epochs.
# You might need to adjust event_id_map or event IDs based on your specific dataset and requirements.
#%% Overall use case for sanity checks
# Parameters
window_duration = 1.0  # Window duration in seconds
step_size = 0.1  # Step size in seconds
start_time = -0.5
end_time = 5.0  # Total duration in seconds
sampling_rate = epochs.info['sfreq']  # Sampling frequency of your data

# Calculate the number of steps (0.1-second increments)
num_steps = int((end_time - start_time - window_duration) / step_size) + 1

# Initialize a list to store predictions
all_predictions = []

# Loop through each start time incrementally by 0.1 seconds
for i in range(num_steps):
    current_time = start_time + i * step_size
    
    # Crop out a 2-second window from the current time
    data_to_predict = epochs.copy().crop(tmin=current_time, tmax=current_time + window_duration).get_data()
    
    # Predict for the current window
    prediction = trained_clf.predict(data_to_predict)
    
    # Append the prediction to the list
    all_predictions.append(prediction)

# Convert to a numpy array if needed
all_predictions = np.array(all_predictions)

# Initialize list to store accuracy for each prediction set
accuracies = []
all_mismatch_details = []
overall_mismatch_counts = Counter()

# Loop over each set of predictions in all_predictions
for i in range(len(all_predictions)):
    predicted_events = np.array(all_predictions[i])  # Get the predictions for the i-th set
    result = compare_events(actual_events, predicted_events)  # Compare with actual events
    
    # Store the accuracy for this set of predictions
    accuracies.append(result['accuracy'])
    
    # Collect mismatch details and counts
    all_mismatch_details.extend(result['mismatch_details'])
    overall_mismatch_counts.update(result['mismatch_counts'])

# Calculate mean accuracy across all predictions
mean_accuracy = np.mean(accuracies)

print("Mean Accuracy:", mean_accuracy)
print("All Mismatch Details:", all_mismatch_details)
print("Overall Mismatch Counts:", overall_mismatch_counts)

# %%
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.time_frequency import tfr_multitaper,tfr_morlet
from matplotlib.colors import TwoSlopeNorm
# ERD Maps Section
freqs = np.arange(5, 34)  # frequencies from 2-35Hz
# vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = (-2.8, -1.8)  # baseline interval (in s)
# cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)  # for cluster test
n_cycles = np.linspace(2.6, 2.6 + 0.2 * (30 - 1), len(freqs))
tfr = tfr_multitaper(
    combined_preprocessing_dict['epochs'].pick(["C3","Cz","C2"]),
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
)

tfr.crop(-3, 5).apply_baseline(baseline, mode="logratio")
tfr.average().plot()

for event in desired_events:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(
        1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
    )
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot(
            [ch],
            cmap="RdBu",
            axes=ax,
            colorbar=False,
            show=False,
            mask=mask,
            mask_style="mask",
        )

        ax.set_title(combined_preprocessing_dict['epochs'].ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    plt.show()
# %%
band_dict =  dict(Alpha=dict(fmin=8, fmax=12),
                    Beta=dict(fmin=15, fmax=28))
tfr = tfr_multitaper(
    combined_preprocessing_dict['epochs'],
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
)

tfr.crop(-2.5, 4).apply_baseline(baseline, mode="zscore")


for title,fmin_fmax in band_dict.items():

    fig, axes = plt.subplots(1, 3, figsize=(7, 4), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    
    tfr_band = tfr.copy().crop(fmin = fmin_fmax['fmin'], fmax =fmin_fmax['fmax'])

    
    plot_dict = dict(Alpha=dict(fmin=8, fmax=12), Beta=dict(fmin=15, fmax=28))

    for event in desired_events:
        tfr_ev = tfr[event]
        tfr_ev.average().plot_topomap(**fmin_fmax, axes=ax)
        ax.set_title(title)
        
    plt.tight_layout()
    plt.show()

# %%
decim = 4
#freqs = np.arange(4, 30, 2)  # define frequencies of interest
#n_cycles = 2

#freqs = np.logspace(*np.log10([7.5, 12.5]), num=25)
n_cycles = np.linspace(2.6, 2.6 + 0.2 * (30 - 1), len(freqs))

#n_cycles = freqs / 2.0  # different number of cycle per frequency


tfr_all = tfr_morlet(
    combined_preprocessing_dict['epochs']["ClosePalm"].pick(["C3", "Cz", "C4"]),
    freqs,
    n_cycles=n_cycles,
    
    return_itc=False,
    average=False,
)
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
    num_classes = len(desired_events)
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
    axes_handle.set_xlim([-2, 5])  # Adjust the x-axis limits to extend to 5 seconds
    axes_handle.legend()
    axes_handle.grid(True)

# %%
