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
