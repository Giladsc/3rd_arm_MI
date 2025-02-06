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
def classifier_training(fold_train_data_x,fold_train_data_y,params_dict, BinaryClassification = False):
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

def run_windowed_pretrained_classifier(clf,x_uncropped,y,w_start,w_length):
    scores_per_time_window = []
    confusion_metrices_per_window=[]
    if len(x_uncropped.shape)==3: #reshape it as if it was a 4d matrix (assuming the 4th dimention is the filterbank)
        x_uncropped=x_uncropped.reshape(list(x_uncropped.shape)+[1])
    for n in w_start:
        fold_data=np.squeeze(x_uncropped[:, :, n:(n + w_length),:]) #using squeeze here so that if the 4th dimention size is 1 it will reduce it to a 3d vector
        #if the classifier uses a filterbank its input should be 4d (trials,channels,timesteps,filter_bands) and if it doesnt its 3d (trials,channels,timesteps)
        fold_score_on_time_window=clf.score(fold_data, y)
        #append the score for the LDA, using this csp to predict the relevant test scores: 
        scores_per_time_window.append(fold_score_on_time_window)
        confusion_mat=confusion_matrix(y,clf.predict(fold_data),labels=clf.classes_)
        confusion_metrices_per_window.append(confusion_mat)
    return scores_per_time_window,confusion_metrices_per_window

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
