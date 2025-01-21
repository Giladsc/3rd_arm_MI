# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt

# MNE functions
import mne
from mne import Epochs,find_events
from mne.decoding import Vectorizer


# Scikit-learn and Pyriemann ML functionalities
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score,train_test_split

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

import mne

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

import gzip
import logging
import struct
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

from pyxdf import load_xdf


print(__doc__)

import pyxdf
import PyQt5

from easygui import *

import pathlib

# For interactive plots
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

from os import listdir
from os.path import isfile, join



def crop_the_data(epochs,train_inds,validation_inds,full_epoch_tmin=1,full_epoch_tmax=2,tmin=1,tmax=2):
    tmin=float(tmin)
    tmax=float(tmax)
    #save uncropped versions of the data: 
    #save the training data:
    train_set_data_uncroped=epochs.get_data()[train_inds]
    train_set_labels_uncroped=epochs.events[train_inds,-1]-2

    #save the validation data: 
    validation_set_data_uncroped=epochs.get_data()[validation_inds]
    validation_Set_labels_uncroped=epochs.events[validation_inds,-1]-2

    #crop the epochs (use the epochs structure)
    epochs_cropped = epochs.copy().crop(tmin=full_epoch_tmin, tmax=full_epoch_tmax)

    #from here on - we extract the data as matrices (not epoch object anymore):

    #save the training data:
    train_set_data=epochs_cropped.get_data()[train_inds]
    train_set_labels=epochs_cropped.events[train_inds,-1]-2

    #save the validation data: 
    validation_set_data=epochs_cropped.get_data()[validation_inds]
    validation_set_labels=epochs_cropped.events[validation_inds,-1]-2

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

def augment_the_data (cropped_epochs ,train_inds, train_set_data_uncroped):

    number_of_events = len(cropped_epochs.events)
#augment the epochs (use the uncroped train set data structure)
    epochs_aug =[]
    index=0
    i=1
    while i<=1.5:
        epochs_aug.append(cropped_epochs.copy().crop(tmin=(i), tmax=(i+0.5)))
        epochs_aug[index].shift_time(tshift = 0, relative=False)
        if (index ==0):
            epochs_cropped = epochs_aug[index]
            train_inds_conj = train_inds
        if (index !=0):
            epochs_cropped = mne.concatenate_epochs ([epochs_cropped,epochs_aug[index]])
            train_inds_conj = np.concatenate([train_inds_conj,train_inds+(index*number_of_events)])
        index = index + 1
        i = i + 0.05
        #from here on - we extract the data as matrices (not epoch object anymore):
        #save the augmented training data:
        train_set_data_aug=epochs_cropped.get_data()[train_inds_conj]
        train_set_labels_aug=epochs_cropped.events[train_inds_conj,-1]-2
    
    augmented_dict={'epochs_cropped_augmented':epochs_cropped,
                'train_set_data_augmented':train_set_data_aug,
                'train_set_labels_augmented':train_set_labels_aug}
    return augmented_dict

def def_csp_lda_cv(train_set_data,train_set_labels,n_folds=10,test_size=0.2,random_state=42):
    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []

    cv = ShuffleSplit(n_folds, test_size=test_size, random_state=random_state)
    cv_split = cv.split(train_set_data)


    #copy the cv object running this cell repeatedly will be possible
    cv_split_copy=list(cv_split)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, train_set_data, train_set_labels, cv=cv, n_jobs=None)    
    return cv_split_copy,clf

def run_csp_lda_cv(train_set_data,train_set_labels,n_folds=10,test_size=0.2,random_state=42):
    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []

    cv = StratifiedShuffleSplit(n_folds, test_size=test_size, random_state=random_state)
    cv_split = cv.split(train_set_data,train_set_labels)


    #copy the cv object running this cell repeatedly will be possible
    cv_split_copy=list(cv_split)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, train_set_data, train_set_labels, cv=cv, n_jobs=None)

    # Printing the results
    class_balance = np.mean(train_set_labels == train_set_labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print('train set shape:',train_set_data.shape)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                            class_balance))
    return cv_split_copy,clf


def plot_accuracy_over_time(scores_windows,w_times,axes_handle):
    #this function accepts the scores windows (a list of n folds - each giving a score on a time window)
    #it converts it to a long dataframe with the following columns: fold_id,Time,Accuracy
    #then it uses the long format to plot using seaborn lineplot and get a confidence interval
    scores_windows_df=pd.DataFrame(scores_windows)
    scores_windows_df.columns=[np.round(w_times[s],2) for s in scores_windows_df]
    scores_windows_df['fold_id']=range(len(scores_windows_df))
    longform_scores_windows_df=pd.melt(scores_windows_df, id_vars='fold_id', value_vars=scores_windows_df.columns)
    longform_scores_windows_df.rename(columns={'variable':'Time','value':'Accuracy'},inplace=True)
    sns.lineplot(data=longform_scores_windows_df,x='Time',y='Accuracy',ax=axes_handle)
    axes_handle.axvline(3, linestyle='--', color='k', label='Onset') ## adjusted 'Onset' to 3 seconds for augmented data
    axes_handle.axvline(2, linestyle=':', color='k', label='ArrowGone') ## adjusted 'Onset' to 3 seconds for augmented data
    axes_handle.axhline(0.5, linestyle='-', color='k', label='Chance')
    axes_handle.set_xlabel('time (s)')
    axes_handle.set_ylabel('classification accuracy')
    axes_handle.set_title('Classification score over time')
    axes_handle.set_ylim([0.25, 0.9])

def augment_data (augmentation_params,data_x_to_augment,y,sfreq):
    augmentation_params['win_step'] = 0.05
    augmentation_params['win_len'] = 0.5
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

def run_windowed_classification_aug(epochs_cropped,cv_split_copy,train_set_data,train_set_labels,train_set_data_uncroped,clf,win_len=0.5,win_step=0.1):

    csp=clf[list(clf.named_steps.keys())[0]]
    lda=clf[list(clf.named_steps.keys())[1]]

    sfreq = epochs_cropped.info['sfreq']
    epochs_cropped_data=epochs_cropped.get_data()
    

    w_length = int(sfreq * win_len)   # running classifier: window length
    w_step = int(sfreq * win_step)  # running classifier: window step size
    w_start = np.arange(0, train_set_data_uncroped.shape[2] - w_length, w_step)
    print('uncroped train set length = ',train_set_data_uncroped.shape[2])

    scores_windows = []

    for train_idx, test_idx in cv_split_copy:
        print('length of train idx',len(train_idx))
 
        
        #seperate the cv fold for labels - train-test
        y_train, y_test = train_set_labels[train_idx], train_set_labels[test_idx] 
        
        #augment the training and test set: 
        
        augmented_fold_x = train_set_data[train_idx]
        augmented_fold_y = train_set_data[test_idx]
        
        #augment_dict_x = augment_the_data (epochs_cropped, train_idx,augmented_fold_x)
        #augment_dict_y = augment_the_data (epochs_cropped,test_idx,augmented_fold_y)

        augmented_x, augmented_y = augment_data ({},augmented_fold_x ,y_train,500)
        
        #augmented_fold_x=train_set_data[train_idx]
        #augmented_fold_y=y_train[train_idx]
        
        #x_epoch_aug = augment_dict_x['epochs_cropped_augmented']
        #x_data_aug = augment_dict_x['train_set_data_augmented']
        #x_lables_aug = augment_dict_x['train_set_labels_augmented']
        #print (x_data_aug.shape)
        #print (x_lables_aug.shape)
        #y_epoch_aug = augment_dict_y['epochs_cropped_augmented']
        #y_data_aug = augment_dict_y['train_set_data_augmented']
        #y_lables_aug = augment_dict_y['train_set_labels_augmented']
        
        #seperate the cv fold features for train-test
        X_train_after_csp = csp.fit_transform(augmented_x, augmented_y)
        X_test_after_csp  = csp.transform(train_set_data[test_idx])


        # fit classifier
        lda.fit(X_train_after_csp, augmented_y)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            #transform the original data using the csp trained on the current cv fold: 
            X_window_test_after_csp = csp.transform(train_set_data_uncroped[test_idx][:, :, n:(n + w_length)])
            #append the score for the LDA, using this csp to predict the relevant test scores: 
            score_this_window.append(lda.score(X_window_test_after_csp, y_test))
        scores_windows.append(score_this_window)

    w_times = (w_start + w_length / 2.) / sfreq + epochs_cropped.tmin
    return scores_windows,w_times

def run_windowed_classification(epochs_cropped,cv_split_copy,train_set_data,train_set_labels,train_set_data_uncroped,clf,win_len=0.5,win_step=0.1):

    csp=clf[list(clf.named_steps.keys())[0]]
    lda=clf[list(clf.named_steps.keys())[1]]

    sfreq = epochs_cropped.info['sfreq']
    epochs_cropped_data=epochs_cropped.get_data()

    w_length = int(sfreq * win_len)   # running classifier: window length
    w_step = int(sfreq * win_step)  # running classifier: window step size
    w_start = np.arange(0, train_set_data_uncroped.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split_copy:
        #seperate the cv fold for labels - train-test
        y_train, y_test = train_set_labels[train_idx], train_set_labels[test_idx] 
        #seperate the cv fold features for train-test
        X_train_after_csp = csp.fit_transform(train_set_data[train_idx], y_train)
        X_test_after_csp  = csp.transform(train_set_data[test_idx])


        # fit classifier
        lda.fit(X_train_after_csp, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            #transform the original data using the csp trained on the current cv fold: 
            X_window_test_after_csp = csp.transform(train_set_data_uncroped[test_idx][:, :, n:(n + w_length)])
            #append the score for the LDA, using this csp to predict the relevant test scores: 
            score_this_window.append(lda.score(X_window_test_after_csp, y_test))
        scores_windows.append(score_this_window)

    w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
    return scores_windows,w_times


logger = logging.getLogger()


def open_xdf(filename):
    """Open XDF file for reading."""
    filename = Path(filename)  # convert to pathlib object
    if filename.suffix == '.xdfz' or filename.suffixes == ['.xdf', '.gz']:
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, 'rb')
    if f.read(4) != b'XDF:':  # magic bytes
        raise IOError('Invalid XDF file {}'.format(filename))
    return f


def match_streaminfos(stream_infos, parameters):
    """Find stream IDs matching specified criteria.

    Parameters
    ----------
    stream_infos : list of dicts
        List of dicts containing information on each stream. This information
        can be obtained using the function resolve_streams.
    parameters : list of dicts
        List of dicts containing key/values that should be present in streams.
        Examples: [{"name": "Keyboard"}] matches all streams with a "name"
                  field equal to "Keyboard".
                  [{"name": "Keyboard"}, {"type": "EEG"}] matches all streams
                  with a "name" field equal to "Keyboard" and all streams with
                  a "type" field equal to "EEG".
    """
    matches = []
    for request in parameters:
        for info in stream_infos:
            for key in request.keys():
                match = info[key] == request[key]
                if not match:
                    break
            if match:
                matches.append(info['stream_id'])

    return list(set(matches))  # return unique values


def resolve_streams(fname):
    """Resolve streams in given XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    stream_infos : list of dicts
        List of dicts containing information on each stream.
    """
    return parse_chunks(parse_xdf(fname))


def parse_xdf(fname):
    """Parse and return chunks contained in an XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    chunks : list
        List of all chunks contained in the XDF file.
    """
    chunks = []
    with open_xdf(fname) as f:
        for chunk in _read_chunks(f):
            chunks.append(chunk)
    return chunks


def _read_chunks(f):
    """Read and yield XDF chunks.

    Parameters
    ----------
    f : file handle
        File handle of XDF file.


    Yields
    ------
    chunk : dict
        XDF chunk.
    """
    while True:
        chunk = dict()
        try:
            chunk["nbytes"] = _read_varlen_int(f)
        except EOFError:
            return
        chunk["tag"] = struct.unpack('<H', f.read(2))[0]
        if chunk["tag"] in [2, 3, 4, 6]:
            chunk["stream_id"] = struct.unpack("<I", f.read(4))[0]
            if chunk["tag"] == 2:  # parse StreamHeader chunk
                xml = ET.fromstring(f.read(chunk["nbytes"] - 6).decode())
                chunk = {**chunk, **_parse_streamheader(xml)}
            else:  # skip remaining chunk contents
                f.seek(chunk["nbytes"] - 6, 1)
        else:
            f.seek(chunk["nbytes"] - 2, 1)  # skip remaining chunk contents
        yield chunk


def _parse_streamheader(xml):
    """Parse stream header XML."""
    return {el.tag: el.text for el in xml if el.tag != "desc"}


def parse_chunks(chunks):
    """Parse chunks and extract information on individual streams."""
    streams = []
    for chunk in chunks:
        if chunk["tag"] == 2:  # stream header chunk
            streams.append(dict(stream_id=chunk["stream_id"],
                                name=chunk.get("name"),  # optional
                                type=chunk.get("type"),  # optional
                                source_id=chunk.get("source_id"),  # optional
                                created_at=chunk.get("created_at"),  # optional
                                uid=chunk.get("uid"),  # optional
                                session_id=chunk.get("session_id"),  # optional
                                hostname=chunk.get("hostname"),  # optional
                                channel_count=int(chunk["channel_count"]),
                                channel_format=chunk["channel_format"],
                                nominal_srate=int(chunk["nominal_srate"])))
    return streams


def read_raw_xdf(fname, stream_id=None):
    """Read XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.
    stream_id : int | str | None
        ID (number) or name of the stream to load (optional). If None, the
        first stream of type "EEG" will be read.

    Returns
    -------
    raw : mne.io.Raw
        XDF file data.
    """
    streams, header = load_xdf(fname)

    if stream_id is not None:
        if isinstance(stream_id, str):
            stream = _find_stream_by_name(streams, stream_id)
        elif isinstance(stream_id, int):
            stream = _find_stream_by_id(streams, stream_id)
    else:
        stream = _find_stream_by_type(streams, stream_type="EEG")

    if stream is not None:
        name = stream["info"]["name"][0]
        n_chans = int(stream["info"]["channel_count"][0])
        fs = float(stream["info"]["nominal_srate"][0])
        logger.info(f"Found EEG stream '{name}' ({n_chans} channels, "
                    f"sampling rate {fs}Hz).")
        labels, types, units = _get_ch_info(stream)
        if not labels:
            labels = [str(n) for n in range(n_chans)]
        if not units:
            units = ["NA" for _ in range(n_chans)]
        info = mne.create_info(ch_names=labels, sfreq=fs, ch_types="eeg")
        # convert from microvolts to volts if necessary
        scale = np.array([1e-6 if u == "microvolts" else 1 for u in units])
        raw = mne.io.RawArray((stream["time_series"] * scale).T, info)
        first_samp = stream["time_stamps"][0]
    else:
        logger.info("No EEG stream found.")
        return

    markers = _find_stream_by_type(streams, stream_type="Markers")
    if markers is not None:
        onsets = markers["time_stamps"] - first_samp
        logger.info(f"Adding {len(onsets)} annotations.")
        descriptions = markers["time_series"]
        descriptions = np.array(descriptions).squeeze()  # fix to prevent dim2 ndarray in MNE annotation parsing. lokinou on 11/12/2020
        annotations = mne.Annotations(onsets, [0] * len(onsets), descriptions)
        raw.set_annotations(annotations)

    return raw


def _find_stream_by_name(streams, stream_name):
    """Find the first stream that matches the given name."""
    for stream in streams:
        if stream["info"]["name"][0] == stream_name:
            return stream


def _find_stream_by_id(streams, stream_id):
    """Find the stream that matches the given ID."""
    for stream in streams:
        if stream["info"]["stream_id"] == stream_id:
            return stream


def _find_stream_by_type(streams, stream_type="EEG"):
    """Find the first stream that matches the given type."""
    for stream in streams:
        if stream["info"]["type"][0] == stream_type:
            return stream


def _get_ch_info(stream):
    labels, types, units = [], [], []
    if stream["info"]["desc"]:
        for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
            types.append(ch["type"][0])
            units.append(ch["unit"][0])
    return labels, types, units


def _read_varlen_int(f):
    """Read a variable-length integer."""
    nbytes = f.read(1)
    if nbytes == b'\x01':
        return ord(f.read(1))
    elif nbytes == b'\x04':
        return struct.unpack('<I', f.read(4))[0]
    elif nbytes == b'\x08':
        return struct.unpack('<Q', f.read(8))[0]
    elif not nbytes:  # EOF
        raise EOFError
    else:
        raise RuntimeError('Invalid variable-length integer encountered.')


if __name__ == "__main__":
    fnames = glob("/Users/clemens/Downloads/testfiles/*.xdf")
    for fname in fnames:
        print("=" * len(fname) + "\n" + fname + "\n" + "=" * len(fname))
        raw = read_raw_xdf(fname)
        if raw is not None:
            print(raw, end="\n\n")
            print(raw.annotations, end="\n\n")
            # raw.plot(block=True)

        chunks = parse_xdf(fname)

        df = pd.DataFrame.from_dict(chunks)
        df = df[["nbytes", "tag", "stream_id"]]
        df["stream_id"] = df["stream_id"].astype("Int64")
        df["tag"] = pd.Categorical(df["tag"], ordered=True)
        df["tag"].cat.rename_categories(["FileHeader", "StreamHeader", "Samples",
                                         "ClockOffset", "Boundary",
                                         "StreamFooter"], inplace=True)

        print("Chunk table\n-----------")
        print(df, end="\n\n")  # detailed chunk table

        print("Chunk type frequencies\n----------------------")
        print(df["tag"].value_counts().sort_index(), end="\n\n")

        print("Chunks per stream\n-----------------")
        print(df["stream_id"].value_counts().sort_index(), end="\n\n")

        print("Unique stream IDs\n-----------------")
        print(sorted(df["stream_id"].dropna().unique()), end="\n\n")
