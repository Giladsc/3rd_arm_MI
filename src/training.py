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
from mne.decoding import CSP, Vectorizer
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

# XDF file format support in MNE
import pyxdf
from .mne_import_xdf import *

# Scikit-learn and Pyriemann for feature extraction and machine learning functionalities
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn, Covariances
from pyriemann.tangentspace import TangentSpace,FGDA
from pyriemann.classification import MDM,FgMDM


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
from sklearn.base import BaseEstimator, TransformerMixin


class PairwiseCSP(BaseEstimator, TransformerMixin):
    """One-vs-One CSP: fits a separate CSP for every pair of classes,
    then concatenates their log-variance features.

    Parameters
    ----------
    n_components : int
        Number of CSP components *per class pair*.
    reg : str | None
        Covariance regularization passed to each MNE CSP instance.
    log : bool
        Whether to apply log-variance transformation.
    norm_trace : bool
        Whether to normalise the covariance trace.
    """

    def __init__(self, n_components=4, reg='oas', log=True, norm_trace=True):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.pairs_ = list(itertools.combinations(self.classes_, 2))
        self.csps_ = []

        for c1, c2 in self.pairs_:
            mask = np.isin(y, [c1, c2])
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=self.log,
                norm_trace=self.norm_trace,
            )
            csp.fit(X[mask], y[mask])
            self.csps_.append(csp)
        return self

    def transform(self, X):
        features = [csp.transform(X) for csp in self.csps_]
        return np.hstack(features)


from braindecode.models import ShallowFBCSPNet
from braindecode.training import CroppedLoss
from braindecode.training.scoring import trial_preds_from_window_preds
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

#%%
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

def classifier_training(fold_train_data_x,fold_train_data_y,params_dict, BinaryClassification = False):
    #note that this is currently the  function that really does the classification and extracts the performance measure (the previous calls to run_lda.... for example, are just tests)
    curr_classifier_name=params_dict['pipeline_name']
    csp = lda = None
    if curr_classifier_name=='csp+lda':  
        #define the classifier components:  
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=params_dict['n_components'], reg='oas', log=True, norm_trace=True)
        scaler = StandardScaler() 
        #define the pipeline: 
        clf = Pipeline([('csp',csp),('scaler', scaler), ('classifier_LDA',lda)])
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
        cov = Covariances(estimator="oas")
        ts = TangentSpace()
        scaler = StandardScaler() 
        lda = LinearDiscriminantAnalysis()
        #define the pipeline: 
        clf = Pipeline([('cov',cov),('ts', ts),('scaler', scaler), ('LDA', lda)])
    elif curr_classifier_name == 'ts+FGDA':
        clf = Pipeline([
            ('cov', Covariances(estimator="oas")),
            ('fgda', FGDA(metric='riemann', tsupdate=False)),
            ('ts', TangentSpace()),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis())
        ])

    elif curr_classifier_name == 'MDM':
        clf = Pipeline([
            ('cov', Covariances(estimator="oas")),
            ('mdm', MDM(metric=dict(mean="riemann", distance="riemann")))
        ])

    elif curr_classifier_name == 'FgMDM':
        clf = Pipeline([
            ('cov', Covariances(estimator="oas")),
            ('fgmdm', FgMDM(metric='riemann', tsupdate=False, n_jobs=1))
        ])
    elif curr_classifier_name=='csp_ovo+lda':
        pairwise_csp = PairwiseCSP(n_components=params_dict['n_components'], reg='oas', log=True, norm_trace=True)
        scaler = StandardScaler()
        lda = LinearDiscriminantAnalysis()
        clf = Pipeline([('pairwise_csp', pairwise_csp), ('scaler', scaler), ('classifier_LDA', lda)])
    elif curr_classifier_name=='csp_ovo+svm':
        pairwise_csp = PairwiseCSP(n_components=params_dict['n_components'], reg=None, log=True, norm_trace=False)
        clf = Pipeline([('pairwise_csp', pairwise_csp), ('ovo_svm', OneVsOneClassifier(SVC(kernel='linear', random_state=42)))])
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
    triggers_label_dict={val:key for key,val in params_dict['events_trigger_dict'].items()} 
    fold_train_data_y_labels=np.array([triggers_label_dict[cur_y] for cur_y in fold_train_data_y])  
    A, B , C = 'RightHand','LeftHand', 'MiddleHand'  # Replace with actual trigger names/values
    combined_labels_train = np.array(['motor_imagery' if label in [A, B,C] else label for label in fold_train_data_y_labels])
    
    
    # Define class weights based on class distribution
    class_weights = {'Rest': 2, 'ActiveRest': 6, 'OpenPalm': 6, 'MiddleHand': 6}

    # Assign a sample weight to each sample based on its class
    #sample_weights = np.array([class_weights[cls] for cls in combined_labels_train])
    if BinaryClassification:
        clf.fit(fold_train_data_x, combined_labels_train)
    else:
        clf.fit(fold_train_data_x, fold_train_data_y_labels)
    # running classifier: test classifier on sliding window
    return clf,csp,lda

def compute_block_weights(block_epoch_counts, alpha=0.7):
    """Compute per-trial sample weights with exponential decay over blocks.

    More recent blocks receive higher weight. Weights are normalised so that
    the mean weight equals 1.0 (preserves the effective sample size signal).

    Parameters
    ----------
    block_epoch_counts : list of int
        Number of trials in each block, in chronological order.
        e.g. [20, 18, 22] for 3 blocks.
    alpha : float in (0, 1)
        Decay factor. Older blocks are weighted by alpha^k where k is the
        number of blocks ago. alpha=1.0 means equal weights (no decay).

    Returns
    -------
    sample_weight : np.ndarray, shape (sum(block_epoch_counts),)
    """
    n_blocks = len(block_epoch_counts)
    weights = []
    for k, n_trials in enumerate(block_epoch_counts):
        block_age = n_blocks - 1 - k
        w = alpha ** block_age
        weights.extend([w] * n_trials)
    sample_weight = np.array(weights, dtype=float)
    sample_weight /= sample_weight.mean()
    return sample_weight


def resample_by_weights(X, y, sample_weight, random_state=42):
    """Resample a dataset by duplicating trials proportional to their weights.

    Model-agnostic alternative to sample_weight — works with any estimator.
    Total dataset size is approximately preserved (exactly preserved when all
    weights are equal).

    Per-trial duplicate counts use stochastic rounding: floor(norm_weight)
    copies, plus one more with probability equal to the fractional part.
    This is unbiased (E[count] == norm_weight) and reduces to an exact,
    deterministic 1-copy-each pass-through when weights are uniform.

    NOTE: an earlier version of this function instead took floor(norm_weight)
    copies and then deterministically topped up the exact remainder count by
    giving one extra copy to the highest-fraction trials. For weights close to
    1.0 (e.g. mean-normalised exponential block-decay weights with few
    blocks), every count floored to 0 or 1 and that top-up step always ended
    up handing back out exactly the deficit it created — silently collapsing
    to "1 copy each" regardless of the actual weight values. Independent
    per-trial stochastic rounding avoids that collapse.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, ...)
    y : np.ndarray, shape (n_trials,)
    sample_weight : np.ndarray, shape (n_trials,)
    random_state : int

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    norm_weights = sample_weight / sample_weight.sum() * n
    counts = np.floor(norm_weights).astype(int)
    fractions = norm_weights - counts
    counts = counts + (rng.random(n) < fractions).astype(int)
    indices = np.repeat(np.arange(n), counts)
    rng.shuffle(indices)
    return X[indices], y[indices]


def assert_same_label_space(y, classes):
    diff = np.setdiff1d(np.unique(y), classes)
    assert diff.size == 0, f"Test labels {diff.tolist()} not in trained classes {classes.tolist()}"
def run_windowed_pretrained_classifier(clf, x_uncropped, y, w_start, w_length, verbose=False):
    scores_per_time_window = []
    confusion_matrices_per_window = []
    classes = np.array(clf.classes_)
    if verbose:
        print("Class order in confusion matrix:", classes)

    # Sanity: label space match
    assert_same_label_space(y, classes)

    for n in w_start:
        fold_data = _slice_window_block(x_uncropped, n, w_length)  # keeps rank consistent
        y_pred = clf.predict(fold_data)
        scores_per_time_window.append(np.mean(y_pred == y))
        cm = confusion_matrix(y, y_pred, labels=classes)
        confusion_matrices_per_window.append((cm, classes))

    return scores_per_time_window, confusion_matrices_per_window
def _slice_window_block(x4, start, length):
    # x4 expected: (epochs, ch, T, filters)  or (epochs, ch, T) -> promote to 4D
    if x4.ndim == 3:
        x4 = x4[..., None]  # add filterbank dim
    window = x4[:, :, start:start+length, :]          # (E, C, L, F)
    if window.shape[-1] == 1:
        window = window[..., 0]                       # back to (E, C, L) if no FB
    return window

def run_windowed_classification_aug_cv(epochs, epochs_cropped, cv_split, params_dict, BinaryClassification=False, labels_override=None):
    """
    Train-and-evaluate windowed classifiers with in-fold augmentation, avoiding leakage.

    epochs:            MNE Epochs (UNCROPPED) used for windowed testing (x_test_uncropped)
    epochs_cropped:    MNE Epochs (CROPPED)  used for training/augmentation source
    cv_split:          iterable of (train_idx, test_idx)
    params_dict:       dict with keys:
                       - 'augmentation_params': {'win_len': seconds, 'win_step': seconds}
                       - 'windowed_prediction_params': {'win_len': seconds, 'win_step': seconds}
                       - 'epoch_tmin': float
                       - 'pipeline_name', etc. (used by classifier_training)
                       - optionally 'binary_positive_classes' for BinaryClassification
    BinaryClassification: bool
    """
    from .preprocessing import augment_data

    augmentation_params = params_dict['augmentation_params']
    windowed_prediction_params = params_dict['windowed_prediction_params']
    win_len = float(windowed_prediction_params['win_len'])
    win_step = float(windowed_prediction_params['win_step'])

    # Pull data once
    epochs_cropped_data = epochs_cropped.get_data()  # training/augmentation view
    epochs_data = epochs.get_data()                  # windowed test view

    # ---------- Alignment & consistency checks ----------
    assert epochs.events.shape[0] == epochs_cropped.events.shape[0], "Epoch count mismatch"
    # Ensure identical trial order & labels (last col usually encodes event id)
    assert np.all(epochs.events[:, -1] == epochs_cropped.events[:, -1]), "Mismatch epochs vs epochs_cropped"

    # Use sfreq from the object you actually window (epochs)
    sfreq = epochs.info['sfreq']

    # Window params in samples
    w_length = int(round(sfreq * win_len))
    w_step_samp = int(round(sfreq * win_step))

    # Defensive: window step must be >= 1 and length must be <= trial length
    assert w_length > 0, f"win_len too small; got {win_len}s -> {w_length} samples"
    assert w_step_samp >= 1, f"win_step too small; got {win_step}s -> {w_step_samp} samples"
    assert epochs_data.shape[2] >= w_length, "Trials shorter than window length"

    # Start indices INCLUDING the last window that fits exactly
    # shape[2] = n_times; valid starts are [0 .. n_times - w_length]
    w_start = np.arange(0, epochs_data.shape[2] - w_length + 1, w_step_samp)

    scores_windows = []
    folds_confusion_matrices_per_window = []

    # ---------- Cross-validation ----------
    for train_idx, test_idx in cv_split:
        # Labels (ints) per fold
        _labels = labels_override if labels_override is not None else epochs_cropped.events[:, -1]
        y_train = _labels[train_idx]
        y_test  = _labels[test_idx]

        # Training features slice (support 3D or 4D with filterbanks)
        if epochs_cropped_data.ndim == 3:
            x_train_source = epochs_cropped_data[train_idx, :, :]
        elif epochs_cropped_data.ndim == 4:
            x_train_source = epochs_cropped_data[train_idx, :, :, :]
        else:
            raise ValueError(f"Unexpected epochs_cropped_data ndim={epochs_cropped_data.ndim}")

        # --- Augment TRAIN ONLY (no leakage) ---
        augmented_x, augmented_y = augment_data(augmentation_params, x_train_source, y_train, sfreq)

        # --- Fit classifier INSIDE the fold ---
        clf, _, _ = classifier_training(
            augmented_x,
            augmented_y,
            params_dict,
            BinaryClassification=BinaryClassification  # propagate flag correctly
        )

        # --- Windowed evaluation on the UNCROPPED TEST trials ---
        x_test_uncropped = epochs_data[test_idx, ...]
        curr_scores_windows, confusion_matrices_per_window = run_windowed_classification_on_fold(
            augmented_x,                 # (not used for predicting; kept for signature parity)
            augmented_y,                 # (same)
            x_test_uncropped,
            y_test,
            params_dict,
            w_start,
            w_length,
            clf,
            BinaryClassification=BinaryClassification
        )

        scores_windows.append(curr_scores_windows)
        folds_confusion_matrices_per_window.append(confusion_matrices_per_window)


    
    # Window end times (s) relative to epoch_tmin — time when prediction is made after processing full window
    w_times = (w_start + w_length ) / sfreq + params_dict['epoch_tmin']
    
    return scores_windows, folds_confusion_matrices_per_window, w_times


def run_permutation_test(
    epochs,
    epochs_cropped,
    params_dict,
    n_permutations=100,
    score_method='majority_vote',
    eval_tmin=0.0,
    eval_tmax=5.0,
    BinaryClassification=False,
    random_state=None,
):
    """
    Validate classifier performance against a null distribution of shuffled labels.

    Runs the full CV pipeline N times with randomly permuted labels to build a
    null distribution, then computes a p-value as the fraction of permutations
    that matched or exceeded the true accuracy.

    Parameters
    ----------
    epochs : mne.Epochs          Uncropped epochs.
    epochs_cropped : mne.Epochs  Cropped epochs used for training.
    params_dict : dict           Pipeline configuration.
    n_permutations : int         Number of label shuffles. Default 100; use >=1000 for publication.
    score_method : str
        'majority_vote' (default) — uses run_full_epoch_classification_cv: majority vote across
            windows in [eval_tmin, eval_tmax], one prediction per trial. Matches what
            run_full_epoch_classification_cv returns and gives a score comparable to what you
            see when calling that function directly.
        'windowed_mean' — uses run_windowed_classification_aug_cv: mean accuracy across all
            windows and folds. Lower than majority_vote for the same data.
    eval_tmin : float  Start of the evaluation window (s, relative to event onset). Default 0.0.
    eval_tmax : float  End of the evaluation window (s). Default 5.0.
    BinaryClassification : bool  Passed through to the CV function.
    random_state : int or None   Seeds numpy RNG for reproducibility.

    Returns
    -------
    true_score : float           Scalar accuracy from real labels.
    perm_scores : np.ndarray     Shape (n_permutations,) — accuracy for each permuted run.
    p_value : float              (count >= true_score + 1) / (n_permutations + 1).
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        from tqdm import tqdm as _tqdm

    if score_method not in ('majority_vote', 'windowed_mean'):
        raise ValueError(f"score_method must be 'majority_vote' or 'windowed_mean', got {score_method!r}")

    rng = np.random.default_rng(random_state)
    true_labels = epochs_cropped.events[:, -1].copy()
    X_for_split = epochs_cropped.get_data()

    def _run_and_score(cv_split_iter, labels_override=None):
        """Returns (scalar_score, fold_confusion_matrices or None)."""
        if score_method == 'majority_vote':
            fold_accs, fold_cms = run_full_epoch_classification_cv(
                epochs, epochs_cropped, cv_split_iter, params_dict,
                tmin=eval_tmin, tmax=eval_tmax,
                BinaryClassification=BinaryClassification,
                labels_override=labels_override,
            )
            return float(np.mean(fold_accs)), fold_cms
        else:
            scores_windows, fold_cms_per_window, w_times = run_windowed_classification_aug_cv(
                epochs, epochs_cropped, cv_split_iter, params_dict,
                BinaryClassification=BinaryClassification,
                labels_override=labels_override,
            )
            arr = np.array(scores_windows)
            lo, hi = eval_tmin, eval_tmax
            mask = (w_times >= lo) & (w_times <= hi)
            if not mask.any():
                raise ValueError(
                    f"No windows in [{lo}, {hi}] s. "
                    f"w_times range: [{w_times[0]:.2f}, {w_times[-1]:.2f}]"
                )
            return float(np.mean(arr[:, mask])), fold_cms_per_window

    print(f"Starting permutation test: {n_permutations} permutations × 10 CV folds "
          f"(score_method='{score_method}', eval window [{eval_tmin}, {eval_tmax}] s). "
          f"Expected runtime ≈ {n_permutations}× a single CV run.")

    cv_true = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)
    true_score, true_fold_cms = _run_and_score(cv_true.split(X_for_split, true_labels))

    def _cms_to_flat(fold_cms):
        """Yield every (cm, classes) pair regardless of format (flat or windowed)."""
        for item in fold_cms:
            if isinstance(item, tuple) and isinstance(item[0], np.ndarray):
                yield item                          # majority_vote: (cm, classes)
            else:
                for sub in item:                    # windowed_mean: (cm, classes) per window
                    yield sub

    perm_scores = np.zeros(n_permutations)
    perm_cm_sum = None
    perm_classes = None
    for i in _tqdm(range(n_permutations), desc="Permutations", unit="perm"):
        perm_labels = rng.permutation(true_labels)
        cv_perm = ShuffleSplit(10, test_size=0.2, random_state=int(rng.integers(0, 2**31)))
        perm_scores[i], perm_fold_cms = _run_and_score(cv_perm.split(X_for_split), labels_override=perm_labels)
        for cm, classes in _cms_to_flat(perm_fold_cms):
            if perm_cm_sum is None:
                perm_cm_sum = cm.astype(float).copy()
                perm_classes = list(classes)
            else:
                perm_cm_sum += cm

    p_value = (np.sum(perm_scores >= true_score) + 1) / (n_permutations + 1)
    return true_score, perm_scores, p_value, true_fold_cms, perm_cm_sum, perm_classes


def run_windowed_classification_aug(epochs_cropped,train_set_data,train_set_labels,train_set_data_uncroped,test_y,params_dict,BinaryClassification):
    augmentation_params=params_dict['augmentation_params']
    windowed_prediction_params=params_dict['windowed_prediction_params']
    win_len=windowed_prediction_params['win_len']
    win_step=windowed_prediction_params['win_step']
    sfreq = epochs_cropped.info['sfreq']
    w_length = int(sfreq * win_len)   # running classifier: window length
    w_step = int(sfreq * win_step)  # running classifier: window step size
    w_start = np.arange(0, train_set_data_uncroped.shape[2] - w_length, w_step)
    # Window end times (s) relative to epoch_tmin — time when prediction is made after processing full window
    w_times = (w_start + w_length ) / sfreq + params_dict['epoch_tmin']

    augmented_x,augmented_y=augment_data(augmentation_params,train_set_data,train_set_labels,sfreq)
    scores_windows,confusion_metrices_per_window,trained_clf=run_windowed_classification_on_fold(augmented_x,augmented_y,train_set_data_uncroped,test_y,params_dict,w_start,w_length, BinaryClassification)         

    return scores_windows,confusion_metrices_per_window,w_times,trained_clf


# %%
def run_windowed_classification_on_fold(fold_train_data_x,fold_train_data_y,fold_test_data_x_uncropped,fold_test_data_y,params_dict,w_start,w_length, clf, BinaryClassification = False):
    triggers_label_dict={val:key for key,val in params_dict['events_trigger_dict'].items()} 
    fold_test_data_y_labels=np.array([triggers_label_dict[cur_y] for cur_y in fold_test_data_y])
    if BinaryClassification:
        A, B , C = 'RightHand','LeftHand', 'MiddleHand'  # Replace with actual trigger names/values
        combined_labels_test = np.array(['motor_imagery' if label in [A, B, C] else label for label in fold_test_data_y_labels])
        fold_windowed_scores,confusion_matrices_per_window=run_windowed_pretrained_classifier(clf,fold_test_data_x_uncropped,combined_labels_test,w_start,w_length)
    else:
        fold_windowed_scores,confusion_matrices_per_window=run_windowed_pretrained_classifier(clf,fold_test_data_x_uncropped,fold_test_data_y_labels,w_start,w_length)
    return fold_windowed_scores,confusion_matrices_per_window


def sanity_check_trained_clf(trained_clf, epochs, params_dict, BinaryClassification=False):
    """
    Sanity-check a pre-trained classifier using the windowed prediction approach
    on the provided (uncropped) epochs — same method used in CV evaluation.

    Parameters
    ----------
    trained_clf : fitted sklearn Pipeline / classifier
        The classifier to evaluate (e.g. from classifier_training()).
    epochs : mne.Epochs
        UNCROPPED epochs to evaluate on (the full trial length).
    params_dict : dict
        Must contain:
        - 'windowed_prediction_params': {'win_len': float, 'win_step': float}
        - 'epoch_tmin': float
        - 'events_trigger_dict': {str: int} mapping event names to trigger codes
    BinaryClassification : bool
        If True, collapse MI classes into a single 'motor_imagery' label.

    Returns
    -------
    scores_windows : list of float
        Accuracy at each time window.
    confusion_matrices_per_window : list of (cm, classes) tuples
        Confusion matrix and class labels at each time window.
    w_times : np.ndarray
        Time (in seconds, relative to epoch onset) for each window.
    """
    windowed_prediction_params = params_dict['windowed_prediction_params']
    win_len = float(windowed_prediction_params['win_len'])
    win_step = float(windowed_prediction_params['win_step'])

    sfreq = epochs.info['sfreq']
    epochs_data = epochs.get_data()

    w_length = int(round(sfreq * win_len))
    w_step_samp = int(round(sfreq * win_step))
    w_start = np.arange(0, epochs_data.shape[2] - w_length + 1, w_step_samp)

    # Map integer trigger codes to string labels
    triggers_label_dict = {val: key for key, val in params_dict['events_trigger_dict'].items()}
    y_labels = np.array([triggers_label_dict[code] for code in epochs.events[:, -1]])

    if BinaryClassification:
        A, B, C = 'RightHand', 'LeftHand', 'MiddleHand'
        y_labels = np.array(['motor_imagery' if label in [A, B, C] else label for label in y_labels])

    scores_windows, confusion_matrices_per_window = run_windowed_pretrained_classifier(
        trained_clf, epochs_data, y_labels, w_start, w_length
    )

    w_times = (w_start + w_length) / sfreq + params_dict['epoch_tmin']

    # Wrap in a list to match the CV output format (list of folds)
    return [scores_windows], [confusion_matrices_per_window], w_times


def evaluate_full_epoch(clf, epochs, params_dict, tmin=0.0, tmax=5.0, BinaryClassification=False):
    """
    Classify each trial using majority vote across all sliding windows within [tmin, tmax].

    The classifier sees the same window size it was trained on; the final
    per-trial label is decided by a majority vote across every window that
    fits inside the specified epoch range.  This gives one decision per
    trial rather than one score per time window.

    Parameters
    ----------
    clf : fitted sklearn pipeline / classifier
        Must expose `.predict()` and `.classes_`.
    epochs : mne.Epochs
        UNCROPPED epochs to evaluate on (must cover [tmin, tmax]).
    params_dict : dict
        Must contain:
        - 'windowed_prediction_params': {'win_len': float, 'win_step': float}
        - 'epoch_tmin': float  (start of the epoch relative to event onset)
        - 'events_trigger_dict': {str: int}
    tmin, tmax : float
        Time range in seconds, relative to event onset (t=0), to use for
        the majority vote.  Default 0–5 s (the MI execution window).
    BinaryClassification : bool
        Collapse MI classes into 'motor_imagery' if True.

    Returns
    -------
    accuracy : float
        Fraction of trials correctly classified by majority vote.
    cm_tuple : (np.ndarray, np.ndarray)
        (confusion_matrix, class_labels).
    trial_predictions : np.ndarray of str
        Majority-vote label for every trial.
    """
    windowed_prediction_params = params_dict['windowed_prediction_params']
    win_len = float(windowed_prediction_params['win_len'])
    win_step = float(windowed_prediction_params['win_step'])

    sfreq = epochs.info['sfreq']
    epoch_tmin = params_dict['epoch_tmin']
    epochs_data = epochs.get_data()

    # Convert tmin/tmax to sample indices relative to the start of the stored epoch
    start_sample = int(round((tmin - epoch_tmin) * sfreq))
    end_sample = int(round((tmax - epoch_tmin) * sfreq))
    start_sample = max(0, start_sample)
    end_sample = min(epochs_data.shape[2], end_sample)

    w_length = int(round(sfreq * win_len))
    w_step_samp = int(round(sfreq * win_step))

    w_start = np.arange(start_sample, end_sample - w_length + 1, w_step_samp)
    if len(w_start) == 0:
        raise ValueError(
            f"No windows fit in [{tmin}, {tmax}]s with win_len={win_len}s. "
            f"Sample range [{start_sample}, {end_sample}], w_length={w_length}."
        )

    # Map integer trigger codes to string labels
    triggers_label_dict = {val: key for key, val in params_dict['events_trigger_dict'].items()}
    y_labels = np.array([triggers_label_dict[code] for code in epochs.events[:, -1]])
    if BinaryClassification:
        A, B, C = 'RightHand', 'LeftHand', 'MiddleHand'
        y_labels = np.array(['motor_imagery' if lbl in [A, B, C] else lbl for lbl in y_labels])

    classes = np.array(clf.classes_)
    assert_same_label_space(y_labels, classes)

    # Collect per-trial predictions from every window → shape (n_windows, n_trials)
    all_window_preds = []
    for n in w_start:
        window_data = _slice_window_block(epochs_data, n, w_length)
        all_window_preds.append(clf.predict(window_data))

    all_window_preds = np.array(all_window_preds).T  # (n_trials, n_windows)

    # Majority vote: one label per trial (np.unique works on string arrays)
    trial_preds = np.array([
        vals[np.argmax(counts)]
        for vals, counts in (np.unique(row, return_counts=True) for row in all_window_preds)
    ])

    accuracy = float(np.mean(trial_preds == y_labels))
    cm = confusion_matrix(y_labels, trial_preds, labels=classes)

    return accuracy, (cm, classes), trial_preds


def run_full_epoch_classification_cv(epochs, epochs_cropped, cv_split, params_dict,
                                      tmin=0.0, tmax=5.0, BinaryClassification=False,
                                      labels_override=None):
    """
    CV loop that trains on the cropped window (with augmentation) and evaluates
    each test trial using majority vote across all windows in [tmin, tmax].

    Returns per-fold trial-level accuracy and confusion matrices rather than
    per-window scores.  Use this alongside (or instead of)
    run_windowed_classification_aug_cv when you want a single classification
    decision per epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        UNCROPPED epochs (used for test evaluation).
    epochs_cropped : mne.Epochs
        CROPPED epochs (used for training / augmentation).
    cv_split : iterable of (train_idx, test_idx)
    params_dict : dict
        Same structure as for run_windowed_classification_aug_cv.
    tmin, tmax : float
        Epoch range (seconds, relative to event onset) to use for the
        majority-vote classification.  Default 0–5 s.
    BinaryClassification : bool

    Returns
    -------
    fold_accuracies : list of float
        One accuracy per CV fold.
    fold_confusion_matrices : list of (cm, classes)
        One confusion matrix per CV fold.
    """
    from .preprocessing import augment_data

    augmentation_params = params_dict['augmentation_params']
    windowed_prediction_params = params_dict['windowed_prediction_params']
    win_len = float(windowed_prediction_params['win_len'])
    win_step = float(windowed_prediction_params['win_step'])

    sfreq = epochs.info['sfreq']
    epoch_tmin = params_dict['epoch_tmin']

    epochs_data = epochs.get_data()
    epochs_cropped_data = epochs_cropped.get_data()

    assert epochs.events.shape[0] == epochs_cropped.events.shape[0], "Epoch count mismatch"
    assert np.all(epochs.events[:, -1] == epochs_cropped.events[:, -1]), "Label mismatch epochs vs epochs_cropped"

    w_length = int(round(sfreq * win_len))
    w_step_samp = int(round(sfreq * win_step))

    # Sample range for [tmin, tmax]
    start_sample = max(0, int(round((tmin - epoch_tmin) * sfreq)))
    end_sample = min(epochs_data.shape[2], int(round((tmax - epoch_tmin) * sfreq)))
    w_start_eval = np.arange(start_sample, end_sample - w_length + 1, w_step_samp)

    if len(w_start_eval) == 0:
        raise ValueError(
            f"No windows fit in [{tmin}, {tmax}]s with win_len={win_len}s."
        )

    triggers_label_dict = {val: key for key, val in params_dict['events_trigger_dict'].items()}

    fold_accuracies = []
    fold_confusion_matrices = []

    for train_idx, test_idx in cv_split:
        _labels = labels_override if labels_override is not None else epochs_cropped.events[:, -1]
        y_train = _labels[train_idx]
        y_test = _labels[test_idx]

        if epochs_cropped_data.ndim == 3:
            x_train_source = epochs_cropped_data[train_idx]
        else:
            x_train_source = epochs_cropped_data[train_idx]

        augmented_x, augmented_y = augment_data(augmentation_params, x_train_source, y_train, sfreq)

        clf, _, _ = classifier_training(augmented_x, augmented_y, params_dict,
                                        BinaryClassification=BinaryClassification)

        x_test = epochs_data[test_idx]
        y_test_labels = np.array([triggers_label_dict[c] for c in y_test])
        if BinaryClassification:
            A, B, C = 'RightHand', 'LeftHand', 'MiddleHand'
            y_test_labels = np.array(['motor_imagery' if lbl in [A, B, C] else lbl
                                       for lbl in y_test_labels])

        classes = np.array(clf.classes_)
        assert_same_label_space(y_test_labels, classes)

        # Collect window predictions for test trials
        all_window_preds = []
        for n in w_start_eval:
            window_data = _slice_window_block(x_test, n, w_length)
            all_window_preds.append(clf.predict(window_data))

        all_window_preds = np.array(all_window_preds).T  # (n_test_trials, n_windows)

        trial_preds = np.array([
            vals[np.argmax(counts)]
            for vals, counts in (np.unique(row, return_counts=True) for row in all_window_preds)
        ])

        fold_acc = float(np.mean(trial_preds == y_test_labels))
        cm = confusion_matrix(y_test_labels, trial_preds, labels=classes)

        fold_accuracies.append(fold_acc)
        fold_confusion_matrices.append((cm, classes))

    return fold_accuracies, fold_confusion_matrices


def run_adaptive_cv(all_epochs, adaptive_block_epochs, cv_split, params_dict,
                    alpha=0.7, tmin=0.0, tmax=5.0, BinaryClassification=False):
    """
    CV evaluation for the co-adaptive classifier with block-weighted training.

    Replicates the block-weighted augment-and-resample training from the online
    retraining cell inside a proper CV loop so each test fold is held out from
    the weighted training set.

    Parameters
    ----------
    all_epochs : mne.Epochs
        Concatenation of all adaptive_block_epochs (same trial order, uncropped).
    adaptive_block_epochs : list of mne.Epochs
        One Epochs object per block, in chronological order.
    cv_split : iterable of (train_idx, test_idx)
    params_dict : dict
        Must contain 'classifier_window_s', 'classifier_window_e',
        'augmentation_params', 'windowed_prediction_params', 'epoch_tmin',
        'events_trigger_dict'.
    alpha : float
        Exponential decay factor (same value used in the online retraining cell).
    tmin, tmax : float
        Epoch range (s, relative to event onset) for majority-vote evaluation.
    BinaryClassification : bool

    Returns
    -------
    fold_accuracies : list of float
    fold_confusion_matrices : list of (cm, classes)
        Same format as run_full_epoch_classification_cv — one entry per fold.
    """
    from .preprocessing import augment_data

    block_counts = [len(e) for e in adaptive_block_epochs]
    sample_weight_all = compute_block_weights(block_counts, alpha=alpha)

    sfreq = all_epochs.info['sfreq']
    epoch_tmin = params_dict['epoch_tmin']

    epochs_data = all_epochs.get_data()
    epochs_cropped_data = all_epochs.copy().crop(
        tmin=params_dict['classifier_window_s'],
        tmax=params_dict['classifier_window_e']
    ).get_data()

    labels = all_epochs.events[:, -1]
    triggers_label_dict = {val: key for key, val in params_dict['events_trigger_dict'].items()}

    win_len = float(params_dict['windowed_prediction_params']['win_len'])
    win_step = float(params_dict['windowed_prediction_params']['win_step'])
    w_length = int(round(sfreq * win_len))
    w_step_samp = int(round(sfreq * win_step))

    start_sample = max(0, int(round((tmin - epoch_tmin) * sfreq)))
    end_sample = min(epochs_data.shape[2], int(round((tmax - epoch_tmin) * sfreq)))
    w_start_eval = np.arange(start_sample, end_sample - w_length + 1, w_step_samp)

    if len(w_start_eval) == 0:
        raise ValueError(f"No windows fit in [{tmin}, {tmax}]s with win_len={win_len}s.")

    fold_accuracies = []
    fold_confusion_matrices = []

    for train_idx, test_idx in cv_split:
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        x_train_source = epochs_cropped_data[train_idx]
        train_weights = sample_weight_all[train_idx]

        augmented_x, augmented_y = augment_data(
            params_dict['augmentation_params'], x_train_source, y_train, sfreq
        )

        aug_factor = len(augmented_x) // len(x_train_source)
        remainder = len(augmented_x) % len(x_train_source)
        aug_weights = np.concatenate([
            np.tile(train_weights, aug_factor),
            train_weights[:remainder]
        ])
        aug_x_r, aug_y_r = resample_by_weights(augmented_x, augmented_y, aug_weights)

        clf, _, _ = classifier_training(aug_x_r, aug_y_r, params_dict,
                                        BinaryClassification=BinaryClassification)

        x_test = epochs_data[test_idx]
        y_test_labels = np.array([triggers_label_dict[c] for c in y_test])
        if BinaryClassification:
            A, B, C = 'RightHand', 'LeftHand', 'MiddleHand'
            y_test_labels = np.array(['motor_imagery' if lbl in [A, B, C] else lbl
                                      for lbl in y_test_labels])

        classes = np.array(clf.classes_)
        assert_same_label_space(y_test_labels, classes)

        all_window_preds = []
        for n in w_start_eval:
            window_data = _slice_window_block(x_test, n, w_length)
            all_window_preds.append(clf.predict(window_data))

        all_window_preds = np.array(all_window_preds).T
        trial_preds = np.array([
            vals[np.argmax(counts)]
            for vals, counts in (np.unique(row, return_counts=True) for row in all_window_preds)
        ])

        fold_acc = float(np.mean(trial_preds == y_test_labels))
        cm = confusion_matrix(y_test_labels, trial_preds, labels=classes)
        fold_accuracies.append(fold_acc)
        fold_confusion_matrices.append((cm, classes))

    return fold_accuracies, fold_confusion_matrices


def run_trial_count_sweep_cv(epochs, epochs_cropped, cv_split, params_dict,
                              n_trials_list=None,
                              n_repeats=5,
                              tmin=0.0, tmax=5.0,
                              BinaryClassification=False,
                              labels_override=None):
    """
    Learning curve sweep: vary the number of training trials and measure accuracy.

    For each value in n_trials_list, draws n_repeats stratified subsamples of that
    size from each CV fold's training set, trains the classifier, and evaluates on
    the same held-out test fold.  Averaging across folds and repeats yields a
    robust accuracy-vs-n_trials learning curve.

    The test set is held constant across all trial counts within each fold so that
    accuracy comparisons are apples-to-apples.

    Parameters
    ----------
    epochs : mne.Epochs
        UNCROPPED epochs used for test evaluation.
    epochs_cropped : mne.Epochs
        CROPPED epochs used for training / augmentation.
    cv_split : iterable of (train_idx, test_idx)
        Must be materialise-able (iterated multiple times internally).
    params_dict : dict
        Same structure as run_full_epoch_classification_cv.
    n_trials_list : list of int or None
        Trial counts to sweep over.  Defaults to [5, 10, 20, 30, 50, 80, 120].
    n_repeats : int
        Random draws per (fold, n_trials) pair.  Ignored when n_trials covers all
        available training data (no randomness).
    tmin, tmax : float
        Epoch range (seconds relative to event onset) for majority-vote evaluation.
    BinaryClassification : bool

    Returns
    -------
    sweep_results : dict
        Keyed by n_trials (int):
        {
          n_trials: {
            'all_accs': list[float],   # one entry per (fold x repeat)
            'mean_acc': float,
            'std_acc':  float,
            'sem_acc':  float,
          }
        }
    """
    from .preprocessing import augment_data
    from scipy.stats import sem as scipy_sem

    if n_trials_list is None:
        n_trials_list = [5, 10, 20, 30, 50, 80, 120]

    augmentation_params = params_dict['augmentation_params']
    windowed_prediction_params = params_dict['windowed_prediction_params']
    win_len = float(windowed_prediction_params['win_len'])
    win_step = float(windowed_prediction_params['win_step'])

    sfreq = epochs.info['sfreq']
    epoch_tmin = params_dict['epoch_tmin']

    epochs_data = epochs.get_data()
    epochs_cropped_data = epochs_cropped.get_data()

    assert epochs.events.shape[0] == epochs_cropped.events.shape[0], "Epoch count mismatch"
    assert np.all(epochs.events[:, -1] == epochs_cropped.events[:, -1]), "Label mismatch"

    w_length = int(round(sfreq * win_len))
    w_step_samp = int(round(sfreq * win_step))

    start_sample = max(0, int(round((tmin - epoch_tmin) * sfreq)))
    end_sample = min(epochs_data.shape[2], int(round((tmax - epoch_tmin) * sfreq)))
    w_start_eval = np.arange(start_sample, end_sample - w_length + 1, w_step_samp)

    if len(w_start_eval) == 0:
        raise ValueError(
            f"No windows fit in [{tmin}, {tmax}]s with win_len={win_len}s."
        )

    triggers_label_dict = {val: key for key, val in params_dict['events_trigger_dict'].items()}
    n_classes = len(params_dict['desired_events'])

    sweep_results = {n: {'all_accs': []} for n in n_trials_list}

    cv_list = list(cv_split)

    for fold_i, (train_idx, test_idx) in enumerate(cv_list):
        _labels = labels_override if labels_override is not None else epochs_cropped.events[:, -1]
        y_train_full = _labels[train_idx]
        y_test = _labels[test_idx]

        x_test = epochs_data[test_idx]
        y_test_labels = np.array([triggers_label_dict[c] for c in y_test])
        if BinaryClassification:
            A, B, C = 'RightHand', 'LeftHand', 'MiddleHand'
            y_test_labels = np.array(
                ['motor_imagery' if lbl in [A, B, C] else lbl for lbl in y_test_labels]
            )

        n_train_available = len(train_idx)

        for n_trials in n_trials_list:
            if n_trials < n_classes:
                # Cannot form a stratified sample with fewer trials than classes
                continue

            if n_trials >= n_train_available:
                # Use all available training data; repeats are identical so run once
                subsample_indices_list = [np.arange(n_train_available)]
            else:
                # Generate n_repeats independent stratified subsamples
                subsample_indices_list = []
                for rep_seed in range(n_repeats):
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_trials,
                                                 random_state=rep_seed)
                    sub_idx, _ = next(sss.split(np.zeros(n_train_available), y_train_full))
                    subsample_indices_list.append(sub_idx)

            for sub_idx in subsample_indices_list:
                x_sub = epochs_cropped_data[train_idx[sub_idx]]
                y_sub = y_train_full[sub_idx]

                augmented_x, augmented_y = augment_data(augmentation_params, x_sub, y_sub, sfreq)

                try:
                    clf, _, _ = classifier_training(augmented_x, augmented_y, params_dict,
                                                    BinaryClassification=BinaryClassification)
                except Exception as e:
                    print(f"  Fold {fold_i + 1}, n_trials={n_trials}: training failed ({e}), skipping")
                    continue

                classes = np.array(clf.classes_)
                assert_same_label_space(y_test_labels, classes)

                all_window_preds = []
                for n in w_start_eval:
                    window_data = _slice_window_block(x_test, n, w_length)
                    all_window_preds.append(clf.predict(window_data))

                all_window_preds = np.array(all_window_preds).T  # (n_test_trials, n_windows)
                trial_preds = np.array([
                    vals[np.argmax(counts)]
                    for vals, counts in (np.unique(row, return_counts=True) for row in all_window_preds)
                ])

                fold_acc = float(np.mean(trial_preds == y_test_labels))
                sweep_results[n_trials]['all_accs'].append(fold_acc)

        print(f"Fold {fold_i + 1}/{len(cv_list)} complete.")

    # Aggregate statistics across all folds and repeats
    for n_trials in n_trials_list:
        accs = sweep_results[n_trials]['all_accs']
        if len(accs) > 0:
            sweep_results[n_trials]['mean_acc'] = float(np.mean(accs))
            sweep_results[n_trials]['std_acc'] = float(np.std(accs))
            sweep_results[n_trials]['sem_acc'] = float(scipy_sem(accs))
        else:
            sweep_results[n_trials]['mean_acc'] = float('nan')
            sweep_results[n_trials]['std_acc'] = float('nan')
            sweep_results[n_trials]['sem_acc'] = float('nan')

    return sweep_results
