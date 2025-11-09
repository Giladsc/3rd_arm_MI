import os,pandas as pd
import numpy as np
import pathlib

#%%
def check_thresholds(thresholded_prediction, thresholds):
    """
    Check if each class's prediction is above the designated threshold for each instance.

    Args:
        thresholded_prediction (np.ndarray): The decision function output for the instances (shape: (n_instances, 3)).
        thresholds (np.ndarray): The thresholds for each class (shape: (3,)).

    Returns:
        List[str]: List of strings indicating the classification for each instance.
    """
    if thresholded_prediction.shape[1] != thresholds.shape[0]:
        raise ValueError("Number of classes in thresholded_prediction and thresholds must be the same.")

    # Perform thresholding
    bool_predictions = thresholded_prediction > thresholds
    
    results = []
    for pred in bool_predictions:
        if np.array_equal(pred, [True, False, False]):
            results.append('Left')
        elif np.array_equal(pred, [False, True, False]):
            results.append('Rest')
        elif np.array_equal(pred, [False, False, True]):
            results.append('Right')
        elif np.array_equal(pred, [True, False, True]):
            results.append('MI')
        elif np.array_equal(pred, [False, False, False]):
            results.append('Nothing')
        else:
            results.append('Indecisive')
    
    return results

#%%

# Function to apply band-pass filter
def filter_band(raw, band):
    LowPass, HighPass = band
    return raw().filter(LowPass, HighPass, method='fir', copy = True)
#%%

# Function to add an annotation to the stream_annotations
def add_stream_annotation(onset, description, duration=0):
    new_annotation = mne.Annotations(onset=[onset], duration=[duration], description=[description])
    stream_annotations += new_annotation  # This merges the new annotation with the existing ones
#%%
def gate_chunk(data_volts):
    """
    Quick artifact gate.
    data_volts : ndarray (n_channels, n_samples) in VOLTS
    Returns: accept (bool), reason (str or None)
    """
    abs_max = np.max(np.abs(data_volts), axis=1)   # per channel
    stds    = np.std(data_volts, axis=1, ddof=0)

    bad_abs  = abs_max > (MAX_ABS_UV * UV2V)
    bad_flat = stds   < (MIN_STD_UV * UV2V)
    bad_emg  = stds   > (MAX_STD_UV * UV2V)
    bad      = bad_abs | bad_flat | bad_emg
    bad_frac = bad.mean()
    if bad_frac > MAX_BAD_FRAC:
        return False, f"reject: {bad_frac*100:.1f}% bad channels"
    return True, None