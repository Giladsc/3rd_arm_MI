#%%

import mne
import os,numpy as np,pandas as pd
import pathlib
from mne_import_xdf import *

# Define paths: 
current_path = pathlib.Path().absolute()  
recording_path = current_path / 'Recordings'

Raw=read_raw_xdf(recording_path / 'sub-Noam_ses-2_task-mi_run-003_eeg.xdf')

# %%
Raw.plot()
# %%
