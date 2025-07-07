# Annika Balaji + Rutwik Sureshkumar Devmurari
# BCI Rehabilitation Lab, Catholic University of America

# Step 1: LOADING FUCTIONS
# The first step in EEG processing and analysis is to load the data.
# The following code snippets demonstrate how to load EEG data from various file formats using MNE-Python Library

# Imports (Import the necessary libraries into your Python environment as well)
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import mne
import sklearn
import pyedflib
import seaborn as sns
import statsmodels.api as sm
import pywt


#1 Upload the data to your Python environment
#Go to File > Open File... (or press Ctrl+O on Windows/Linux, Cmd+O on macOS).
#Navigate to the location of your data file on your computer.
#Select the file and click "Open".
#The file's content will appear in the editor pane. VS Code will usually provide syntax highlighting based on the file extension (e.g., for JSON, XML, CSV).


#2 Load the data using MNE-Python
# Note: The preload=True argument is used to load the data into memory for faster access.
# Choose the appropriate file format based on your data source, and thus the corresponding loading function.

mne.io.read_raw_edf('sample.edf', preload=True)
# mne.io.read_raw_edf('sample.edf', preload=True).plot()
# This is a sample code to read an EDF file using MNE-Python. Replace 'sample.edf' with your actual file name.

mne.io.read_raw_eeglab('sample.set', preload=True)
# This is a sample code to read an EEGLAB file using MNE-Python. Replace 'sample.set' with your actual file name.

mne.io.read_raw_matlab('sample.mat', preload=True)
# This is a sample code to read a MATLAB file using MNE-Python. Replace 'sample.mat' with your actual file name.

mne.io.read_raw_csv('sample.csv', preload=True)
# This is a sample code to read a CSV file using MNE-Python. Replace 'sample.csv' with your actual file name.

mne.io.read_raw_fif('sample.fif', preload=True)
# This is a sample code to read a FIF file using MNE-Python. Replace 'sample.fif' with your actual file name.

mne.io.read_raw_bti('sample.bti', preload=True)
# This is a sample code to read a BTi file using MNE-Python. Replace 'sample.bti' with your actual file name.



