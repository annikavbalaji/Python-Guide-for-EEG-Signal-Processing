import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import os

# Load EEG data from EDF (background, before arithmetic task)
edf_path = '../Downloads/Subject00_1.edf'
raw = mne.io.read_raw_edf(edf_path, preload=True)
print(raw.info)

# Preprocessing functions
def bandpass_filter(raw, l_freq=1, h_freq=30):
    return raw.filter(l_freq, h_freq, fir_design='firwin')

def rereference(raw, ref='average'):
    return raw.set_eeg_reference(ref)

def run_ica(raw, n_components=None, random_state=42):
    n_chan = len(raw.ch_names)
    if n_components is None or n_components > n_chan:
        n_components = n_chan
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw)
    return ica

def apply_ica(raw, ica, exclude=None):
    if exclude is not None:
        ica.exclude = exclude
    return ica.apply(raw.copy())

def downsample(raw, sfreq):
    return raw.resample(sfreq)

def interpolate_bads(raw):
    raw.interpolate_bads(reset_bads=True)
    return raw

# Preprocessing pipeline
raw = bandpass_filter(raw)
raw = rereference(raw)
ica = run_ica(raw)
raw = apply_ica(raw, ica)
raw = downsample(raw, 128)
raw.info['bads'] = []
raw = interpolate_bads(raw)

# Save preprocessed data
preproc_fif = 'Subject00_1_preprocessed_raw.fif'
raw.save(preproc_fif, overwrite=True)
print(f'Preprocessed data saved to {preproc_fif}')

# Analysis functions
def plot_psd(raw):
    fig = raw.plot_psd(fmax=30, show=False)
    plt.savefig('Subject00_1_psd.png')
    plt.close(fig)

def pearson_corr_matrix(raw):
    data = raw.get_data()
    corr = np.corrcoef(data)
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, xticklabels=raw.ch_names, yticklabels=raw.ch_names, cmap='coolwarm', center=0)
    plt.title('Pearson Correlation Matrix')
    plt.tight_layout()
    plt.savefig('Subject00_1_pearson_corr.png')
    plt.close()
    return corr

def granger_causality_matrix(raw, maxlag=5):
    data = raw.get_data()
    n_ch = data.shape[0]
    gc_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j:
                try:
                    test = grangercausalitytests(data[[i, j], :].T, maxlag=maxlag, verbose=False)
                    gc_matrix[i, j] = np.min([test[lag][0]['ssr_ftest'][1] for lag in test])
                except Exception:
                    gc_matrix[i, j] = np.nan
    plt.figure(figsize=(14, 12))
    sns.heatmap(gc_matrix, xticklabels=raw.ch_names, yticklabels=raw.ch_names, cmap='magma')
    plt.title('Granger Causality Matrix (min p-value)')
    plt.tight_layout()
    plt.savefig('Subject00_1_granger_matrix.png')
    plt.close()
    return gc_matrix

# Run analysis and save results
plot_psd(raw)
pearson_corr = pearson_corr_matrix(raw)
granger_matrix = granger_causality_matrix(raw, maxlag=5)
np.save('Subject00_1_pearson_corr.npy', pearson_corr)
np.save('Subject00_1_granger_matrix.npy', granger_matrix)
print('Analysis complete. Results saved as PNG and NPY files.')
