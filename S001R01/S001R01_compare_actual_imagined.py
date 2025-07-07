import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import os

# Load preprocessed data
raw = mne.io.read_raw_fif('S001R01_preprocessed_raw.fif', preload=True)

# Extract events and annotations
try:
    events, event_id = mne.events_from_annotations(raw)
    print('Event IDs:', event_id)
    print('Number of events:', len(events))
except Exception as e:
    print('Could not extract events:', e)
    events, event_id = None, None

# Define movement/imagery event IDs (update as needed)
actual_keys = ['actual', 'move', 'movement']
imagined_keys = ['imagined', 'imagine', 'imagery']
actual_ids = [k for k in event_id if any(x in k.lower() for x in actual_keys)] if event_id else []
imagined_ids = [k for k in event_id if any(x in k.lower() for x in imagined_keys)] if event_id else []

# Helper to get epochs for a set of event keys
def get_epochs(keys):
    if not keys:
        return None
    ids = {k: event_id[k] for k in keys}
    return mne.Epochs(raw, events, ids, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)

actual_epochs = get_epochs(actual_ids)
imagined_epochs = get_epochs(imagined_ids)

# Function to compute mean PSD, correlation, and Granger matrix for epochs
def compute_metrics(epochs, label):
    if epochs is None or len(epochs) == 0:
        print(f'No epochs for {label}')
        return None, None, None
    # PSD
    psd, freqs = mne.time_frequency.psd_welch(epochs, fmin=1, fmax=40, n_fft=256)
    mean_psd = psd.mean(axis=0).mean(axis=1)  # mean over epochs and freqs
    # Correlation
    data = epochs.get_data().mean(axis=0)  # mean over epochs
    corr = np.corrcoef(data)
    # Granger
    n_ch = data.shape[0]
    gc_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j:
                try:
                    test = grangercausalitytests(data[[i, j], :].T, maxlag=5, verbose=False)
                    gc_matrix[i, j] = np.min([test[lag][0]['ssr_ftest'][1] for lag in test])
                except Exception:
                    gc_matrix[i, j] = np.nan
    # Save plots
    ch_names = epochs.ch_names
    plt.figure(figsize=(12, 8))
    plt.bar(ch_names, mean_psd)
    plt.title(f'Mean PSD ({label})')
    plt.ylabel('Power')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'S001R01_psd_{label}.png')
    plt.close()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, xticklabels=ch_names, yticklabels=ch_names, cmap='coolwarm', center=0)
    plt.title(f'Pearson Correlation ({label})')
    plt.tight_layout()
    plt.savefig(f'S001R01_corr_{label}.png')
    plt.close()
    plt.figure(figsize=(12, 10))
    sns.heatmap(gc_matrix, xticklabels=ch_names, yticklabels=ch_names, cmap='magma')
    plt.title(f'Granger Causality ({label})')
    plt.tight_layout()
    plt.savefig(f'S001R01_granger_{label}.png')
    plt.close()
    return mean_psd, corr, gc_matrix

actual_metrics = compute_metrics(actual_epochs, 'actual')
imagined_metrics = compute_metrics(imagined_epochs, 'imagined')

print('Analysis complete. Compare the generated PNGs for actual vs. imagined movement.')
