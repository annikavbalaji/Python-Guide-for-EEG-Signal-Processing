import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from statsmodels.tsa.stattools import grangercausalitytests
from mne.preprocessing import ICA
try:
    from mne.connectivity import spectral_connectivity
except ImportError:
    spectral_connectivity = None
    print("WARNING: mne.connectivity.spectral_connectivity is not available in your MNE version. Connectivity analysis will be skipped.")
import os

# For reproducibility
np.random.seed(42)

# Load EEG data from CSV
csv_path = '../Downloads/s01_ex01_s01.csv'
raw_df = pd.read_csv(csv_path)
print('Shape:', raw_df.shape)
print(raw_df.head())
print(raw_df.info())

# Preprocessing functions
def bandpass_filter(raw, l_freq=1, h_freq=40):
    return raw.filter(l_freq, h_freq, fir_design='firwin')

def rereference(raw, ref='average'):
    return raw.set_eeg_reference(ref)

def run_ica(raw, n_components=None, random_state=42):
    # Set n_components to number of channels if not specified or if too high
    n_chan = len(raw.ch_names)
    if n_components is None or n_components > n_chan:
        n_components = n_chan
    ica = ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw)
    return ica

def apply_ica(raw, ica, exclude=None):
    if exclude is not None:
        ica.exclude = exclude
    return ica.apply(raw.copy())

def epoch_data(raw, events, event_id, tmin=-0.2, tmax=0.5):
    return mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)

def baseline_correct(epochs):
    return epochs.apply_baseline((None, 0))

def downsample(raw, sfreq):
    return raw.resample(sfreq)

def select_channels(raw, picks):
    return raw.pick(picks)

def interpolate_bads(raw):
    raw.interpolate_bads(reset_bads=True)
    return raw

def df_to_raw(df, sfreq, ch_names=None, ch_types='eeg'):
    data = df.values.T
    if ch_names is None:
        ch_names = df.columns.tolist()
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw

# Example: Assume sampling frequency is 256 Hz (update if known)
sfreq = 256
raw = df_to_raw(raw_df, sfreq)

# Bandpass filter
raw = bandpass_filter(raw)
# Re-reference
raw = rereference(raw)
# ICA
ica = run_ica(raw)
# (Manual inspection step would go here)
raw = apply_ica(raw, ica)
# Downsample (optional)
raw = downsample(raw, 128)
# Interpolate bads (if any)
raw.info['bads'] = []  # Set bads if known
raw = interpolate_bads(raw)

# Save preprocessed data
preproc_fif = 's01_ex01_s01_preprocessed_raw.fif'
raw.save(preproc_fif, overwrite=True)
print(f'Preprocessed data saved to {preproc_fif}')

# Analysis functions
def plot_psd(raw):
    fig = raw.plot_psd(fmax=50, show=False)
    plt.savefig('psd.png')
    plt.close(fig)

def plot_erp(epochs):
    evoked = epochs.average()
    fig = evoked.plot(spatial_colors=True, show=False)
    plt.savefig('erp.png')
    plt.close(fig)

def plot_time_frequency(epochs, picks=None):
    power = mne.time_frequency.tfr_morlet(epochs, freqs=np.arange(2, 40, 2), n_cycles=2, picks=picks, return_itc=False)
    fig = power.plot_topo(baseline=(None, 0), mode='logratio', title='Average power', show=False)
    plt.savefig('tfr.png')
    plt.close(fig)

def plot_connectivity(raw, method='coh'):
    if spectral_connectivity is None:
        print("Connectivity analysis skipped: spectral_connectivity not available.")
        return
    con, freqs, times, n_epochs, _ = spectral_connectivity(
        [raw.get_data()], method=method, sfreq=raw.info['sfreq'], fmin=8, fmax=13, faverage=True, verbose=False)
    plt.figure(figsize=(10,8))
    sns.heatmap(con[:, :, 0], xticklabels=raw.ch_names, yticklabels=raw.ch_names, cmap='viridis')
    plt.title(f'Connectivity ({method})')
    plt.savefig('connectivity.png')
    plt.close()

def pearson_corr_matrix(raw):
    data = raw.get_data()
    corr = np.corrcoef(data)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, xticklabels=raw.ch_names, yticklabels=raw.ch_names, cmap='coolwarm', center=0)
    plt.title('Pearson Correlation Matrix')
    plt.savefig('pearson_corr.png')
    plt.close()
    return corr

def granger_causality_matrix(raw, maxlag=5):
    data = raw.get_data()
    n_ch = data.shape[0]
    gc_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j:
                test = grangercausalitytests(data[[i, j], :].T, maxlag=maxlag, verbose=False)
                gc_matrix[i, j] = np.min([test[lag][0]['ssr_ftest'][1] for lag in test])
    plt.figure(figsize=(10,8))
    sns.heatmap(gc_matrix, xticklabels=raw.ch_names, yticklabels=raw.ch_names, cmap='magma')
    plt.title('Granger Causality Matrix (min p-value)')
    plt.savefig('granger_matrix.png')
    plt.close()
    return gc_matrix

# Run analysis functions and compile results
plot_psd(raw)

# If events are available, create epochs and run ERP/time-frequency
if 'stim' in raw_df.columns or 'event' in raw_df.columns:
    # Example: assuming event column is named 'stim' and event_id is 1
    events = mne.find_events(raw, stim_channel='stim')
    event_id = dict(stim=1)
    epochs = epoch_data(raw, events, event_id)
    plot_erp(epochs)
    plot_time_frequency(epochs)
else:
    print('No event markers found; skipping ERP/time-frequency analysis.')

plot_connectivity(raw, method='coh')
pearson_corr = pearson_corr_matrix(raw)
granger_matrix = granger_causality_matrix(raw, maxlag=5)

np.save('s01_ex01_s01_pearson_corr.npy', pearson_corr)
np.save('s01_ex01_s01_granger_matrix.npy', granger_matrix)
print('Correlation matrices saved.')
