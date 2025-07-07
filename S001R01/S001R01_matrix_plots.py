import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# Load Granger and Pearson matrices
G = np.load('S001R01_granger_matrix.npy')
P = np.load('S001R01_pearson_corr.npy')

# Try to get channel names from the preprocessed raw file
try:
    raw = mne.io.read_raw_fif('S001R01_preprocessed_raw.fif', preload=False)
    ch_names = raw.ch_names
except Exception:
    ch_names = None

# Plot Granger matrix
plt.figure(figsize=(14, 12))
sns.heatmap(G, xticklabels=ch_names, yticklabels=ch_names, cmap='magma')
plt.title('Granger Causality Matrix (min p-value)')
plt.tight_layout()
plt.savefig('S001R01_granger_matrix_only.png')
plt.close()

# Plot Pearson matrix
plt.figure(figsize=(14, 12))
sns.heatmap(P, xticklabels=ch_names, yticklabels=ch_names, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.savefig('S001R01_pearson_corr_only.png')
plt.close()

print('Saved: S001R01_granger_matrix_only.png, S001R01_pearson_corr_only.png')
