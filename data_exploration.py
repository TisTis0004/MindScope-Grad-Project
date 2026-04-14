# %% [markdown]
# # EEG Data Exploration & Statistics
# This interactive notebook/script explores the compiled `.pt` cache files for Train and Validation sets.
# It computes statistical properties of the EEG waves to detect any significant domain shifts between Train and Val.
# You can run this file interactively in VSCode by clicking "Run Cell" above each `%%` block.

# %%
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Set the paths to your Train and Val manifests
TRAIN_MANIFEST = "cache_windows_binary_10_sec/manifest.jsonl"
VAL_MANIFEST = "cache_windows_binary_10_sec_eval/manifest.jsonl"

def load_manifest_paths(manifest_path, max_files=100):
    """Loads a random subset of .pt files from the manifest to keep analysis fast."""
    paths = []
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return paths
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                paths.append(json.loads(line)["pt_path"])
                
    # Randomly sample if too large to save memory/time
    if len(paths) > max_files:
        np.random.seed(42)
        paths = np.random.choice(paths, max_files, replace=False).tolist()
    return paths

train_files = load_manifest_paths(TRAIN_MANIFEST, max_files=100)
val_files = load_manifest_paths(VAL_MANIFEST, max_files=50)

print(f"Sampling {len(train_files)} Train files and {len(val_files)} Val files for statistics...")

# %%
def extract_statistics(file_paths, split_name):
    """Extracts raw EEG statistics from a list of cached .pt tensors."""
    stats_list = []
    
    for path in tqdm(file_paths, desc=f"Processing {split_name}"):
        if not os.path.exists(path): continue
            
        data = torch.load(path, map_location="cpu")
        x = data["x"].numpy()  # Shape: (Num_Windows, Channels, Time)
        y = data["y"].numpy()  # Shape: (Num_Windows)
        
        for i in range(len(y)):
            window = x[i]
            # Discard dead/missing channels (zeros) for stats calculation
            active_mask = np.abs(window).sum(axis=1) > 1e-6
            if not np.any(active_mask):
                continue
            
            active_channels = window[active_mask]
            
            stats_list.append({
                "split": split_name,
                "label": "Seizure" if y[i] == 1 else "Background",
                "mean_amplitude": float(np.mean(active_channels)),
                "std_amplitude": float(np.std(active_channels)),
                "max_amplitude": float(np.max(active_channels)),
                "min_amplitude": float(np.min(active_channels)),
                "missing_channels_count": int(np.sum(~active_mask))
            })
            
    return pd.DataFrame(stats_list)

# %%
# Extract all statistics
df_train = extract_statistics(train_files, "Train")
df_val = extract_statistics(val_files, "Val")

# Combine for comparison
df_all = pd.concat([df_train, df_val], ignore_index=True)
print("\n--- Summary Statistics ---")
print(df_all.groupby(["split", "label"]).mean())

# %% [markdown]
# # Visualization: Feature Distributions
# We will use KDE plots to check if the overall distribution of standard deviations (wave energy) is the same across Train and Val.
# If these don't align, the model is experiencing domain shift.

# %%
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_all, x="std_amplitude", hue="split", common_norm=False, fill=True)
plt.title("Distribution of Window Standard Deviations (Energy) - Train vs Val")
plt.xlabel("Standard Deviation of Amplitude")
plt.xlim(0, df_all["std_amplitude"].quantile(0.95)) # Zoom in on the meat of the distribution
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_all, x="mean_amplitude", hue="split", common_norm=False, fill=True)
plt.title("Distribution of Window Means (DC Offset) - Train vs Val")
plt.xlabel("Mean Amplitude")
plt.xlim(df_all["mean_amplitude"].quantile(0.05), df_all["mean_amplitude"].quantile(0.95))
plt.show()

# %% [markdown]
# # Seizure vs Background Statistics
# Does the difference between Seizure and Background visually match between the two splits?

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_all, x="split", y="std_amplitude", hue="label", showfliers=False)
plt.title("Amplitude Energy (Std Dev): Seizure vs Background in Train vs Val")
plt.show()

# %% [markdown]
# # Missing Channels
# Did the validation set have more missing channels than the training set?
# %%
missing_summary = df_all.groupby("split")["missing_channels_count"].value_counts(normalize=True).unstack().fillna(0)
print("\n--- Percentage of Windows with 'N' Missing Channels ---")
print(missing_summary * 100)

missing_summary.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
plt.title("Missing Channels per Window (Train vs Val)")
plt.ylabel("Percentage of Windows")
plt.legend(title="Number of Missing Channels")
plt.show()
