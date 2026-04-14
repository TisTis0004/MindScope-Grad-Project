import json
import shutil
import torch
import numpy as np
import pywt
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm

def get_psd_features(data, fs):
    """Calculates Power Spectral Density in clinical bands."""
    freqs, psd = welch(data, fs=fs, nperseg=fs*2)
    # Bands: 1-4Hz (Delta), 4-8Hz (Theta), 8-12Hz (Alpha), 12-30Hz (Beta)
    bands = [(1, 4), (4, 8), (8, 12), (12, 30)]
    band_powers = []
    for low, high in bands:
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers.append(np.mean(psd[:, idx], axis=1))
    return np.stack(band_powers, axis=1) # (channels, 4)

def get_wavelet_features(data):
    """Daubechies D4 Wavelet transform features."""
    coeffs = pywt.wavedec(data, 'db4', level=4)
    features = []
    # Detail coefficients (skipping approximation at index 0)
    for c in coeffs[1:]: 
        features.append(np.log(np.sum(np.square(c), axis=-1) + 1e-6))
        features.append(np.std(c, axis=-1))
        features.append(np.mean(np.abs(c), axis=-1))
    return np.stack(features, axis=1)

def process_cache_to_features(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy label map if it exists
    label_map_src = input_path / "label_map.json"
    if label_map_src.exists():
        shutil.copy(label_map_src, output_path / "label_map.json")
    
    files = list(input_path.glob("*.pt"))
    manifest_path = output_path / "manifest.jsonl"
    
    total_windows = 0
    n_cached_records = 0
    
    # 2. Extract features and write manifest
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for pt_file in tqdm(files, desc="Extracting Features"):
            data_dict = torch.load(pt_file)
            windows = data_dict['x'].numpy() # Shape: [N, C, T]
            labels = data_dict['y']
            fs = data_dict['meta']['fs']
            
            all_window_features = []
            
            for i in range(windows.shape[0]):
                win = windows[i] # (Channels, Samples)
                
                # Time Domain: Stats
                f_mean = np.mean(win, axis=1)
                f_std = np.std(win, axis=1)
                
                # Frequency Domain: PSD
                f_psd = get_psd_features(win, fs)
                
                # Time-Frequency: Wavelets
                f_wav = get_wavelet_features(win)
                
                # Combine all features for this window
                combined = np.column_stack([f_mean, f_std, f_psd, f_wav]) 
                all_window_features.append(combined.flatten())
                
            # Convert to Tensor (M, 1276, 1)
            feature_matrix = torch.tensor(np.array(all_window_features)).float()
            if feature_matrix.ndim == 2:
                feature_matrix = feature_matrix.unsqueeze(-1)
                
            out_pt_path = output_path / pt_file.name
            
            # Save the new feature dictionary
            torch.save({
                'x': feature_matrix,
                'y': labels,
                'meta': data_dict['meta']
            }, out_pt_path)
            
            # Update counters and write to manifest
            n_windows = feature_matrix.shape[0]
            total_windows += n_windows
            n_cached_records += 1
            
            mf.write(json.dumps({"pt_path": str(out_pt_path), "n": n_windows}, ensure_ascii=False) + "\n")

    print("\nExtraction Complete.")
    print(f"Processed records: {n_cached_records}")
    print(f"Total feature windows: {total_windows}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    process_cache_to_features(
        input_dir="cache_windows_binary_10_sec_eval", 
        output_dir="feature_dataset_v1_eval"
    )