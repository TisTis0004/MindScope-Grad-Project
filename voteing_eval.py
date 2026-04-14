"""
Enhanced Seizure Detection Evaluation with Post-Processing.

This script applies three post-processing techniques used in top TUSZ submissions:
1. Optimal threshold search (instead of hardcoded 0.50)
2. Probability-based median smoothing (instead of hard binary voting)
3. Minimum seizure duration filter (reject isolated false positives)

Run after training: python voteing_eval.py
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from helper.train_helper import build_model, build_loaders, CHECKPOINT_PATH
from tqdm import tqdm
from helper.T import EEGToSpectrogram
from scipy.ndimage import median_filter


def find_optimal_threshold(y_true, y_prob, low=0.20, high=0.80, step=0.01):
    """
    Sweep probability thresholds to find the one that maximizes F1-macro.
    This is crucial because training is 50/50 balanced but eval is ~80/20.
    """
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(low, high, step):
        preds = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def smooth_probabilities_per_file(y_prob, file_lengths, kernel_size=5):
    """
    Apply median smoothing to seizure probabilities WITHIN each patient file.
    
    WHY: Seizures evolve gradually over seconds. A single high-probability spike
    surrounded by low probabilities is likely noise. Median smoothing preserves
    genuine seizure stretches while suppressing isolated spikes.
    
    We smooth probabilities (not binary predictions) to preserve calibration.
    """
    smoothed = np.zeros_like(y_prob)
    current_idx = 0
    
    for n in file_lengths:
        file_probs = y_prob[current_idx:current_idx + n]
        
        if n >= kernel_size:
            # Median filter smooths within this patient's recording
            file_probs = median_filter(file_probs, size=kernel_size, mode='reflect')
        
        smoothed[current_idx:current_idx + n] = file_probs
        current_idx += n
    
    return smoothed


def apply_min_duration_filter(preds, file_lengths, min_consecutive=3):
    """
    Reject seizure predictions shorter than min_consecutive windows.
    
    WHY: Real seizures last at least 10-30 seconds. With 10s windows and 5s stride,
    a real seizure should span at least 3+ consecutive windows. Isolated 1-2 window
    predictions are almost always false positives from artifacts.
    
    This is standard in every competitive TUSZ submission.
    """
    filtered = np.zeros_like(preds)
    current_idx = 0
    
    for n in file_lengths:
        file_preds = preds[current_idx:current_idx + n]
        file_filtered = np.zeros_like(file_preds)
        
        # Find runs of consecutive seizure predictions
        i = 0
        while i < n:
            if file_preds[i] == 1:
                # Count the length of this run
                run_start = i
                while i < n and file_preds[i] == 1:
                    i += 1
                run_length = i - run_start
                
                # Only keep if the run is long enough
                if run_length >= min_consecutive:
                    file_filtered[run_start:i] = 1
            else:
                i += 1
        
        filtered[current_idx:current_idx + n] = file_filtered
        current_idx += n
    
    return filtered


def print_metrics(name, y_true, y_pred, y_prob=None):
    """Print comprehensive metrics for a prediction set."""
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  F1-macro:      {f1_mac:.4f}")
    print(f"  Seizure  — Precision: {prec[1]:.4f}, Recall: {rec[1]:.4f}, F1: {f1_per[1]:.4f}")
    print(f"  BackGnd  — Precision: {prec[0]:.4f}, Recall: {rec[0]:.4f}, F1: {f1_per[0]:.4f}")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"  AUC:           {auc:.4f}")
        except:
            pass
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"    FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")
    return f1_mac


def evaluate_with_voting():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Best Model
    print(f"Loading best model from {CHECKPOINT_PATH}...")
    model = build_model(device, weights=CHECKPOINT_PATH)
    model.eval()

    # 2. Load Validation Data (sequential, not shuffled)
    transform = EEGToSpectrogram().to(device)
    transform.eval()
    _, val_loader = build_loaders(transform=None)
    
    # Get file boundaries for per-patient processing
    val_dataset = val_loader.dataset 
    file_lengths = [n for _, n in val_dataset.items]

    all_targets = []
    all_probs = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).long()
            
            x = transform(x)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(seizure)
            
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_probs).numpy()

    print(f"\nTotal samples: {len(y_true)}")
    print(f"Seizure: {(y_true == 1).sum()}, Background: {(y_true == 0).sum()}")
    print(f"Class ratio: {(y_true == 0).sum() / (y_true == 1).sum():.1f}:1 background:seizure")

    # =====================================================
    # STAGE 1: Raw predictions with fixed threshold (baseline)
    # =====================================================
    raw_preds = (y_prob > 0.50).astype(int)
    print_metrics("RAW (threshold=0.50)", y_true, raw_preds, y_prob)

    # =====================================================
    # STAGE 2: Optimal threshold search
    # =====================================================
    best_thresh, best_f1 = find_optimal_threshold(y_true, y_prob)
    print(f"\n>>> Optimal threshold: {best_thresh:.2f}")
    thresh_preds = (y_prob >= best_thresh).astype(int)
    print_metrics(f"OPTIMAL THRESHOLD ({best_thresh:.2f})", y_true, thresh_preds, y_prob)

    # =====================================================
    # STAGE 3: Probability smoothing + optimal threshold
    # =====================================================
    smoothed_prob = smooth_probabilities_per_file(y_prob, file_lengths, kernel_size=5)
    smooth_thresh, smooth_f1 = find_optimal_threshold(y_true, smoothed_prob)
    smooth_preds = (smoothed_prob >= smooth_thresh).astype(int)
    print_metrics(f"SMOOTHED + THRESHOLD ({smooth_thresh:.2f})", y_true, smooth_preds, smoothed_prob)

    # =====================================================
    # STAGE 4: Smoothing + threshold + min duration filter
    # =====================================================
    # Try different minimum durations
    for min_dur in [2, 3, 4]:
        filtered_preds = apply_min_duration_filter(smooth_preds, file_lengths, min_consecutive=min_dur)
        label = f"FULL PIPELINE (smooth + thresh={smooth_thresh:.2f} + min_dur={min_dur})"
        f1 = print_metrics(label, y_true, filtered_preds, smoothed_prob)

    # =====================================================
    # SUMMARY
    # =====================================================
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  Raw (0.50):          {f1_score(y_true, raw_preds, average='macro'):.4f}")
    print(f"  Optimal thresh:      {best_f1:.4f}  (thresh={best_thresh:.2f})")
    print(f"  Smoothed + thresh:   {smooth_f1:.4f}  (thresh={smooth_thresh:.2f})")
    
    for min_dur in [2, 3, 4]:
        fp = apply_min_duration_filter(smooth_preds, file_lengths, min_consecutive=min_dur)
        f1_v = f1_score(y_true, fp, average='macro')
        print(f"  + min_duration={min_dur}:    {f1_v:.4f}")


if __name__ == "__main__":
    evaluate_with_voting()