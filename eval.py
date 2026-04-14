import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from helper.train_helper import build_model, build_loaders, CHECKPOINT_PATH
from tqdm import tqdm

from helper.T import EEGToSpectrogram

def sweep_thresholds():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Best Model
    print(f"Loading best model from {CHECKPOINT_PATH}...")
    model = build_model(device, weights=CHECKPOINT_PATH)
    model.eval()

    # 2. Load Validation Data
    transform = EEGToSpectrogram().to(device)
    transform.eval()  # Super important to turn off SpecAugment!
    _, val_loader = build_loaders(transform=None)

    all_targets = []
    all_probs = []

    print("Running validation set to get probabilities...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).long()
            
            x = transform(x)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1] # Get probability of Class 1 (Seizure)
            
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # 3. Sweep Thresholds mathematically (Fast)
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    best_t = 0.5
    best_f1 = 0.0

    print("\n--- THRESHOLD SWEEP RESULTS ---")
    for t in thresholds:
        # Convert probabilities to 0 or 1 based on current threshold
        y_pred = (y_prob > t).astype(int)
        
        # Calculate Macro F1
        f1 = f1_score(y_true, y_pred, average="macro")
        
        print(f"Threshold: {t:.2f} | F1 Macro: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("\n=====================================")
    print(f"BEST THRESHOLD: {best_t:.2f} with F1: {best_f1:.4f}")
    
    # Print the exact confusion matrix for the best threshold
    best_preds = (y_prob > best_t).astype(int)
    cm = confusion_matrix(y_true, best_preds)
    print("Confusion Matrix at Best Threshold:")
    print(cm)
    print("=====================================")

if __name__ == "__main__":
    sweep_thresholds()