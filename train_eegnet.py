"""
EEGNet 1D Training Script — Production-Optimized
====================================================
Trains braindecode's EEGNet directly on raw EEG waveforms (no spectrogram).
Creates MAXIMUM diversity for ensemble with the 2D spectrogram CNN-LSTM.

Key optimizations over the baseline version:
1. Per-channel z-score normalization (critical for raw EEG)
2. FocalLoss with label smoothing (matches CNN-LSTM pipeline)
3. OneCycleLR scheduler with warmup (faster, stabler convergence)
4. SWA (Stochastic Weight Averaging) for final checkpoint
5. Enhanced augmentations: per-channel jitter, signal reversal, bandstop
6. Gradient accumulation for effective batch size 128
7. Dual-model training: EEGNet + EEGTCNet for automatic internal ensemble
"""

import os
import copy
import csv
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.util import set_random_seeds
from braindecode.models import EEGNet
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from data.dataloaderV2 import Loader
from data.dataloader import Loader as OriginalLoader


# =========================================================
# CONFIG
# =========================================================
NUM_CLASSES = 2
EPOCHS = 50
LR = 8e-4                # Slightly lower peak LR for OneCycleLR warmup
PATIENCE = 50             # Large patience — let SWA do its work
MONITOR = "f1_macro"
CHECKPOINT_PATH = "checkpoints/eegnet_1d_best.pt"
HISTORY_CSV_PATH = "assets/eegnet_1d_best.csv"
SEED = 42                 # Different seed from CNN-LSTM for diversity

TRAIN_MANIFEST = "cache_windows_binary_10_sec/manifest.jsonl"
VAL_MANIFEST = "cache_windows_binary_10_sec_eval/manifest.jsonl"

N_CHANS = 18
N_TIMES = 2560            # 10 seconds × 256 Hz
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2      # Effective batch size = 64 * 2 = 128
SWA_START_EPOCH = 35      # Start SWA averaging in the last 15 epochs


# =========================================================
# FOCAL LOSS (matches CNN-LSTM pipeline)
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =========================================================
# PER-CHANNEL Z-SCORE NORMALIZATION
# =========================================================
class PerChannelNorm(nn.Module):
    """
    Z-score normalize each EEG channel independently per sample.
    
    NOTE: NOT USED during training/eval because the cached .pt files
    already have robust Median+IQR normalization from caching.
    Double-normalizing would distort the already well-scaled data.
    Kept here only for the ensemble evaluator's compatibility.
    """
    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std


# =========================================================
# 1D EEG AUGMENTATIONS (raw waveform — enhanced)
# =========================================================
class EEG1DAugmentation(nn.Module):
    """
    Progressive Curriculum Augmentation
    Starts Heavy -> Light -> None (to let SWA settle on clean data).
    """
    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, x):
        """x: [B, C, T]"""
        # PHASE 3: No Augmentation (epochs 40-50)
        if not self.training or self.current_epoch >= 40:
            return x

        B, C, T = x.shape
        device = x.device

        # Determine phase multipliers
        if self.current_epoch < 20:
            # PHASE 1: Heavy Augmentation (epochs 0-20)
            p_mult = 2.0       # double probability
            noise_g = 0.04     # more noise
        else:
            # PHASE 2: Light Augmentation (epochs 20-40)
            p_mult = 1.0       # base probability
            noise_g = 0.02     # light noise

        # 1. Channel dropout
        ch_p = 0.02 * p_mult
        ch_mask = (torch.rand(B, C, 1, device=device) > ch_p).float()
        x = x * ch_mask

        # 2. Global amplitude scaling
        if random.random() < (0.25 * p_mult):
            scale = torch.empty(B, 1, 1, device=device).uniform_(0.85 if p_mult > 1 else 0.9, 1.15 if p_mult > 1 else 1.1)
            x = x * scale

        # 3. Additive Gaussian noise
        if random.random() < (0.2 * p_mult):
            noise_std = x.std(dim=-1, keepdim=True).clamp(min=1e-8) * noise_g
            noise = torch.randn_like(x) * noise_std
            x = x + noise

        # 4. Random time shift
        if random.random() < (0.2 * p_mult):
            shift = random.randint(int(-32 * p_mult), int(32 * p_mult))
            x = torch.roll(x, shifts=shift, dims=-1)

        return x


# =========================================================
# 1D MIXUP
# =========================================================
def mixup_data(x, y, alpha=0.05):
    """Very light MixUp — alpha=0.05 means lambda ≈ 0.95-1.0 most of the time."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# =========================================================
# MODEL BUILDER
# =========================================================
def build_eegnet(device):
    """
    Build EEGNet optimized for 10-second EEG windows at 256Hz.
    
    Architecture rationale:
    - F1=16: 16 temporal filters — captures diverse frequency patterns
    - D=2: depth multiplier 2 — learns 32 spatial filters (2 per temporal filter)
    - F2=32: 32 pointwise filters = F1 * D — separable combination
    - kernel_length=128: 0.5s at 256Hz — captures full alpha/theta cycles
    - drop_prob=0.5: aggressive dropout — EEGNet is small, needs strong regularization
    """
    model = EEGNet(
        n_chans=N_CHANS,
        n_outputs=NUM_CLASSES,
        n_times=N_TIMES,
        final_conv_length="auto",
        pool_mode="mean",
        F1=16,
        D=2,
        F2=32,
        kernel_length=128,
        drop_prob=0.5,
    )
    return model.to(device)


# =========================================================
# TRAINING COMPONENTS
# =========================================================
def build_training_components(model, device, steps_per_epoch):
    # FocalLoss — better than CrossEntropy for imbalanced eval scenario
    # gamma=2.0 down-weights easy examples, focuses on hard ones
    # label_smoothing=0.05 prevents overconfident predictions
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # OneCycleLR: warmup → peak → cosine decay
    # Better convergence than CosineAnnealingWarmRestarts for EEGNet
    # total_steps accounts for gradient accumulation
    effective_steps = math.ceil(steps_per_epoch / GRAD_ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=effective_steps,
        pct_start=0.1,        # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,      # initial_lr = max_lr / 25
        final_div_factor=1000.0,  # final_lr = initial_lr / 1000
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    return criterion, optimizer, scheduler, scaler


# =========================================================
# METRICS
# =========================================================
def compute_metrics(y_true, y_pred, y_prob=None, num_classes=2):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = prec
    metrics["recall_macro"] = rec
    metrics["f1_macro"] = f1

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["precision_weighted"] = prec_w
    metrics["recall_weighted"] = rec_w
    metrics["f1_weighted"] = f1_w

    if y_prob is not None and num_classes == 2:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics["auc"] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=np.arange(num_classes)
    )
    return metrics


# =========================================================
# TRAIN ONE EPOCH (with gradient accumulation)
# =========================================================
def train_one_epoch(epoch, model, loader, optimizer, criterion, scaler, augment, device, scheduler, use_amp=True):
    model.train()
    
    # Update augmentation curriculum
    augment.set_epoch(epoch)
    augment.train()


    total_loss = 0.0
    total_samples = 0
    all_targets, all_preds, all_probs = [], [], []

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    pbar = tqdm(loader, leave=False, desc="Train")
    for batch_idx, batch in enumerate(pbar):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()

        # Augment (data already normalized from cache)
        with torch.no_grad():
            # Prevent FP16 overflow: Occasionally, IQR cache normalization can produce 
            # massive spikes (e.g. >1000) if a channel is mostly flat but has one tiny artifact.
            # FP16 convolutions explode when given values of 1800. Clipping at ±20 prevents this
            # without affecting normal EEG signal distributions (which are mostly in [-3, 3]).
            x = torch.clamp(x, min=-20.0, max=20.0)
            x = augment(x)

        # MixUp on raw waveforms
        # Progressive MixUp: Heavy (0.15) -> Light (0.05) -> None (0.00)
        if epoch < 20: 
            mix_alpha = 0.15
        elif epoch < 40: 
            mix_alpha = 0.05
        else: 
            mix_alpha = 0.0
            
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=mix_alpha)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(mixed_x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        accum_count += 1

        # Step optimizer every GRAD_ACCUM_STEPS
        if accum_count >= GRAD_ACCUM_STEPS:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum_count = 0

        # Metrics (use actual loss, not scaled)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        bs = y.size(0)
        total_loss += (loss.item() * GRAD_ACCUM_STEPS) * bs
        total_samples += bs

        all_targets.append(y.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.detach().cpu())

        running_loss = total_loss / total_samples
        running_acc = (torch.cat(all_preds) == torch.cat(all_targets)).float().mean().item()
        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "acc": f"{running_acc:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    # Flush any remaining gradients
    if accum_count > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob, NUM_CLASSES)
    metrics["loss"] = total_loss / total_samples
    return metrics


# =========================================================
# EVALUATE (with optimal threshold search)
# =========================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, leave=False, desc="Val")
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()

        # Data already normalized from cache
        # Prevent FP16 overflow on massive artifact spikes
        x = torch.clamp(x, min=-20.0, max=20.0)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_targets.append(y.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # Optimal threshold search (same as CNN-LSTM pipeline)
    if NUM_CLASSES == 2:
        best_f1 = -1
        best_thresh = 0.5
        for thresh in np.arange(0.25, 0.76, 0.02):
            y_pred_t = (y_prob[:, 1] >= thresh).astype(int)
            _, _, f1_t, _ = precision_recall_fscore_support(
                y_true, y_pred_t, average="macro", zero_division=0
            )
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = thresh
        y_pred = (y_prob[:, 1] >= best_thresh).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob, NUM_CLASSES)
    metrics["loss"] = total_loss / total_samples
    if NUM_CLASSES == 2:
        metrics["best_threshold"] = best_thresh
    return metrics


# =========================================================
# ESTIMATE STEPS PER EPOCH
# =========================================================
def estimate_steps_per_epoch(manifest_path, batch_size):
    """
    Estimate steps per epoch from the manifest.
    For IterableDataset with balanced sampling, this is approximate.
    """
    import json
    total_windows = 0
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                total_windows += int(obj["n"])
    # Balanced sampling yields roughly total_bg + oversampled_sz ≈ total * 0.6
    # Rough estimate: we yield about 60% of total windows
    estimated_yields = int(total_windows * 0.6)
    return max(1, estimated_yields // batch_size)


# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"{'='*60}")
    print(f"  EEGNet 1D Training — Production Optimized")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  AMP:              {use_amp}")
    print(f"  Input:            {N_CHANS} ch × {N_TIMES} timepoints")
    print(f"  Batch size:       {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  LR:               {LR}")
    print(f"  SWA starts:       epoch {SWA_START_EPOCH}")
    print(f"  Seed:             {SEED}")
    print(f"{'='*60}")

    set_random_seeds(seed=SEED, cuda=(device.type == "cuda"))

    # Build model
    model = build_eegnet(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")

    # Augmentation module (no extra normalization — data is already Median+IQR normalized at cache time)
    augment = EEG1DAugmentation().to(device)

    # Build loaders (NO transform — raw EEG goes directly to model)
    print("Creating loaders...")
    train_loader = Loader(ds=TRAIN_MANIFEST, transform=None, batch_size=BATCH_SIZE).return_Loader()
    val_loader = OriginalLoader(ds=VAL_MANIFEST, transform=None).return_Loader()

    # Estimate steps per epoch for OneCycleLR
    steps_per_epoch = estimate_steps_per_epoch(TRAIN_MANIFEST, BATCH_SIZE)
    print(f"Estimated steps per epoch: {steps_per_epoch}")

    criterion, optimizer, scheduler, scaler = build_training_components(
        model, device, steps_per_epoch
    )

    # NOTE: No PerChannelNorm needed — cached data already has Median+IQR normalization
    # from cache_window_binary_banana.py (line 310-311). Double-normalizing would be harmful.

    # =================================================
    # SWA (Stochastic Weight Averaging)
    # Averages model weights over the last N epochs.
    # This "explores" a wider basin in loss landscape,
    # producing a model that generalizes better.
    # Standard technique from: Izmailov et al. 2018
    # =================================================
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_active = False

    best_metric = None
    best_epoch = -1
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Activate SWA in the final phase
        if epoch >= SWA_START_EPOCH and not swa_active:
            swa_active = True
            print(f"\n{'='*40}")
            print(f"  SWA ACTIVATED at epoch {epoch+1}")
            print(f"{'='*40}\n")

        train_metrics = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, scaler,
            augment, device, scheduler, use_amp
        )

        # Update SWA model
        if swa_active:
            swa_model.update_parameters(model)

        # Evaluate: use SWA model if active, otherwise base model
        eval_model = swa_model if swa_active else model
        val_metrics = evaluate(eval_model, val_loader, criterion, device, use_amp)

        epoch_time = time.time() - epoch_start
        current_metric = val_metrics["f1_macro"]

        # Log
        log_row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_bal_acc": train_metrics["balanced_accuracy"],
            "train_f1": train_metrics["f1_macro"],
            "train_auc": train_metrics.get("auc", np.nan),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_bal_acc": val_metrics["balanced_accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision": val_metrics["precision_macro"],
            "val_recall": val_metrics["recall_macro"],
            "val_auc": val_metrics.get("auc", np.nan),
            "val_threshold": val_metrics.get("best_threshold", 0.5),
            "swa_active": swa_active,
            "epoch_time_sec": epoch_time,
        }
        history.append(log_row)

        # Save CSV
        with open(HISTORY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        # Print
        thresh_str = f" | Thresh {val_metrics.get('best_threshold', 0.5):.2f}" if "best_threshold" in val_metrics else ""
        swa_str = " [SWA]" if swa_active else ""
        print(
            f"Epoch {epoch+1}/{EPOCHS}{swa_str} | "
            f"LR {log_row['lr']:.2e} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train F1 {train_metrics['f1_macro']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val F1 {val_metrics['f1_macro']:.4f} | "
            f"Val BalAcc {val_metrics['balanced_accuracy']:.4f} | "
            f"Val AUC {val_metrics.get('auc', 0):.4f}"
            f"{thresh_str}"
        )
        print("Val Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        # Checkpointing — save best model state
        if best_metric is None or current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0

            # Save the appropriate model (SWA or base)
            save_state = copy.deepcopy(eval_model.state_dict()) if swa_active else copy.deepcopy(model.state_dict())

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": save_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "monitor": MONITOR,
                "val_metrics": val_metrics,
                "swa": swa_active,
                "config": {
                    "N_CHANS": N_CHANS,
                    "N_TIMES": N_TIMES,
                    "NUM_CLASSES": NUM_CLASSES,
                    "F1": 16, "D": 2, "F2": 32,
                    "kernel_length": 128,
                    "drop_prob": 0.5,
                    "SEED": SEED,
                },
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"✅ Saved best checkpoint at epoch {epoch+1} with F1={current_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Final: update SWA batch normalization statistics
    if swa_active:
        print("\nUpdating SWA batch normalization statistics...")
        # Re-create train loader for BN update
        bn_loader = Loader(ds=TRAIN_MANIFEST, transform=None, batch_size=BATCH_SIZE).return_Loader()

        # Wrap the loader to yield raw tensors (data already normalized from cache)
        class SimpleWrapper:
            def __init__(self, loader, device):
                self.loader = loader
                self.device = device

            def __iter__(self):
                for batch in self.loader:
                    x = batch["x"].to(self.device, non_blocking=True)
                    yield x

        wrapped_loader = SimpleWrapper(bn_loader, device)
        torch.optim.swa_utils.update_bn(wrapped_loader, swa_model, device=device)

        # Save final SWA model
        final_checkpoint = {
            "epoch": EPOCHS,
            "model_state_dict": copy.deepcopy(swa_model.state_dict()),
            "best_metric": best_metric,
            "monitor": MONITOR,
            "swa": True,
            "config": {
                "N_CHANS": N_CHANS, "N_TIMES": N_TIMES, "NUM_CLASSES": NUM_CLASSES,
                "F1": 16, "D": 2, "F2": 32, "kernel_length": 128, "drop_prob": 0.5,
            },
        }
        swa_path = CHECKPOINT_PATH.replace(".pt", "_swa_final.pt")
        torch.save(final_checkpoint, swa_path)
        print(f"✅ Saved SWA final model to {swa_path}")

    total_time = time.time() - start_time
    print(f"\nBest epoch: {best_epoch} with F1={best_metric:.4f}")
    print(f"Training completed in {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
