import copy
import csv
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from braindecode.models import EEGNet

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data.dataloader import Loader  # noqa: E402


# =========================================================
# CONFIG
# =========================================================
NUM_CLASSES = 4
EPOCHS = 30
LR = 1e-3
PATIENCE = 10
MONITOR = "f1_macro"   # options: val_loss, accuracy, f1_macro, balanced_accuracy, auc
CHECKPOINT_PATH = 'checkpoints/eeg_4_classes.pt'
HISTORY_CSV_PATH = 'assets/eegnet_history.csv'
SEED = 20200220

TRAIN_MANIFEST = 'cahce_windows/manifest.jsonl'
VAL_MANIFEST = "cache_windows_eval/manifest.jsonl"


# =========================================================
# HELPERS
# =========================================================
def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def is_better(current, best, mode="max"):
    if best is None:
        return True
    return current > best if mode == "max" else current < best


def compute_classification_metrics(y_true, y_pred, y_prob=None, num_classes=None, topk=2):
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["precision_macro"] = precision_macro
    metrics["recall_macro"] = recall_macro
    metrics["f1_macro"] = f1_macro
    metrics["precision_weighted"] = precision_weighted
    metrics["recall_weighted"] = recall_weighted
    metrics["f1_weighted"] = f1_weighted

    if y_prob is not None and num_classes == 2:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            metrics["auc"] = np.nan

        try:
            metrics[f"top{topk}_accuracy"] = top_k_accuracy_score(
                y_true, y_prob, k=min(topk, 2), labels=np.arange(num_classes)
            )
        except Exception:
            metrics[f"top{topk}_accuracy"] = np.nan

    elif y_prob is not None and num_classes is not None and num_classes > 2:
        try:
            metrics[f"top{topk}_accuracy"] = top_k_accuracy_score(
                y_true, y_prob, k=topk, labels=np.arange(num_classes)
            )
        except Exception:
            metrics[f"top{topk}_accuracy"] = np.nan

        try:
            metrics["auc_ovr_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["auc_ovr_macro"] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes) if num_classes is not None else None
    )

    return metrics


def save_history_to_csv(history, csv_path):
    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def build_model(device, weights=None):
    model = EEGNet(
        n_chans=41,
        n_outputs=NUM_CLASSES,
        n_times=2500,
    )

    if weights:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model.to(device)


def build_loaders(transform=None):
    train_loader_obj = Loader(transform=transform)
    train_loader = train_loader_obj.return_Loader()

    val_loader_obj = Loader(transform=transform, ds=VAL_MANIFEST)
    val_loader = val_loader_obj.return_Loader()

    return train_loader, val_loader


def get_loader_counts(loader):
    counts = Counter()
    for batch in loader:
        y = batch["y"]
        if torch.is_tensor(y):
            y = y.cpu().tolist()
        counts.update(map(int, y))
    return dict(sorted(counts.items()))

#for 9 classes or to make the training not biased for the bcgz
def get_class_weights(device):
    counts = torch.tensor([118873, 4391, 8970, 1376], dtype=torch.float)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return weights.to(device)


def build_training_components(model, device):
    criterion = nn.CrossEntropyLoss(weight = get_class_weights(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    return criterion, optimizer, scheduler, scaler


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp=True, num_classes=2, topk=2):
    model.train()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_preds = []
    all_probs = []

    pbar = tqdm(loader, leave=False, desc="Train")

    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

        running_loss = total_loss / total_samples
        running_acc = (torch.cat(all_preds) == torch.cat(all_targets)).float().mean().item()

        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "acc": f"{running_acc:.4f}",
            "lr": f"{get_current_lr(optimizer):.2e}",
        })

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=num_classes,
        topk=topk,
    )
    metrics["loss"] = total_loss / total_samples

    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True, num_classes=2, topk=2, desc="Eval"):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_preds = []
    all_probs = []

    pbar = tqdm(loader, leave=False, desc=desc)

    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=num_classes,
        topk=topk,
    )
    metrics["loss"] = total_loss / total_samples

    return metrics


def get_monitored_metric(val_metrics):
    if MONITOR == "val_loss":
        return val_metrics["loss"]
    if MONITOR == "accuracy":
        return val_metrics["accuracy"]
    if MONITOR == "balanced_accuracy":
        return val_metrics["balanced_accuracy"]
    if MONITOR == "f1_macro":
        return val_metrics["f1_macro"]
    if MONITOR == "auc":
        return val_metrics.get("auc", val_metrics.get("auc_ovr_macro", np.nan))
    raise ValueError(f"Unsupported MONITOR: {MONITOR}")


def build_log_row(epoch, optimizer, train_metrics, val_metrics, epoch_time, topk):
    log_row = {
        "epoch": epoch + 1,
        "lr": get_current_lr(optimizer),
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["accuracy"],
        "train_bal_acc": train_metrics["balanced_accuracy"],
        "train_precision_macro": train_metrics["precision_macro"],
        "train_recall_macro": train_metrics["recall_macro"],
        "train_f1_macro": train_metrics["f1_macro"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["accuracy"],
        "val_bal_acc": val_metrics["balanced_accuracy"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "epoch_time_sec": epoch_time,
    }

    if f"top{topk}_accuracy" in train_metrics:
        log_row[f"train_top{topk}_acc"] = train_metrics[f"top{topk}_accuracy"]
    if f"top{topk}_accuracy" in val_metrics:
        log_row[f"val_top{topk}_acc"] = val_metrics[f"top{topk}_accuracy"]

    if "auc" in train_metrics:
        log_row["train_auc"] = train_metrics["auc"]
    if "auc" in val_metrics:
        log_row["val_auc"] = val_metrics["auc"]

    return log_row


def build_epoch_message(epoch, total_epochs, log_row, train_metrics, val_metrics, topk):
    msg = (
        f"Epoch {epoch + 1}/{total_epochs} | "
        f"LR {log_row['lr']:.2e} | "
        f"Train Loss {train_metrics['loss']:.4f} | "
        f"Train Acc {train_metrics['accuracy']:.4f} | "
        f"Train F1 {train_metrics['f1_macro']:.4f} | "
        f"Val Loss {val_metrics['loss']:.4f} | "
        f"Val Acc {val_metrics['accuracy']:.4f} | "
        f"Val BalAcc {val_metrics['balanced_accuracy']:.4f} | "
        f"Val F1 {val_metrics['f1_macro']:.4f}"
    )

    if "auc" in val_metrics and not np.isnan(val_metrics["auc"]):
        msg += f" | Val AUC {val_metrics['auc']:.4f}"

    if f"top{topk}_accuracy" in val_metrics and not np.isnan(val_metrics[f"top{topk}_accuracy"]):
        msg += f" | Val Top{topk} {val_metrics[f'top{topk}_accuracy']:.4f}"

    return msg


def make_checkpoint(epoch, model, optimizer, scheduler, scaler, best_metric, train_metrics, val_metrics, history, use_amp):
    return {
        "epoch": epoch + 1,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else None,
        "best_metric": best_metric,
        "monitor": MONITOR,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
        "config": {
            "NUM_CLASSES": NUM_CLASSES,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "PATIENCE": PATIENCE,
            "MONITOR": MONITOR,
            "SEED": SEED,
        },
    }