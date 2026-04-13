import copy
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
)
from tqdm import tqdm
from models.ResBiLSTM import BinarySeizureCNN, MultiClassSeizureCNN

# =========================================================
# CONFIG
# =========================================================
NUM_CLASSES = 9
EPOCHS = 30
LR = 1e-3
PATIENCE = 10
MONITOR = "f1_macro"
CHECKPOINT_PATH = "best_model_checkpoint.pt"
HISTORY_CSV_PATH = "training_history.csv"


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def is_better(current, best, mode="max"):
    if best is None:
        return True
    return current > best if mode == "max" else current < best


def compute_classification_metrics(
    y_true, y_pred, y_prob=None, num_classes=None, topk=2
):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"] = (
        p_m,
        r_m,
        f1_m,
    )

    if y_prob is not None:
        if num_classes == 2:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                metrics["auc"] = np.nan
        elif num_classes > 2:
            try:
                metrics["auc_ovr_macro"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            except:
                metrics["auc_ovr_macro"] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=np.arange(num_classes)
    )
    return metrics


def build_model(device, weights=None, num_classes=2, task="binary"):
    if task == "binary":
        model = BinarySeizureCNN()
    else:
        model = MultiClassSeizureCNN(num_classes=num_classes)

    if weights:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        model.load_state_dict(
            ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        )
    return model.to(device)


def get_dynamic_class_weights(class_counts, device, task="binary"):
    N, K = sum(class_counts), len(class_counts)
    weights = []
    if task == "binary":
        weights = [N / (K * c) if c > 0 else 0.0 for c in class_counts]
    else:
        max_N = max(class_counts)
        weights = [(N / (K * c)) * (max_N / c) if c > 0 else 0.0 for c in class_counts]
    w_tensor = torch.tensor(weights, dtype=torch.float)
    return (w_tensor / w_tensor.mean()).to(device)


def build_training_components(model, device, class_counts, task):
    weights = get_dynamic_class_weights(class_counts, device, task)
    if task == "binary":
        pos_weight = torch.tensor([weights[1] / weights[0]], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    return criterion, optimizer, scheduler, scaler


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    use_amp=True,
    num_classes=2,
    topk=2,
    total_batches=None,
):
    model.train()
    total_loss, total_samples, total_correct = 0.0, 0, 0
    all_targets, all_preds, all_probs = [], [], []
    running_f1 = 0.0

    # leave=True ensures the bar stays visible after finishing
    pbar = tqdm(loader, leave=True, desc="Train", total=total_batches, mininterval=0.5)

    for i, batch in enumerate(pbar):
        x, y = (
            batch["x"].to(device, non_blocking=True),
            batch["y"].to(device, non_blocking=True).long(),
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = (
                criterion(logits.squeeze(-1), y.float())
                if num_classes == 2
                else criterion(logits, y)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Predictions for metrics
        if num_classes == 2:
            prob_pos = torch.sigmoid(logits.squeeze(-1))
            preds = (prob_pos >= 0.5).long()
            probs = torch.stack([1 - prob_pos, prob_pos], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        # Update running stats
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (preds == y).sum().item()

        # Accumulate for F1
        all_targets.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

        # Update F1 every 10 batches to save CPU
        if i % 10 == 0 or i == total_batches:
            temp_y = torch.cat(all_targets).numpy()
            temp_p = torch.cat(all_preds).numpy()
            running_f1 = f1_score(temp_y, temp_p, average="macro", zero_division=0)

        # Live stats on progress bar
        pbar.set_postfix(
            {
                "loss": f"{total_loss/total_samples:.4f}",
                "acc": f"{total_correct/total_samples:.4f}",
                "f1": f"{running_f1:.4f}",
                "lr": f"{get_current_lr(optimizer):.1e}",
            }
        )

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, num_classes, topk)
    metrics["loss"] = total_loss / total_samples
    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp=True,
    num_classes=2,
    topk=2,
    desc="Eval",
    total_batches=None,
):
    model.eval()
    total_loss, total_samples, total_correct = 0.0, 0, 0
    all_targets, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, leave=True, desc=desc, total=total_batches)
    for batch in pbar:
        x, y = (
            batch["x"].to(device, non_blocking=True),
            batch["y"].to(device, non_blocking=True).long(),
        )
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = (
                criterion(logits.squeeze(-1), y.float())
                if num_classes == 2
                else criterion(logits, y)
            )

        if num_classes == 2:
            prob_pos = torch.sigmoid(logits.squeeze(-1))
            preds = (prob_pos >= 0.5).long()
            probs = torch.stack([1 - prob_pos, prob_pos], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        total_correct += (preds == y).sum().item()

        all_targets.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

        pbar.set_postfix(
            {
                "loss": f"{total_loss/total_samples:.4f}",
                "acc": f"{total_correct/total_samples:.4f}",
            }
        )

    metrics = compute_classification_metrics(
        torch.cat(all_targets).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_probs).numpy(),
        num_classes,
        topk,
    )
    metrics["loss"] = total_loss / total_samples
    return metrics


def save_history_to_csv(history, csv_path):
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def get_monitored_metric(val_metrics):
    if MONITOR == "val_loss":
        return val_metrics["loss"]
    if MONITOR == "f1_macro":
        return val_metrics["f1_macro"]
    return val_metrics.get(MONITOR, val_metrics["accuracy"])


def build_log_row(epoch, optimizer, train_metrics, val_metrics, epoch_time, topk):
    return {
        "epoch": epoch + 1,
        "lr": get_current_lr(optimizer),
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "train_f1_macro": train_metrics["f1_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "epoch_time_sec": epoch_time,
    }


def build_epoch_message(epoch, total_epochs, log_row, train_metrics, val_metrics, topk):
    return (
        f"Epoch {epoch+1}/{total_epochs} | "
        f"Train Loss {train_metrics['loss']:.4f} | "
        f"Val Loss {val_metrics['loss']:.4f} | "
        f"Val F1 {val_metrics['f1_macro']:.4f}"
    )


def make_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_metric,
    train_metrics,
    val_metrics,
    history,
    use_amp,
):
    return {
        "epoch": epoch + 1,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "monitor": MONITOR,
        "history": history,
    }
