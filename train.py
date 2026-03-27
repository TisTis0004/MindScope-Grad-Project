import os
import time

import numpy as np
import torch
from braindecode.util import set_random_seeds

from helper.train_helper import (
    CHECKPOINT_PATH,
    EPOCHS,
    HISTORY_CSV_PATH,
    MONITOR,
    NUM_CLASSES,
    PATIENCE,
    SEED,
    build_epoch_message,
    build_loaders,
    build_log_row,
    build_model,
    build_training_components,
    evaluate,
    get_monitored_metric,
    is_better,
    make_checkpoint,
    save_history_to_csv,
    train_one_epoch,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    mode = "min" if MONITOR == "val_loss" else "max"
    topk = min(2, NUM_CLASSES)

    print(f"Using device: {device}")
    print(f"AMP enabled: {use_amp}")

    set_random_seeds(seed=SEED, cuda=(device.type == "cuda"))

    model = build_model(device, weights="best_model_checkpoint.pt")
    criterion, optimizer, scheduler, scaler = build_training_components(model, device)

    transform = None
    train_loader, val_loader = build_loaders(transform=transform)

    best_metric = None
    best_epoch = -1
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            num_classes=NUM_CLASSES,
            topk=topk,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            num_classes=NUM_CLASSES,
            topk=topk,
            desc="Val",
        )

        scheduler.step()

        current_metric = get_monitored_metric(val_metrics)
        epoch_time = time.time() - epoch_start

        log_row = build_log_row(
            epoch=epoch,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch_time=epoch_time,
            topk=topk,
        )

        history.append(log_row)
        save_history_to_csv(history, HISTORY_CSV_PATH)

        print(build_epoch_message(epoch, EPOCHS, log_row, train_metrics, val_metrics, topk))
        print("Val Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        if is_better(current_metric, best_metric, mode=mode):
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = make_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                history=history,
                use_amp=use_amp,
            )

            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Saved best checkpoint at epoch {epoch + 1} with {MONITOR}={best_metric:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} "
            f"with {checkpoint['monitor']}={checkpoint['best_metric']:.6f}"
        )

    total_time = time.time() - start_time
    print(f"Best epoch: {best_epoch}")
    print(f"Training completed in {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()