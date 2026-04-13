import os
import time
import torch
from braindecode.util import set_random_seeds
from data.dataloader import Loader
from helper.train_helper import *

from dotenv import load_dotenv

load_dotenv()

TASK = "binary"  # or "multiclass"
SEED = int(os.getenv("SEED_1"))


if TASK == "binary":
    NUM_CLASSES, CLASS_COUNTS = 2, [118873, 14737]
    TRAIN_MANIFEST = "cache_windows_train_8_classes/manifest.jsonl"
    VAL_MANIFEST = "cache_windows_eval_8_classes/manifest.jsonl"
    WEIGHTS_SAVE_NAME = "binary_best_model.pt"
else:
    NUM_CLASSES, CLASS_COUNTS = 8, [7598, 5601, 17837, 36500, 1276, 1226, 1589, 292]
    TRAIN_MANIFEST = "cache_windows_train_8_classes/stage2_filtered_manifest.jsonl"
    VAL_MANIFEST = "cache_windows_eval_8_classes/stage2_eval_filtered_manifest.jsonl"
    WEIGHTS_SAVE_NAME = "multiclass_best_model.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(seed=SEED, cuda=(device.type == "cuda"))

    model = build_model(device, weights=None, num_classes=NUM_CLASSES, task=TASK)
    if os.path.exists(WEIGHTS_SAVE_NAME):
        print(f"Loading existing weights from {WEIGHTS_SAVE_NAME}")
        model = build_model(
            device, weights=WEIGHTS_SAVE_NAME, num_classes=NUM_CLASSES, task=TASK
        )

    criterion, optimizer, scheduler, scaler = build_training_components(
        model, device, CLASS_COUNTS, TASK
    )

    train_loader_obj = Loader(ds_path=TRAIN_MANIFEST, balanced=False, batch_size=128)
    val_loader_obj = Loader(ds_path=VAL_MANIFEST, balanced=False, batch_size=128)

    train_loader = train_loader_obj.return_Loader()
    val_loader = val_loader_obj.return_Loader()

    best_metric, history = None, []
    mode = "min" if MONITOR == "val_loss" else "max"

    print(f"\nStarting {TASK} training...")
    for epoch in range(EPOCHS):
        start = time.time()

        train_m = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            num_classes=NUM_CLASSES,
            total_batches=train_loader_obj.total_batches,
        )

        val_m = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=NUM_CLASSES,
            total_batches=val_loader_obj.total_batches,
        )

        scheduler.step()

        curr_m = get_monitored_metric(val_m)
        log_row = build_log_row(
            epoch, optimizer, train_m, val_m, time.time() - start, 2
        )
        history.append(log_row)

        # Print epoch summary
        print(f"\n" + "=" * 50)
        print(build_epoch_message(epoch, EPOCHS, log_row, train_m, val_m, 2))

        # --- RESTORED: PRINT CONFUSION MATRIX ---
        print("\nVal Confusion Matrix:")
        print(val_m["confusion_matrix"])
        print("=" * 50 + "\n")

        if is_better(curr_m, best_metric, mode):
            best_metric = curr_m
            torch.save(
                make_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_metric,
                    train_m,
                    val_m,
                    history,
                    True,
                ),
                CHECKPOINT_PATH,
            )
            print(f"*** New Best {MONITOR}: {best_metric:.4f} - Checkpoint Saved ***")
        else:
            print(f"No improvement in {MONITOR}.")


if __name__ == "__main__":
    main()
