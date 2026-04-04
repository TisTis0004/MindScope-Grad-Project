import json
import torch
from pathlib import Path
from tqdm import tqdm
from data.dataloader import Loader
from models.models import BinarySeizureCNN

# Configs
BINARY_WEIGHTS = "Binary_Imbalanced_Finetune_Run_2.pt"
THRESHOLD = 0.90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_manifest(input_manifest, output_manifest):
    print(f"Purifying {input_manifest} -> {output_manifest}...")

    # Load Model
    model = BinarySeizureCNN().to(DEVICE)
    checkpoint = torch.load(BINARY_WEIGHTS, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Data
    loader_obj = Loader(ds=input_manifest, task="binary", batch_size=128, shuffle=False)
    loader = loader_obj.return_Loader()

    # Dictionary to keep track of which windows pass the threshold per file
    # file_idx -> list of window_idx
    passed_windows = {}

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["x"].to(DEVICE)
            file_idx = batch["file_idx"].tolist()
            window_idx = batch["window_idx"].tolist()

            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)

            # Find indices where the probability >= 0.9
            passed_mask = probs >= THRESHOLD

            for i, passed in enumerate(passed_mask):
                if passed:
                    f_id = file_idx[i]
                    w_id = window_idx[i]
                    if f_id not in passed_windows:
                        passed_windows[f_id] = []
                    passed_windows[f_id].append(w_id)

    # Write new manifest
    original_items = loader_obj.ds.items  # (pt_path, valid_indices)

    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_original = 0
    total_kept = 0

    with out_path.open("w", encoding="utf-8") as f:
        for f_id, pt_info in enumerate(original_items):
            pt_path = pt_info[0]
            original_n = len(pt_info[1])
            total_original += original_n

            if f_id in passed_windows:
                kept_indices = sorted(passed_windows[f_id])
                total_kept += len(kept_indices)
                d = {"pt_path": str(pt_path), "n": original_n, "indices": kept_indices}
                f.write(json.dumps(d) + "\n")

    print(
        f"Done! Kept {total_kept}/{total_original} windows ({(total_kept/total_original)*100:.2f}% survival)."
    )


if __name__ == "__main__":
    # Purify Training Set
    filter_manifest(
        input_manifest="cache_windows_train_8_classes/manifest.jsonl",
        output_manifest="cache_windows_train_8_classes/stage2_filtered_manifest.jsonl",
    )
    # Purify Validation Set
    filter_manifest(
        input_manifest="cache_windows_eval_8_classes/manifest.jsonl",
        output_manifest="cache_windows_eval_8_classes/stage2_eval_filtered_manifest.jsonl",
    )
