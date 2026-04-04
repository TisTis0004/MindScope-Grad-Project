import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataloader import PTStreamWindowsDataset, collate_xy
from helper.train_helper import build_model


def create_filtered_manifest(
    model_weights: str,
    input_manifest: str,
    output_manifest: str,
    threshold: float = 0.9,
    device: str = "cuda",
):
    print(f"Loading binary model from {model_weights}...")
    # 2 output classes for Binary Stage
    model = build_model(device, weights=model_weights, num_classes=2)
    model.eval()

    print(f"Loading dataset from {input_manifest}...")
    # Load as binary to match the model
    dataset = PTStreamWindowsDataset(input_manifest, task="binary")
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_xy
    )

    # Dictionary to keep track of which windows passed the threshold per file
    filtered_data = {}

    print(f"Filtering with confidence threshold \u03b8 = {threshold}...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Filtering"):
            x = batch["x"].to(device, non_blocking=True)
            file_indices = batch["file_idx"].cpu().numpy()
            window_indices = batch["window_idx"].cpu().numpy()

            # Get probabilities
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

            # Prob of class 1 (Seizure)
            seizure_probs = probs[:, 1].cpu().numpy()

            for i in range(len(seizure_probs)):
                if seizure_probs[i] >= threshold:
                    fi = file_indices[i]
                    wi = window_indices[i]
                    pt_path = str(dataset.items[fi][0])

                    if pt_path not in filtered_data:
                        filtered_data[pt_path] = []
                    filtered_data[pt_path].append(int(wi))

    print(f"Writing Stage 2 manifest to {output_manifest}...")
    with open(output_manifest, "w", encoding="utf-8") as f:
        for pt_path, indices in filtered_data.items():
            # Sort indices to maintain temporal order
            indices = sorted(indices)
            obj = {"pt_path": pt_path, "n": len(indices), "indices": indices}
            f.write(json.dumps(obj) + "\n")

    print("Filtering complete! Ready for Stage 2.")


if __name__ == "__main__":
    create_filtered_manifest(
        model_weights="binary_best_model.pt",
        input_manifest="cache_windows_val_8_classes/manifest.jsonl",
        output_manifest="cache_windows_val_8_classes/stage2_val_filtered_manifest.jsonl",
        threshold=0.9,
    )
