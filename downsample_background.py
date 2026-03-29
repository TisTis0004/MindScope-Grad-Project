from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import torch


def load_manifest(manifest_path: str | Path):
    manifest_path = Path(manifest_path)
    items = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_manifest(items, manifest_path: str | Path):
    manifest_path = Path(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_classes(manifest_items):
    counts = Counter()
    for item in manifest_items:
        obj = torch.load(item["pt_path"], map_location="cpu")
        y = obj["y"]
        counts.update(y.tolist())
    return counts


def downsample_background(
    manifest_path: str | Path,
    output_manifest_path: str | Path | None = None,
    background_class: int = 0,
    bg_multiplier: int = 5,
    seed: int = 42,
):
    torch.manual_seed(seed)

    manifest_items = load_manifest(manifest_path)
    if not manifest_items:
        raise ValueError("Manifest is empty")

    class_counts = count_classes(manifest_items)
    print("Original class counts:", dict(class_counts))

    seizure_counts = {
        cls: cnt for cls, cnt in class_counts.items()
        if cls != background_class
    }
    if not seizure_counts:
        raise ValueError("No seizure classes found")

    largest_seizure = max(seizure_counts.values())
    target_bg = bg_multiplier * largest_seizure
    current_bg = class_counts.get(background_class, 0)

    print(f"Largest seizure class count: {largest_seizure}")
    print(f"Target background count: {target_bg}")
    print(f"Current background count: {current_bg}")

    if current_bg <= target_bg:
        print("Background is already within target. Nothing to do.")
        return

    # Collect all background indices globally
    bg_locations = []  # list of (file_idx, sample_idx)
    for file_idx, item in enumerate(manifest_items):
        obj = torch.load(item["pt_path"], map_location="cpu")
        y = obj["y"]
        bg_idx = (y == background_class).nonzero(as_tuple=False).flatten().tolist()
        for sample_idx in bg_idx:
            bg_locations.append((file_idx, sample_idx))

    print(f"Total background samples found: {len(bg_locations)}")

    perm = torch.randperm(len(bg_locations))
    keep_bg_global = set(
        tuple(bg_locations[i]) for i in perm[:target_bg].tolist()
    )

    new_manifest = []

    for file_idx, item in enumerate(manifest_items):
        pt_path = Path(item["pt_path"])
        obj = torch.load(pt_path, map_location="cpu")

        x = obj["x"]
        y = obj["y"]

        keep_mask = torch.ones(len(y), dtype=torch.bool)

        bg_idx = (y == background_class).nonzero(as_tuple=False).flatten().tolist()
        for sample_idx in bg_idx:
            if (file_idx, sample_idx) not in keep_bg_global:
                keep_mask[sample_idx] = False

        x_new = x[keep_mask]
        y_new = y[keep_mask]

        obj["x"] = x_new
        obj["y"] = y_new

        torch.save(obj, pt_path)

        new_n = int(len(y_new))
        new_manifest.append({
            "pt_path": str(pt_path),
            "n": new_n,
        })

        print(f"{pt_path.name}: {len(y)} -> {new_n}")

    if output_manifest_path is None:
        output_manifest_path = manifest_path

    save_manifest(new_manifest, output_manifest_path)

    new_counts = count_classes(new_manifest)
    print("New class counts:", dict(new_counts))


if __name__ == "__main__":
    downsample_background(
        manifest_path="cache_windows/manifest.jsonl",
        output_manifest_path="cache_windows/manifest.jsonl",
        background_class=0,
        bg_multiplier=5,
        seed=42,
    )
    
# New class counts: {0: 44850, 3: 1376, 1: 4391, 2: 8970} new counts 5 : 1 