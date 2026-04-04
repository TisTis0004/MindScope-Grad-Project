import json
import random
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class PTStreamWindowsDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        transform=None,
        task="multiclass",
        balance_data=False,
    ):
        self.transform = transform
        self.task = task  # "binary" or "multiclass"
        manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        self.items: List[Tuple[Path, List[int]]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                pt_path = Path(obj["pt_path"])

                # Check if specific indices are provided (for Stage 2 filtered data)
                if "indices" in obj:
                    valid_indices = obj["indices"]
                else:
                    valid_indices = list(range(int(obj["n"])))

                if valid_indices:
                    self.items.append((pt_path, valid_indices))

        if not self.items:
            raise RuntimeError("Manifest is empty or contains no valid indices.")

        self.index: List[Tuple[int, int]] = []
        for fi, (_, indices) in enumerate(self.items):
            for li in indices:
                self.index.append((fi, li))

        if balance_data and self.task == "binary":
            print("Scanning dataset to balance classes (this takes a few seconds)...")
            class_0_idx = []
            class_1_idx = []

            global_idx = 0
            for fi, (pt_path, indices) in enumerate(self.items):
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                y_arr = data["y"]
                for li in indices:
                    y_val = y_arr[li].item()
                    y_mapped = 1 if y_val > 0 else 0
                    if y_mapped == 1:
                        class_1_idx.append(global_idx)
                    else:
                        class_0_idx.append(global_idx)
                    global_idx += 1

            min_size = len(class_1_idx)
            print(
                f"Original - Class 0: {len(class_0_idx)} | Class 1: {len(class_1_idx)}"
            )

            # Undersample bckg to match seiz
            class_0_sampled = random.sample(class_0_idx, min_size)
            balanced_indices = class_0_sampled + class_1_idx
            random.shuffle(balanced_indices)

            self.index = [self.index[i] for i in balanced_indices]
            print(f"Balanced - Total windows loaded: {len(self.index)} (1:1 ratio)")

        self._last_fi = None
        self._last_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, li = self.index[idx]

        if self._last_fi != fi:
            self._last_data = torch.load(
                self.items[fi][0], map_location="cpu", weights_only=False
            )
            self._last_fi = fi

        x = self._last_data["x"][li]
        y = self._last_data["y"][li].long()

        if self.task == "binary":
            # 0 is bckg. If y > 0, it's a seiz, so map it to 1.
            y = torch.tensor(1 if y.item() > 0 else 0, dtype=torch.long)
        elif self.task == "multiclass":
            # Original seizures are 8 so 0-7
            y = torch.tensor(max(0, y.item() - 1), dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return {
            "x": x,
            "y": y,
            "file_idx": fi,
            "window_idx": li,
        }


def collate_xy(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    file_idx = torch.tensor([b["file_idx"] for b in batch])
    window_idx = torch.tensor([b["window_idx"] for b in batch])
    return {"x": x, "y": y, "file_idx": file_idx, "window_idx": window_idx}


class Loader:
    def __init__(
        self,
        ds="cache_windows/manifest.jsonl",
        transform=None,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_xy,
        task="multiclass",
        balance_data=False,
    ):
        self.ds = PTStreamWindowsDataset(
            ds, transform, task=task, balance_data=balance_data
        )
        self.dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    def return_Loader(self):
        return self.dl
