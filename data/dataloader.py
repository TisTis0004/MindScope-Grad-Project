from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class PTStreamWindowsDataset(Dataset):
    """
    Reads manifest.jsonl with lines: {"pt_path": "...", "n": N}
    Each pt file contains:
      x: [N, C, T]
      y: [N]
    Loads only one pt file at a time (last-file cache).
    """

    def __init__(self, manifest_path: Path | str, transform=None):
        self.transform = transform
        manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        self.items: List[Tuple[Path, int]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append((Path(obj["pt_path"]), int(obj["n"])))

        if not self.items:
            raise RuntimeError("Manifest is empty")

        self.index: List[Tuple[int, int]] = []
        for fi, (_, n) in enumerate(self.items):
            for li in range(n):
                self.index.append((fi, li))

        self._last_fi = None
        self._last_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, li = self.index[idx]

        if self._last_fi != fi:
            self._last_data = torch.load(self.items[fi][0], map_location="cpu")
            self._last_fi = fi

        x = self._last_data["x"][li]  # [C, T]
        y = self._last_data["y"][li].long()  # class index for CrossEntropyLoss

        if self.transform is not None:
            x = self.transform(x)

        return {"x": x, "y": y}


def collate_xy(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"x": x, "y": y}


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
    ):
        ds = PTStreamWindowsDataset(ds, transform)
        for i in range(10):
            s = ds[i]
            print(i, s["y"].item(), s["y"].dtype)
        self.dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    def return_Loader(self):
        return self.dl


if __name__ == "__main__":
    ds = PTStreamWindowsDataset(
        "cache_windows_train_8_classes/manifest.jsonl", transform=None
    )

    for i in range(len(ds)):
        print(ds[i]["y"]) if ds[i]["y"].item() != 0 else 0

    # sample = ds[10]["x"]
    # label = ds[10]["y"]
    # print(sample)
    # print("=" * 20)
    # print(label)
    # print("=" * 20)
    # print(sample.shape)
