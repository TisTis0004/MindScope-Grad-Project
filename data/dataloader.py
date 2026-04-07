from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


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

        # Load labels once so sampler can use them
        self.labels = []
        for path, _ in self.items:
            data = torch.load(path, map_location="cpu")
            ys = data["y"].long()
            self.labels.extend(ys.tolist())

        self._last_fi = None
        self._last_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, li = self.index[idx]

        if self._last_fi != fi:
            self._last_data = torch.load(self.items[fi][0], map_location="cpu")
            self._last_fi = fi

        x = self._last_data["x"][li]         # [C, T]
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
        use_weighted_sampler=False,
    ):
        ds = PTStreamWindowsDataset(ds, transform)

        # for i in range(min(10, len(ds))):
        #     s = ds[i]
        #     print(i, s["y"].item(), s["y"].dtype)

        sampler = None
        if use_weighted_sampler:
            labels = np.array(ds.labels)

            class_counts = np.bincount(labels)
            print("Class counts:", class_counts)

            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )

        print('creating the loader ')
        self.dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle if sampler is None else False),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    def return_Loader(self):
        return self.dl