from __future__ import annotations
import json
import random
import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader

class BalancedBufferEEGDataset(IterableDataset):
    """
    Reads files sequentially (fast disk I/O) but splits them into two RAM buffers
    to guarantee a balanced 50/50 mix of Seizure vs Background.
    """
    def __init__(
        self, 
        manifest_path: Path | str, 
        transform=None, 
        buffer_capacity: int = 2000,
        seizure_label: int = 1
    ):
        super().__init__()
        self.transform = transform
        self.manifest_path = Path(manifest_path)
        self.buffer_capacity = buffer_capacity
        self.seizure_label = seizure_label
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)

        # Load all file paths into memory
        self.files: List[Path] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self.files.append(Path(obj["pt_path"]))

        if not self.files:
            raise RuntimeError("Manifest is empty")

    def __iter__(self):
        # 1. Handle Multi-processing (Prevent workers from reading the same files)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            worker_files = self.files.copy()
        else:
            # Split files among workers so they don't duplicate work
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_files = self.files[worker_id * per_worker : (worker_id + 1) * per_worker]

        # Shuffle the files for this epoch so we don't always read patients in the same order
        random.shuffle(worker_files)

        # 2. Create our Two Buckets
        bg_bucket = []
        sz_bucket = []

        # 3. Main Loop: Read files and yield balanced data
        for file_path in worker_files:
            try:
                # mmap=True is CRITICAL here. It prevents massive RAM spikes and disk thrashing.
                data = torch.load(file_path, map_location="cpu", mmap=True)
                
                # We pull everything into RAM for this specific file
                x_all = data["x"]
                y_all = data["y"].long()
                
                # Split into buckets based on label
                sz_mask = (y_all == self.seizure_label)
                bg_mask = ~sz_mask
                
                # Add to buckets (as a list of dictionaries)
                for i in range(len(y_all)):
                    sample = {"x": x_all[i], "y": y_all[i]}
                    if y_all[i] == self.seizure_label:
                        sz_bucket.append(sample)
                    else:
                        bg_bucket.append(sample)
                        
                # Keep buckets from overflowing RAM
                if len(bg_bucket) > self.buffer_capacity:
                    random.shuffle(bg_bucket)
                    bg_bucket = bg_bucket[:self.buffer_capacity]
                    
                if len(sz_bucket) > self.buffer_capacity:
                    random.shuffle(sz_bucket)
                    sz_bucket = sz_bucket[:self.buffer_capacity]

                # 4. Yield Data! 
                # We only yield if we have SOME data in both buckets to ensure balance.
                while len(bg_bucket) > 0 and len(sz_bucket) > 0:
                    # Pick 1 background randomly
                    bg_idx = random.randint(0, len(bg_bucket) - 1)
                    # Pick 1 seizure randomly
                    sz_idx = random.randint(0, len(sz_bucket) - 1)

                    # Pop the background (we have plenty, so we discard it after using)
                    bg_sample = bg_bucket.pop(bg_idx)
                    
                    # DO NOT pop the seizure immediately! (Oversampling strategy)
                    # We have very few seizures, so we leave it in the bucket to be reused
                    # We only pop it 10% of the time so it eventually cycles out
                    sz_sample = sz_bucket[sz_idx]
                    if random.random() < 0.1: 
                        sz_bucket.pop(sz_idx)

                    # Apply transforms if you have them
                    if self.transform is not None:
                        bg_sample["x"] = self.transform(bg_sample["x"])
                        sz_sample["x"] = self.transform(sz_sample["x"])

                    # Yield them one by one. The DataLoader will batch them automatically.
                    yield bg_sample
                    yield sz_sample

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue


def collate_xy(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"x": x, "y": y}


class Loader:
    def __init__(
        self,
        ds="cache_windows/manifest.jsonl",
        transform=None,
        batch_size=32, # This will now contain roughly 16 BG and 16 SZ per batch
        num_workers=2,
        pin_memory=True, # Highly recommended for GPU training
    ):
        # We use our new BalancedBufferDataset
        ds = BalancedBufferEEGDataset(ds, transform=transform, buffer_capacity=5000)

        print('Creating the Iterable Loader...')
        
        self.dl = DataLoader(
            ds,
            batch_size=batch_size,
            # shuffle=False AND sampler=None are REQUIRED for IterableDatasets
            shuffle=False, 
            sampler=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_xy,
            # drop_last=True helps avoid weird batch sizes at the very end of an epoch
            drop_last=True 
        )

    def return_Loader(self):
        return self.dl