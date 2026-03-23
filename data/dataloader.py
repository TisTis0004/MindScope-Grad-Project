from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset ,DataLoader


import torch
import torch.nn as nn

class STFTSpec(nn.Module):
    def __init__(self, fs, n_fft=256, hop_length=64, fmin=1.0, fmax=40.0, eps=1e-8):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

        # store window as buffer (moves with model / picklable)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        # precompute freq mask once
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs)  # [F]
        mask = (freqs >= fmin) & (freqs <= fmax)
        self.register_buffer("freq_mask", mask, persistent=False)

    def forward(self, x):
        # x: [C, T]
        S = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window.to(x.device),
            return_complex=True,
        )  # [C, F, TT]

        P = (S.real**2 + S.imag**2)      # [C, F, TT]
        P = P[:, self.freq_mask, :]      # keep only 1–40 Hz -> [C, F_band, TT]
        return torch.log(P + self.eps)


class SpecZNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return (x - mean) / std
    
class PTStreamWindowsDataset(Dataset):
    """
    Reads manifest.jsonl with lines: {"pt_path": "...", "n": N}
    Each pt file contains:
      x: [N, C, T]
      y: [N]
    This Dataset loads ONLY ONE pt file at a time (last-file cache).
    """
    def __init__(self, manifest_path:Path,transform):
        self.transform=transform
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

        # global index: (file_id, local_idx)
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

        x = self._last_data["x"][li]          # [C, T]
        y = self._last_data["y"][li].float()  # scalar float for BCE
        if self.transform is not None:
            x = self.transform(x)
        return {"x": x, "y": y}

def collate_xy(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)  # [B,C,T]
    y = torch.stack([b["y"] for b in batch], dim=0)  # [B]
    return {"x": x, "y": y}

class Loader():
    def __init__(self,ds='cache_windows/manifest.jsonl',transform=None,
        batch_size=32,
        shuffle=False, # it's the issue
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_xy):
        
        
       ds=PTStreamWindowsDataset(ds , transform)
       self.dl=DataLoader(ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_xy) 
    def return_Loader(self):
        return self.dl
    
if __name__=='__main__':
    ds = PTStreamWindowsDataset("cache_windows/manifest.jsonl",transform = lambda x: SpecZNorm()(STFTSpec()(x)))

    sample = ds[0]["x"]
    print(sample.shape)
        