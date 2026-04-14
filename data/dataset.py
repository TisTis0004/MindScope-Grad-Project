from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TUHZRecord:
    stem: str
    csv_path: Path
    csv_bi_path: Path
    edf_path: Path
    subject_dir: Optional[str] = None
    session_dir: Optional[str] = None
    montage_dir: Optional[str] = None


class TUHZDataset(Dataset):
    """
    TUHZ hierarchy (as you described):
      split_root/
        <subject_folder>/
          <session_folder>/ (e.g., s001_2002)
            <montage_folder>/ (e.g., 03_tcp_ar_a)
              <stem>.csv
              <stem>.csv_bi
              <stem>.edf

    Each recording is a triplet.
    """

    def __init__(
        self,
        split_root: str | Path,
        return_paths_only: bool = True,
        allowed_montages: Optional[set[str]] = None,
        strict: bool = True,
    ) -> None:
        self.split_root = Path(split_root)
        if not self.split_root.exists():
            raise FileNotFoundError(f"split_root not found: {self.split_root}")

        self.return_paths_only = return_paths_only
        self.allowed_montages = allowed_montages
        self.strict = strict

        self.records: List[TUHZRecord] = self._index_split()

        if len(self.records) == 0:
            raise RuntimeError(
                f"No recordings found under {self.split_root}. "
                f"Check path + naming patterns."
            )

    def _index_split(self) -> List[TUHZRecord]:
        # Index by finding all .edf and checking its sibling .csv and .csv_bi
        edf_files = sorted(self.split_root.rglob("*.edf"))
        records: List[TUHZRecord] = []

        for edf_path in edf_files:
            montage_dir = edf_path.parent.name
            session_dir = (
                edf_path.parent.parent.name if edf_path.parent.parent else None
            )
            subject_dir = (
                edf_path.parent.parent.parent.name
                if edf_path.parent.parent and edf_path.parent.parent.parent
                else None
            )

            if (
                self.allowed_montages is not None
                and montage_dir not in self.allowed_montages
            ):
                continue

            stem = edf_path.stem  # e.g., aaaaabnu_s002_t000
            csv_path = edf_path.with_suffix(".csv")
            csv_bi_path = edf_path.parent / f"{stem}.csv_bi"

            ok = csv_path.exists() and csv_bi_path.exists()

            if not ok:
                if self.strict:
                    raise FileNotFoundError(
                        f"Missing triplet for stem={stem}\n"
                        f"Expected:\n  {edf_path}\n  {csv_path}\n  {csv_bi_path}\n"
                    )
                else:
                    continue

            records.append(
                TUHZRecord(
                    stem=stem,
                    csv_path=csv_path,
                    csv_bi_path=csv_bi_path,
                    edf_path=edf_path,
                    subject_dir=subject_dir,
                    session_dir=session_dir,
                    montage_dir=montage_dir,
                )
            )

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        if self.return_paths_only:
            return {
                "stem": rec.stem,
                "csv_path": str(rec.csv_path),
                "csv_bi_path": str(rec.csv_bi_path),
                "edf_path": str(rec.edf_path),
                "subject": rec.subject_dir,
                "session": rec.session_dir,
                "montage": rec.montage_dir,
            }

        raise NotImplementedError(
            "Set return_paths_only=True, or implement loaders for EDF/CSV here."
        )
        
    def to_json(self, out_path: str | Path) -> None:
        out_path = Path(out_path)

        data = []
        for rec in self.records:
            data.append({
                "stem": rec.stem,
                "csv_path": str(rec.csv_path),
                "csv_bi_path": str(rec.csv_bi_path),
                "edf_path": str(rec.edf_path),
                "subject": rec.subject_dir,
                "session": rec.session_dir,
                "montage": rec.montage_dir,
            })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(data)} records to {out_path}")
    

def csv_has_seizure(csv_path: str | Path) -> bool:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return False

    df = pd.read_csv(
        csv_path,
        comment="#",          # ⬅️ THIS skips all metadata rows
        usecols=["label"],    # ⬅️ safer than column index
        dtype=str
    )

    labels = df["label"].str.strip().str.lower()

    # TUH seizure labels are usually cpsz, spsz, fnsz, gnsz, etc.
    return labels.str.contains("sz").any()


dataset = TUHZDataset(
    r"D:\EEG_DATA\tuh_eval", allowed_montages=None
)  # We can use allowed montages only as follows if needed {"03_tcp_ar_a", "02_tcp_le"} ما بعرف اذا بتفيد بس عمومًا موجودة
dataset.to_json("tuh_train_index_eval.json")
with open('tuh_train_index_eval.json', 'r') as f:
    data=json.load(f)

seizure_records = []

for rec in data:
    if csv_has_seizure(rec["csv_path"]):
        seizure_records.append(rec)


outpath=Path('eeg_seizure_only_eval.json')
with outpath.open('w' , encoding='utf-8') as f:
    json.dump(seizure_records , f, indent=2)
print(f"Seizure recordings: {len(seizure_records)} / {len(data)}")
