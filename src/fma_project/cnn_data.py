"""Frozen-cohort numeric input loader for the planned binary spectrogram CNN."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np


class SpectrogramDataset:
    """NumPy dataset protocol compatible with a future PyTorch DataLoader.

    Each item is (float32 [1,128,1292], float32 [1], integer track ID).
    It performs a fixed dB mapping, without fitting dataset normalization.
    """
    def __init__(self, project_root, split="training", bundle=None):
        self.project_root = Path(project_root).resolve()
        if split not in {"training", "validation", "test"}:
            raise ValueError("Unknown split")
        base = self.project_root / "data/processed/task2"
        self.bundle = Path(bundle) if bundle else base / json.loads((base/"LATEST_FROZEN.json").read_text())["bundle"]
        ledger = json.loads((self.bundle/"bundle.json").read_text())
        if ledger["status"] != "frozen":
            raise ValueError("CNN inputs require a frozen cohort")
        raw = (self.bundle/"manifest.csv").read_bytes()
        if hashlib.sha256(raw).hexdigest() != ledger["files"]["manifest.csv"]:
            raise ValueError("Frozen manifest changed")
        with (self.bundle/"manifest.csv").open(encoding="utf-8",newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [row for row in rows if row["set_split"] == split]
        self.split = split
        self.cohort_id = ledger["candidate_id"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.project_root/row["array_path"]).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError("Array path leaves project")
        array = np.load(path,allow_pickle=False)
        if array.shape != (128,1292) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid array for {row['track_id']}")
        if array.min() < -80.001 or array.max() > .001:
            raise ValueError("Unexpected dB range")
        normalized = np.clip((array+np.float32(80))/np.float32(80),0,1)[None,...]
        label = np.asarray([int(row["hit"])],dtype=np.float32)
        return normalized,label,int(row["track_id"])

    def batch(self, indexes):
        records = [self[index] for index in indexes]
        return (np.stack([r[0] for r in records]),np.stack([r[1] for r in records]),
                np.asarray([r[2] for r in records],dtype=np.int64))
