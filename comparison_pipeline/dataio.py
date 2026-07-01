from pathlib import Path
import gzip
import json

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset


class ParquetPacketDataset(Dataset):
    def __init__(self, path, limit=None):
        files = data_files(path)
        if not files:
            raise FileNotFoundError(f"No parquet/jsonl files found under {path}")
        frames = []
        for file in files:
            if str(file).endswith(".jsonl.gz"):
                frames.append(pd.DataFrame(read_jsonl_gz(file)))
            else:
                frames.append(pd.read_parquet(file))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["feature", "label"])
        if limit is not None:
            df = df.head(int(limit))
        self.features = df["feature"].tolist()
        self.labels = df["label"].astype("int64").tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"feature": self.features[idx], "label": self.labels[idx]}


class JsonlPacketIterableDataset(IterableDataset):
    def __init__(self, path, limit=None):
        self.files = jsonl_files(path)
        if not self.files:
            raise FileNotFoundError(f"No jsonl.gz files found under {path}")
        self.limit = None if limit is None else int(limit)

    def __iter__(self):
        seen = 0
        for file in self.files:
            with gzip.open(file, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
                    seen += 1
                    if self.limit is not None and seen >= self.limit:
                        return


def parquet_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
        return sorted(p for p in path.rglob("*.parquet") if p.is_file())
    return []


def data_files(path):
    files = parquet_files(path)
    if files:
        return files
    path = Path(path)
    candidates = []
    if path.suffix == ".parquet":
        candidates.append(path.with_suffix(".jsonl.gz"))
    if path.is_dir():
        candidates.extend(sorted(path.glob("*.jsonl.gz")))
    elif str(path).endswith(".jsonl.gz") and path.exists():
        candidates.append(path)
    return [p for p in candidates if p.exists()]


def jsonl_files(path):
    path = Path(path)
    candidates = []
    if path.suffix == ".parquet":
        candidates.append(path.with_suffix(".jsonl.gz"))
    if path.is_dir():
        candidates.extend(sorted(path.glob("*.jsonl.gz")))
    elif str(path).endswith(".jsonl.gz"):
        candidates.append(path)
    return [p for p in candidates if p.exists()]


def packet_dataset(path, limit=None):
    if jsonl_files(path):
        return JsonlPacketIterableDataset(path, limit=limit)
    return ParquetPacketDataset(path, limit=limit)


def read_jsonl_gz(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def collate_packet_batch(batch):
    features = torch.stack([torch.tensor([row["feature"]], dtype=torch.float32) for row in batch])
    labels = torch.tensor([row["label"] for row in batch], dtype=torch.long)
    return {"feature": features, "label": labels}
