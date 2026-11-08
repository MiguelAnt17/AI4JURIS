"""Load the processed dataset and wrap it into tokenized torch Datasets."""

import json

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import DATA_DIR, DATASET_VARIANTS, LABEL_SET_VARIANTS


def load_label_map(label_set: str = "grupo3") -> dict:
    _, label_json_filename = LABEL_SET_VARIANTS[label_set]
    with open(DATA_DIR / label_json_filename, encoding="utf-8") as f:
        return json.load(f)


def load_splits(dataset_variant: str = "sumario", label_set: str = "grupo3") -> pd.DataFrame:
    """Load the dataset CSV with a numeric `label_id` column."""
    csv_filename, _ = LABEL_SET_VARIANTS[label_set]
    path = DATA_DIR / (csv_filename if label_set != "grupo3" else DATASET_VARIANTS[dataset_variant])
    df = pd.read_csv(path)
    label2id = load_label_map(label_set)["label2id"]
    df["label_id"] = df["Decisao"].map(label2id)
    return df


class SumarioDataset(Dataset):
    """Tokenizes `Sumario` on the fly."""

    def __init__(self, texts: list[str], label_ids: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.label_ids = label_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
        return item


def build_datasets(df: pd.DataFrame, tokenizer, max_length: int) -> dict[str, SumarioDataset]:
    """Build one dataset per split (train/val/test)."""
    datasets = {}
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split]
        datasets[split] = SumarioDataset(
            texts=subset["Sumario"].tolist(),
            label_ids=subset["label_id"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )
    return datasets
