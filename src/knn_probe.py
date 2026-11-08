"""KNN classifier over cached frozen embeddings."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.capture import pool_mean
from src.config import ModelConfig
from src.evaluate import evaluate_predictions
from src.runtime import SEED


@torch.no_grad()
def extract_mean_pooled_embeddings(
    model, dataset, device: torch.device, batch_size: int,
) -> torch.Tensor:
    """Extract mean-pooled embeddings with a frozen encoder."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_embeddings = []
    for batch in tqdm(loader, desc="extracting mean-pooled embeddings", leave=False):
        attention_mask = batch["attention_mask"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs, output_hidden_states=True)
        all_embeddings.append(pool_mean(outputs.hidden_states[-1], attention_mask).cpu())
    return torch.cat(all_embeddings, dim=0)


@dataclass
class KnnGridLog:
    n_neighbors: int
    weights: str
    metric: str
    val_accuracy: float
    val_macro_f1: float
    val_macro_f1_minority: float


def train_knn_probe(
    train_embeddings: np.ndarray,
    train_label_ids: list[int],
    val_embeddings: np.ndarray,
    val_label_ids: list[int],
    labels: list[str],
    n_neighbors_grid: tuple[int, ...] = (3, 5, 7, 9, 11, 15, 21, 31),
    weights: str = "distance",
    metric: str = "cosine",
) -> tuple[KNeighborsClassifier, list[KnnGridLog], KnnGridLog]:
    """Grid-search n_neighbors on validation macro-F1 and return the best model."""
    history = []
    best_log: KnnGridLog | None = None
    best_model: KNeighborsClassifier | None = None

    for k in n_neighbors_grid:
        clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
        clf.fit(train_embeddings, train_label_ids)

        val_pred = clf.predict(val_embeddings).tolist()
        metrics = evaluate_predictions(val_label_ids, val_pred, labels)

        log = KnnGridLog(
            n_neighbors=k,
            weights=weights,
            metric=metric,
            val_accuracy=metrics["accuracy"],
            val_macro_f1=metrics["macro_f1"],
            val_macro_f1_minority=metrics["macro_f1_minority"],
        )
        history.append(log)
        print(
            f"n_neighbors={k} - val_accuracy={metrics['accuracy']:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_macro_f1_minority={metrics['macro_f1_minority']:.4f}"
        )

        if best_log is None or log.val_macro_f1 > best_log.val_macro_f1:
            best_log = log
            best_model = clf

    return best_model, history, best_log


def predict_full_proba(clf: KNeighborsClassifier, embeddings: np.ndarray, num_labels: int) -> np.ndarray:
    """Return probabilities with one column per label."""
    proba = clf.predict_proba(embeddings)
    if len(clf.classes_) == num_labels and list(clf.classes_) == list(range(num_labels)):
        return proba
    full = np.zeros((embeddings.shape[0], num_labels), dtype=proba.dtype)
    full[:, clf.classes_] = proba
    return full


def history_to_dicts(history: list[KnnGridLog]) -> list[dict]:
    return [asdict(log) for log in history]


def training_args_payload(
    model_cfg: ModelConfig,
    device: torch.device,
    n_neighbors_grid: tuple[int, ...],
    weights: str,
    metric: str,
    best_n_neighbors: int,
) -> dict:
    """Hyperparameter log for the KNN probe run."""
    return {
        "training_regime": "frozen_encoder_knn_probe",
        "model_hf_id": model_cfg.hf_id,
        "model_revision": model_cfg.revision,
        "seed": SEED,
        "precision": "fp32",
        "attn_implementation": "eager",
        "device": str(device),
        "encoder_frozen": True,
        "pooling": "mean_pooling_last_hidden_state",
        "classifier": "sklearn.neighbors.KNeighborsClassifier",
        "n_neighbors_grid": list(n_neighbors_grid),
        "weights": weights,
        "metric": metric,
        "best_n_neighbors": best_n_neighbors,
    }
