"""Linear probe trained on frozen encoder embeddings."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.capture import pool_last_n_layers_cls
from src.config import ModelConfig
from src.evaluate import evaluate_predictions
from src.runtime import SEED


def build_frozen_encoder_and_tokenizer(model_cfg: ModelConfig, device: torch.device):
    """Load a frozen encoder in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id, revision=model_cfg.revision)
    model = AutoModel.from_pretrained(
        model_cfg.hf_id,
        revision=model_cfg.revision,
        output_hidden_states=True,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, tokenizer


@torch.no_grad()
def extract_embeddings(
    model, dataset, device: torch.device, batch_size: int, n_layers: int = 4,
) -> torch.Tensor:
    """Extract [CLS] embeddings with a frozen encoder."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_embeddings = []
    for batch in tqdm(loader, desc="extracting embeddings", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**batch, output_hidden_states=True)
        all_embeddings.append(pool_last_n_layers_cls(outputs.hidden_states, n_layers=n_layers).cpu())
    return torch.cat(all_embeddings, dim=0)


@dataclass
class ProbeEpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_macro_f1_minority: float


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_label_ids: list[int],
    val_embeddings: torch.Tensor,
    val_label_ids: list[int],
    num_labels: int,
    labels: list[str],
    device: torch.device,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    batch_size: int = 64,
) -> tuple[nn.Linear, list[ProbeEpochLog]]:
    """Train a linear head on cached embeddings."""
    head = nn.Linear(train_embeddings.shape[1], num_labels).to(device)
    optimizer = AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_embeddings, torch.tensor(train_label_ids, dtype=torch.long)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_embeddings, torch.tensor(val_label_ids, dtype=torch.long)),
        batch_size=batch_size, shuffle=False,
    )

    history = []
    for epoch in range(1, num_epochs + 1):
        head.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs} [train]", leave=False)
        for step, (x, y) in enumerate(train_bar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")
        train_loss = running_loss / len(train_loader)

        head.eval()
        val_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"epoch {epoch}/{num_epochs} [val]", leave=False):
                x, y = x.to(device), y.to(device)
                logits = head(x)
                val_loss += criterion(logits, y).item()
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_true.extend(y.cpu().tolist())
        val_loss /= len(val_loader)
        metrics = evaluate_predictions(all_true, all_preds, labels)

        epoch_log = ProbeEpochLog(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=metrics["accuracy"],
            val_macro_f1=metrics["macro_f1"],
            val_macro_f1_minority=metrics["macro_f1_minority"],
        )
        history.append(epoch_log)
        print(
            f"epoch {epoch}/{num_epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_macro_f1_minority={metrics['macro_f1_minority']:.4f}"
        )

    return head, history


def history_to_dicts(history: list[ProbeEpochLog]) -> list[dict]:
    return [asdict(epoch_log) for epoch_log in history]


def training_args_payload(
    model_cfg: ModelConfig,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs_run: int | None = None,
) -> dict:
    """Hyperparameter log for the linear probe run."""
    return {
        "training_regime": "frozen_encoder_linear_probe",
        "model_hf_id": model_cfg.hf_id,
        "model_revision": model_cfg.revision,
        "seed": SEED,
        "precision": "fp32",
        "attn_implementation": "eager",
        "device": str(device),
        "encoder_frozen": True,
        "pooling": "last_4_layers_cls_concat",
        "optimizer": "AdamW",
        "lr_scheduler": None,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_epochs_configured": num_epochs,
        "epochs_run": epochs_run if epochs_run is not None else num_epochs,
    }
