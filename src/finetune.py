"""Fine-tuning loop shared by all encoder models."""

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import ModelConfig
from src.evaluate import evaluate_predictions, per_class_f1
from src.runtime import SEED


def build_model_and_tokenizer(model_cfg: ModelConfig, num_labels: int, device: torch.device):
    """Load the model with a fresh classification head, plus its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id, revision=model_cfg.revision)
    from_pretrained_kwargs = dict(
        revision=model_cfg.revision,
        num_labels=num_labels,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        hidden_dropout_prob=model_cfg.dropout,
        attention_probs_dropout_prob=model_cfg.dropout,
    )
    config = AutoConfig.from_pretrained(model_cfg.hf_id, revision=model_cfg.revision)
    if hasattr(config, "classifier_dropout"):
        from_pretrained_kwargs["classifier_dropout"] = model_cfg.dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.hf_id, **from_pretrained_kwargs,
    ).to(device)
    return model, tokenizer


def training_args_payload(model_cfg: ModelConfig, device: torch.device, epochs_run: int | None = None) -> dict:
    """Hyperparameter log saved with each run."""
    return {
        "training_regime": "full_finetune",
        "model_hf_id": model_cfg.hf_id,
        "model_revision": model_cfg.revision,
        "seed": SEED,
        "precision": "fp32",
        "attn_implementation": "eager",
        "device": str(device),
        "encoder_frozen": False,
        "pooling": "last_layer_cls",
        "dropout": model_cfg.dropout,
        "optimizer": "AdamW",
        "lr_scheduler": "linear_warmup_decay",
        "learning_rate": model_cfg.learning_rate,
        "weight_decay": model_cfg.weight_decay,
        "weight_decay_excludes_bias_and_layernorm": True,
        "label_smoothing": model_cfg.label_smoothing,
        "max_grad_norm": model_cfg.max_grad_norm,
        "warmup_ratio": model_cfg.warmup_ratio,
        "batch_size": model_cfg.batch_size(device.type),
        "gradient_accumulation_steps": model_cfg.gradient_accumulation_steps,
        "class_weighted_loss": "balanced (inverse class frequency, see compute_class_weights)",
        "early_stopping_patience": model_cfg.early_stopping_patience,
        "num_train_epochs_configured": model_cfg.num_train_epochs,
        "epochs_run": epochs_run if epochs_run is not None else model_cfg.num_train_epochs,
    }


def compute_class_weights(label_ids: list[int], num_labels: int) -> torch.Tensor:
    """Compute balanced class weights."""
    counts = torch.bincount(torch.tensor(label_ids, dtype=torch.long), minlength=num_labels).float()
    n_samples = counts.sum()
    return n_samples / (num_labels * counts)


_NO_WEIGHT_DECAY = ("bias", "LayerNorm.weight")


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if any(nd in name for nd in _NO_WEIGHT_DECAY) else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_accuracy: float
    train_macro_f1: float
    train_micro_f1: float
    train_macro_f1_minority: float
    train_f1_per_class: dict[str, float]
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_micro_f1: float
    val_macro_f1_minority: float
    val_f1_per_class: dict[str, float]


def _save_pretrained_checkpoint(model: nn.Module, checkpoint_dir: Path) -> None:
    model.save_pretrained(checkpoint_dir)


def _trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone the trainable parameters to CPU."""
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}


def _save_resume_state(
    resume_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_macro_f1: float,
    best_epoch: int | None,
    best_state_dict: dict | None,
    epochs_without_improvement: int,
    history: list[EpochLog],
) -> None:
    """Save the state needed to resume an interrupted run."""
    state = {
        "epoch": epoch,
        "model_state_dict": _trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp_path = resume_path.with_suffix(".pt.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(resume_path)


def _load_resume_state(resume_path: Path) -> dict:
    return torch.load(resume_path, map_location="cpu", weights_only=False)


def train(
    model,
    train_dataset,
    val_dataset,
    model_cfg: ModelConfig,
    device: torch.device,
    labels: list[str],
    checkpoint_dir: Path | None = None,
    save_checkpoint: Callable[[nn.Module, Path], None] | None = None,
) -> list[EpochLog]:
    """Train with AdamW and a linear schedule, returning the per-epoch history."""
    batch_size = model_cfg.batch_size(device.type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weights(train_dataset.label_ids, num_labels=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=model_cfg.label_smoothing)

    optimizer = AdamW(_param_groups(model, model_cfg.weight_decay), lr=model_cfg.learning_rate)

    steps_per_epoch = math.ceil(len(train_loader) / model_cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * model_cfg.num_train_epochs
    warmup_steps = int(total_steps * model_cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_macro_f1 = -1.0
    best_state_dict = None
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    start_epoch = 0
    resume_path = None

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint = save_checkpoint or _save_pretrained_checkpoint
        resume_path = checkpoint_dir / "resume_state.pt"

    if resume_path is not None and resume_path.exists():
        state = _load_resume_state(resume_path)
        model.load_state_dict({k: v.to(device) for k, v in state["model_state_dict"].items()}, strict=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        best_val_macro_f1 = state["best_val_macro_f1"]
        best_epoch = state["best_epoch"]
        best_state_dict = state["best_state_dict"]
        epochs_without_improvement = state["epochs_without_improvement"]
        history = state["history"]
        start_epoch = state["epoch"]

        rng_state = state["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if rng_state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

        print(
            f"Resuming from {resume_path}: epoch {start_epoch}/{model_cfg.num_train_epochs} "
            f"already completed (best val_macro_f1={best_val_macro_f1:.4f} at epoch {best_epoch})."
        )

    already_early_stopped = (
        start_epoch > 0
        and model_cfg.early_stopping_patience is not None
        and epochs_without_improvement >= model_cfg.early_stopping_patience
    )
    if already_early_stopped:
        print(
            f"Recorded run had already early-stopped at epoch {start_epoch} -- nothing to resume. "
            f"Delete {resume_path} to force a fresh run."
        )

    epoch_range = () if already_early_stopped else range(start_epoch + 1, model_cfg.num_train_epochs + 1)
    for epoch in epoch_range:
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        train_preds, train_true = [], []

        train_bar = tqdm(
            train_loader, desc=f"epoch {epoch}/{model_cfg.num_train_epochs} [train]", leave=False,
        )
        for step, batch in enumerate(train_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop("labels")
            outputs = model(**batch)
            raw_loss = criterion(outputs.logits, target)
            loss = raw_loss / model_cfg.gradient_accumulation_steps
            loss.backward()
            running_loss += raw_loss.item()
            train_bar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

            train_preds.extend(outputs.logits.detach().argmax(dim=-1).cpu().tolist())
            train_true.extend(target.cpu().tolist())

            is_last_step = (step + 1) == len(train_loader)
            if (step + 1) % model_cfg.gradient_accumulation_steps == 0 or is_last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = running_loss / len(train_loader)
        train_metrics = evaluate_predictions(train_true, train_preds, labels)
        val_loss, val_preds, val_true = _evaluate_loss(model, val_loader, device, criterion)
        metrics = evaluate_predictions(val_true, val_preds, labels)

        epoch_log = EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_metrics["accuracy"],
            train_macro_f1=train_metrics["macro_f1"],
            train_micro_f1=train_metrics["micro_f1"],
            train_macro_f1_minority=train_metrics["macro_f1_minority"],
            train_f1_per_class=per_class_f1(train_true, train_preds, labels),
            val_loss=val_loss,
            val_accuracy=metrics["accuracy"],
            val_macro_f1=metrics["macro_f1"],
            val_micro_f1=metrics["micro_f1"],
            val_macro_f1_minority=metrics["macro_f1_minority"],
            val_f1_per_class=per_class_f1(val_true, val_preds, labels),
        )
        history.append(epoch_log)
        print(
            f"epoch {epoch}/{model_cfg.num_train_epochs} - train_loss={train_loss:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} train_micro_f1={train_metrics['micro_f1']:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
            f"val_micro_f1={metrics['micro_f1']:.4f} "
            f"val_macro_f1_minority={metrics['macro_f1_minority']:.4f}"
        )

        if metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = _trainable_state_dict(model)
            epochs_without_improvement = 0
            if checkpoint_dir is not None:
                save_checkpoint(model, checkpoint_dir)
                print(f"  new best val_macro_f1={best_val_macro_f1:.4f} -- checkpoint saved to {checkpoint_dir}")
        else:
            epochs_without_improvement += 1

        if resume_path is not None:
            _save_resume_state(
                resume_path, epoch, model, optimizer, scheduler, best_val_macro_f1, best_epoch,
                best_state_dict, epochs_without_improvement, history,
            )

        if (
            model_cfg.early_stopping_patience is not None
            and epochs_without_improvement >= model_cfg.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: val_macro_f1 hasn't improved for "
                f"{model_cfg.early_stopping_patience} epochs (best={best_val_macro_f1:.4f} "
                f"at epoch {best_epoch})."
            )
            break

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()}, strict=False)

    return history


@torch.no_grad()
def _evaluate_loss(model, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    for batch in tqdm(loader, desc="validation", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch.pop("labels")
        outputs = model(**batch)
        total_loss += criterion(outputs.logits, target).item()
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        all_true.extend(target.cpu().tolist())
    return total_loss / len(loader), all_preds, all_true


def history_to_dicts(history: list[EpochLog]) -> list[dict]:
    return [asdict(epoch_log) for epoch_log in history]
