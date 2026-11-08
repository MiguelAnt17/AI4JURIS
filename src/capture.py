"""Save logits, hidden states and attentions as .npy files for XAI."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def pool_last_n_layers_cls(hidden_states: tuple[torch.Tensor, ...], n_layers: int = 4) -> torch.Tensor:
    """Concatenate the [CLS] token of the last N hidden-state layers end to end."""
    stacked = torch.stack(hidden_states[-n_layers:], dim=0)
    cls_stacked = stacked[:, :, 0, :]
    n_layers, batch_size, hidden_size = cls_stacked.shape
    return cls_stacked.permute(1, 0, 2).reshape(batch_size, n_layers * hidden_size)


def pool_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-weighted mean of the last hidden layer's token embeddings."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / token_counts


@torch.no_grad()
def capture_logits_and_cls(model, dataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    """Save logits, probabilities and [CLS] embeddings for a full split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_logits, all_probs, all_hidden_cls, all_cls_embeddings = [], [], [], []
    for batch in tqdm(loader, desc="capturing logits/hidden states", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**batch, output_hidden_states=True)

        all_logits.append(outputs.logits.cpu().numpy())
        all_probs.append(outputs.logits.softmax(dim=-1).cpu().numpy())

        hidden_cls = torch.stack([layer[:, 0, :] for layer in outputs.hidden_states], dim=1)
        all_hidden_cls.append(hidden_cls.cpu().numpy())

        all_cls_embeddings.append(pool_last_n_layers_cls(outputs.hidden_states).cpu().numpy())

    return {
        "logits": np.concatenate(all_logits, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "hidden_states_cls": np.concatenate(all_hidden_cls, axis=0),
        "cls_embeddings": np.concatenate(all_cls_embeddings, axis=0),
    }


@torch.no_grad()
def capture_hidden_states_and_mean_pooled(
    model, dataset, device: torch.device, batch_size: int
) -> dict[str, np.ndarray]:
    """Save mean-pooled embeddings from a frozen encoder."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_hidden_cls, all_mean_pooled = [], []
    for batch in tqdm(loader, desc="capturing hidden states", leave=False):
        attention_mask = batch["attention_mask"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs, output_hidden_states=True)

        hidden_cls = torch.stack([layer[:, 0, :] for layer in outputs.hidden_states], dim=1)
        all_hidden_cls.append(hidden_cls.cpu().numpy())

        all_mean_pooled.append(pool_mean(outputs.hidden_states[-1], attention_mask).cpu().numpy())

    return {
        "hidden_states_cls": np.concatenate(all_hidden_cls, axis=0),
        "cls_embeddings": np.concatenate(all_mean_pooled, axis=0),
    }


@torch.no_grad()
def capture_full_attention_subset(model, dataset, device: torch.device, n_samples: int) -> dict[str, np.ndarray]:
    """Save full attention and hidden states for a small subset."""
    model.eval()
    n_samples = min(n_samples, len(dataset))

    all_attentions, all_hidden_states = [], []
    for idx in tqdm(range(n_samples), desc="capturing attention subset", leave=False):
        batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[idx].items() if k != "labels"}
        outputs = model(**batch, output_hidden_states=True, output_attentions=True)

        all_attentions.append(torch.stack(outputs.attentions, dim=0).squeeze(1).cpu().numpy())
        all_hidden_states.append(torch.stack(outputs.hidden_states, dim=0).squeeze(1).cpu().numpy())

    return {
        "attention_full": np.stack(all_attentions, axis=0),
        "hidden_states_full": np.stack(all_hidden_states, axis=0),
    }
