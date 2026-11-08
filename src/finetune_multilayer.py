"""Fine-tuning with a head over the concatenated [CLS] of the last N layers."""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from src.capture import pool_last_n_layers_cls
from src.config import ModelConfig


class MultiLayerClsClassifier(nn.Module):
    """Encoder with a linear head over the last N layers' [CLS] tokens."""

    def __init__(
        self, encoder: nn.Module, hidden_size: int, num_labels: int, dropout: float, n_layers: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_layers * hidden_size, num_labels)

    def forward(self, output_hidden_states: bool = False, output_attentions: bool = False, **inputs):
        encoder_outputs = self.encoder(
            **inputs, output_hidden_states=True, output_attentions=output_attentions,
        )
        pooled = pool_last_n_layers_cls(encoder_outputs.hidden_states, n_layers=self.n_layers)
        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )


def build_model_and_tokenizer(
    model_cfg: ModelConfig, num_labels: int, device: torch.device, n_layers: int = 4,
):
    """Load the encoder and multi-layer head, plus its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id, revision=model_cfg.revision)
    encoder = AutoModel.from_pretrained(
        model_cfg.hf_id,
        revision=model_cfg.revision,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        hidden_dropout_prob=model_cfg.dropout,
        attention_probs_dropout_prob=model_cfg.dropout,
    )
    model = MultiLayerClsClassifier(
        encoder, encoder.config.hidden_size, num_labels, model_cfg.dropout, n_layers=n_layers,
    ).to(device)
    return model, tokenizer
