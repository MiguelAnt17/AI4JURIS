"""LoRA variant of src/finetune_multilayer.py's build_model_and_tokenizer."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from src.config import ModelConfig
from src.finetune_multilayer import MultiLayerClsClassifier


def build_model_and_tokenizer(
    model_cfg: ModelConfig,
    num_labels: int,
    device: torch.device,
    n_layers: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Load the encoder wrapped with LoRA adapters plus the multi-layer [CLS] head."""
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id, revision=model_cfg.revision)
    encoder = AutoModel.from_pretrained(
        model_cfg.hf_id,
        revision=model_cfg.revision,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        hidden_dropout_prob=model_cfg.dropout,
        attention_probs_dropout_prob=model_cfg.dropout,
    )
    hidden_size = encoder.config.hidden_size

    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", target_modules="all-linear",
    )
    encoder = get_peft_model(encoder, lora_config)

    model = MultiLayerClsClassifier(
        encoder, hidden_size, num_labels, model_cfg.dropout, n_layers=n_layers,
    ).to(device)
    return model, tokenizer
