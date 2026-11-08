"""LoRA variant of src/finetune.py's build_model_and_tokenizer."""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.config import ModelConfig


def build_model_and_tokenizer(
    model_cfg: ModelConfig,
    num_labels: int,
    device: torch.device,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load a sequence-classification model wrapped with LoRA adapters."""
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
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg.hf_id, **from_pretrained_kwargs)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config).to(device)
    return model, tokenizer
