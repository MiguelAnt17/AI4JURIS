"""Central registry of models and paths shared across inference & fine-tuning notebooks."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "1_raw"
DATA_DIR = PROJECT_ROOT / "data" / "2_processed"
EXPLANATIONS_DIR = PROJECT_ROOT / "data" / "3_explanations"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATASET_VARIANTS = {
    "sumario": "preprocessed_grupo3.csv",
}

LABEL_SET = "grupo3"

LABEL_SET_VARIANTS = {
    "grupo3": ("preprocessed_grupo3.csv", "label2id_grupo3.json"),
    "grupo3_8class": ("preprocessed_grupo3_8class.csv", "label2id_grupo3_8class.json"),
    "grupo3_8class_similarity": (
        "preprocessed_grupo3_8class_similarity.csv", "label2id_grupo3_8class_similarity.json",
    ),
    "grupo3_6class": ("preprocessed_grupo3_6class.csv", "label2id_grupo3_6class.json"),
}

MINORITY_LABELS = [
    "NÃO PROVIDO",
    "REVOGADA EM PARTE",
    "PROCEDENTE",
    "PARCIALMENTE PROCEDENTE",
    "PROVIDO",
]


@dataclass(frozen=True)
class ModelConfig:
    slug: str
    hf_id: str
    display_name: str
    revision: str | None = None
    max_length: int = 512
    batch_size_cuda: int = 4
    batch_size_other: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 15
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    dropout: float = 0.1
    early_stopping_patience: int | None = None
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0

    def batch_size(self, device_type: str) -> int:
        return self.batch_size_cuda if device_type == "cuda" else self.batch_size_other

    def results_dir(self, task: str, dataset_variant: str = "sumario", label_set: str = "grupo3") -> Path:
        """Return (and create) outputs/<slug>/<task>[/<dataset_variant>][/<label_set>]."""
        out = OUTPUTS_DIR / self.slug / task
        if dataset_variant != "sumario":
            out = out / dataset_variant
        if label_set != "grupo3":
            out = out / label_set
        out.mkdir(parents=True, exist_ok=True)
        return out


MODEL_REGISTRY = {
    "albertina": ModelConfig(
        slug="albertina",
        hf_id="PORTULAN/albertina-100m-portuguese-ptpt-encoder",
        display_name="Albertina PT-PT (100M)",
        revision="d15e086005cdab26d746d9113984cc654439df2b",
        early_stopping_patience=2
    ),
    "robertalexpt": ModelConfig(
        slug="robertalexpt",
        hf_id="eduagarcia/RoBERTaLexPT-base",
        display_name="RoBERTaLexPT-base",
        revision="9dea30875c93dc2d33b35c2338dfc93c86715b9e",
        early_stopping_patience=2
    ),
    "legalbert": ModelConfig(
        slug="legalbert",
        hf_id="dominguesm/legal-bert-base-cased-ptbr",
        display_name="Legal-BERT PT-BR (base)",
        revision="356e007ddd856c3028bd61962041952b6aa2a53b",
        early_stopping_patience=2
    ),
    "bertimbau": ModelConfig(
        slug="bertimbau",
        hf_id="neuralmind/bert-base-portuguese-cased",
        display_name="BERTimbau (base)",
        revision="94d69c95f98f7d5b2a8700c420230ae10def0baa",
        early_stopping_patience=2
    ),
    "albertina_900m": ModelConfig(
        slug="albertina_900m",
        hf_id="PORTULAN/albertina-900m-portuguese-ptpt-encoder",
        display_name="Albertina PT-PT (900M)",
        revision="64c3e48b28593633cccc1465694a9734726e1df8",
        batch_size_cuda=2,
        batch_size_other=2,
        gradient_accumulation_steps=8,
        early_stopping_patience=2
    ),
    "serafim_900m": ModelConfig(
        slug="serafim_900m",
        hf_id="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",
        display_name="Serafim PT-PT (900M sentence-encoder)",
        revision="8b6aac1431bf56dd4b8f6a27a98ea78d3a7c5d31",
        batch_size_cuda=2,
        batch_size_other=4,
        early_stopping_patience=2
    ),
}


def get_model_config(key: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown model key '{key}'. Available: {list(MODEL_REGISTRY)}"
        ) from None


@dataclass(frozen=True)
class LinearBaselineConfig:
    slug: str
    display_name: str
    classifier_kwargs: dict
    scoring: str = "f1_macro"

    def results_dir(self, task: str = "baseline", dataset_variant: str = "sumario", label_set: str = "grupo3") -> Path:
        """Return the results directory for this model."""
        out = OUTPUTS_DIR / self.slug / task
        if dataset_variant != "sumario":
            out = out / dataset_variant
        if label_set != "grupo3":
            out = out / label_set
        out.mkdir(parents=True, exist_ok=True)
        return out


BASELINE_REGISTRY = {
    "tfidf_svm": LinearBaselineConfig(
        slug="tfidf_svm",
        display_name="TF-IDF + Linear SVM",
        classifier_kwargs={"class_weight": "balanced"},
        scoring="f1_macro",
    ),
    "tfidf_logreg": LinearBaselineConfig(
        slug="tfidf_logreg",
        display_name="TF-IDF + Logistic Regression",
        classifier_kwargs={"class_weight": "balanced", "max_iter": 2000},
        scoring="f1_macro",
    ),
}


def get_baseline_config(key: str) -> LinearBaselineConfig:
    try:
        return BASELINE_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown baseline key '{key}'. Available: {list(BASELINE_REGISTRY)}"
        ) from None
