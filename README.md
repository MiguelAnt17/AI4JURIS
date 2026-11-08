# AI4JURIS

Explainability (XAI) research on Portuguese jurisprudence: BERT-style language models are fine-tuned to predict a court's decision label (`Decisao`) from a case summary (`Sumario`), as a basis for later LIME / Integrated Gradients / Attention Rollout analysis.

> **Data is not included.** The court decisions contain sensitive information, so neither the raw nor the processed data are in this repository. You need your own access to the source (see [`data/1_raw/README.md`](data/1_raw/README.md)).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
cp .env.example .env     # fill in HF_TOKEN (and JURIS_API_BEARER_TOKEN to download data)
```

Optional extras: `pip install -e ".[eda]"` (plots, spaCy), `".[xai]"` (LIME, Captum).

## Project layout

```
src/         core code: config, data loading, fine-tuning, probes, evaluation
scripts/     command-line runners (data extraction, baseline, fine-tuning)
notebooks/   numbered pipeline, by stage (below)
data/        dataset docs + label maps only (data files are git-ignored)
outputs/     per-model results (weights and embeddings are git-ignored)
```

## Pipeline

1. **Data** – `scripts/extract_jurisprudencia.py` downloads court records; `notebooks/eda/` explores them.
2. **Preprocessing** – `notebooks/preprocessing/` builds the labelled dataset and the merged-label variants (`grupo3`, `grupo3_8class`, `grupo3_6class`, ...); `notebooks/split/` makes the 75/10/15 split.
3. **Models** – `notebooks/baselines/` (TF-IDF + SVM / LogReg), `notebooks/inference/` (frozen embeddings) and `notebooks/fine_tuning/` (Albertina, BERTimbau, Legal-BERT, Serafim, ...). The same can be run from the CLI:
   ```bash
   python scripts/baseline_variant.py --label-set grupo3_6class
   python scripts/finetune_variant.py --model bertimbau --label-set grupo3_6class
   ```
4. **XAI** – planned (LIME, Integrated Gradients, Attention Rollout); not implemented yet.

Models and dataset/label variants are defined in [`src/config.py`](src/config.py).

## Reproducibility

Seed `42` everywhere, best-effort deterministic PyTorch, pinned dependencies and pinned Hugging Face model revisions. Results are reproducible on the same hardware, not bit-exact across different machines (CUDA vs. MPS/CPU use different batch sizes).

## License

Code: [MIT](LICENSE). Pretrained models and the underlying court data have their own licences/terms — check them before reuse.
