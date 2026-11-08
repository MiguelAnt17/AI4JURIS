"""Fine-tune one model on one label set from the command line."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_REGISTRY  # noqa: E402
from src.data import build_datasets, load_label_map, load_splits  # noqa: E402
from src.evaluate import classification_report_text, evaluate_predictions, full_test_report  # noqa: E402
from src.finetune import build_model_and_tokenizer, history_to_dicts, train, training_args_payload  # noqa: E402
from src.plotting import plot_confusion_matrix, plot_per_class_f1_curves, plot_training_curves  # noqa: E402
from src.runtime import get_device, set_all_seeds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--label-set", default="grupo3")
    parser.add_argument("--dataset-variant", default="sumario")
    args = parser.parse_args()

    set_all_seeds()
    model_cfg = MODEL_REGISTRY[args.model]
    device = get_device()

    df = load_splits(dataset_variant=args.dataset_variant, label_set=args.label_set)
    labels = load_label_map(args.label_set)["labels"]

    print(f"model: {args.model}  label_set: {args.label_set}  dataset_variant: {args.dataset_variant}")
    print(f"device: {device}")
    print(f"split sizes: {df['split'].value_counts().to_dict()}")
    print(f"labels ({len(labels)}): {labels}")

    model, tokenizer = build_model_and_tokenizer(model_cfg, num_labels=len(labels), device=device)
    datasets = build_datasets(df, tokenizer, max_length=model_cfg.max_length)
    print({split: len(ds) for split, ds in datasets.items()})

    results_dir = model_cfg.results_dir("finetune", dataset_variant=args.dataset_variant, label_set=args.label_set)
    checkpoint_dir = results_dir / "checkpoint"
    print(f"results_dir: {results_dir}")

    history = train(
        model,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        model_cfg=model_cfg,
        device=device,
        labels=labels,
        checkpoint_dir=checkpoint_dir,
    )

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_loader_batch_size = model_cfg.batch_size(device.type)

    import torch
    from torch.utils.data import DataLoader

    test_loader = DataLoader(datasets["test"], batch_size=test_loader_batch_size, shuffle=False)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels")
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu())
    logits = torch.cat(all_logits, dim=0)

    test_label_ids = test_df["label_id"].tolist()
    test_pred_ids = logits.argmax(dim=-1).tolist()
    test_metrics = evaluate_predictions(test_label_ids, test_pred_ids, labels)
    test_report = full_test_report(test_label_ids, test_pred_ids, labels)

    metrics_payload = {
        "training_args": training_args_payload(model_cfg, device, epochs_run=history[-1].epoch if history else 0),
        "epoch_history": history_to_dicts(history),
        "test": test_metrics,
        "classification_report": test_report["classification_report"],
        "confusion_matrix": test_report["confusion_matrix"],
        "confusion_matrix_labels": test_report["confusion_matrix_labels"],
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report_text(test_label_ids, test_pred_ids, labels))

    plot_training_curves(history, results_dir / "training_curves.png", title=f"{model_cfg.display_name} — {args.label_set}")
    plot_per_class_f1_curves(history, labels, results_dir / "per_class_f1_curves.png", title=f"{model_cfg.display_name} — {args.label_set}")
    plot_confusion_matrix(
        test_report["confusion_matrix"], labels, results_dir / "confusion_matrix.png",
        title=f"{model_cfg.display_name} — {args.label_set}",
    )

    print()
    print("=== TEST METRICS ===")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
