"""Run the TF-IDF + LogReg baseline on one label set from the command line."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_baseline_config  # noqa: E402
from src.data import load_label_map, load_splits  # noqa: E402
from src.evaluate import (  # noqa: E402
    classification_report_text, evaluate_predictions, full_test_report, linear_model_size,
)
from src.plotting import plot_confusion_matrix  # noqa: E402
from src.runtime import SEED, set_all_seeds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", default="grupo3")
    args = parser.parse_args()

    set_all_seeds()
    cfg = get_baseline_config("tfidf_logreg")
    results_dir = cfg.results_dir(label_set=args.label_set)

    df = load_splits(label_set=args.label_set)
    labels = load_label_map(args.label_set)["labels"]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"label_set: {args.label_set}")
    print(f"split sizes: {df['split'].value_counts().to_dict()}")
    print(f"labels ({len(labels)}): {labels}")
    print(f"results_dir: {results_dir}")

    PARAM_GRID = {
        "tfidf__max_features": [20000, 50000, 100000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__l1_ratio": [0.0, 0.5, 1.0],
        "clf__C": [1, 3, 10, 30, 100],
    }
    N_CV_FOLDS = 5

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(
            random_state=SEED, solver="saga", **{**cfg.classifier_kwargs, "max_iter": 5000},
        )),
    ])
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    grid_rows = []
    for params in ParameterGrid(PARAM_GRID):
        pipeline.set_params(**params)
        scores = cross_val_score(
            pipeline, train_df["Sumario"], train_df["label_id"],
            scoring=cfg.scoring, cv=cv, n_jobs=-1,
        )
        row = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}
        row["mean_cv_macro_f1"] = scores.mean()
        row["std_cv_macro_f1"] = scores.std()
        grid_rows.append(row)
        print(f"  {row}")

    grid_results_df = pd.DataFrame(grid_rows).sort_values("mean_cv_macro_f1", ascending=False).reset_index(drop=True)

    best_row = grid_results_df.iloc[0]

    def _native(key, value):
        if key.endswith("ngram_range"):
            return tuple(value)
        return value.item() if hasattr(value, "item") else value

    best_params = {k: _native(k, best_row[k]) for k in PARAM_GRID}
    pipeline.set_params(**best_params)
    pipeline.fit(train_df["Sumario"], train_df["label_id"])
    print(f"best params: {best_params}")
    print(f"mean CV macro-F1: {best_row['mean_cv_macro_f1']:.4f}")

    val_pred_ids = pipeline.predict(val_df["Sumario"]).tolist()
    val_metrics = evaluate_predictions(val_df["label_id"].tolist(), val_pred_ids, labels)

    test_pred_ids = pipeline.predict(test_df["Sumario"]).tolist()
    test_label_ids = test_df["label_id"].tolist()
    test_metrics = evaluate_predictions(test_label_ids, test_pred_ids, labels)
    test_report = full_test_report(test_label_ids, test_pred_ids, labels)
    model_size = linear_model_size(pipeline.named_steps["tfidf"], pipeline.named_steps["clf"])

    joblib.dump(pipeline, results_dir / "model.joblib")

    metrics_payload = {
        "param_grid": {k: (list(map(list, v)) if k.endswith("ngram_range") else v) for k, v in PARAM_GRID.items()},
        "cv_folds": N_CV_FOLDS,
        "grid_results": grid_rows,
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
        "val": val_metrics,
        "test": test_metrics,
        "model_size": model_size,
        "classification_report": test_report["classification_report"],
        "confusion_matrix": test_report["confusion_matrix"],
        "confusion_matrix_labels": test_report["confusion_matrix_labels"],
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report_text(test_label_ids, test_pred_ids, labels))

    plot_confusion_matrix(
        test_report["confusion_matrix"], labels, results_dir / "confusion_matrix.png",
        title=f"{cfg.display_name} — {args.label_set}",
    )

    print()
    print("=== TEST METRICS ===")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
