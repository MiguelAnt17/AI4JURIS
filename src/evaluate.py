"""Shared evaluation metrics."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import MINORITY_LABELS


def evaluate_predictions(y_true_ids: list[int], y_pred_ids: list[int], labels: list[str]) -> dict:
    """Compute accuracy, macro/micro F1 and related metrics."""
    minority_ids = [i for i, label in enumerate(labels) if label in MINORITY_LABELS]

    return {
        "accuracy": accuracy_score(y_true_ids, y_pred_ids),
        "macro_f1": f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0),
        "macro_f1_minority": f1_score(
            y_true_ids, y_pred_ids, labels=minority_ids, average="macro", zero_division=0
        ),
        "micro_f1": f1_score(y_true_ids, y_pred_ids, average="micro", zero_division=0),
    }


def per_class_f1(y_true_ids: list[int], y_pred_ids: list[int], labels: list[str]) -> dict[str, float]:
    """F1 for each class, keyed by label."""
    label_ids = list(range(len(labels)))
    scores = f1_score(y_true_ids, y_pred_ids, labels=label_ids, average=None, zero_division=0)
    return dict(zip(labels, scores.tolist()))


def full_test_report(y_true_ids: list[int], y_pred_ids: list[int], labels: list[str]) -> dict:
    """Per-class report and confusion matrix for the test set."""
    label_ids = list(range(len(labels)))
    report = classification_report(
        y_true_ids, y_pred_ids, labels=label_ids, target_names=labels,
        zero_division=0, output_dict=True,
    )
    matrix = confusion_matrix(y_true_ids, y_pred_ids, labels=label_ids)
    return {
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
    }


def linear_model_size(vectorizer, classifier) -> dict:
    """Parameter count of a fitted TF-IDF + linear classifier."""
    return {
        "vocabulary_size": len(vectorizer.vocabulary_),
        "total_parameters": int(classifier.coef_.size + classifier.intercept_.size),
    }


def classification_report_text(y_true_ids: list[int], y_pred_ids: list[int], labels: list[str]) -> str:
    """Per-class report as plain text."""
    label_ids = list(range(len(labels)))
    return classification_report(
        y_true_ids, y_pred_ids, labels=label_ids, target_names=labels, zero_division=0,
    )
