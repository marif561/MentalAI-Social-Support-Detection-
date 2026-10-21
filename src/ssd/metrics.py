"""Evaluation metrics (paper Section 3.5). Macro-F1 is the primary metric."""
from __future__ import annotations

from typing import Dict, List, Sequence

from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate(y_true: Sequence[int], y_pred: Sequence[int], labels: List[str]) -> Dict:
    ids = list(range(len(labels)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0),
        "report": classification_report(y_true, y_pred, labels=ids, target_names=labels,
                                        output_dict=True, zero_division=0),
    }


def format_report(result: Dict, labels: List[str]) -> str:
    rows = [f"{'class':<18}{'P':>8}{'R':>8}{'F1':>8}{'n':>8}"]
    for lab in labels:
        r = result["report"][lab]
        rows.append(f"{lab:<18}{r['precision']:>8.4f}{r['recall']:>8.4f}"
                    f"{r['f1-score']:>8.4f}{int(r['support']):>8d}")
    rows.append(f"macro-F1 {result['macro_f1']:.4f} · weighted-F1 {result['weighted_f1']:.4f}"
                f" · accuracy {result['accuracy']:.4f}")
    return "\n".join(rows)
