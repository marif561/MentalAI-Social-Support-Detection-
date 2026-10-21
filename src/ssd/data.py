"""Data loading for the three SSD-2026 subtasks."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .preprocess import clean_text

TASK_LABELS: Dict[int, List[str]] = {
    1: ["Non-Support", "Support"],
    2: ["Group", "Individual"],
    3: ["Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"],
}

# Accept common spellings of the labels found in shared-task files.
_ALIASES = {
    "non-support": "Non-Support", "non support": "Non-Support", "nonsupport": "Non-Support",
    "not support": "Non-Support", "not_support": "Non-Support", "non_support": "Non-Support",
    "0": "Non-Support", "support": "Support", "1": "Support",
    "group": "Group", "individual": "Individual",
    "nation": "Nation", "other": "Other", "lgbtq": "LGBTQ", "lgbtq+": "LGBTQ",
    "black community": "Black Community", "black_community": "Black Community",
    "religion": "Religion", "women": "Women",
}


def normalise_label(label, task: int) -> str:
    key = str(label).strip().lower()
    out = _ALIASES.get(key, str(label).strip())
    if out not in TASK_LABELS[task]:
        raise ValueError(f"Unknown Task {task} label {label!r}; expected {TASK_LABELS[task]}")
    return out


def load_task(path: str, task: int, text_col: str = "text", label_col: str | None = "label",
              clean: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Load a CSV/TSV for one subtask.

    Rows with an empty label are dropped, which handles hierarchical files where
    Task 2/3 labels only exist for supportive comments. Duplicate texts and rows with
    missing text are removed, as described in the paper.
    """
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    if text_col not in df.columns:
        raise KeyError(f"Column {text_col!r} not in {path}; available: {list(df.columns)}")
    df = df.dropna(subset=[text_col]).copy()
    df["text_clean"] = df[text_col].map(clean_text) if clean else df[text_col].astype(str)
    df = df[df["text_clean"].str.len() > 0]

    labels = TASK_LABELS[task]
    if label_col and label_col in df.columns:
        df = df.dropna(subset=[label_col])
        df = df[df[label_col].astype(str).str.strip() != ""]
        df["label_name"] = [normalise_label(l, task) for l in df[label_col]]
        df["label_id"] = df["label_name"].map({l: i for i, l in enumerate(labels)})
        df = df.drop_duplicates(subset=["text_clean", "label_name"])
    return df.reset_index(drop=True), labels
