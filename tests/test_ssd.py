from pathlib import Path

import pytest

from ssd.data import TASK_LABELS, load_task, normalise_label
from ssd.metrics import evaluate
from ssd.preprocess import clean_text

DATA = Path(__file__).resolve().parents[1] / "data"


def test_clean_text():
    out = clean_text("Stay STRONG @friend!!!! see https://t.co/abc #Fuerza ¡Ánimo! ❤️")
    assert "@" not in out and "http" not in out
    assert out.startswith("stay strong")
    assert "fuerza" in out and "ánimo" in out and "❤" in out
    assert "!!!" not in out


@pytest.mark.parametrize("task", [1, 2, 3])
def test_load_samples(task):
    df, labels = load_task(str(DATA / f"sample_task{task}.csv"), task)
    assert labels == TASK_LABELS[task]
    assert set(df["label_name"]) == set(labels)
    assert df["label_id"].between(0, len(labels) - 1).all()


def test_label_aliases():
    assert normalise_label("not support", 1) == "Non-Support"
    assert normalise_label("1", 1) == "Support"
    assert normalise_label("black_community", 3) == "Black Community"
    with pytest.raises(ValueError):
        normalise_label("Support", 3)


def test_empty_labels_dropped(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("text,label\nhello there,Group\nno target here,\n", encoding="utf-8")
    df, _ = load_task(str(f), 2)
    assert len(df) == 1


def test_macro_f1():
    res = evaluate([0, 0, 1, 1], [0, 1, 1, 1], ["Non-Support", "Support"])
    assert res["accuracy"] == 0.75
    assert abs(res["macro_f1"] - (2 / 3 + 0.8) / 2) < 1e-9
