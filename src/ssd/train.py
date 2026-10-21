"""Fine-tune XLM-RoBERTa for one SSD-2026 subtask and language.

Hyperparameters follow Table 1 of the paper: XLM-RoBERTa-base, max length 256,
batch size 16, AdamW (lr 2e-5), 3 epochs, dropout 0.1, weight decay 0.01, linear scheduler.

If no --dev file is given, 10% of the training set is held out (stratified) for
model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from .data import load_task
from .metrics import evaluate, format_report


def to_dataset(df, tokenizer, max_len):
    cols = {"text": df["text_clean"].tolist()}
    if "label_id" in df:
        cols["labels"] = df["label_id"].astype(int).tolist()
    ds = Dataset.from_dict(cols)
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["text"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--lang", default="en", help="tag used in the output folder name")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--config", default="configs/xlmr.yaml")
    ap.add_argument("--model", default=None, help="override the pretrained model name")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    model_name = args.model or cfg["model"]
    out = Path(args.out or f"outputs/task{args.task}_{args.lang}")
    set_seed(args.seed)

    train_df, labels = load_task(args.train, args.task, args.text_col, args.label_col)
    if args.dev:
        dev_df, _ = load_task(args.dev, args.task, args.text_col, args.label_col)
    else:
        train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=args.seed,
                                            stratify=train_df["label_id"])
    print(f"Task {args.task} ({args.lang}): {len(train_df)} train / {len(dev_df)} dev, labels={labels}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
        hidden_dropout_prob=cfg["dropout"], attention_probs_dropout_prob=cfg["dropout"])

    max_len = cfg["max_length"]
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["epochs"],
        lr_scheduler_type=cfg["scheduler"],
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        logging_steps=50, report_to=[], seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics(p):
        res = evaluate(p.label_ids, np.argmax(p.predictions, -1), labels)
        return {"macro_f1": res["macro_f1"], "weighted_f1": res["weighted_f1"], "accuracy": res["accuracy"]}

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=to_dataset(train_df, tok, max_len),
        eval_dataset=to_dataset(dev_df, tok, max_len),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out / "model"))
    tok.save_pretrained(str(out / "model"))

    results = {}
    dev_pred = np.argmax(trainer.predict(to_dataset(dev_df, tok, max_len)).predictions, -1)
    results["dev"] = evaluate(dev_df["label_id"], dev_pred, labels)
    print("\nDev\n" + format_report(results["dev"], labels))

    if args.test:
        test_df, _ = load_task(args.test, args.task, args.text_col, args.label_col)
        pred = np.argmax(trainer.predict(to_dataset(test_df, tok, max_len)).predictions, -1)
        test_df["prediction"] = [labels[i] for i in pred]
        keep = [c for c in test_df.columns if c not in ("text_clean", "label_id")]
        test_df[keep].to_csv(out / "test_predictions.csv", index=False)
        if "label_id" in test_df:
            results["test"] = evaluate(test_df["label_id"], pred, labels)
            print("\nTest\n" + format_report(results["test"], labels))

    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved model and metrics to {out}/")


if __name__ == "__main__":
    main()
