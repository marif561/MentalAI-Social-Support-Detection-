"""Label new comments with a model trained by ``ssd.train``."""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocess import clean_text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="folder saved by ssd.train (…/model)")
    ap.add_argument("--input", required=True, help="CSV with a text column")
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    id2label = model.config.id2label

    df = pd.read_csv(args.input)
    texts = df[args.text_col].fillna("").map(clean_text).tolist()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            enc = tok(texts[i:i + args.batch_size], truncation=True, max_length=args.max_length,
                      padding=True, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits, -1).cpu()
            preds += [id2label[int(j)] for j in p.argmax(-1)]
            probs += p.max(-1).values.tolist()
    df["prediction"] = preds
    df["confidence"] = probs
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
