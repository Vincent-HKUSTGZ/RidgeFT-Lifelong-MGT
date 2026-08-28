"""End-to-end RidgeFT sample on an MGT-Academic-style corpus.

Pipeline (mirrors the paper's P5 protocol on one topic):

  1. Load the 6 attribution classes from the ``AI_Polish_clean`` layout,
     balance to the smallest class count (optionally subsampled with
     ``--per-class``), split 80/10/10 with a fixed seed.
  2. Fine-tune a DeBERTa encoder on the 5 INITIAL classes only
     (2 epochs, batch 32, max_len 384, AdamW lr 2e-5), then FREEZE it.
  3. Encode every split to CLS features once.
  4. RidgeFT: fit_base on the initial classes, update_manyshot with the
     held-out generator, report full / old / new macro-F1.

This is a demonstration sample, not the full experiment harness: it runs
one (topic, encoder, protocol) cell. Use ``--per-class 1000`` for a quick
run; drop it for the paper's full balanced split.

Usage:
  python examples/run_mgt_academic.py \
      --data-root /path/to/AI_Polish_clean \
      --encoder microsoft/deberta-v3-base \
      --topic STEM --per-class 1000 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from ridgeft import RidgeFTModel

LABELS = ["Human", "gpt35", "Mixtral", "Moonshot", "Llama3", "gpt-4omini"]
NEW_LABEL = "gpt-4omini"          # P5: newest generator arrives incrementally
TOPIC_MAPPING = {
    "Physics": "STEM", "Math": "STEM", "Chemistry": "STEM", "Biology": "STEM",
    "Electrical_engineering": "STEM", "Computer_science": "STEM",
    "Statistics": "STEM", "Medicine": "STEM",
    "Literature": "Humanities", "History": "Humanities", "Law": "Humanities",
    "Art": "Humanities", "Philosophy": "Humanities",
    "Economy": "Social_sciences", "Management": "Social_sciences",
    "Education": "Social_sciences",
}


# --------------------------------------------------------------- data ----
def load_topic(data_root: Path, topic: str, seed: int,
               per_class: int | None) -> dict:
    """Balanced 80/10/10 split of the 6 classes for one topic."""
    subjects = [s for s, t in TOPIC_MAPPING.items() if t == topic]
    rng = random.Random(seed)

    per_label: dict[str, list[str]] = {}
    for label in LABELS:
        texts: list[str] = []
        for sub in subjects:
            if label == "Human":
                paths = sorted((data_root / "Human" / sub).glob("*.json"))
            else:
                p = data_root / f"{label}_new" / f"{sub}_task3.json"
                paths = [p] if p.exists() else []
            for p in paths:
                for r in json.loads(p.read_text(encoding="utf-8")):
                    t = (r.get("text") or "").strip()
                    if t:
                        texts.append(t)
        rng.shuffle(texts)
        per_label[label] = texts

    n = min(len(v) for v in per_label.values())
    if per_class is not None:
        n = min(n, per_class)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)

    splits = {"train": [], "val": [], "test": []}
    for cid, label in enumerate(LABELS):
        rows = per_label[label][:n]
        for split, lo, hi in (("train", 0, n_tr), ("val", n_tr, n_tr + n_va),
                              ("test", n_tr + n_va, n)):
            splits[split] += [{"text": t, "label": cid} for t in rows[lo:hi]]
    for k in splits:
        rng.shuffle(splits[k])
    print(f"[data] topic={topic}  {n}/class  "
          f"train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}")
    return splits


# ------------------------------------------------------------ encoder ----
class Encoder(nn.Module):
    """Backbone + linear head; the CLS vector is the RidgeFT feature."""

    def __init__(self, name: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def cls(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        return self.head(self.cls(input_ids, attention_mask))


class TextDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_loader(rows, tokenizer, max_length: int, batch_size: int,
                shuffle: bool) -> DataLoader:
    def collate(batch):
        enc = tokenizer([r["text"] for r in batch], truncation=True,
                        max_length=max_length, padding=True,
                        return_tensors="pt")
        out = {"input_ids": enc["input_ids"],
               "attention_mask": enc["attention_mask"],
               "labels": torch.tensor([r["label"] for r in batch])}
        return out

    return DataLoader(TextDataset(rows), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)


def finetune(model: Encoder, rows, tokenizer, device: str, *,
             epochs: int = 2, batch_size: int = 32, lr: float = 2e-5,
             max_length: int = 384) -> Encoder:
    """Fine-tune on the initial classes only, then the caller freezes it."""
    model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader = make_loader(rows, tokenizer, max_length, batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = ce(model(**batch), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            total += loss.item()
        print(f"[encoder] epoch {ep + 1}/{epochs}  loss={total / max(1, len(loader)):.4f}")
    return model


@torch.inference_mode()
def encode(model: Encoder, rows, tokenizer, device: str, *,
           batch_size: int = 64, max_length: int = 384) -> np.ndarray:
    model.to(device).eval()
    loader = make_loader(rows, tokenizer, max_length, batch_size, shuffle=False)
    feats = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels")
        feats.append(model.cls(**batch).float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# --------------------------------------------------------------- main ----
def macro_f1(y_true, y_pred, class_ids) -> float:
    per = f1_score(y_true, y_pred, labels=class_ids, average=None,
                   zero_division=0)
    return float(np.mean(per))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="path to AI_Polish_clean")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--topic", default="STEM",
                    choices=["STEM", "Humanities", "Social_sciences"])
    ap.add_argument("--per-class", type=int, default=None,
                    help="subsample per class for a quick run (e.g. 1000)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    new_id = LABELS.index(NEW_LABEL)
    init_ids = [i for i in range(len(LABELS)) if i != new_id]

    ds = load_topic(Path(args.data_root), args.topic, args.seed,
                    args.per_class)
    y = {k: np.array([r["label"] for r in ds[k]], dtype=np.int64)
         for k in ("train", "val", "test")}

    # ---- 1. fine-tune the encoder on the INITIAL classes, then freeze ----
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = Encoder(args.encoder, num_labels=len(init_ids))
    init_rows = [r for r in ds["train"] if r["label"] != new_id]
    remap = {c: i for i, c in enumerate(init_ids)}
    init_rows = [{"text": r["text"], "label": remap[r["label"]]}
                 for r in init_rows]
    model = finetune(model, init_rows, tokenizer, args.device)

    # ---- 2. encode all splits once with the frozen encoder ---------------
    H_train = encode(model, ds["train"], tokenizer, args.device)
    H_test = encode(model, ds["test"], tokenizer, args.device)

    # ---- 3. RidgeFT: base fit on initial classes -------------------------
    base_mask = np.isin(y["train"], init_ids)
    ridgeft = RidgeFTModel.fit_base(H_train[base_mask], y["train"][base_mask])

    s0_mask = np.isin(y["test"], init_ids)
    pred0 = ridgeft.predict(H_test[s0_mask])
    print(f"[S0] base {len(init_ids)}-class macro-F1 = "
          f"{macro_f1(y['test'][s0_mask], pred0, init_ids):.4f}")

    # ---- 4. the new generator arrives: closed-form update ----------------
    new_mask = y["train"] == new_id
    ridgeft.update_manyshot(H_train[new_mask], y["train"][new_mask])

    pred = ridgeft.predict(H_test)
    all_ids = init_ids + [new_id]
    print(f"[S1] +{NEW_LABEL}:")
    print(f"     full macro-F1 = {macro_f1(y['test'], pred, all_ids):.4f}")
    print(f"     old  macro-F1 = {macro_f1(y['test'], pred, init_ids):.4f}")
    print(f"     new       F1  = {macro_f1(y['test'], pred, [new_id]):.4f}")


if __name__ == "__main__":
    main()
