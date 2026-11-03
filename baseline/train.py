import argparse, json, os
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/deberta-v3-base")
ap.add_argument("--model_out", default="baseline/outputs/model")   # gitignored
ap.add_argument("--results_out", default="baseline/results")       # committed
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
set_seed(args.seed)
os.makedirs(args.results_out, exist_ok=True)

# label 1 = injection, 0 = benign
ds = load_dataset("deepset/prompt-injections")
split = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
train_ds, val_ds, test_ds = split["train"], split["test"], ds["test"]
print(len(train_ds), len(val_ds), len(test_ds))

tok = AutoTokenizer.from_pretrained(args.model)
def enc(b): return tok(b["text"], truncation=True, max_length=args.max_len)
train_ds, val_ds, test_ds = [d.map(enc, batched=True) for d in (train_ds, val_ds, test_ds)]

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=2, id2label={0: "benign", 1: "injection"},
    label2id={"benign": 0, "injection": 1})

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(p.label_ids, preds), "precision": pr, "recall": rc, "f1": f1}

targs = TrainingArguments(
    output_dir="baseline/outputs/ckpt", num_train_epochs=args.epochs,
    learning_rate=args.lr, per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=32, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1",
    save_total_limit=1, report_to="none", seed=args.seed,
    fp16=torch.cuda.is_available(), logging_steps=10)

trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()

test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
print(test_metrics)
trainer.save_model(args.model_out); tok.save_pretrained(args.model_out)
with open(f"{args.results_out}/test_metrics.json", "w") as f:
    json.dump({"base_model": args.model, "seed": args.seed, **test_metrics}, f, indent=2)