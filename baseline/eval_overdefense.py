import json
import pandas as pd
from datasets import load_dataset
from transformers import pipeline

clf = pipeline("text-classification", model="baseline/outputs/model", truncation=True, max_length=256)

ds = load_dataset("leolee99/NotInject")
rows = []
for name, split in ds.items():
    print(name, split.column_names)          # check the text column name
    col = "prompt" if "prompt" in split.column_names else split.column_names[0]
    preds = clf(list(split[col]), batch_size=32)
    fp = sum(p["label"] == "injection" for p in preds)
    rows.append({"split": name, "n": len(preds), "false_positives": fp, "fpr": fp / len(preds)})

df = pd.DataFrame(rows); print(df)
df.to_csv("baseline/results/notinject_overdefense.csv", index=False)
