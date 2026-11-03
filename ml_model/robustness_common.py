"""Shared utilities for Deliverable 2 robustness experiments."""
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
# Support the repository layout shown in the user's project (CSV at root),
# plus common Deliverable 1 output layouts.
INPUT_CSV_CANDIDATES = [
    ROOT / "notinject_results.csv",
    ROOT / "final_deliverable_1" / "notinject_results.csv",
    ROOT / "final_deliverable_1" / "csv_results" / "notinject_results.csv",
]
OUTPUT_ROOT = ROOT / "final_deliverable_2"
CSV_DIR = OUTPUT_ROOT / "csv_results"
GRAPH_DIR = OUTPUT_ROOT / "graphs"
THRESHOLD = 0.5


def prepare_output_dirs():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def load_benign_prompts(limit=50):
    input_csv = next((p for p in INPUT_CSV_CANDIDATES if p.exists()), None)
    if input_csv is None:
        searched = "\\n".join(str(p) for p in INPUT_CSV_CANDIDATES)
        raise FileNotFoundError(
            "Could not find notinject_results.csv. Checked these locations:\\n" + searched
        )
    print(f"Loading benign prompts from: {input_csv}")
    df = pd.read_csv(input_csv)
    if "prompt" not in df.columns:
        raise ValueError("Input CSV must contain a 'prompt' column.")
    prompts = df["prompt"].dropna().astype(str).drop_duplicates().tolist()
    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        raise ValueError("No usable prompts found in NotInject results.")
    return prompts[:min(limit, len(prompts))]


class Detector:
    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.malicious_id = self._find_malicious_id()
        print(f"Model ready on {self.device}; malicious class id={self.malicious_id}")

    def _find_malicious_id(self):
        labels = self.model.config.id2label
        for idx, label in labels.items():
            name = str(label).lower()
            if any(word in name for word in ("malicious", "injection", "attack")):
                return int(idx)
        # This is the same binary-class assumption used in the existing experiment.
        if self.model.config.num_labels == 2:
            print("WARNING: label names did not identify the malicious class; using class 1 as in Deliverable 1.")
            return 1
        raise ValueError(f"Cannot identify malicious class from id2label={labels}")

    def score(self, text):
        encoded = self.tokenizer(
            str(text), return_tensors="pt", truncation=True, max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]
        malicious_score = float(probs[self.malicious_id].item())
        prediction = "INJECTION" if malicious_score >= THRESHOLD else "SAFE"
        return prediction, malicious_score


def save_plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
