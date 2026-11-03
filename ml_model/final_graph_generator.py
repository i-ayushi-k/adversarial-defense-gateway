"""
Final graph generator for Adversarial-Defense / Review 2.
Run from project root: python ml_model/final_graph_generator.py
Uses existing CSVs only; missing optional datasets are skipped.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "final_deliverable_1" / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def locate_csv(name):
    for p in (ROOT/name, ROOT/"ml_model"/name, ROOT/"final_deliverable_1"/"csv_results"/name):
        if p.exists():
            return p
    return None

def save_graph(name):
    target = OUT_DIR/name
    plt.tight_layout()
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", target)

count = 0

# Statistical flip-rate graph with confidence intervals
p = locate_csv("statistical_analysis_results.csv")
if p:
    d = pd.read_csv(p)
    if {"trigger", "flip_rate_percent"}.issubset(d.columns):
        x = range(len(d))
        y = pd.to_numeric(d["flip_rate_percent"])
        plt.figure(figsize=(11, 6))
        plt.bar(list(x), y)
        if {"ci_lower_percent", "ci_upper_percent"}.issubset(d.columns):
            lo = pd.to_numeric(d["ci_lower_percent"])
            hi = pd.to_numeric(d["ci_upper_percent"])
            plt.errorbar(list(x), y, yerr=[y-lo, hi-y], fmt="none", capsize=5, color="black")
        plt.xticks(list(x), d["trigger"].astype(str), rotation=18, ha="right")
        plt.ylabel("Prediction Flip Rate (%)")
        plt.xlabel("Injected Trigger")
        plt.title("Statistical Analysis: Trigger-Induced Prediction Flips")
        plt.ylim(0, 100)
        save_graph("statistical_flip_rate_graph.png")
        count += 1
    if {"trigger", "mean_score_increase"}.issubset(d.columns):
        plt.figure(figsize=(11, 6))
        plt.bar(d["trigger"].astype(str), pd.to_numeric(d["mean_score_increase"]))
        plt.xticks(rotation=18, ha="right")
        plt.ylabel("Mean Increase in Malicious Score")
        plt.xlabel("Injected Trigger")
        plt.title("Trigger-Induced Increase in Malicious Classification Score")
        save_graph("statistical_score_increase_graph.png")
        count += 1
else:
    print("Skip statistical graphs: statistical_analysis_results.csv missing.")

# Controlled trigger false-positive rate
p = locate_csv("trigger_bias_controlled_summary.csv")
if p:
    d = pd.read_csv(p)
    if {"trigger", "trigger_induced_fp_rate"}.issubset(d.columns):
        plt.figure(figsize=(11, 6))
        plt.bar(d["trigger"].astype(str), pd.to_numeric(d["trigger_induced_fp_rate"]))
        plt.xticks(rotation=18, ha="right")
        plt.ylabel("Trigger-Induced False Positive Rate (%)")
        plt.xlabel("Injected Trigger Phrase")
        plt.title("Controlled Trigger Bias Analysis")
        plt.ylim(0, 100)
        save_graph("trigger_bias_controlled_graph.png")
        count += 1
else:
    print("Skip controlled graph: summary CSV missing.")

# NotInject benign score distribution (if a score column exists)
p = locate_csv("notinject_results.csv")
if p:
    d = pd.read_csv(p)
    score = next((c for c in ("confidence", "malicious_score", "injection_score") if c in d.columns), None)
    if score:
        vals = pd.to_numeric(d[score], errors="coerce").dropna()
        plt.figure(figsize=(10, 6))
        plt.hist(vals, bins=25)
        plt.xlabel(score.replace("_", " ").title())
        plt.ylabel("Number of Benign Samples")
        plt.title("NotInject Baseline: Score Distribution")
        save_graph("notinject_baseline_score_distribution.png")
        count += 1
    else:
        print("NotInject CSV has no recognized score column; skipped baseline graph.")
else:
    print("Skip baseline graph: notinject_results.csv missing.")

for name in ("threshold_analysis_results.csv", "trigger_comparison_results.csv", "adversarial_results.csv"):
    if not locate_csv(name):
        print("Optional data missing; graph not created:", name)
    else:
        print("Optional CSV found; inspect its columns before plotting:", name)

print(f"\\nDone: generated {count} graph(s).")
print("Graphs folder:", OUT_DIR)
