
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "trigger_bias_controlled_results.csv"

THRESHOLD = 0.80

print("Loading controlled trigger results...")

df = pd.read_csv(INPUT_FILE)

modified = df[df["condition"] == "Modified"].copy()

# Apply selected threshold
modified["threshold_prediction"] = (
    modified["malicious_score"] >= THRESHOLD
)

# Calculate per-trigger statistics
summary = modified.groupby("trigger").agg(
    total_prompts=("prompt_id", "count"),
    injection_predictions=("threshold_prediction", "sum"),
    mean_malicious_score=("malicious_score", "mean"),
    maximum_score=("malicious_score", "max"),
    minimum_score=("malicious_score", "min")
).reset_index()

summary["injection_detection_rate"] = (
    summary["injection_predictions"] /
    summary["total_prompts"] * 100
)

summary.to_csv(
    "trigger_comparison_results.csv",
    index=False
)

print("\n========== TRIGGER COMPARISON ==========")
print(summary.round(4).to_string(index=False))

# ---------------- GRAPH ----------------

plt.figure(figsize=(10, 6))

plt.bar(
    summary["trigger"],
    summary["injection_detection_rate"]
)

plt.xlabel("Trigger Phrase")
plt.ylabel("Injection Detection Rate (%)")

plt.title(
    f"Trigger Comparison at Threshold {THRESHOLD}"
)

plt.ylim(0, 100)
plt.xticks(rotation=15, ha="right")

plt.tight_layout()

plt.savefig(
    "trigger_comparison_graph.png",
    dpi=300
)

plt.show()

print("\nTrigger comparison completed!")
