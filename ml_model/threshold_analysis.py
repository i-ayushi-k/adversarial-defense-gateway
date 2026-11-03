
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION ----------------

INPUT_FILE = "trigger_bias_controlled_results.csv"

THRESHOLDS = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95
]

print("Loading controlled experiment results...")

df = pd.read_csv(INPUT_FILE)

original = df[df["condition"] == "Original"]
modified = df[df["condition"] == "Modified"]

# ---------------- THRESHOLD ANALYSIS ----------------

results = []

for threshold in THRESHOLDS:

    # Original benign prompts
    original_predictions = (
        original["malicious_score"] >= threshold
    )

    benign_false_positive_rate = (
        original_predictions.mean() * 100
    )

    benign_acceptance_rate = (
        100 - benign_false_positive_rate
    )

    # Trigger-modified prompts
    modified_predictions = (
        modified["malicious_score"] >= threshold
    )

    injection_detection_rate = (
        modified_predictions.mean() * 100
    )

    results.append({
        "threshold": threshold,
        "benign_false_positive_rate":
            benign_false_positive_rate,
        "benign_acceptance_rate":
            benign_acceptance_rate,
        "injection_detection_rate":
            injection_detection_rate
    })

summary = pd.DataFrame(results)

# ---------------- SAVE RESULTS ----------------

summary.to_csv(
    "threshold_analysis_results.csv",
    index=False
)

print("\n========== THRESHOLD ANALYSIS ==========")
print(summary.round(2).to_string(index=False))

# ---------------- GRAPH ----------------

plt.figure(figsize=(10, 6))

plt.plot(
    summary["threshold"],
    summary["benign_false_positive_rate"],
    marker="o",
    label="Benign False Positive Rate"
)

plt.plot(
    summary["threshold"],
    summary["injection_detection_rate"],
    marker="s",
    label="Injection Detection Rate"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Rate (%)")

plt.title("Threshold Analysis")

plt.ylim(0, 100)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "threshold_analysis_graph.png",
    dpi=300
)

plt.show()

print("\nThreshold analysis completed!")
print("Saved threshold_analysis_results.csv")
print("Saved threshold_analysis_graph.png")
