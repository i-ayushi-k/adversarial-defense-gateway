
import pandas as pd
import matplotlib.pyplot as plt
import ast

print("Loading evaluation results...")

df = pd.read_csv("notinject_results.csv")

# Convert word_list into a Python list
def get_trigger_count(value):
    try:
        parsed = ast.literal_eval(value)
        return len(parsed) if isinstance(parsed, list) else 0
    except (ValueError, SyntaxError):
        return 0

df["trigger_count"] = df["word_list"].apply(get_trigger_count)

# Identify injection predictions
df["is_injection"] = (
    df["prediction"].astype(str).str.upper() == "INJECTION"
)

# Group by number of trigger words
summary = df.groupby("trigger_count").agg(
    total_samples=("prediction", "count"),
    false_positives=("is_injection", "sum")
).reset_index()

summary["false_positive_rate"] = (
    summary["false_positives"] / summary["total_samples"] * 100
)

summary["benign_acceptance_rate"] = (
    100 - summary["false_positive_rate"]
)

print("\n========== TRIGGER BIAS RESULTS ==========")
print(summary.to_string(index=False))

# Save numerical results
summary.to_csv("trigger_bias_summary.csv", index=False)

# Generate graph
plt.figure(figsize=(8, 5))

plt.bar(
    summary["trigger_count"],
    summary["false_positive_rate"]
)

plt.xlabel("Number of Trigger Words")
plt.ylabel("False Positive Rate (%)")
plt.title("Trigger Bias Analysis - NotInject")

plt.xticks(summary["trigger_count"])

for x, y in zip(
    summary["trigger_count"],
    summary["false_positive_rate"]
):
    plt.text(x, y + 1, f"{y:.2f}%", ha="center")

plt.ylim(0, 100)
plt.tight_layout()

plt.savefig("trigger_bias_graph.png", dpi=300)
plt.show()

print("\nSaved trigger_bias_summary.csv")
print("Saved trigger_bias_graph.png")
