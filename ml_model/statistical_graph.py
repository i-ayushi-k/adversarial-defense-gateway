
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("statistical_analysis_results.csv")

# ---------------- FLIP RATE GRAPH ----------------

plt.figure(figsize=(11, 6))

x = np.arange(len(df))

y = df["flip_rate_percent"]

lower_error = y - df["ci_lower_percent"]
upper_error = df["ci_upper_percent"] - y

plt.bar(x, y, alpha=0.8)

plt.errorbar(
    x,
    y,
    yerr=[lower_error, upper_error],
    fmt="none",
    capsize=6,
    color="black"
)

plt.xticks(
    x,
    df["trigger"],
    rotation=15,
    ha="right"
)

plt.ylabel("Prediction Flip Rate (%)")
plt.xlabel("Injected Trigger")
plt.title("Statistical Analysis of Trigger-Induced Classification Bias")

plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "statistical_flip_rate_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------- SCORE INCREASE GRAPH ----------------

plt.figure(figsize=(11, 6))

plt.bar(
    x,
    df["mean_score_increase"],
    alpha=0.8
)

plt.xticks(
    x,
    df["trigger"],
    rotation=15,
    ha="right"
)

plt.ylabel("Mean Increase in Malicious Score")
plt.xlabel("Injected Trigger")
plt.title("Trigger-Induced Increase in Malicious Classification Score")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "statistical_score_increase_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Statistical graphs generated successfully!")
