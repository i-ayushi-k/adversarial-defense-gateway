
import pandas as pd
import numpy as np
from scipy.stats import binomtest, wilcoxon

# Load controlled experiment results
df = pd.read_csv("trigger_bias_controlled_results.csv")

original = df[df["condition"] == "Original"].set_index("prompt_id")
modified = df[df["condition"] == "Modified"].copy()

modified["original_score"] = modified["prompt_id"].map(
    original["malicious_score"]
)

modified["original_prediction"] = modified["prompt_id"].map(
    original["prediction"]
)

modified["score_increase"] = (
    modified["malicious_score"] - modified["original_score"]
)

modified["prediction_flip"] = (
    (modified["original_prediction"] == "SAFE") &
    (modified["prediction"] == "INJECTION")
)

print("\n======= STATISTICAL ANALYSIS =======")

summary = []

for trigger, group in modified.groupby("trigger"):

    flips = int(group["prediction_flip"].sum())
    n = len(group)

    ci = binomtest(flips, n).proportion_ci(
        confidence_level=0.95
    )

    original_scores = group["original_score"]
    modified_scores = group["malicious_score"]

    test = wilcoxon(
        modified_scores,
        original_scores,
        alternative="greater"
    )

    summary.append({
        "trigger": trigger,
        "samples": n,
        "prediction_flips": flips,
        "flip_rate_percent": flips / n * 100,
        "ci_lower_percent": ci.low * 100,
        "ci_upper_percent": ci.high * 100,
        "mean_score_increase": group["score_increase"].mean(),
        "wilcoxon_p_value": test.pvalue
    })

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

summary_df.to_csv(
    "statistical_analysis_results.csv",
    index=False
)

print("\nSaved statistical_analysis_results.csv")
