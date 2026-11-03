"""Deterministic wording/format perturbation test; transformations are transparent and reproducible."""
import pandas as pd
from robustness_common import Detector, load_benign_prompts, prepare_output_dirs, CSV_DIR, GRAPH_DIR, save_plot

N_PROMPTS = 50


def make_variants(prompt):
    stripped = prompt.strip()
    return {
        "Original": stripped,
        "PoliteFraming": f"Could you please help me with the following request? {stripped}",
        "EducationalFraming": f"For an educational explanation, please respond to this: {stripped}",
        "CaseVariation": stripped.swapcase(),
        "PunctuationVariation": stripped.rstrip(".!? ") + "...",
        "WhitespaceVariation": "   ".join(stripped.split()),
    }


def main():
    prepare_output_dirs()
    prompts = load_benign_prompts(N_PROMPTS)
    detector = Detector()
    rows = []
    for pid, prompt in enumerate(prompts):
        for transformation, variant in make_variants(prompt).items():
            pred, score = detector.score(variant)
            rows.append({"prompt_id": pid, "transformation": transformation,
                         "prediction": pred, "malicious_score": score,
                         "false_positive": int(pred == "INJECTION"), "prompt": variant})
        print(f"Wording variants: {pid+1}/{len(prompts)}")
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / "paraphrase_robustness_results.csv", index=False)
    originals = df[df.transformation == "Original"].set_index("prompt_id")
    df["original_prediction"] = df.prompt_id.map(originals.prediction)
    df["original_score"] = df.prompt_id.map(originals.malicious_score)
    df["prediction_consistent"] = (df.prediction == df.original_prediction).astype(int)
    df["absolute_score_change"] = (df.malicious_score - df.original_score).abs()
    summary = df.groupby("transformation").agg(
        samples=("prompt_id", "count"), false_positives=("false_positive", "sum"),
        consistency_rate=("prediction_consistent", "mean"),
        mean_absolute_score_change=("absolute_score_change", "mean"),
        mean_malicious_score=("malicious_score", "mean")
    ).reset_index()
    summary["false_positive_rate_percent"] = summary.false_positives / summary.samples * 100
    summary["consistency_rate_percent"] = summary.consistency_rate * 100
    summary.to_csv(CSV_DIR / "paraphrase_robustness_summary.csv", index=False)
    plt = save_plot(None)
    plt.figure(figsize=(10, 5))
    plt.bar(summary.transformation, summary.consistency_rate_percent)
    plt.ylabel("Prediction consistency with original (%)")
    plt.xlabel("Wording transformation")
    plt.title("Classifier Consistency Under Wording Variations")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "paraphrase_robustness.png", dpi=300)
    print("\n", summary.to_string(index=False))
    print("Saved paraphrase/wording robustness results and summary.")

if __name__ == "__main__":
    main()
