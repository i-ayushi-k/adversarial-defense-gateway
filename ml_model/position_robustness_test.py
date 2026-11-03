"""Position sensitivity test on benign segments; measures whole-prompt classifier output."""
import pandas as pd
from robustness_common import Detector, load_benign_prompts, prepare_output_dirs, CSV_DIR, GRAPH_DIR, save_plot

N_PROMPTS = 50
N_POSITIONS = 5


def main():
    prepare_output_dirs()
    prompts = load_benign_prompts(N_PROMPTS)
    detector = Detector()
    # Neutral, benign context segments keep the total number of segments constant.
    contexts = [
        "Please provide a concise explanation of this everyday topic.",
        "I am collecting notes for a general educational assignment.",
        "Use clear language and include one ordinary example.",
        "The information is for personal learning and documentation.",
    ]
    rows = []
    for pid, target in enumerate(prompts):
        for pos in range(N_POSITIONS):
            segments = contexts[:]
            segments.insert(pos, target)
            full_prompt = "\n\n".join(f"Segment {i+1}: {s}" for i, s in enumerate(segments))
            pred, score = detector.score(full_prompt)
            rows.append({"prompt_id": pid, "target_position": pos + 1,
                         "prediction": pred, "malicious_score": score,
                         "false_positive": int(pred == "INJECTION"), "prompt": full_prompt})
        print(f"Position prompts: {pid+1}/{len(prompts)}")
    df = pd.DataFrame(rows)
    out = CSV_DIR / "position_robustness_results.csv"
    df.to_csv(out, index=False)
    summary = df.groupby("target_position").agg(
        samples=("prompt_id", "count"),
        false_positives=("false_positive", "sum"),
        mean_malicious_score=("malicious_score", "mean"),
    ).reset_index()
    summary["false_positive_rate_percent"] = summary.false_positives / summary.samples * 100
    summary.to_csv(CSV_DIR / "position_robustness_summary.csv", index=False)
    plt = save_plot(None)
    plt.figure(figsize=(8, 5))
    plt.plot(summary.target_position, summary.false_positive_rate_percent, marker="o")
    plt.xlabel("Position of target benign segment")
    plt.ylabel("False-positive rate (%)")
    plt.title("Position Sensitivity on Benign Prompts")
    plt.xticks(range(1, N_POSITIONS + 1))
    plt.ylim(0, 100)
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "position_robustness.png", dpi=300)
    print("\n", summary.to_string(index=False))
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
