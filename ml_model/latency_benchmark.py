"""Inference latency and throughput benchmark for the existing detector."""
import time
import statistics
import pandas as pd
from robustness_common import Detector, load_benign_prompts, prepare_output_dirs, CSV_DIR, GRAPH_DIR, save_plot

N_PROMPTS = 50
WARMUP_RUNS = 5
REPEATS = 3


def main():
    prepare_output_dirs()
    prompts = load_benign_prompts(N_PROMPTS)
    detector = Detector()
    for prompt in prompts[:min(WARMUP_RUNS, len(prompts))]:
        detector.score(prompt)
    rows = []
    for pid, prompt in enumerate(prompts):
        durations = []
        for _ in range(REPEATS):
            if detector.device.type == "cuda":
                import torch
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction, score = detector.score(prompt)
            if detector.device.type == "cuda":
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
        rows.append({"prompt_id": pid, "prediction": prediction,
                     "malicious_score": score, "mean_latency_ms": statistics.mean(durations),
                     "median_latency_ms": statistics.median(durations),
                     "min_latency_ms": min(durations), "max_latency_ms": max(durations),
                     "input_characters": len(prompt)})
        print(f"Latency prompts: {pid+1}/{len(prompts)}")
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / "latency_benchmark_results.csv", index=False)
    total_seconds = df.mean_latency_ms.sum() / 1000
    summary = pd.DataFrame([{
        "samples": len(df), "device": str(detector.device),
        "mean_latency_ms": df.mean_latency_ms.mean(),
        "median_latency_ms": df.median_latency_ms.median(),
        "p95_latency_ms": df.mean_latency_ms.quantile(0.95),
        "min_latency_ms": df.mean_latency_ms.min(),
        "max_latency_ms": df.mean_latency_ms.max(),
        "estimated_sequential_throughput_prompts_per_second": len(df) / total_seconds if total_seconds else 0,
        "warmup_runs_excluded": min(WARMUP_RUNS, len(prompts)), "repeats_per_prompt": REPEATS
    }])
    summary.to_csv(CSV_DIR / "latency_benchmark_summary.csv", index=False)
    plt = save_plot(None)
    plt.figure(figsize=(9, 5))
    plt.hist(df.mean_latency_ms, bins=15)
    plt.xlabel("Mean inference latency per prompt (ms)")
    plt.ylabel("Number of prompts")
    plt.title("Detector Inference Latency Distribution")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "latency_benchmark.png", dpi=300)
    print("\n", summary.to_string(index=False))
    print("Saved latency benchmark results and summary.")

if __name__ == "__main__":
    main()
