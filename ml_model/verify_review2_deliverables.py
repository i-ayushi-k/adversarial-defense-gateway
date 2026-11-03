"""
Review 2 deliverables verification
Run from project root:
    python ml_model/verify_review2_deliverables.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELIVERABLES = ROOT / "final_deliverable_1"

REQUIRED_FILES = [
    ("CSV", DELIVERABLES / "csv_results" / "notinject_results.csv"),
    ("CSV", DELIVERABLES / "csv_results" / "trigger_bias_controlled_results.csv"),
    ("CSV", DELIVERABLES / "csv_results" / "trigger_bias_controlled_summary.csv"),
    ("CSV", DELIVERABLES / "csv_results" / "statistical_analysis_results.csv"),
    ("Graph", DELIVERABLES / "graphs" / "statistical_flip_rate_graph.png"),
    ("Graph", DELIVERABLES / "graphs" / "statistical_score_increase_graph.png"),
    ("Graph", DELIVERABLES / "graphs" / "trigger_bias_controlled_graph.png"),
    ("Graph", DELIVERABLES / "graphs" / "notinject_baseline_score_distribution.png"),
    ("Report", DELIVERABLES / "final_experiment_report.txt"),
    ("Index", DELIVERABLES / "consolidated_file_index.csv"),
]

OPTIONAL_FILES = [
    ("Optional CSV", DELIVERABLES / "csv_results" / "threshold_analysis_results.csv"),
    ("Optional CSV", DELIVERABLES / "csv_results" / "trigger_comparison_results.csv"),
    ("Optional CSV", DELIVERABLES / "csv_results" / "adversarial_results.csv"),
]

print("=" * 60)
print("REVIEW 2 DELIVERABLES VERIFICATION")
print("=" * 60)
print("Project:", ROOT)
print("Deliverables:", DELIVERABLES, "\\n")

missing_required = []
for category, path in REQUIRED_FILES:
    exists = path.is_file()
    status = "OK" if exists else "MISSING"
    print(f"[{status:7}] {category:8} {path.relative_to(ROOT)}")
    if not exists:
        missing_required.append(path)

print("\\nOptional outputs:")
for category, path in OPTIONAL_FILES:
    status = "PRESENT" if path.is_file() else "not found"
    print(f"[{status:10}] {category:12} {path.relative_to(ROOT)}")

graphs_dir = DELIVERABLES / "graphs"
csv_dir = DELIVERABLES / "csv_results"
graph_count = len(list(graphs_dir.glob("*.png"))) if graphs_dir.exists() else 0
csv_count = len(list(csv_dir.glob("*.csv"))) if csv_dir.exists() else 0

print("\\nCounts:")
print("CSV files in csv_results:", csv_count)
print("PNG graphs in graphs:", graph_count)

print("\\n" + "=" * 60)
if missing_required:
    print(f"CHECK RESULT: INCOMPLETE — {len(missing_required)} required item(s) missing.")
    print("Fix the missing paths above, then run this verifier again.")
else:
    print("CHECK RESULT: REQUIRED FILES PRESENT.")
    print("Note: this verifies file presence only, not scientific correctness/content.")
print("=" * 60)
