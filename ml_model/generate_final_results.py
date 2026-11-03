
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "final_deliverable_1"
CSV_OUT = OUTPUT / "csv_results"
GRAPH_OUT = OUTPUT / "graphs"

CSV_OUT.mkdir(parents=True, exist_ok=True)
GRAPH_OUT.mkdir(parents=True, exist_ok=True)

print("\n========== FINAL RESULTS COLLECTION ==========")

copied_csv = 0
copied_graphs = 0

# Search project root and ml_model
search_dirs = [ROOT, ROOT / "ml_model"]

for folder in search_dirs:

    for file in folder.glob("*.csv"):

        if file.parent == CSV_OUT:
            continue

        destination = CSV_OUT / file.name
        shutil.copy2(file, destination)

        copied_csv += 1
        print("CSV:", file.name)

    for file in folder.glob("*.png"):

        if file.parent == GRAPH_OUT:
            continue

        destination = GRAPH_OUT / file.name
        shutil.copy2(file, destination)

        copied_graphs += 1
        print("GRAPH:", file.name)

# Generate index
index = []

for file in CSV_OUT.glob("*.csv"):

    try:
        df = pd.read_csv(file)

        index.append({
            "filename": file.name,
            "rows": len(df),
            "columns": len(df.columns)
        })

    except Exception as e:
        print("Skipped:", file.name, e)

if index:

    pd.DataFrame(index).to_csv(
        OUTPUT / "consolidated_file_index.csv",
        index=False
    )

# Generate report
with open(
    OUTPUT / "final_experiment_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write("PROMPT INJECTION CLASSIFIER BIAS ANALYSIS\n")
    report.write("=" * 50 + "\n")

    report.write(
        "\nGenerated: "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    report.write("\n\nCOLLECTED CSV FILES\n")

    for file in CSV_OUT.glob("*.csv"):
        report.write("\n" + file.name)

    report.write("\n\nCOLLECTED GRAPHS\n")

    for file in GRAPH_OUT.glob("*.png"):
        report.write("\n" + file.name)

print("\n========== COLLECTION COMPLETE ==========")
print("CSV copied:", copied_csv)
print("Graphs copied:", copied_graphs)
print("Output:", OUTPUT)
