import csv
import pathlib
import sys
import importlib.util

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "bench", SCRIPT_DIR / "7_benchmark_experiments.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench"] = bench
_spec.loader.exec_module(bench)

DATA_DIR = SCRIPT_DIR / "data"


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            r["budget"]      = int(r["budget"])
            r["r2"]          = float(r["r2"])
            r["r2_lo"]       = float(r["r2_lo"])
            r["r2_hi"]       = float(r["r2_hi"])
            r["nrmse"]       = float(r["nrmse"])
            r["n_valid"]     = int(r["n_valid"])
            r["n_total"]     = int(r["n_total"])
            r["mean_time_s"] = float(r["mean_time_s"])
            rows.append(r)
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tags = sys.argv[1:]
        paths = [DATA_DIR / f"fig7_benchmark_{tag}.csv" for tag in tags]
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.exit(f"ERROR: missing files: {missing}")
    else:
        paths = sorted(DATA_DIR.glob("fig7_benchmark_*.csv"))
        if not paths:
            sys.exit(f"ERROR: no fig7_benchmark_*.csv files found in {DATA_DIR}. "
                     f"Download and extract all job artifacts first.")

    print(f"Merging {len(paths)} file(s):")

    all_rows = []
    seen_budgets = set()
    for path in paths:
        rows = load_rows(path)
        budgets_here = sorted(set(r["budget"] for r in rows))
        overlap = seen_budgets.intersection(budgets_here)
        if overlap:
            print(f"  WARNING: budgets {overlap} already seen — "
                 f"duplicate rows will be kept, check for overlapping jobs.")
        seen_budgets.update(budgets_here)
        print(f"  {path.name}: budgets={budgets_here}  ({len(rows)} rows)")
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["budget"], r["parameter"], r["method"]))

    print(f"\nTotal merged rows: {len(all_rows)}")
    print(f"All budgets present: {sorted(seen_budgets)}")
    expected = set(bench._ALL_BUDGETS)
    missing_budgets = expected - seen_budgets
    if missing_budgets:
        print(f"WARNING: missing budgets from full sweep: {sorted(missing_budgets)}")
    else:
        print("All expected budgets present: OK")

    merged_path = DATA_DIR / "fig7_benchmark.csv"
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bench.BENCH_COLS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved merged CSV: {merged_path}")

    bench.save_threshold_csv(all_rows)
    bench.plot_results(all_rows)
    print("\nDone. fig7_benchmark.csv, fig7_threshold.csv, and "
         "fig7_benchmark.pdf/.png are now the final, complete outputs.")