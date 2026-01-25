import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

PRED_ROOT = Path("../pred_video/det_678") # det_123 | det_678
TARGET_METRICS = ["mAP50", "mAP50_95", "macro_f1", "micro_f1"]
REFERENCE_METRICS = TARGET_METRICS + ["macro_precision", "macro_recall", "micro_precision", "micro_recall"]
TOP_K = 5

def find_metrics_files(root: Path) -> List[Path]:
    return list(root.rglob("metrics_by_threshold.csv"))

def read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

def to_float(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")

def sort_key(row: Dict[str, Any]) -> Tuple:
    """Define sorting logic: higher scores first"""
    return tuple(-to_float(row.get(metric, "nan")) for metric in TARGET_METRICS)

def main():
    files = find_metrics_files(PRED_ROOT)
    if not files:
        print(f"No metrics_by_threshold.csv under {PRED_ROOT}")
        return

    # Step 1: Merge all metrics_by_threshold.csv
    combined_rows = []
    for fp in files:
        rows = read_csv(fp)
        trio = fp.parts[-3]
        parts = trio.split("_")
        case_ID, model_name = parts[0], parts[1]
        train_ID = "_".join(parts[2:])
        for r in rows:
            r["case_ID"] = r.get("case_ID", case_ID)
            r["model_name"] = r.get("model_name", model_name)
            r["train_ID"] = r.get("train_ID", train_ID)
            combined_rows.append(r)

    # Step 2: Save merged file
    out_dir = PRED_ROOT / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / "combined_metrics.csv"
    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=combined_rows[0].keys())
        writer.writeheader()
        writer.writerows(combined_rows)

    # Step 3: Find best threshold per model
    model_groups = defaultdict(list)
    for r in combined_rows:
        key = (r["case_ID"], r["model_name"], r["train_ID"])
        model_groups[key].append(r)

    best_per_model = []
    for key, rows in model_groups.items():
        best_row = sorted(rows, key=sort_key)[0]
        best_per_model.append(best_row)

    # Step 4: Sort all models
    best_per_model_sorted = sorted(best_per_model, key=sort_key)

    # Step 5: Save best_by_metric.csv
    best_csv = out_dir / "best_by_metric.csv"
    with open(best_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_per_model_sorted[0].keys())
        writer.writeheader()
        writer.writerows(best_per_model_sorted)

    # Step 6: Save topk_by_metric.csv
    topk_csv = out_dir / "topk_by_metric.csv"
    with open(topk_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank"] + list(best_per_model_sorted[0].keys()))
        writer.writeheader()
        for idx, row in enumerate(best_per_model_sorted[:TOP_K], 1):
            writer.writerow({"rank": idx, **row})

    print(f"[OK] combined         -> {combined_csv}")
    print(f"[OK] best_by_metric   -> {best_csv}")
    print(f"[OK] topk_by_metric   -> {topk_csv}")

if __name__ == "__main__":
    main()