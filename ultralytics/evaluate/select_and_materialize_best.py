# -*- coding: utf-8 -*-
"""
select_and_materialize_best.py

從單一模型的 all/metrics_by_threshold.csv 中選出最佳 threshold：
排序鍵：mAP50（desc）→ mAP50_95（desc）→ macro_f1（desc）

選出後，自動呼叫 materialize_full_outputs.py，輸出 confusion matrices 與 4 支影片等完整結果。

也可加上 --dry-run 僅列印最佳門檻而不實作。
"""

import argparse
from pathlib import Path
import csv
import math
import subprocess

def to_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("-inf")
    except Exception:
        return float("-inf")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--train-id", required=True)
    p.add_argument("--root", type=Path, required=True, help="Dataset root (contains patient_data/)")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iou-thres", type=float, default=0.5)
    p.add_argument("--showall-conf", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--pred-root", type=Path, default=Path("../pred_video"))
    p.add_argument("--dry-run", action="store_true", help="Only print best threshold.")
    return p.parse_args()

def main():
    opt = parse_args()

    csv_path = opt.pred_root / opt.case_id / f"{opt.case_id}_{opt.model_name}_{opt.train_id}" / "all" / "metrics_by_threshold.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Sort: mAP50 desc -> mAP50_95 desc -> macro_f1 desc
    rows.sort(key=lambda r: (
        to_float(r.get("mAP50", "-inf")),
        to_float(r.get("mAP50_95", "-inf")),
        to_float(r.get("macro_f1", "-inf"))
    ), reverse=True)

    best = rows[0]
    best_thr = float(best["threshold"])
    print(f"[BEST] threshold={best_thr:.4f} | mAP50={float(best['mAP50']):.6f} "
          f"| mAP50_95={float(best['mAP50_95']):.6f} | macro_f1={float(best['macro_f1']):.6f}")

    if opt.dry_run:
        return

    # Call materialize_full_outputs.py
    cmd = [
        "python", "materialize_full_outputs.py",
        "--case-id", opt.case_id,
        "--model-name", opt.model_name,
        "--train-id", opt.train_id,
        "--root", str(opt.root),
        "--weights", str(opt.weights),
        "--device", opt.device,
        "--iou-thres", str(opt.iou_thres),
        "--primary-conf", str(best_thr),
        "--showall-conf", str(opt.showall_conf),
        "--fps", str(opt.fps),
        "--out-root", str(opt.pred_root),
    ]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
