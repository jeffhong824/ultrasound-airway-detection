# -*- coding: utf-8 -*-
"""
run_batch_models.py  ⭢  一條龍：
1. evaluate_model_thresholds.py   →  產出 metrics_by_threshold.csv
2. 擷取最佳 threshold (mAP50 → mAP50_95 → macro_f1)
3. materialize_full_outputs.py    →  產出 confusion matrix + 4 支影片

若 DRY_RUN=True 則僅列印最佳 threshold，不會產生影片。
"""

from pathlib import Path
import subprocess
import csv, math, sys

# ================= 你要比較的模型 =================
MODELS = [
    # --- det_123 ---
    # dict(case_ID="det_123", model_name="yolov8n", train_ID="20250630-211022"),
    # dict(case_ID="det_123", model_name="yolo11n", train_ID="20250630-012050"),
    # dict(case_ID="det_123", model_name="yolo12n", train_ID="20250630-095658"),

    # --- det_678 ---
    dict(case_ID="det_678", model_name="yolov8n", train_ID="20250705-145127"),
    dict(case_ID="det_678", model_name="yolo11n", train_ID="20250705-110402"),
    dict(case_ID="det_678", model_name="yolo12n", train_ID="20250702-030540"),
]

# ================= 共用參數 =================
BASE_DATA   = Path("../../yolo_dataset")              # <case>/v1/  下面要有 patient_data/ 與 subID_test.txt
RUNS_DIR    = Path("../runs/train")                   # Ultralytics 預設訓練輸出
PRED_ROOT   = Path("../pred_video")                   # 所有推論結果集中存放處
WEIGHTS_TM  = "{model_name}-{case_ID}-{train_ID}/weights/best.pt"

DEVICE      = "cuda:0"
IOU_THRES   = 0.5
THR_START   = 0.0
THR_STOP    = 1.0
THR_STEP    = 0.1
SHOW_ALL    = 0.0
FPS         = 10
DRY_RUN     = False        # True：只找最佳 threshold；False：順便產影片

# ------------------------------------------------------------------
def to_float(x: str) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("-inf")
    except Exception:
        return float("-inf")

def best_threshold(csv_path: Path) -> float:
    """讀 metrics_by_threshold.csv，依 mAP50→mAP50_95→macro_f1 取最佳閾值"""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise RuntimeError(f"{csv_path} 無任何資料")

    rows.sort(
        key=lambda r: (
            to_float(r.get("mAP50", "-inf")),
            to_float(r.get("mAP50_95", "-inf")),
            to_float(r.get("macro_f1", "-inf")),
        ),
        reverse=True,
    )
    best = rows[0]
    thr  = float(best["threshold"])
    print(f"   ↳ BEST threshold={thr:.4f} | mAP50={float(best['mAP50']):.6f} "
          f"| mAP50_95={float(best['mAP50_95']):.6f} | macro_f1={float(best['macro_f1']):.6f}")
    return thr

# ------------------------------------------------------------------
def main() -> None:
    for m in MODELS:
        case_ID, model_name, train_ID = m["case_ID"], m["model_name"], m["train_ID"]
        root     = BASE_DATA / case_ID / "v1"
        weights  = RUNS_DIR  / WEIGHTS_TM.format(model_name=model_name, case_ID=case_ID, train_ID=train_ID)

        # ---------- Step-1: evaluate thresholds ----------
        eval_cmd = [
            "python", "evaluate_model_thresholds.py",
            "--case-id",   case_ID,
            "--model-name",model_name,
            "--train-id",  train_ID,
            "--root",      str(root),
            "--weights",   str(weights),
            "--device",    DEVICE,
            "--iou-thres", str(IOU_THRES),
            "--thr-start", str(THR_START),
            "--thr-stop",  str(THR_STOP),
            "--thr-step",  str(THR_STEP),
        ]
        print("\n================ EVALUATE =================")
        print(">>", " ".join(eval_cmd))
        subprocess.check_call(eval_cmd)

        # 生成的 CSV 路徑
        csv_path = (
            PRED_ROOT / case_ID / f"{case_ID}_{model_name}_{train_ID}" / "all" /
            "metrics_by_threshold.csv"
        )
        if not csv_path.exists():
            print(f"❌ 找不到 {csv_path}，略過 materialize")
            continue

        # ---------- Step-2: 找最佳 threshold ----------
        best_thr = best_threshold(csv_path)

        if DRY_RUN:
            continue  # 只列最佳 threshold，不跑影片

        # ---------- Step-3: materialize full outputs ----------
        mat_cmd = [
            "python", "materialize_full_outputs.py",
            "--case-id",   case_ID,
            "--model-name",model_name,
            "--train-id",  train_ID,
            "--root",      str(root),
            "--weights",   str(weights),
            "--device",    DEVICE,
            "--iou-thres", str(IOU_THRES),
            "--primary-conf", str(best_thr),
            "--showall-conf", str(SHOW_ALL),
            "--fps",      str(FPS),
            "--out-root", str(PRED_ROOT),
        ]
        print("\n================ MATERIALIZE ===============")
        print(">>", " ".join(mat_cmd))
        try:
            subprocess.check_call(mat_cmd)
        except subprocess.CalledProcessError as e:
            # 不讓整批中斷，印錯誤即可
            print(f"⚠️ materialize_full_outputs 失敗（{case_ID}-{model_name}-{train_ID}）：{e}")
            continue

    print("\n🎉 全部模型處理完畢！")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
