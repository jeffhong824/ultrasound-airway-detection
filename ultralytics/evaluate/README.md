# Airway Detection Evaluation – Threshold Sweep & Best Materialization

本專案提供一個**兩階段**的評估流程：

1. **快速掃描階段（No Confusion Matrix, No Videos）**  
   對單一模型只做一次 `conf=0.0` 的推論，接著離線掃描 **0.00 ~ 1.00** 的 confidence threshold，  
   針對每個 threshold 輸出 **ALL 層級** 的
   - Macro/Micro：Precision、Recall、F1 （基於事件級 TP/FP/FN）
   - mAP50、mAP50_95（在該 threshold 過濾後的預測集合上進行 ranking-based 積分）
   產物：`pred_video/<case>_<model>_<train>/all/metrics_by_threshold.csv`

2. **最佳化實體化階段（Materialize Best）**  
   依序以 **mAP50 → mAP50_95 → Macro‑F1** 排序挑選**最佳 primary threshold**，  
   再以此門檻重新評估並輸出完整成果（含 **per‑patient** / **ALL** 的 confusion matrices、詳細 metrics、以及 4 支影片）。

---

## 目錄

- `common_eval.py`：共用工具（配對、mAP、混淆矩陣、繪圖等）
- `evaluate_model_thresholds.py`：單一模型掃描 0.0~1.0 門檻，輸出 `metrics_by_threshold.csv`
- `run_batch_models.py`：批次跑多個模型
- `aggregate_compare.py`：彙整多模型；輸出 `combined_metrics.csv` 與 `best_by_metric.csv` / `topk_by_metric.csv`
- `select_and_materialize_best.py`：由單一模型的 `metrics_by_threshold.csv` 選最佳門檻並呼叫 `materialize_full_outputs.py`
- `materialize_full_outputs.py`：以指定門檻輸出完整資訊（confusion matrix + videos）

---

## 安裝需求

- Python 3.9+
- Ultralytics YOLO（需與你的訓練版本相容）
- OpenCV-Python、NumPy、tqdm

```bash
pip install ultralytics opencv-python numpy tqdm
```
---

## 步驟一：掃描單一模型的 thresholds

```bash
python evaluate_model_thresholds.py \
  --case-id det_123 \
  --model-name yolov8n \
  --train-id 20250630-211022 \
  --root "../../yolo_dataset/det_123/v1" \
  --weights "../runs/train/yolov8n-det_123-20250630-211022/weights/best.pt" \
  --device cuda:0 \
  --iou-thres 0.5 \
  --thr-start 0.0 --thr-stop 1.0 --thr-step 0.1
```
```bash
python evaluate_model_thresholds.py --case-id det_123 --model-name yolov8n --train-id 20250630-211022 --root "../../yolo_dataset/det_123/v1" --weights "../runs/train/yolov8n-det_123-20250630-211022/weights/best.pt" --device cuda:0 --iou-thres 0.5 --thr-start 0.0 --thr-stop 1.0 --thr-step 0.01
```

## 步驟二（選擇性）：批次對多模型掃描

```bash
python run_batch_models.py
```

## 步驟三（選擇性）：跨模型彙整與比較

```bash
python aggregate_compare.py
```

## 步驟四：挑選最佳門檻並產出完整結果

```bash
python select_and_materialize_best.py \
  --case-id det_123 \
  --model-name yolo12n \
  --train-id 20250630-095658 \
  --root "D:/.../yolo_dataset/det_123/v1" \
  --weights "./runs/train/yolo12n-det_123-20250630-095658/weights/best.pt" \
  --device cuda:0 \
  --iou-thres 0.5 \
  --showall-conf 0.0 \
  --fps 10
```

```bash
python select_and_materialize_best.py --case-id det_123 --model-name yolo12n --train-id 20250630-095658 --root "../../yolo_dataset/det_123/v1" --weights "../runs/train/yolo12n-det_123-20250630-095658/weights/best.pt" --device cuda:0 --iou-thres 0.5 --showall-conf 0.0 --fps 10
```