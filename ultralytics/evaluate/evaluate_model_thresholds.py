# -*- coding: utf-8 -*-
"""
evaluate_model_thresholds.py

單一模型：一次以 conf=0.0 做推論，離線掃描 thresholds（0.0 ~ 1.0），
可選是否只保留每個 class confidence 最高的 box。
輸出 Macro/Micro（P/R/F1）與 mAP50 / mAP50_95。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any
import csv
import warnings
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from common_eval import (
    get_class_names, keep_max_per_class, iou_xyxy,
    update_stats, compute_macro_micro, compute_map
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--train-id", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--img-ext", nargs="+", default=[".png", ".jpg"])
    p.add_argument("--iou-thres", type=float, default=0.5)

    p.add_argument("--thr-start", type=float, default=0.0)
    p.add_argument("--thr-stop",  type=float, default=1.0)
    p.add_argument("--thr-step",  type=float, default=0.01)

    p.add_argument("--keep-max-per-class", action="store_true", default=True,
                   help="是否保留每張圖中每類別最大信心值的框（可與 top1-per-class 二選一）")
    p.add_argument("--top1-per-class", action="store_true", default=False,
                   help="只保留每張圖每類別 confidence 最高的框")
    p.add_argument("--out-root", type=Path, default=Path("../pred_video"))

    return p.parse_args()

def main():
    print("📌 開始執行 evaluate_model_thresholds.py")
    warnings.filterwarnings("ignore")
    opt = parse_args()
    print(f"🔧 輸入參數: case={opt.case_id}, model={opt.model_name}, train_id={opt.train_id}")
    print(f"🔧 使用模型權重: {opt.weights.resolve()}")
    print(f"🔧 資料來源路徑: {opt.root.resolve()}")
    print(f"🔍 使用 Top-1 per Class 模式: {opt.top1_per_class}")

    # 驗證 weights 路徑存在
    assert opt.weights.exists(), f"❌ 找不到模型權重檔案: {opt.weights}"
    assert (opt.root / "subID_test.txt").exists(), f"❌ 找不到 subID_test.txt：{opt.root/'subID_test.txt'}"

    CLASS_NAMES = get_class_names(opt.case_id)
    class_ids = sorted(CLASS_NAMES.keys())
    is_seg = ("seg" in opt.model_name.lower()) or ("seg" in opt.case_id.lower())
    print(f"📚 類別數量: {len(class_ids)} | 類別名稱: {[CLASS_NAMES[c] for c in class_ids]}")

    out_dir = opt.out_root / opt.case_id / f"{opt.case_id}_{opt.model_name}_{opt.train_id}" / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "metrics_by_threshold.csv"
    print(f"📁 評估結果將輸出至: {out_csv.resolve()}")

    patient_root = opt.root / "patient_data"
    id_txt = opt.root / "subID_test.txt"
    patient_ids = [ln.strip() for ln in id_txt.read_text().splitlines() if ln.strip()]
    print(f"🧪 測試病患數量: {len(patient_ids)}")

    # 載入模型
    print("🧠 載入模型中 ...")
    model = YOLO(str(opt.weights)).to(opt.device)
    print("✅ 模型載入完成")

    print("📥 開始推論所有圖片 (conf=0.0)")

    images = []
    raw_preds: Dict[str, Dict[str, np.ndarray]] = {}
    gts: List[Dict[str, Any]] = []

    for pid in patient_ids:
        img_paths = sorted([p for p in (patient_root/pid).rglob("*")
                            if p.suffix.lower() in opt.img_ext])
        for img_path in tqdm(img_paths, desc=f"[Load {pid}]"):
            img = cv2.imread(str(img_path)); h, w = img.shape[:2]
            res = model(img, conf=0.0, verbose=False)[0]

            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4))
            cls   = res.boxes.cls.cpu().numpy()   if res.boxes is not None else np.empty((0,))
            sc    = res.boxes.conf.cpu().numpy()  if res.boxes is not None else np.empty((0,))
            image_id = f"{pid}/{img_path.name}"
            raw_preds[image_id] = dict(boxes=boxes, scores=sc, classes=cls)
            images.append((pid, img_path, w, h))

            # ground truth
            label_path = img_path.with_suffix(".txt")
            gt_boxes, gt_cls = [], []
            if label_path.exists():
                for ln in label_path.read_text().splitlines():
                    vals = list(map(float, ln.strip().split()))
                    if is_seg:
                        c = int(vals[0]); poly = np.array(vals[1:]).reshape(-1, 2)
                        poly[:, 0] *= w; poly[:, 1] *= h
                        x_min, y_min = poly.min(axis=0); x_max, y_max = poly.max(axis=0)
                        gt_cls.append(c); gt_boxes.append([x_min, y_min, x_max, y_max])
                    else:
                        if len(vals) != 5: continue
                        c, x, y, bw, bh = vals
                        gt_cls.append(int(c))
                        gt_boxes.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
            gts.append(dict(image_id=image_id, boxes=np.array(gt_boxes), classes=list(map(int, gt_cls))))

    thresholds = np.arange(opt.thr_start, opt.thr_stop + 1e-12, opt.thr_step)

    fieldnames = [
        "case_ID","model_name","train_ID","iou_thres","threshold",
        "macro_precision","macro_recall","macro_f1",
        "micro_precision","micro_recall","micro_f1",
        "mAP50","mAP50_95",
        "TP_total","FP_total","FN_total","Support_total"
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for thr in thresholds:
            preds_t = []
            stats_all: Dict[int, Dict[str, int]] = {}

            for pid, img_path, w, h in images:
                image_id = f"{pid}/{img_path.name}"
                pred = raw_preds[image_id]
                boxes = pred["boxes"]; scores = pred["scores"]; classes = pred["classes"]

                mask = scores >= thr
                boxes = boxes[mask]; scores_f = scores[mask]; classes_f = classes[mask]

                # ★ Top-1 per class
                if opt.top1_per_class:
                    filtered_boxes, filtered_scores, filtered_classes = [], [], []
                    for cls_id in np.unique(classes_f):
                        cls_mask = (classes_f == cls_id)
                        cls_scores = scores_f[cls_mask]
                        cls_boxes = boxes[cls_mask]
                        if len(cls_scores) > 0:
                            top_idx = np.argmax(cls_scores)
                            filtered_boxes.append(cls_boxes[top_idx])
                            filtered_scores.append(cls_scores[top_idx])
                            filtered_classes.append(cls_id)
                    boxes = np.array(filtered_boxes)
                    scores_f = np.array(filtered_scores)
                    classes_f = np.array(filtered_classes)

                # ★ 原本邏輯（若非 top1 時才執行）
                if opt.keep_max_per_class and not opt.top1_per_class and len(boxes) > 0:
                    boxes, classes_f, scores_f = keep_max_per_class(boxes, classes_f, scores_f)

                preds_t.append(dict(image_id=image_id, boxes=boxes.copy(),
                                    scores=scores_f.copy(), classes=classes_f.copy()))

                gt_entry = next(d for d in gts if d["image_id"] == image_id)
                gt_boxes = gt_entry["boxes"]; gt_cls = gt_entry["classes"]

                matched = set()
                for p_box, p_cls in zip(boxes, classes_f):
                    best_iou, best_idx = 0.0, -1
                    for gi, (g_box, g_c) in enumerate(zip(gt_boxes, gt_cls)):
                        if gi in matched or g_c != int(p_cls):
                            continue
                        iou = iou_xyxy(p_box, g_box)
                        if iou > best_iou:
                            best_iou, best_idx = iou, gi
                    if best_iou >= opt.iou_thres and best_idx >= 0:
                        update_stats(stats_all, int(p_cls), tp=1)
                        matched.add(best_idx)
                    else:
                        update_stats(stats_all, int(p_cls), fp=1)
                for gi, g_c in enumerate(gt_cls):
                    if gi not in matched:
                        update_stats(stats_all, int(g_c), fn=1)

            agg = compute_macro_micro(stats_all)
            map_res = compute_map(preds_t, gts, class_ids)

            writer.writerow(dict(
                case_ID=opt.case_id, model_name=opt.model_name, train_ID=opt.train_id,
                iou_thres=opt.iou_thres, threshold=round(float(thr), 6),
                macro_precision=agg["macro"]["precision"],
                macro_recall=agg["macro"]["recall"],
                macro_f1=agg["macro"]["f1"],
                micro_precision=agg["micro"]["precision"],
                micro_recall=agg["micro"]["recall"],
                micro_f1=agg["micro"]["f1"],
                mAP50=map_res["map50"], mAP50_95=map_res["map5095"],
                TP_total=agg["totals"]["TP"], FP_total=agg["totals"]["FP"],
                FN_total=agg["totals"]["FN"], Support_total=agg["totals"]["Support"],
            ))

    print(f"[✅ 完成] 指標已輸出至: {out_csv}")

if __name__ == "__main__":
    main()
