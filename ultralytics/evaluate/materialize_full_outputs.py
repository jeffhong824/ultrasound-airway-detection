# -*- coding: utf-8 -*-
"""
materialize_full_outputs.py

以指定的 primary threshold 與（可選）show-all threshold，輸出完整資訊：
- 每病患：pred.mp4 / compare.mp4 / pred_all.mp4 / compare_all.mp4 / metrics.csv
- 每病患：confusion_primary.csv / confusion_show_all.csv
- 全體：all/metrics_summary.csv / all/confusion_primary.csv / all/confusion_show_all.csv
- 全體：all/*.mp4（四支影片）

改良點：
- 加入大量 debug 訊息與檢查點。
- 檢查並保證 VideoWriter 成功開啟；若尺寸不一致，自動 resize 到影片尺寸。
- cv2.imread 失敗會跳過並警告，不會整體中斷。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import csv
import warnings
import math

import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from common_eval import (
    get_class_names, keep_max_per_class, keep_max_mask_per_class,
    iou_xyxy, update_confusion, stats_from_confusion, compute_map,
    draw_boxes, draw_segmentation
)

# ---- 全域顏色（det 與 seg 皆可共用）----
COLORS = {
    0: (0, 100, 255),
    1: (0, 255, 255),
    2: (255, 100, 255),
    3: (255, 150, 100),
    4: (100, 255, 150),
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--train-id", required=True)
    p.add_argument("--root", type=Path, required=True,
                   help="Dataset root，底下需含 patient_data/ 與 subID_test.txt")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--img-ext", nargs="+", default=[".png", ".jpg"])
    p.add_argument("--iou-thres", type=float, default=0.5)
    p.add_argument("--primary-conf", type=float, required=True,
                   help="用來輸出完整資料的最佳 threshold。")
    p.add_argument("--showall-conf", type=float, default=0.0,
                   help="show-all 影片的 threshold，預設 0.0")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out-root", type=Path, default=Path("../pred_video"))
    return p.parse_args()


# ------------------ 輔助函式 ------------------
def open_writer(path: Path, fps: int, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """建立 VideoWriter 並檢查是否成功開啟。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(
            f"❌ 無法開啟 VideoWriter：{path}\n"
            f"   - 檔案資料夾是否有寫入權限？\n"
            f"   - fourcc='mp4v' 是否受系統支援？\n"
            f"   - frame_size={frame_size} 是否正確（寬,高）且固定？"
        )
    print(f"🎬 建立影片：{path} | fps={fps} | size={frame_size}")
    return writer


def ensure_size(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """若 frame 尺寸與目標尺寸不同，resize 成目標尺寸。"""
    h, w = frame.shape[:2]
    tw, th = target_size
    if (w, h) != (tw, th):
        frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
    return frame


def write_confusion_csv(path: Path, conf_mat: np.ndarray, class_names: Dict[int, str]):
    """輸出 (K+1)x(K+1) 混淆矩陣到 CSV，最後一列/欄是 BG。"""
    K = len(class_names)
    labels = [class_names[i] for i in range(K)] + ["BG"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GT\\Pred"] + labels + ["Row_Sum"])
        for i in range(K + 1):
            row_vals = conf_mat[i, :].astype(int).tolist()
            row_name = labels[i] if i < K else "BG"
            writer.writerow([row_name] + row_vals + [int(sum(row_vals))])
        col_sum = conf_mat.sum(axis=0).astype(int).tolist()
        writer.writerow(["Col_Sum"] + col_sum + [int(conf_mat.sum())])


# ------------------ 主程式 ------------------
def main():
    warnings.filterwarnings("ignore")
    opt = parse_args()

    print("========== materialize_full_outputs.py ==========")
    print(f"case={opt.case_id} | model={opt.model_name} | train={opt.train_id}")
    print(f"weights: {opt.weights.resolve()}")
    print(f"root   : {opt.root.resolve()}")
    print(f"primary_conf={opt.primary_conf} | showall_conf={opt.showall_conf} | iou_thres={opt.iou_thres}")
    print("=================================================")

    assert opt.weights.exists(), f"❌ 找不到權重檔：{opt.weights}"
    assert (opt.root / "subID_test.txt").exists(), f"❌ 找不到 subID_test.txt：{opt.root/'subID_test.txt'}"

    CLASS_NAMES = get_class_names(opt.case_id)
    class_ids = sorted(CLASS_NAMES.keys())
    print(f"📚 類別 ({len(class_ids)}): {[CLASS_NAMES[i] for i in class_ids]}")

    is_seg = ("seg" in opt.model_name.lower()) or ("seg" in opt.case_id.lower())
    print(f"🧩 模式：{'Segmentation' if is_seg else 'Detection'}")

    out_root = opt.out_root / opt.case_id / f"{opt.case_id}_{opt.model_name}_{opt.train_id}"
    (out_root / "all").mkdir(parents=True, exist_ok=True)
    print(f"📁 輸出根目錄：{out_root.resolve()}")

    # 欄位
    fieldnames = [
        "Patient","Threshold","Class",
        "TP","FP","FN","TN","Support",
        "Precision","Recall","F1","AP",
        "mAP50","mAP50_95"
    ]

    # 讀資料
    patient_root = opt.root / "patient_data"
    id_txt = opt.root / "subID_test.txt"
    patient_ids = [ln.strip() for ln in id_txt.read_text().splitlines() if ln.strip()]
    print(f"🧪 病患數：{len(patient_ids)}")

    # 載入模型
    print("🧠 載入 YOLO 權重中 ...")
    model = YOLO(str(opt.weights)).to(opt.device)
    print("✅ 模型載入完成")

    # ALL 影片寫入器與尺寸
    writers_all = {"pred": None, "cmp": None, "pred_all": None, "cmp_all": None}
    all_size_pred: Tuple[int, int] = None      # (w, h)
    all_size_cmp: Tuple[int, int] = None       # (2w, h)
    all_frames_counter = {"pred":0, "cmp":0, "pred_all":0, "cmp_all":0}

    # ALL 混淆矩陣
    K = len(CLASS_NAMES); BG = K
    conf_total_primary = np.zeros((K + 1, K + 1), dtype=int)
    conf_total_all     = np.zeros((K + 1, K + 1), dtype=int)

    # ALL mAP 收集
    all_preds_primary, all_preds_all, all_gts = [], [], []

    # ========== 逐病患 ==========
    for p_idx, pid in enumerate(patient_ids, start=1):
        pid_dir = patient_root / pid
        img_paths = sorted([p for p in pid_dir.rglob("*") if p.suffix.lower() in opt.img_ext])

        print(f"\n[{p_idx}/{len(patient_ids)}] 病患 {pid} | 影像數：{len(img_paths)}")
        out_pid = out_root / pid
        out_pid.mkdir(parents=True, exist_ok=True)

        # 單一病患影片 writer 與尺寸
        vw_pred = vw_cmp = vw_pred_all = vw_cmp_all = None
        size_pred: Tuple[int, int] = None
        size_cmp : Tuple[int, int] = None
        patient_frames = {"pred":0, "cmp":0, "pred_all":0, "cmp_all":0}

        conf_primary_pid = np.zeros((K + 1, K + 1), dtype=int)
        conf_all_pid     = np.zeros((K + 1, K + 1), dtype=int)

        preds_primary_pid, preds_all_pid, gts_pid = [], [], []

        for img_path in tqdm(img_paths, desc=f"[{pid}]"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ 無法讀取影像：{img_path}")
                continue

            h, w = img.shape[:2]

            # 推論
            res_primary = model(img, conf=opt.primary_conf, verbose=False)[0]
            res_all     = model(img, conf=opt.showall_conf,  verbose=False)[0]

            # 取出 detection
            pb   = res_primary.boxes.xyxy.cpu().numpy() if res_primary.boxes is not None else np.empty((0,4))
            pc   = res_primary.boxes.cls.cpu().numpy()   if res_primary.boxes is not None else np.empty((0,))
            psc  = res_primary.boxes.conf.cpu().numpy()  if res_primary.boxes is not None else np.empty((0,))

            pb_all  = res_all.boxes.xyxy.cpu().numpy() if res_all.boxes is not None else np.empty((0,4))
            pc_all  = res_all.boxes.cls.cpu().numpy()   if res_all.boxes is not None else np.empty((0,))
            psc_all = res_all.boxes.conf.cpu().numpy()  if res_all.boxes is not None else np.empty((0,))

            # 每類別保留最大信心框
            pb, pc, psc = keep_max_per_class(pb, pc, psc)
            pb_all, pc_all, psc_all = keep_max_per_class(pb_all, pc_all, psc_all)

            # 讀 GT
            gt_boxes, gt_cls = [], []
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                for ln in label_path.read_text().splitlines():
                    vals = list(map(float, ln.strip().split()))
                    if is_seg:
                        c = int(vals[0]); poly = np.array(vals[1:]).reshape(-1, 2)
                        poly[:, 0] *= w; poly[:, 1] *= h
                        x_min, y_min = poly.min(axis=0); x_max, y_max = poly.max(axis=0)
                        gt_cls.append(c); gt_boxes.append([x_min, y_min, x_max, y_max])
                    else:
                        if len(vals) != 5:
                            continue
                        c, x, y, bw, bh = vals
                        gt_cls.append(int(c))
                        gt_boxes.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
            gt_boxes = np.array(gt_boxes)
            image_id = f"{pid}/{img_path.name}"

            # 收集 mAP
            all_preds_primary.append(dict(image_id=image_id, boxes=pb.copy(),     scores=psc.copy(),     classes=pc.copy()))
            all_preds_all.append(   dict(image_id=image_id, boxes=pb_all.copy(), scores=psc_all.copy(), classes=pc_all.copy()))
            all_gts.append(dict(image_id=image_id, boxes=gt_boxes.copy(), classes=list(map(int, gt_cls))))

            preds_primary_pid.append(dict(image_id=image_id, boxes=pb.copy(),     scores=psc.copy(),     classes=pc.copy()))
            preds_all_pid.append(   dict(image_id=image_id, boxes=pb_all.copy(), scores=psc_all.copy(), classes=pc_all.copy()))
            gts_pid.append(dict(image_id=image_id, boxes=gt_boxes.copy(), classes=list(map(int, gt_cls))))

            # 混淆矩陣
            update_confusion(conf_primary_pid, pb,     pc,     psc,     gt_boxes, gt_cls, opt.iou_thres, K)
            update_confusion(conf_all_pid,     pb_all, pc_all, psc_all, gt_boxes, gt_cls, opt.iou_thres, K)

            # 視覺化 frame
            if is_seg:
                # 這裡僅建立 bbox 視覺化，也可選擇將 mask 繪出
                pred_img     = draw_boxes(img.copy(), pb,     pc,     psc,     COLORS, CLASS_NAMES, pid)
                pred_all_img = draw_boxes(img.copy(), pb_all, pc_all, psc_all, COLORS, CLASS_NAMES, pid)
                gt_img       = draw_boxes(img.copy(), gt_boxes, gt_cls, [1]*len(gt_cls), COLORS, CLASS_NAMES, pid)
            else:
                pred_img     = draw_boxes(img.copy(), pb,     pc,     psc,     COLORS, CLASS_NAMES, pid)
                pred_all_img = draw_boxes(img.copy(), pb_all, pc_all, psc_all, COLORS, CLASS_NAMES, pid)
                gt_img       = draw_boxes(img.copy(), gt_boxes, gt_cls, [1]*len(gt_cls), COLORS, CLASS_NAMES, pid)

            cmp_img     = np.hstack([pred_img, gt_img])
            cmp_all_img = np.hstack([pred_all_img, gt_img])

            # ---- 初始化單一病患影片 ----
            if vw_pred is None:
                size_pred = (pred_img.shape[1], pred_img.shape[0])    # (w, h)
                size_cmp  = (cmp_img.shape[1],  cmp_img.shape[0])
                vw_pred     = open_writer(out_pid / "pred.mp4",        opt.fps, size_pred)
                vw_cmp      = open_writer(out_pid / "compare.mp4",     opt.fps, size_cmp)
                vw_pred_all = open_writer(out_pid / "pred_all.mp4",    opt.fps, size_pred)
                vw_cmp_all  = open_writer(out_pid / "compare_all.mp4", opt.fps, size_cmp)

            # ---- 初始化 ALL 聚合影片 ----
            if writers_all["pred"] is None:
                all_size_pred = size_pred
                all_size_cmp  = size_cmp
                writers_all["pred"]     = open_writer(out_root / "all/pred.mp4",        opt.fps, all_size_pred)
                writers_all["cmp"]      = open_writer(out_root / "all/compare.mp4",     opt.fps, all_size_cmp)
                writers_all["pred_all"] = open_writer(out_root / "all/pred_all.mp4",    opt.fps, all_size_pred)
                writers_all["cmp_all"]  = open_writer(out_root / "all/compare_all.mp4", opt.fps, all_size_cmp)

            # ---- 寫入影片（若尺寸不同就 resize）----
            vw_pred.write(ensure_size(pred_img,     size_pred));     patient_frames["pred"]     += 1
            vw_cmp.write( ensure_size(cmp_img,      size_cmp));      patient_frames["cmp"]      += 1
            vw_pred_all.write(ensure_size(pred_all_img, size_pred)); patient_frames["pred_all"] += 1
            vw_cmp_all.write( ensure_size(cmp_all_img,  size_cmp));  patient_frames["cmp_all"]  += 1

            writers_all["pred"].write(    ensure_size(pred_img,     all_size_pred)); all_frames_counter["pred"]     += 1
            writers_all["cmp"].write(     ensure_size(cmp_img,      all_size_cmp));  all_frames_counter["cmp"]      += 1
            writers_all["pred_all"].write(ensure_size(pred_all_img, all_size_pred)); all_frames_counter["pred_all"] += 1
            writers_all["cmp_all"].write( ensure_size(cmp_all_img,  all_size_cmp));  all_frames_counter["cmp_all"]  += 1

        # 關閉單一病患影片
        for vw in [vw_pred, vw_cmp, vw_pred_all, vw_cmp_all]:
            if vw: vw.release()
        print(f"🎞 病患 {pid} 影片完成：pred={patient_frames['pred']} | cmp={patient_frames['cmp']} | "
              f"pred_all={patient_frames['pred_all']} | cmp_all={patient_frames['cmp_all']}")

        # 疊加至 ALL
        conf_total_primary += conf_primary_pid
        conf_total_all     += conf_all_pid

        # per-patient mAP 與度量
        map_primary_pid = compute_map(preds_primary_pid, gts_pid, class_ids)
        map_all_pid     = compute_map(preds_all_pid,    gts_pid, class_ids)
        stats_primary = stats_from_confusion(conf_primary_pid, K)
        stats_all     = stats_from_confusion(conf_all_pid, K)

        # confusion matrix CSV（per-patient）
        write_confusion_csv(out_pid / "confusion_primary.csv", conf_primary_pid, CLASS_NAMES)
        write_confusion_csv(out_pid / "confusion_show_all.csv", conf_all_pid, CLASS_NAMES)

        # per-patient metrics.csv
        with open(out_pid/"metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader()
            # primary per-class
            for cls in range(K):
                v = stats_primary["per_class"].get(cls, {})
                writer.writerow(dict(
                    Patient=pid, Threshold="primary", Class=CLASS_NAMES.get(cls, str(cls)),
                    TP=v.get("TP",0), FP=v.get("FP",0), FN=v.get("FN",0), TN=v.get("TN",0),
                    Support=v.get("Support",0),
                    Precision=v.get("precision",0.0), Recall=v.get("recall",0.0), F1=v.get("f1",0.0),
                    AP=map_primary_pid["aps_by_class_50"].get(cls, float("nan")),
                    mAP50="", mAP50_95=""
                ))
            # Macro / Micro
            writer.writerow(dict(
                Patient=pid, Threshold="primary", Class="Macro",
                TP="", FP="", FN="", TN="", Support="",
                Precision=stats_primary["macro"]["precision"],
                Recall=stats_primary["macro"]["recall"],
                F1=stats_primary["macro"]["f1"],
                AP="", mAP50=map_primary_pid["map50"], mAP50_95=map_primary_pid["map5095"]
            ))
            writer.writerow(dict(
                Patient=pid, Threshold="primary", Class="Micro",
                TP=stats_primary["totals"]["TP"],
                FP=stats_primary["totals"]["FP"],
                FN=stats_primary["totals"]["FN"],
                TN=stats_primary["totals"]["N_eval"] - stats_primary["totals"]["TP"]
                  - stats_primary["totals"]["FP"] - stats_primary["totals"]["FN"],
                Support="",
                Precision=stats_primary["micro"]["precision"],
                Recall=stats_primary["micro"]["recall"],
                F1=stats_primary["micro"]["f1"],
                AP="", mAP50="", mAP50_95=""
            ))
            # show_all per-class
            for cls in range(K):
                v = stats_all["per_class"].get(cls, {})
                writer.writerow(dict(
                    Patient=pid, Threshold="show_all", Class=CLASS_NAMES.get(cls, str(cls)),
                    TP=v.get("TP",0), FP=v.get("FP",0), FN=v.get("FN",0), TN=v.get("TN",0),
                    Support=v.get("Support",0),
                    Precision=v.get("precision",0.0), Recall=v.get("recall",0.0), F1=v.get("f1",0.0),
                    AP=map_all_pid["aps_by_class_50"].get(cls, float("nan")),
                    mAP50="", mAP50_95=""
                ))
            writer.writerow(dict(
                Patient=pid, Threshold="show_all", Class="Macro",
                TP="", FP="", FN="", TN="", Support="",
                Precision=stats_all["macro"]["precision"],
                Recall=stats_all["macro"]["recall"],
                F1=stats_all["macro"]["f1"],
                AP="", mAP50=map_all_pid["map50"], mAP50_95=map_all_pid["map5095"]
            ))
            writer.writerow(dict(
                Patient=pid, Threshold="show_all", Class="Micro",
                TP=stats_all["totals"]["TP"],
                FP=stats_all["totals"]["FP"],
                FN=stats_all["totals"]["FN"],
                TN=stats_all["totals"]["N_eval"] - stats_all["totals"]["TP"]
                  - stats_all["totals"]["FP"] - stats_all["totals"]["FN"],
                Support="",
                Precision=stats_all["micro"]["precision"],
                Recall=stats_all["micro"]["recall"],
                F1=stats_all["micro"]["f1"],
                AP="", mAP50="", mAP50_95=""
            ))

    # ========== ALL 層級 ==========
    map_primary = compute_map(all_preds_primary, all_gts, class_ids)
    map_all     = compute_map(all_preds_all,    all_gts, class_ids)
    stats_total_primary = stats_from_confusion(conf_total_primary, K)
    stats_total_all     = stats_from_confusion(conf_total_all, K)

    # ALL confusion CSV
    write_confusion_csv(out_root/"all/confusion_primary.csv", conf_total_primary, CLASS_NAMES)
    write_confusion_csv(out_root/"all/confusion_show_all.csv", conf_total_all, CLASS_NAMES)

    # ALL metrics_summary.csv
    with open(out_root/"all/metrics_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader()
        # primary per-class
        for cls in range(K):
            v = stats_total_primary["per_class"][cls]
            writer.writerow(dict(
                Patient="ALL", Threshold="primary", Class=CLASS_NAMES.get(cls, str(cls)),
                TP=v["TP"], FP=v["FP"], FN=v["FN"], TN=v["TN"],
                Support=v["Support"],
                Precision=v["precision"], Recall=v["recall"], F1=v["f1"],
                AP=map_primary["aps_by_class_50"].get(cls, float("nan")),
                mAP50="", mAP50_95=""
            ))
        writer.writerow(dict(
            Patient="ALL", Threshold="primary", Class="Macro",
            TP="", FP="", FN="", TN="", Support="",
            Precision=stats_total_primary["macro"]["precision"],
            Recall=stats_total_primary["macro"]["recall"],
            F1=stats_total_primary["macro"]["f1"],
            AP="", mAP50=map_primary["map50"], mAP50_95=map_primary["map5095"]
        ))
        writer.writerow(dict(
            Patient="ALL", Threshold="primary", Class="Micro",
            TP=stats_total_primary["totals"]["TP"],
            FP=stats_total_primary["totals"]["FP"],
            FN=stats_total_primary["totals"]["FN"],
            TN=stats_total_primary["totals"]["N_eval"] - stats_total_primary["totals"]["TP"]
              - stats_total_primary["totals"]["FP"] - stats_total_primary["totals"]["FN"],
            Support="",
            Precision=stats_total_primary["micro"]["precision"],
            Recall=stats_total_primary["micro"]["recall"],
            F1=stats_total_primary["micro"]["f1"],
            AP="", mAP50="", mAP50_95=""
        ))
        # show_all per-class
        for cls in range(K):
            v = stats_total_all["per_class"][cls]
            writer.writerow(dict(
                Patient="ALL", Threshold="show_all", Class=CLASS_NAMES.get(cls, str(cls)),
                TP=v["TP"], FP=v["FP"], FN=v["FN"], TN=v["TN"],
                Support=v["Support"],
                Precision=v["precision"], Recall=v["recall"], F1=v["f1"],
                AP=map_all["aps_by_class_50"].get(cls, float("nan")),
                mAP50="", mAP50_95=""
            ))
        writer.writerow(dict(
            Patient="ALL", Threshold="show_all", Class="Macro",
            TP="", FP="", FN="", TN="", Support="",
            Precision=stats_total_all["macro"]["precision"],
            Recall=stats_total_all["macro"]["recall"],
            F1=stats_total_all["macro"]["f1"],
            AP="", mAP50=map_all["map50"], mAP50_95=map_all["map5095"]
        ))
        writer.writerow(dict(
            Patient="ALL", Threshold="show_all", Class="Micro",
            TP=stats_total_all["totals"]["TP"],
            FP=stats_total_all["totals"]["FP"],
            FN=stats_total_all["totals"]["FN"],
            TN=stats_total_all["totals"]["N_eval"] - stats_total_all["totals"]["TP"]
              - stats_total_all["totals"]["FP"] - stats_total_all["totals"]["FN"],
            Support="",
            Precision=stats_total_all["micro"]["precision"],
            Recall=stats_total_all["micro"]["recall"],
            F1=stats_total_all["micro"]["f1"],
            AP="", mAP50="", mAP50_95=""
        ))

    # 關閉 ALL 影片
    for name, vw in writers_all.items():
        if vw:
            vw.release()
    print(f"\n🎞 ALL 影片 frame 數：{all_frames_counter}")

    print("\n===== Dataset mAP (Primary) =====")
    print(f"mAP@0.5      : {map_primary['map50']:.6f}")
    print(f"mAP@0.5:0.95 : {map_primary['map5095']:.6f}")
    print("===== Dataset mAP (Show-all) ====")
    print(f"mAP@0.5      : {map_all['map50']:.6f}")
    print(f"mAP@0.5:0.95 : {map_all['map5095']:.6f}")

    print("\n輸出完成：")
    print(f"- 每病患：pred.mp4 / compare.mp4 / pred_all.mp4 / compare_all.mp4 / metrics.csv")
    print(f"- 每病患：confusion_primary.csv / confusion_show_all.csv")
    print(f"- 全體：all/*.mp4 四支影片、all/metrics_summary.csv、all/confusion_*.csv")


if __name__ == "__main__":
    main()
