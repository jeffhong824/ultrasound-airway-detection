# -*- coding: utf-8 -*-
"""
common_eval.py
共用工具：類別名稱、幾何、TP/FP/FN 配對、Macro/Micro 指標、AP/mAP、繪圖等。
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2

# --------------------------- Class names ---------------------------
def get_class_names(case_id: str) -> Dict[int, str]:
    if case_id == "det_123":
        return {0: "Mentum", 1: "Hyoid"}
    elif case_id == "det_678":
        return {0: "Hyoid_Bone", 1: "Epiglottis", 2: "Epiglottis2VC"}
    elif case_id == "seg_45":
        return {0: "Tongue_Upper", 1: "Tongue_Lower"}
    else:
        raise ValueError(f"Unknown case_ID: {case_id}")

# --------------------------- Geometry ---------------------------
def iou_xyxy(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(xB-xA, 0) * max(yB-yA, 0)
    if inter <= 0:
        return 0.0
    boxAA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBA = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return float(inter / (boxAA + boxBA - inter + 1e-16))

def keep_max_per_class(boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray):
    """每張影像每個類別僅保留最高分框。"""
    max_dict = {}
    for b, c, s in zip(boxes, classes, scores):
        c = int(c)
        if c not in max_dict or s > max_dict[c][2]:
            max_dict[c] = (b, c, s)
    if not max_dict:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
    B, C, S = zip(*max_dict.values())
    return np.array(B), np.array(C), np.array(S)

def keep_max_mask_per_class(masks: np.ndarray, classes: np.ndarray, scores: np.ndarray, image_shape):
    """Segmentation 版本：每類保留最高分 mask。"""
    max_dict = {}
    for i, (c, s) in enumerate(zip(classes, scores)):
        c = int(c)
        if c not in max_dict or s > max_dict[c][1]:
            max_dict[c] = (masks[i], s)
    if not max_dict:
        return np.empty((0, *image_shape)), np.empty((0,)), np.empty((0,))
    masks_max, scores_max = zip(*max_dict.values())
    classes_max = list(max_dict.keys())
    return np.stack(masks_max), np.array(classes_max), np.array(scores_max)

# --------------------------- Count-based stats ---------------------------
def update_stats(stats: Dict[int, Dict[str, int]], cls: int, tp=0, fp=0, fn=0):
    if cls not in stats:
        stats[cls] = {"TP": 0, "FP": 0, "FN": 0}
    stats[cls]["TP"] += tp; stats[cls]["FP"] += fp; stats[cls]["FN"] += fn

def compute_macro_micro(stats: Dict[int, Dict[str, int]]) -> Dict[str, Any]:
    precisions, recalls, f1s = [], [], []
    TP_tot = FP_tot = FN_tot = 0
    for v in stats.values():
        TP, FP, FN = v["TP"], v["FP"], v["FN"]
        p = TP / (TP + FP) if (TP + FP) else 0.0
        r = TP / (TP + FN) if (TP + FN) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f)
        TP_tot += TP; FP_tot += FP; FN_tot += FN
    macro = dict(
        precision=float(np.mean(precisions)) if precisions else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        f1=float(np.mean(f1s)) if f1s else 0.0,
    )
    micro_p = TP_tot / (TP_tot + FP_tot) if (TP_tot + FP_tot) else 0.0
    micro_r = TP_tot / (TP_tot + FN_tot) if (TP_tot + FN_tot) else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    micro = dict(precision=micro_p, recall=micro_r, f1=micro_f)
    return {
        "macro": macro,
        "micro": micro,
        "totals": {"TP": TP_tot, "FP": FP_tot, "FN": FN_tot, "Support": TP_tot + FN_tot}
    }

# --------------------------- AP / mAP ---------------------------
def _compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    mrec  = np.concatenate(([0.0], recall, [1.0]))
    mpre  = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def compute_map(
    all_preds: List[Dict[str, Any]],
    all_gts:   List[Dict[str, Any]],
    class_ids: List[int],
    iou_thresholds: np.ndarray = np.arange(0.5, 0.96, 0.05),
) -> Dict[str, Any]:
    """
    在「輸入的預測集合」上計算 AP grid 與 mAP。
    注意：若你想報告「完整 conf=0.0 全集合」的純正 AP，請在過濾前另計一遍。
    """
    gt_dict = {d["image_id"]: d for d in all_gts}

    npos_per_class = {c: 0 for c in class_ids}
    for g in all_gts:
        for c in g["classes"]:
            if c in npos_per_class:
                npos_per_class[c] += 1

    preds_by_class = {c: [] for c in class_ids}
    for p in all_preds:
        img_id = p["image_id"]
        for b, s, c in zip(p["boxes"], p["scores"], p["classes"]):
            if c in preds_by_class:
                preds_by_class[int(c)].append((float(s), img_id, b))

    aps_grid = {c: [] for c in class_ids}

    for iou_thr in iou_thresholds:
        for cls in class_ids:
            preds = preds_by_class[cls]
            npos  = npos_per_class[cls]
            if npos == 0:
                aps_grid[cls].append(np.nan)
                continue

            preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
            tp_list, fp_list = [], []
            matched_cache: Dict[str, set] = {}

            for score, img_id, box in preds_sorted:
                gt_entry = gt_dict.get(img_id)
                if gt_entry is None:
                    tp_list.append(0); fp_list.append(1); continue
                gboxes, gcls = gt_entry["boxes"], gt_entry["classes"]
                if len(gboxes) == 0:
                    tp_list.append(0); fp_list.append(1); continue
                if img_id not in matched_cache:
                    matched_cache[img_id] = set()

                best_iou, best_j = 0.0, -1
                for j, (gb, gc) in enumerate(zip(gboxes, gcls)):
                    if gc != cls or j in matched_cache[img_id]:
                        continue
                    iou = iou_xyxy(box, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= iou_thr and best_j >= 0:
                    tp_list.append(1); fp_list.append(0)
                    matched_cache[img_id].add(best_j)
                else:
                    tp_list.append(0); fp_list.append(1)

            if len(tp_list) == 0:
                aps_grid[cls].append(0.0)
                continue

            tp_cum = np.cumsum(tp_list)
            fp_cum = np.cumsum(fp_list)
            recalls    = tp_cum / (npos + 1e-16)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
            ap = _compute_ap_from_pr(precisions, recalls)
            aps_grid[cls].append(ap)

    aps50   = {c: vals[0] if len(vals)>0 else np.nan for c, vals in aps_grid.items()}
    apsmean = {c: np.nanmean(vals) for c, vals in aps_grid.items()}
    valid   = [c for c in class_ids if npos_per_class[c] > 0]
    map50   = float(np.nanmean([aps50[c]   for c in valid])) if valid else 0.0
    map5095 = float(np.nanmean([apsmean[c] for c in valid])) if valid else 0.0

    return dict(
        map50=map50,
        map5095=map5095,
        aps50=aps50,
        aps_by_class_50=aps50, 
        aps_grid=aps_grid
    )

# --------------------------- Confusion matrix ---------------------------
def update_confusion(conf_mat: np.ndarray,
                     pred_boxes: np.ndarray, pred_cls: np.ndarray, pred_scores: np.ndarray,
                     gt_boxes: np.ndarray,   gt_cls:   List[int],
                     iou_thr: float,
                     K: int):
    """
    事件級（event-level）混淆矩陣建構。矩陣大小 (K+1)x(K+1)，index K 為 BG。
    """
    BG = K
    used_gt = set()

    if len(pred_scores) > 0:
        order = np.argsort(-pred_scores)
        pred_boxes  = pred_boxes[order]
        pred_cls    = pred_cls[order]
        pred_scores = pred_scores[order]

    for pbox, pcl in zip(pred_boxes, pred_cls):
        best_iou, best_j = 0.0, -1
        for gj, (gbox, gcl) in enumerate(zip(gt_boxes, gt_cls)):
            if gj in used_gt:
                continue
            iou = iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou, best_j = iou, gj
        if best_iou >= iou_thr and best_j >= 0:
            conf_mat[int(gt_cls[best_j]), int(pcl)] += 1
            used_gt.add(best_j)
        else:
            conf_mat[BG, int(pcl)] += 1  # FP

    for gj, gcl in enumerate(gt_cls):
        if gj not in used_gt:
            conf_mat[int(gcl), BG] += 1  # FN

def stats_from_confusion(conf_mat: np.ndarray, K: int) -> Dict[str, Any]:
    """由 (K+1)x(K+1) 混淆矩陣推導 per-class 與 macro/micro 指標。"""
    BG = K
    N_eval = int(conf_mat.sum())

    per_class: Dict[int, Dict[str, float]] = {}
    total_TP = total_FP = total_FN = 0

    for i in range(K):
        TP = int(conf_mat[i, i])
        FP = int(conf_mat[BG, i] + conf_mat[:K, i].sum() - conf_mat[i, i])
        FN = int(conf_mat[i, BG] + conf_mat[i, :K].sum() - conf_mat[i, i])
        TN = N_eval - TP - FP - FN
        support = int(conf_mat[i, :].sum())

        p = TP / (TP + FP) if (TP + FP) else 0.0
        r = TP / (TP + FN) if (TP + FN) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0

        per_class[i] = dict(TP=TP, FP=FP, FN=FN, TN=TN, Support=support,
                            precision=p, recall=r, f1=f)

        total_TP += TP; total_FP += FP; total_FN += FN

    macro_p = float(np.mean([per_class[i]["precision"] for i in range(K)])) if K else 0.0
    macro_r = float(np.mean([per_class[i]["recall"]    for i in range(K)])) if K else 0.0
    macro_f = float(np.mean([per_class[i]["f1"]        for i in range(K)])) if K else 0.0

    micro_p = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0.0
    micro_r = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return dict(
        per_class=per_class,
        macro=dict(precision=macro_p, recall=macro_r, f1=macro_f),
        micro=dict(precision=micro_p, recall=micro_r, f1=micro_f),
        totals=dict(TP=total_TP, FP=total_FP, FN=total_FN, N_eval=N_eval),
    )

# --------------------------- Drawing ---------------------------
def draw_boxes(img, boxes, classes, confs, colors, names, patient_id=None):
    for xyxy, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, xyxy)
        color = colors.get(int(cls), (255, 255, 255))
        label = f"{names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if patient_id:
        cv2.putText(img, f"Patient: {patient_id}",
                    (img.shape[1]//2 - 100, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

def draw_segmentation(img, masks, classes, confs, colors, names,
                      patient_id=None, alpha=0.4, is_gt=False):
    overlay = img.copy()
    for idx, mask in enumerate(masks):
        cls = int(classes[idx])
        conf = confs[idx] if not is_gt else 1.0
        color = tuple(min(c+100, 255) for c in colors.get(cls, (255, 255, 255)))
        mask_bin = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(mask_bin)
        label = f"{names[cls]}" if is_gt else f"{names[cls]} {conf:.2f}"
        cv2.putText(overlay, label, (x, max(y-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(overlay, label, (x, max(y-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)
    if patient_id:
        txt = f"Patient: {patient_id}"
        pos = (overlay.shape[1]//2 - 100, overlay.shape[0] - 10)
        cv2.putText(overlay, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(overlay, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
