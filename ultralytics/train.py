import os
import tempfile
import json
import logging
import wandb
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
import torch
import psutil
from typing import Dict
import time

train_start_time = time.time()

# =====================================
# Configuration & Logging Setup
# =====================================
load_dotenv()

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["WANDB_MODE"] = "online"
case_ID = "seg_45" # det_123 | det_678 | seg_45
epochs = 1000
batch = 128
model_name = "yolov8n-seg" # yolov8n | yolo11n | yolo12n | yolov8n-seg 
os.environ["WANDB_PROJECT"] = f"ultrasound-{case_ID}"
os.environ["WANDB_RUN_NAME"] = f"{model_name}-{epochs}epochs"

# wandb.login()
# wandb.init(project=os.environ["WANDB_PROJECT"], name=os.environ["WANDB_RUN_NAME"])

now = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"{model_name}-{case_ID}-{now}"
is_segmentation = "seg" in model_name.lower()
os.environ["WANDB_RUN_NAME"] = run_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =====================================
# Evaluate function for val/test
# =====================================
# def evaluate(model: YOLO, split: str = "val") -> Dict:
#     logging.info(f"Evaluating on {split} split")
#     metrics = model.val(split=split)
#     mp, mr, map50, map = metrics.mean_results()

#     wandb.log({
#         f"{split}/mAP50": map50,
#         f"{split}/mAP50-95": map,
#         f"{split}/precision": mp,
#         f"{split}/recall": mr
#     })
#     return {
#         "precision": mp,
#         "recall": mr,
#         "mAP50": map50,
#         "mAP50-95": map
#     }

def evaluate(model: YOLO, split: str = "val") -> Dict:

    logging.info(f"🔍 Evaluating on {split} split ...")

    # 進行驗證
    metrics = model.val(split=split, batch=batch, imgsz=640)

    # 取得 mean results（整體）
    try:
        mp, mr, map50, map = metrics.mean_results()
    except Exception as e:
        logging.warning(f"❌ Failed to compute mean results: {e}")
        return {}

    # 嘗試擷取 per-class 結果
    try:
        per_class_metrics = metrics.box.mean_class_results  # shape = (num_classes, 6)
    except Exception as e:
        logging.warning(f"⚠️ Cannot extract per-class results: {e}")
        per_class_metrics = None

    names = model.names if hasattr(model, "names") else {i: str(i) for i in range(per_class_metrics.shape[0])}

    # 建立 W&B Table
    tmp_path = os.path.join(tempfile.gettempdir(), "wandb-media")
    os.makedirs(tmp_path, exist_ok=True)
    class_table = wandb.Table(columns=["class_id", "class_name", "precision", "recall", "AP50", "AP75", "F1", "IoU"])

    if per_class_metrics is not None:
        for class_id, row in enumerate(per_class_metrics):
            class_table.add_data(
                class_id,
                names.get(class_id, str(class_id)),
                float(row[0]),  # precision
                float(row[1]),  # recall
                float(row[2]),  # AP@0.5
                float(row[3]),  # AP@0.75
                float(row[4]),  # F1 score
                float(row[5]),  # IoU
            )

    # 推論速度
    speed_data = metrics.speed or {}

    # 額外指標
    extra_metrics = {}
    for k in ["ar100", "ar10", "ar1"]:
        if hasattr(metrics, k):
            extra_metrics[f"{split}/{k.upper()}"] = float(getattr(metrics, k))

    # log 到 W&B
    wandb.log({
        f"{split}/mAP50": float(map50),
        f"{split}/mAP50-95": float(map),
        f"{split}/precision": float(mp),
        f"{split}/recall": float(mr),
        f"{split}/inference_speed(ms)": float(speed_data.get("inference", 0)),
        f"{split}/preprocess_speed(ms)": float(speed_data.get("preprocess", 0)),
        f"{split}/postprocess_speed(ms)": float(speed_data.get("postprocess", 0)),
        f"{split}/loss_speed(ms)": float(speed_data.get("loss", 0)),
        f"{split}/num_classes": len(names),
        f"{split}/per_class_metrics": class_table,
        **extra_metrics
    })

    logging.info(f"✅ {split} evaluation complete. mAP50={map50:.4f}, mAP50-95={map:.4f}")

    return {
        "precision": mp,
        "recall": mr,
        "mAP50": map50,
        "mAP50-95": map,
        "per_class": per_class_metrics,
        "inference_speed": speed_data,
    }

# def log_train_metrics(trainer):
#     mi = trainer.metrics  # type: dict

#     logs = {
#         "train/box_loss": float(mi.get("box_loss", 0)),
#         "train/cls_loss": float(mi.get("cls_loss", 0)),
#         "train/dfl_loss": float(mi.get("dfl_loss", 0)),
#         "train/lr": float(trainer.optimizer.param_groups[0]["lr"]),
#         "train/epoch": trainer.epoch + 1
#     }
#     wandb.log(logs)

def log_train_metrics(trainer):

    now = time.time()
    elapsed = now - train_start_time

    # 自動 unpack training loss
    if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
        try:
            box_loss, cls_loss, dfl_loss = map(float, trainer.loss_items)
        except Exception:
            box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0
    else:
        box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0

    # 從 trainer.metrics 展平結果取得 val 結果
    metrics_dict = trainer.metrics or {}

    logs = {
        "epoch": trainer.epoch + 1,
        "time": round(elapsed, 3),
        "train/box_loss": box_loss,
        "train/cls_loss": cls_loss,
        "train/dfl_loss": dfl_loss,
        "val/box_loss": float(metrics_dict.get("val/box_loss", 0)),
        "val/cls_loss": float(metrics_dict.get("val/cls_loss", 0)),
        "val/dfl_loss": float(metrics_dict.get("val/dfl_loss", 0)),
        "metrics/precision(B)": float(metrics_dict.get("metrics/precision(B)", 0)),
        "metrics/recall(B)": float(metrics_dict.get("metrics/recall(B)", 0)),
        "metrics/mAP50(B)": float(metrics_dict.get("metrics/mAP50(B)", 0)),
        "metrics/mAP50-95(B)": float(metrics_dict.get("metrics/mAP50-95(B)", 0)),
    }

    for i, pg in enumerate(trainer.optimizer.param_groups):
        logs[f"lr/pg{i}"] = float(pg["lr"])

    wandb.log(logs, step=trainer.epoch)

# =====================================
# Main Training Function
# =====================================
def main():
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT"], name=run_name, config={
        "model": f"{model_name}.pt",
        "epochs": epochs,
        "imgsz": 640,
        "batch": batch,
        "optimizer": "AdamW",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "rect": True,
        "warmup_epochs": 3.0,
    })

    if is_segmentation:
        model = YOLO(f"./checkpoints/{model_name}.pt", task="segment")
    else:
        model = YOLO(f"./checkpoints/{model_name}.pt")  # 自動偵測 task=detection
    model.add_callback("on_train_epoch_end", log_train_metrics)

    results = model.train(
        data=f"../yolo_dataset/{case_ID}/v1/{case_ID}.yaml",
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device="cuda:0",
        val=True,
        amp=True,
        pretrained=True,
        plots=True,
        patience=100, # EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        rect=True,
        dropout=0.0,
        save=True,
        save_period=-1,
        workers=8,
        cache=False,
        project="./runs/train",
        name=run_name,
        exist_ok=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        cos_lr=False,
        close_mosaic=0,
        resume=False,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        overlap_mask=True,
    )

    evaluate(model, "val")
    evaluate(model, "test")

    best_model = YOLO(f"runs/train/{run_name}/weights/best.pt")
    logging.info("🔁 Re-evaluating using best.pt")
    evaluate(best_model, "val")
    evaluate(best_model, "test")

    export_path = model.export(format="onnx", save_dir="./exports")
    artifact = wandb.Artifact("exported_model", type="model")
    artifact.add_file(export_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    main()
