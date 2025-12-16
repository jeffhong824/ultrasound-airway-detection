import os
import argparse
import wandb
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Optional

# Load environment variables from .env file / 從 .env 檔案載入環境變數
try:
    from dotenv import load_dotenv
    # Load .env file from ultralytics directory / 從 ultralytics 目錄載入 .env 檔案
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, skip / 未安裝 python-dotenv，跳過
    pass

def log_train_metrics(trainer):
    """回调函数：在每个epoch结束时记录训练指标到W&B"""
    train_start_time = wandb.run.summary.get("train_start_time", datetime.now().timestamp())
    now = datetime.now().timestamp()
    elapsed = now - train_start_time
    
    # 自动unpack training loss
    if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
        try:
            box_loss, cls_loss, dfl_loss = map(float, trainer.loss_items)
        except Exception:
            box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0
    else:
        box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0
    
    # 从 trainer.metrics 取得 val 结果
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
    
    # 记录学习率
    for i, pg in enumerate(trainer.optimizer.param_groups):
        logs[f"lr/pg{i}"] = float(pg["lr"])
    
    wandb.log(logs, step=trainer.epoch)


def evaluate_detailed(model: YOLO, split: str = "val", batch: int = 16, imgsz: int = 640) -> Dict:
    """详细的评估函数，记录per-class指标"""
    logging.info(f"🔍 Evaluating on {split} split ...")
    
    # 进行验证
    metrics = model.val(split=split, batch=batch, imgsz=imgsz)
    
    # 取得mean results
    try:
        mp, mr, map50, map = metrics.mean_results()
    except Exception as e:
        logging.warning(f"❌ Failed to compute mean results: {e}")
        return {}
    
    # 尝试取得per-class结果
    try:
        per_class_metrics = metrics.box.mean_class_results  # shape = (num_classes, 6)
    except Exception as e:
        logging.warning(f"⚠️ Cannot extract per-class results: {e}")
        per_class_metrics = None
    
    names = model.names if hasattr(model, "names") else {i: str(i) for i in range(per_class_metrics.shape[0])}
    
    # 建立W&B Table
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
    
    # 推理速度
    speed_data = metrics.speed or {}
    
    # 额外指标
    extra_metrics = {}
    for k in ["ar100", "ar10", "ar1"]:
        if hasattr(metrics, k):
            extra_metrics[f"{split}/{k.upper()}"] = float(getattr(metrics, k))
    
    # log到W&B
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


if __name__=='__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    train_start_time = datetime.now().timestamp()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=(
        'yolo8n','yolo8s','yolo8m','yolo8l','yolo8x',
        'yolo11n','yolo11s','yolo11m','yolo11l','yolo11x',
        'yolo12n','yolo12s','yolo12m','yolo12l','yolo12x',
        'yolo8n-seg','yolo8s-seg','yolo8m-seg','yolo8l-seg','yolo8x-seg',
        'yolo11n-seg','yolo11s-seg','yolo11m-seg','yolo11l-seg','yolo11x-seg',
        'yolo12n-seg','yolo12s-seg','yolo12m-seg','yolo12l-seg','yolo12x-seg',
        'runs'
    ))
    parser.add_argument('database', type=str, choices=('det_123','seg_45','det_678'))
    parser.add_argument('--db_version', type=int, default=1, choices=(1,2,3))
    parser.add_argument('--es', action='store_true', help='Use ES (Endoscopy) dataset suffix')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs_type', type=str, default='detect', choices=('detect','segment'))
    parser.add_argument('--runs_num', type=int, default=1, help='Only used when model=runs, indicate which previous run to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--project', type=str, default=None, help='W&B project name (auto-generated if not specified)')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name identifier (e.g., exp1, baseline, etc.)')
    
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    
    # Loss function weights
    parser.add_argument('--box', type=float, default=7.5, help='Box loss gain (typical: 7.5, for small objects: 8-12)')
    parser.add_argument('--cls', type=float, default=0.5, help='Class loss gain (typical: 0.5, for class imbalance: 0.7-1.0)')
    parser.add_argument('--dfl', type=float, default=1.5, help='DFL loss gain (typical: 1.5, can increase to 1.5-2.0)')
    parser.add_argument('--pose', type=float, default=12.0, help='Pose loss gain')
    parser.add_argument('--kobj', type=float, default=2.0, help='Keypoint obj loss gain')
    
    # Classification loss type
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of BCE Loss (better for small objects)')
    parser.add_argument('--focal_gamma', type=float, default=1.5, help='Focal Loss gamma parameter (default: 1.5, range: 1.0-2.5)')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal Loss alpha parameter (default: 0.25)')
    
    # Dimension weights for bbox loss (custom feature)
    parser.add_argument('--use_dim_weights', action='store_true', help='Enable dimension-specific weights for bbox loss')
    parser.add_argument('--dim_weights', type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0], 
                       metavar=('W_L', 'W_T', 'W_R', 'W_B'),
                       help='Weights for [left, top, right, bottom] dimensions. Example: --dim_weights 2.0 1.0 2.0 1.0')
    
    # Data augmentation
    parser.add_argument('--hsv_h', type=float, default=0, help='Image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='Image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='Image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0, help='Image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='Image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='Image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='Image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Image perspective (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0, help='Image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0, help='Image mixup (probability)')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='Copy-paste augmentation (probability)')
    
    # Training config
    parser.add_argument('--close_mosaic', type=int, default=0, help='Disable mosaic augmentation for final epochs')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='Warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='Warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='Warmup initial bias lr')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--rect', action='store_true', help='Rectangular training')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--cache', type=str, default=None, help='Cache images (ram/disk)')
    parser.add_argument('--no_amp', action='store_true', help='Disable Automatic Mixed Precision')
    parser.add_argument('--cos_lr', action='store_true', help='Use cosine LR scheduler')
    parser.add_argument('--deterministic', type=bool, default=True, help='Deterministic training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Detection/NMS parameters
    parser.add_argument('--conf', type=float, default=None, help='Object confidence threshold for training')
    parser.add_argument('--iou', type=float, default=None, help='IoU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=None, help='Maximum number of detections per image')
    parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference')
    parser.add_argument('--agnostic_nms', action='store_true', help='Class-agnostic NMS (merge overlapping boxes regardless of class)')
    
    args = parser.parse_args()
    if args.runs_num==1:
        args.runs_num = ''

    # Get project root from environment variable / 從環境變數獲取專案根目錄
    DA_folder = os.getenv('PROJECT_ROOT')
    if not DA_folder:
        # Fallback: try to detect from script location / 備選：嘗試從腳本位置偵測
        script_dir = Path(__file__).resolve().parent.parent.parent
        DA_folder = str(script_dir)
    
    # Ensure path uses forward slashes for cross-platform compatibility / 確保路徑使用正斜線以跨平台兼容
    DA_folder = str(Path(DA_folder).resolve())
    assert os.path.isdir(DA_folder), f'DA_folder not exist: {DA_folder}. Please set PROJECT_ROOT in .env file.'
    if args.model!='runs':
        # Load a COCO-pretrained model
        mdl_file = os.path.join(DA_folder, 'ultralytics', 'weights', f'{args.model}.pt')
    else:
        # Load previous pretrained model
        mdl_file = os.path.join(DA_folder, 'ultralytics', 'runs', args.runs_type, f'train{args.runs_num}', 'weights', 'last.pt')

    assert os.path.isfile(mdl_file), f'Pretrained model not found: {mdl_file}'
    model = YOLO(mdl_file)
    
    # Store dimension weights and focal loss settings for later use
    use_dim_weights_flag = args.use_dim_weights
    dim_weights_value = args.dim_weights if args.use_dim_weights else None
    use_focal_loss_flag = args.use_focal_loss
    focal_gamma_value = args.focal_gamma
    focal_alpha_value = args.focal_alpha
    
    if use_dim_weights_flag:
        logging.info(f"✅ Dimension weights will be enabled: {dim_weights_value} [left, top, right, bottom]")
    
    if use_focal_loss_flag:
        logging.info(f"✅ Focal Loss will be enabled: gamma={focal_gamma_value}, alpha={focal_alpha_value}")
    
    # Callback to set dimension weights and focal loss after trainer is created
    def set_custom_loss_callback(trainer):
        """Set dimension weights and focal loss after trainer initialization and recreate loss function"""
        updated = False
        
        # Set dimension weights
        if use_dim_weights_flag and dim_weights_value:
            if isinstance(trainer.args, dict):
                trainer.args['use_dim_weights'] = True
                trainer.args['dim_weights'] = dim_weights_value
            else:
                setattr(trainer.args, 'use_dim_weights', True)
                setattr(trainer.args, 'dim_weights', dim_weights_value)
            updated = True
        
        # Set focal loss settings
        if use_focal_loss_flag:
            if isinstance(trainer.args, dict):
                trainer.args['use_focal_loss'] = True
                trainer.args['focal_gamma'] = focal_gamma_value
                trainer.args['focal_alpha'] = focal_alpha_value
            else:
                setattr(trainer.args, 'use_focal_loss', True)
                setattr(trainer.args, 'focal_gamma', focal_gamma_value)
                setattr(trainer.args, 'focal_alpha', focal_alpha_value)
            updated = True
        
        # Recreate loss function with custom settings
        if updated and hasattr(trainer.model, 'init_criterion'):
            trainer.model.criterion = None  # Clear existing criterion
            trainer.model.criterion = trainer.model.init_criterion()
            info_msg = []
            if use_dim_weights_flag:
                info_msg.append(f"dimension weights: {dim_weights_value}")
            if use_focal_loss_flag:
                info_msg.append(f"focal loss (gamma={focal_gamma_value}, alpha={focal_alpha_value})")
            logging.info(f"✅ Loss function recreated with {', '.join(info_msg)}")
        elif updated:
            logging.warning("⚠️ Cannot recreate loss function, custom settings may not be applied")
    
    # Add callback to set custom loss settings after trainer initialization
    if use_dim_weights_flag or use_focal_loss_flag:
        model.add_callback("on_train_start", set_custom_loss_callback)

    # Setup W&B if enabled
    if args.wandb:
        wandb.login()
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 在run名称中包含实验标识
        exp_suffix = f"-{args.exp_name}" if args.exp_name else ""
        run_name = f"{args.model}-{args.database}{exp_suffix}-{now}"
        project = args.project if args.project else f"ultrasound-{args.database}"
        wandb.init(
            project=project,
            name=run_name,
            config={
                # Experiment info
                "exp_name": args.exp_name,
                "model": args.model,
                "database": args.database,
                "db_version": args.db_version,
                "es": args.es,
                
                # Training hyperparameters
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "device": args.device,
                "patience": args.patience,
                "seed": args.seed,
                "deterministic": args.deterministic,
                
                # Optimizer parameters
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "lrf": args.lrf,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                
                # Loss weights
                "box": args.box,
                "cls": args.cls,
                "dfl": args.dfl,
                
                # Classification loss type
                "use_focal_loss": args.use_focal_loss,
                "focal_gamma": args.focal_gamma if args.use_focal_loss else None,
                "focal_alpha": args.focal_alpha if args.use_focal_loss else None,
                
                # Dimension weights (custom)
                "use_dim_weights": args.use_dim_weights,
                "dim_weights": args.dim_weights,
                
                # Data augmentation - HSV
                "hsv_h": args.hsv_h,
                "hsv_s": args.hsv_s,
                "hsv_v": args.hsv_v,
                
                # Data augmentation - Geometric
                "degrees": args.degrees,
                "translate": args.translate,
                "scale": args.scale,
                "shear": args.shear,
                "perspective": args.perspective,
                
                # Data augmentation - Flip
                "flipud": args.flipud,
                "fliplr": args.fliplr,
                
                # Data augmentation - Advanced
                "mosaic": args.mosaic,
                "mixup": args.mixup,
                "copy_paste": args.copy_paste,
                "close_mosaic": args.close_mosaic,
                
                # Detection/NMS parameters
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "agnostic_nms": args.agnostic_nms,
                
                # Advanced training
                "rect": args.rect,
                "dropout": args.dropout,
                "cos_lr": args.cos_lr,
                "warmup_epochs": args.warmup_epochs,
                "warmup_momentum": args.warmup_momentum,
                "warmup_bias_lr": args.warmup_bias_lr,
                "amp": not args.no_amp,
                "half": args.half,
                "workers": args.workers,
                "cache": args.cache,
            }
        )
        # 记录训练开始时间
        wandb.run.summary["train_start_time"] = train_start_time
        
        # 添加训练回调函数（记录详细的训练指标）- 只在W&B启用时添加
        model.add_callback("on_train_epoch_end", log_train_metrics)
    else:
        # 不使用W&B时，也可以添加回调但跳过记录
        logging.info("W&B is disabled. Training will proceed without logging.")

    # Train the model - handle ES suffix
    suffix = '_ES' if args.es else ''
    yaml_file = os.path.join(DA_folder, 'yolo_dataset', args.database, f'v{args.db_version}', f'{args.database}{suffix}.yaml')
    assert os.path.isfile(yaml_file), f'DB YAML file not found: {yaml_file}'
    
    model.train(
        data=yaml_file, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        batch=args.batch,
        device=args.device,
        val=True,
        plots=True,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        rect=args.rect,
        dropout=args.dropout,
        save=True,
        save_period=-1,
        workers=args.workers,
        cache=args.cache,
        project="./runs/train",
        name=f"{args.model}-{args.database}-v{args.db_version}" + (f"-{args.exp_name}" if args.exp_name else ""),
        exist_ok=True,
        verbose=True,
        seed=args.seed,
        deterministic=args.deterministic,
        single_cls=False,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        amp=not args.no_amp,
        # Data augmentation parameters
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        # Detection parameters
        conf=args.conf if args.conf is not None else 0.25,  # Default confidence threshold
        iou=args.iou if args.iou is not None else 0.45,    # Default IoU threshold for NMS
        max_det=args.max_det,
        half=args.half,
        agnostic_nms=args.agnostic_nms,
    )
    
    if args.wandb:
        # 使用最佳模型进行详细评估
        run_name = f"{args.model}-{args.database}-v{args.db_version}" + (f"-{args.exp_name}" if args.exp_name else "")
        best_model = YOLO(f"runs/train/{run_name}/weights/best.pt")
        logging.info("🔁 Re-evaluating using best.pt")
        
        # 评估val和test
        val_results = evaluate_detailed(best_model, "val", batch=args.batch, imgsz=args.imgsz)
        test_results = evaluate_detailed(best_model, "test", batch=args.batch, imgsz=args.imgsz)
        
        # 计算并记录 fitness (val 和 test)
        if val_results:
            map50_val = val_results.get("mAP50", 0)
            map_val = val_results.get("mAP50-95", 0)
            fitness_val = map50_val * 0.1 + map_val * 0.9  # 与 best_epoch.py 使用相同公式
            
            wandb.log({
                "fitness/val": fitness_val,
            })
            wandb.run.summary["fitness_val"] = fitness_val
            wandb.run.summary["val/mAP50"] = map50_val
            wandb.run.summary["val/mAP50-95"] = map_val
            logging.info(f"✅ Val Fitness calculated: {fitness_val:.6f}")
        
        if test_results:
            map50_test = test_results.get("mAP50", 0)
            map_test = test_results.get("mAP50-95", 0)
            fitness_test = map50_test * 0.1 + map_test * 0.9
            
            wandb.log({
                "fitness/test": fitness_test,
            })
            wandb.run.summary["fitness_test"] = fitness_test
            wandb.run.summary["test/mAP50"] = map50_test
            wandb.run.summary["test/mAP50-95"] = map_test
            logging.info(f"✅ Test Fitness calculated: {fitness_test:.6f}")
        
        # 导出ONNX模型并上传到W&B
        try:
            export_path = best_model.export(format="onnx", save_dir="./exports")
            artifact = wandb.Artifact("exported_model", type="model")
            artifact.add_file(export_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            logging.warning(f"⚠️ Failed to export model: {e}")
        
        wandb.finish()

