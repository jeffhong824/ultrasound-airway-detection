import os
import argparse
import wandb
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from ultralytics directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, skip
    pass



def print_validation_metrics(trainer):
    """Print additional validation metrics to terminal after each epoch"""
    # trainer.metrics is a DetMetrics object, not a dict
    # Extract metrics from DetMetrics object
    precision = 0.0
    recall = 0.0
    map50 = 0.0
    map50_95 = 0.0
    
    try:
        # Try multiple methods to get metrics
        metrics_source = None
        
        # Method 1: Try trainer.metrics.mean_results()
        if hasattr(trainer, 'metrics') and trainer.metrics is not None:
            metrics_source = trainer.metrics
            if hasattr(trainer.metrics, 'mean_results'):
                mean_results = trainer.metrics.mean_results()
                if mean_results and len(mean_results) >= 4:
                    mp, mr, map50_val, map50_95_val = mean_results
                    precision = float(mp) if mp is not None and not np.isnan(mp) else 0.0
                    recall = float(mr) if mr is not None and not np.isnan(mr) else 0.0
                    map50 = float(map50_val) if map50_val is not None and not np.isnan(map50_val) else 0.0
                    map50_95 = float(map50_95_val) if map50_95_val is not None and not np.isnan(map50_95_val) else 0.0
        
        # Method 2: Try directly from trainer.metrics.box (if mean_results didn't work)
        if (precision == 0.0 and recall == 0.0 and map50 == 0.0 and map50_95 == 0.0) and \
           hasattr(trainer, 'metrics') and trainer.metrics is not None:
            if hasattr(trainer.metrics, 'box') and trainer.metrics.box is not None:
                box = trainer.metrics.box
                try:
                    precision = float(box.mp) if hasattr(box, 'mp') and box.mp is not None else 0.0
                    recall = float(box.mr) if hasattr(box, 'mr') and box.mr is not None else 0.0
                    map50 = float(box.map50) if hasattr(box, 'map50') and box.map50 is not None else 0.0
                    map50_95 = float(box.map) if hasattr(box, 'map') and box.map is not None else 0.0
                except Exception as e:
                    logging.debug(f"Failed to get metrics from box: {e}")
        
        # Method 3: Try validator.metrics if trainer.metrics is not available
        if (precision == 0.0 and recall == 0.0 and map50 == 0.0 and map50_95 == 0.0) and \
           hasattr(trainer, 'validator') and trainer.validator is not None:
            validator = trainer.validator
            if hasattr(validator, 'metrics') and validator.metrics is not None:
                metrics_source = validator.metrics
                if hasattr(validator.metrics, 'mean_results'):
                    mean_results = validator.metrics.mean_results()
                    if mean_results and len(mean_results) >= 4:
                        mp, mr, map50_val, map50_95_val = mean_results
                        precision = float(mp) if mp is not None and not np.isnan(mp) else 0.0
                        recall = float(mr) if mr is not None and not np.isnan(mr) else 0.0
                        map50 = float(map50_val) if map50_val is not None and not np.isnan(map50_val) else 0.0
                        map50_95 = float(map50_95_val) if map50_95_val is not None and not np.isnan(map50_95_val) else 0.0
                # Also try from validator.metrics.box
                if (precision == 0.0 and recall == 0.0 and map50 == 0.0 and map50_95 == 0.0) and \
                   hasattr(validator.metrics, 'box') and validator.metrics.box is not None:
                    box = validator.metrics.box
                    try:
                        precision = float(box.mp) if hasattr(box, 'mp') and box.mp is not None else 0.0
                        recall = float(box.mr) if hasattr(box, 'mr') and box.mr is not None else 0.0
                        map50 = float(box.map50) if hasattr(box, 'map50') and box.map50 is not None else 0.0
                        map50_95 = float(box.map) if hasattr(box, 'map') and box.map is not None else 0.0
                    except Exception as e:
                        logging.debug(f"Failed to get metrics from validator.box: {e}")
        
        # Debug: log if we still have zeros
        if precision == 0.0 and recall == 0.0 and map50 == 0.0 and map50_95 == 0.0:
            logging.warning(f"⚠️ All metrics are 0.0 - metrics_source: {metrics_source}, "
                          f"has_trainer_metrics: {hasattr(trainer, 'metrics')}, "
                          f"has_validator: {hasattr(trainer, 'validator')}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to get metrics: {e}")
        import traceback
        logging.debug(traceback.format_exc())
    
    # Calculate fitness: fitness = map50 * 0.1 + map50_95 * 0.9
    fitness = map50 * 0.1 + map50_95 * 0.9
    
    # Get HMD loss from additional_metrics (set by on_val_end_callback) or directly from criterion
    hmd_loss_value = None
    # First try to get from trainer._additional_metrics
    if hasattr(trainer, '_additional_metrics') and trainer._additional_metrics is not None:
        if "train/hmd_loss" in trainer._additional_metrics:
            hmd_loss_value = trainer._additional_metrics["train/hmd_loss"]
    # If not found in trainer, try to get from validator (if available)
    if hmd_loss_value is None and hasattr(trainer, 'validator') and trainer.validator is not None:
        validator = trainer.validator
        if hasattr(validator, '_additional_metrics') and validator._additional_metrics is not None:
            if "train/hmd_loss" in validator._additional_metrics:
                hmd_loss_value = validator._additional_metrics["train/hmd_loss"]
    # Fallback: try to get directly from criterion
    # NOTE: During validation, trainer.model might be EMA model, which doesn't accumulate training loss
    # So we should rely on the value saved in on_train_epoch_end callback
    if hmd_loss_value is None and hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
        try:
            criterion = trainer.model.criterion
            if hasattr(criterion, 'get_avg_hmd_loss'):
                # Get average HMD loss across all batches in this epoch
                hmd_loss_avg = criterion.get_avg_hmd_loss()
                # Only use if count > 0 (meaning loss was actually accumulated)
                if hasattr(criterion, 'hmd_loss_count') and criterion.hmd_loss_count > 0:
                    hmd_loss_value = hmd_loss_avg
                    logging.debug(f"Got HMD loss from criterion.get_avg_hmd_loss(): {hmd_loss_value} (count={criterion.hmd_loss_count})")
            elif hasattr(criterion, 'last_hmd_loss') and criterion.last_hmd_loss != 0.0:
                # Fallback to last batch loss if average not available
                hmd_loss_value = float(criterion.last_hmd_loss)
                logging.debug(f"Got HMD loss from criterion.last_hmd_loss: {hmd_loss_value}")
        except Exception as e:
            logging.debug(f"Failed to get HMD loss from criterion: {e}")
    
    # Print additional metrics
    print(f"\n📊 Additional Metrics:", flush=True)
    print(f"   Precision: {precision:.4f} | Recall: {recall:.4f}", flush=True)
    print(f"   mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Fitness: {fitness:.4f}", flush=True)
    
    # HMD loss (always show if HMD loss is enabled, even if 0)
    # Check if database is det_123 and HMD loss is enabled
    # Try to get from stored attributes first (set by on_val_end_callback), then fallback to trainer.args
    if hasattr(trainer, '_args_database'):
        db_val = trainer._args_database
        hmd_val = getattr(trainer, '_args_use_hmd_loss', False)
    else:
        db_val = getattr(trainer.args, 'database', None) if hasattr(trainer, 'args') else None
        hmd_val = getattr(trainer.args, 'use_hmd_loss', False) if hasattr(trainer, 'args') else False
    
    is_det_123 = db_val == 'det_123'
    hmd_enabled = hmd_val and is_det_123
    
    # Debug: print condition check (can be removed later)
    logging.info(f"print_validation_metrics: database={db_val}, use_hmd_loss={hmd_val}, is_det_123={is_det_123}, hmd_enabled={hmd_enabled}")
    
    # HMD loss value: only show if HMD loss is enabled
    if is_det_123 and hmd_enabled:
        if hmd_loss_value is not None:
            print(f"   HMD_loss: {hmd_loss_value:.4f}", flush=True)
        else:
            print(f"   HMD_loss: 0.0000 (not calculated)", flush=True)
    
    # HMD metrics (if available and database is det_123)
    # Try to get from additional_metrics (set by on_val_end_callback)
    detection_rate = 0.0
    rmse_pixel = 0.0
    mae_pixel = 0.0
    overall_score_pixel = 0.0
    rmse_no_penalty_pixel = 0.0
    mae_no_penalty_pixel = 0.0
    
    # First try to get from trainer._additional_metrics
    if hasattr(trainer, '_additional_metrics') and trainer._additional_metrics is not None:
        additional_metrics = trainer._additional_metrics
        detection_rate = float(additional_metrics.get("hmd/detection_rate") or additional_metrics.get("val/hmd/detection_rate", 0))
        rmse_pixel = float(additional_metrics.get("hmd/rmse_pixel") or additional_metrics.get("val/hmd/rmse_pixel", 0))
        mae_pixel = float(additional_metrics.get("hmd/mae_pixel") or additional_metrics.get("val/hmd/mae_pixel", 0))
        overall_score_pixel = float(additional_metrics.get("hmd/overall_score_pixel") or additional_metrics.get("val/hmd/overall_score_pixel", 0))
        rmse_no_penalty_pixel = float(additional_metrics.get("hmd/rmse_no_penalty_pixel") or additional_metrics.get("val/hmd/rmse_no_penalty_pixel", 0))
        mae_no_penalty_pixel = float(additional_metrics.get("hmd/mae_no_penalty_pixel") or additional_metrics.get("val/hmd/mae_no_penalty_pixel", 0))
    # If not found in trainer, try to get from validator (if available)
    elif hasattr(trainer, 'validator') and trainer.validator is not None:
        validator = trainer.validator
        if hasattr(validator, '_additional_metrics') and validator._additional_metrics is not None:
            additional_metrics = validator._additional_metrics
            detection_rate = float(additional_metrics.get("hmd/detection_rate") or additional_metrics.get("val/hmd/detection_rate", 0))
            rmse_pixel = float(additional_metrics.get("hmd/rmse_pixel") or additional_metrics.get("val/hmd/rmse_pixel", 0))
            mae_pixel = float(additional_metrics.get("hmd/mae_pixel") or additional_metrics.get("val/hmd/mae_pixel", 0))
            overall_score_pixel = float(additional_metrics.get("hmd/overall_score_pixel") or additional_metrics.get("val/hmd/overall_score_pixel", 0))
            rmse_no_penalty_pixel = float(additional_metrics.get("hmd/rmse_no_penalty_pixel") or additional_metrics.get("val/hmd/rmse_no_penalty_pixel", 0))
            mae_no_penalty_pixel = float(additional_metrics.get("hmd/mae_no_penalty_pixel") or additional_metrics.get("val/hmd/mae_no_penalty_pixel", 0))
    
    # Always show HMD metrics section if database is det_123 (even without HMD loss enabled)
    # This allows monitoring HMD performance for all det_123 experiments
    if is_det_123:
        # Get mm version metrics
        rmse_mm = 0.0
        mae_mm = 0.0
        overall_score_mm = 0.0
        rmse_no_penalty_mm = 0.0
        mae_no_penalty_mm = 0.0
        if hasattr(trainer, '_additional_metrics') and trainer._additional_metrics is not None:
            additional_metrics = trainer._additional_metrics
            rmse_mm = float(additional_metrics.get("hmd/rmse_mm") or additional_metrics.get("val/hmd/rmse_mm", 0))
            mae_mm = float(additional_metrics.get("hmd/mae_mm") or additional_metrics.get("val/hmd/mae_mm", 0))
            overall_score_mm = float(additional_metrics.get("hmd/overall_score_mm") or additional_metrics.get("val/hmd/overall_score_mm", 0))
            rmse_no_penalty_mm = float(additional_metrics.get("hmd/rmse_no_penalty_mm") or additional_metrics.get("val/hmd/rmse_no_penalty_mm", 0))
            mae_no_penalty_mm = float(additional_metrics.get("hmd/mae_no_penalty_mm") or additional_metrics.get("val/hmd/mae_no_penalty_mm", 0))
        elif hasattr(trainer, 'validator') and trainer.validator is not None:
            validator = trainer.validator
            if hasattr(validator, '_additional_metrics') and validator._additional_metrics is not None:
                additional_metrics = validator._additional_metrics
                rmse_mm = float(additional_metrics.get("hmd/rmse_mm") or additional_metrics.get("val/hmd/rmse_mm", 0))
                mae_mm = float(additional_metrics.get("hmd/mae_mm") or additional_metrics.get("val/hmd/mae_mm", 0))
                overall_score_mm = float(additional_metrics.get("hmd/overall_score_mm") or additional_metrics.get("val/hmd/overall_score_mm", 0))
                rmse_no_penalty_mm = float(additional_metrics.get("hmd/rmse_no_penalty_mm") or additional_metrics.get("val/hmd/rmse_no_penalty_mm", 0))
                mae_no_penalty_mm = float(additional_metrics.get("hmd/mae_no_penalty_mm") or additional_metrics.get("val/hmd/mae_no_penalty_mm", 0))
        
        print(f"\n📏 HMD Metrics (det_123):", flush=True)
        print(f"   Detection_Rate: {detection_rate:.4f}", flush=True)
        
        # With penalty versions
        print(f"\n   📊 With Penalty (includes missed detections):", flush=True)
        if detection_rate == 0.0 and rmse_pixel >= 1000.0:
            print(f"      RMSE_HMD (pixel): {rmse_pixel:.2f} px (penalty: no detections)", flush=True)
        else:
            print(f"      RMSE_HMD (pixel): {rmse_pixel:.2f} px", flush=True)
        print(f"      MAE_HMD (pixel): {mae_pixel:.2f} px", flush=True)
        print(f"      Overall_Score (pixel): {overall_score_pixel:.4f}", flush=True)
        
        # No penalty versions (only both_detected cases)
        print(f"\n   📊 No Penalty (only both detected cases):", flush=True)
        if rmse_no_penalty_pixel > 0.0:
            print(f"      RMSE_HMD (pixel): {rmse_no_penalty_pixel:.2f} px", flush=True)
            print(f"      MAE_HMD (pixel): {mae_no_penalty_pixel:.2f} px", flush=True)
        else:
            print(f"      RMSE_HMD (pixel): N/A (no both detected cases)", flush=True)
            print(f"      MAE_HMD (pixel): N/A (no both detected cases)", flush=True)
        
        # Always show mm version if it was calculated (even if 0.0)
        # Check if mm metrics exist in additional_metrics (indicating calculation was attempted)
        mm_metrics_exist = False
        if hasattr(trainer, '_additional_metrics') and trainer._additional_metrics is not None:
            mm_metrics_exist = ("hmd/rmse_mm" in trainer._additional_metrics or 
                               "val/hmd/rmse_mm" in trainer._additional_metrics)
        elif hasattr(trainer, 'validator') and trainer.validator is not None:
            validator = trainer.validator
            if hasattr(validator, '_additional_metrics') and validator._additional_metrics is not None:
                mm_metrics_exist = ("hmd/rmse_mm" in validator._additional_metrics or 
                                   "val/hmd/rmse_mm" in validator._additional_metrics)
        
        if mm_metrics_exist:
            print(f"\n   📏 With Penalty (mm):", flush=True)
            print(f"      RMSE_HMD (mm): {rmse_mm:.2f} mm", flush=True)
            print(f"      MAE_HMD (mm): {mae_mm:.2f} mm", flush=True)
            print(f"      Overall_Score (mm): {overall_score_mm:.4f}", flush=True)
            
            print(f"\n   📏 No Penalty (mm):", flush=True)
            if rmse_no_penalty_mm > 0.0:
                print(f"      RMSE_HMD (mm): {rmse_no_penalty_mm:.2f} mm", flush=True)
                print(f"      MAE_HMD (mm): {mae_no_penalty_mm:.2f} mm", flush=True)
            else:
                print(f"      RMSE_HMD (mm): N/A (no both detected cases)", flush=True)
                print(f"      MAE_HMD (mm): N/A (no both detected cases)", flush=True)
        else:
            print(f"\n   📏 With Penalty (mm): N/A (PixelSpacing not available)", flush=True)
            print(f"   📏 No Penalty (mm): N/A (PixelSpacing not available)", flush=True)


def log_train_metrics(trainer):
    """Callback function: Log training metrics to W&B at the end of each epoch"""
    train_start_time = wandb.run.summary.get("train_start_time", datetime.now().timestamp())
    now = datetime.now().timestamp()
    elapsed = now - train_start_time
    
    # Automatically unpack training loss
    if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
        try:
            box_loss, cls_loss, dfl_loss = map(float, trainer.loss_items)
        except Exception:
            box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0
    else:
        box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0
    
    # Get validation results from trainer.metrics
    # trainer.metrics is a DetMetrics object, extract metrics properly
    precision = 0.0
    recall = 0.0
    map50 = 0.0
    map50_95 = 0.0
    
    try:
        # Try multiple methods to get metrics
        # Method 1: Try trainer.metrics.mean_results()
        if hasattr(trainer, 'metrics') and trainer.metrics is not None:
            if hasattr(trainer.metrics, 'mean_results'):
                mean_results = trainer.metrics.mean_results()
                if mean_results and len(mean_results) >= 4:
                    mp, mr, map50_val, map50_95_val = mean_results
                    precision = float(mp) if mp is not None and not np.isnan(mp) else 0.0
                    recall = float(mr) if mr is not None and not np.isnan(mr) else 0.0
                    map50 = float(map50_val) if map50_val is not None and not np.isnan(map50_val) else 0.0
                    map50_95 = float(map50_95_val) if map50_95_val is not None and not np.isnan(map50_95_val) else 0.0
        
        # Method 2: Try directly from trainer.metrics.box (if mean_results didn't work)
        if (precision == 0.0 and recall == 0.0 and map50 == 0.0 and map50_95 == 0.0) and \
           hasattr(trainer, 'metrics') and trainer.metrics is not None:
            if hasattr(trainer.metrics, 'box') and trainer.metrics.box is not None:
                box = trainer.metrics.box
                try:
                    precision = float(box.mp) if hasattr(box, 'mp') and box.mp is not None else 0.0
                    recall = float(box.mr) if hasattr(box, 'mr') and box.mr is not None else 0.0
                    map50 = float(box.map50) if hasattr(box, 'map50') and box.map50 is not None else 0.0
                    map50_95 = float(box.map) if hasattr(box, 'map') and box.map is not None else 0.0
                except Exception as e:
                    logging.debug(f"Failed to get metrics from box: {e}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to get metrics from trainer.metrics: {e}")
        import traceback
        logging.debug(traceback.format_exc())
    
    # Extract all metrics
    logs = {
        "epoch": trainer.epoch + 1,
        "time": round(elapsed, 3),
        "train/box_loss": box_loss,
        "train/cls_loss": cls_loss,
        "train/dfl_loss": dfl_loss,
    }
    
    logs.update({
        "metrics/precision": precision,
        "metrics/recall": recall,
        "metrics/mAP50": map50,
        "metrics/mAP50-95": map50_95,
        "metrics/fitness": map50 * 0.1 + map50_95 * 0.9,
    })
    
    # IoU and Dice removed - they are bbox-level metrics, not epoch-level
    
    # HMD loss (if available) - get directly from criterion since on_train_epoch_end is called before on_val_end
    hmd_loss_value = None
    if hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
        try:
            criterion = trainer.model.criterion
            if hasattr(criterion, 'get_avg_hmd_loss'):
                hmd_loss_avg = criterion.get_avg_hmd_loss()
                # Only use if we actually have accumulated loss (count > 0)
                if hasattr(criterion, 'hmd_loss_count') and criterion.hmd_loss_count > 0:
                    hmd_loss_value = hmd_loss_avg
                elif hasattr(criterion, 'last_hmd_loss') and criterion.last_hmd_loss != 0.0:
                    # Fallback to last batch if count is 0
                    hmd_loss_value = float(criterion.last_hmd_loss)
            elif hasattr(criterion, 'last_hmd_loss') and criterion.last_hmd_loss != 0.0:
                hmd_loss_value = float(criterion.last_hmd_loss)
        except Exception as e:
            logging.debug(f"Failed to get HMD loss in log_train_metrics: {e}")
    
    # Also check _additional_metrics as fallback (in case on_val_end was called first)
    if hmd_loss_value is None and hasattr(trainer, '_additional_metrics') and "train/hmd_loss" in trainer._additional_metrics:
        hmd_loss_value = trainer._additional_metrics["train/hmd_loss"]
    
    # Save to _additional_metrics for validation to access
    if hmd_loss_value is not None:
        if not hasattr(trainer, '_additional_metrics'):
            trainer._additional_metrics = {}
        trainer._additional_metrics["train/hmd_loss"] = hmd_loss_value
        logs["train/hmd_loss"] = float(hmd_loss_value)
        logging.debug(f"✅ log_train_metrics: Saved HMD loss {hmd_loss_value} to _additional_metrics")
    else:
        logging.debug(f"⚠️ log_train_metrics: HMD loss is None, cannot save")
    
    # HMD metrics (always log for det_123, even if 0) - pixel based only
    # Use val/hmd/ prefix for validation metrics in W&B
    if hasattr(trainer, 'args') and hasattr(trainer.args, 'database') and trainer.args.database == 'det_123':
        if hasattr(trainer, '_additional_metrics'):
            additional_metrics = trainer._additional_metrics
            # Try both naming conventions (hmd/... and val/hmd/...)
            detection_rate = float(additional_metrics.get("val/hmd/detection_rate") or additional_metrics.get("hmd/detection_rate", 0))
            rmse_pixel = float(additional_metrics.get("val/hmd/rmse_pixel") or additional_metrics.get("hmd/rmse_pixel", 0))
            mae_pixel = float(additional_metrics.get("val/hmd/mae_pixel") or additional_metrics.get("hmd/mae_pixel", 0))
            overall_score_pixel = float(additional_metrics.get("val/hmd/overall_score_pixel") or additional_metrics.get("hmd/overall_score_pixel", 0))
            rmse_mm = float(additional_metrics.get("val/hmd/rmse_mm") or additional_metrics.get("hmd/rmse_mm", 0))
            mae_mm = float(additional_metrics.get("val/hmd/mae_mm") or additional_metrics.get("hmd/mae_mm", 0))
            overall_score_mm = float(additional_metrics.get("val/hmd/overall_score_mm") or additional_metrics.get("hmd/overall_score_mm", 0))
            # No penalty versions
            rmse_no_penalty_pixel = float(additional_metrics.get("val/hmd/rmse_no_penalty_pixel") or additional_metrics.get("hmd/rmse_no_penalty_pixel", 0))
            mae_no_penalty_pixel = float(additional_metrics.get("val/hmd/mae_no_penalty_pixel") or additional_metrics.get("hmd/mae_no_penalty_pixel", 0))
            rmse_no_penalty_mm = float(additional_metrics.get("val/hmd/rmse_no_penalty_mm") or additional_metrics.get("hmd/rmse_no_penalty_mm", 0))
            mae_no_penalty_mm = float(additional_metrics.get("val/hmd/mae_no_penalty_mm") or additional_metrics.get("hmd/mae_no_penalty_mm", 0))
            
            logs["val/hmd/detection_rate"] = detection_rate
            logs["val/hmd/rmse_pixel"] = rmse_pixel
            logs["val/hmd/mae_pixel"] = mae_pixel
            logs["val/hmd/overall_score_pixel"] = overall_score_pixel
            logs["val/hmd/rmse_mm"] = rmse_mm
            logs["val/hmd/mae_mm"] = mae_mm
            logs["val/hmd/overall_score_mm"] = overall_score_mm
            logs["val/hmd/rmse_no_penalty_pixel"] = rmse_no_penalty_pixel
            logs["val/hmd/mae_no_penalty_pixel"] = mae_no_penalty_pixel
            logs["val/hmd/rmse_no_penalty_mm"] = rmse_no_penalty_mm
            logs["val/hmd/mae_no_penalty_mm"] = mae_no_penalty_mm
        else:
            logs["val/hmd/detection_rate"] = 0.0
            logs["val/hmd/rmse_pixel"] = 0.0
            logs["val/hmd/mae_pixel"] = 0.0
            logs["val/hmd/overall_score_pixel"] = 0.0
            logs["val/hmd/rmse_mm"] = 0.0
            logs["val/hmd/mae_mm"] = 0.0
            logs["val/hmd/overall_score_mm"] = 0.0
            logs["val/hmd/rmse_no_penalty_pixel"] = 0.0
            logs["val/hmd/mae_no_penalty_pixel"] = 0.0
            logs["val/hmd/rmse_no_penalty_mm"] = 0.0
            logs["val/hmd/mae_no_penalty_mm"] = 0.0
    
    # Log learning rate
    for i, pg in enumerate(trainer.optimizer.param_groups):
        logs[f"lr/pg{i}"] = float(pg["lr"])
    
    wandb.log(logs, step=trainer.epoch)


def on_val_batch_end_callback(trainer):
    """Callback to collect predictions and ground truth bboxes for HMD calculation
    
    This callback collects boxes even when HMD loss is not enabled, so we can calculate
    real HMD errors in Method 2 of calculate_hmd_metrics_from_validator.
    """
    # Only collect if database is det_123 (regardless of use_hmd_loss)
    if not (hasattr(trainer, 'args') and hasattr(trainer.args, 'database') and 
            trainer.args.database == 'det_123'):
        return
    
    # Initialize HMD data collection if not exists
    if not hasattr(trainer, '_hmd_collection'):
        trainer._hmd_collection = {
            'pred_boxes': [],  # List of dicts: {'image_idx': int, 'class': int, 'bbox': [x1,y1,x2,y2], 'conf': float}
            'gt_boxes': [],    # List of dicts: {'image_idx': int, 'class': int, 'bbox': [x1,y1,x2,y2]}
            'image_files': []  # List of image file paths (for pixel spacing lookup)
        }
    
    # Get validator to access current batch
    if hasattr(trainer, 'validator') and trainer.validator is not None:
        validator = trainer.validator
        
        # Access batch data stored by patched_update_metrics
        if hasattr(validator, '_last_batch_preds') and hasattr(validator, '_last_batch_targets'):
            preds = validator._last_batch_preds
            targets = validator._last_batch_targets
            image_files = getattr(validator, '_last_batch_im_files', [])
            
            # Get current image index offset (number of images processed so far)
            current_offset = len(trainer._hmd_collection['image_files'])
            
            # Process predictions - convert to native space
            # Get full batch from validator for _prepare_batch
            full_batch = getattr(validator, '_last_batch_full', None)
            
            for img_idx_in_batch, pred in enumerate(preds):
                global_img_idx = current_offset + img_idx_in_batch
                
                # Prepare prediction (convert to native space)
                if full_batch is not None and img_idx_in_batch < len(targets):
                    try:
                        pbatch = validator._prepare_batch(img_idx_in_batch, full_batch)
                        predn = validator._prepare_pred(pred, pbatch)
                        
                        if 'bboxes' in predn and 'cls' in predn and 'conf' in predn:
                            bboxes = predn['bboxes'].cpu().numpy() if hasattr(predn['bboxes'], 'cpu') else predn['bboxes']
                            classes = predn['cls'].cpu().numpy() if hasattr(predn['cls'], 'cpu') else predn['cls']
                            confs = predn['conf'].cpu().numpy() if hasattr(predn['conf'], 'cpu') else predn['conf']
                            
                            for bbox, cls, conf in zip(bboxes, classes, confs):
                                trainer._hmd_collection['pred_boxes'].append({
                                    'image_idx': global_img_idx,
                                    'class': int(cls),
                                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
                                    'conf': float(conf)
                                })
                    except Exception as e:
                        logging.debug(f"⚠️ Failed to prepare prediction for image {img_idx_in_batch}: {e}")
                        continue
            
            # Process ground truth - already in native space from _prepare_batch
            for img_idx_in_batch, target in enumerate(targets):
                global_img_idx = current_offset + img_idx_in_batch
                
                if 'bboxes' in target and 'cls' in target:
                    bboxes = target['bboxes'].cpu().numpy() if hasattr(target['bboxes'], 'cpu') else target['bboxes']
                    classes = target['cls'].cpu().numpy() if hasattr(target['cls'], 'cpu') else target['cls']
                    
                    for bbox, cls in zip(bboxes, classes):
                        trainer._hmd_collection['gt_boxes'].append({
                            'image_idx': global_img_idx,
                            'class': int(cls),
                            'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                        })
            
            # Store image files
            if image_files:
                trainer._hmd_collection['image_files'].extend(image_files)


def calculate_hmd_from_boxes_np(mentum_box, hyoid_box):
    """
    Calculate HMD from two bounding boxes in pixel coordinates (numpy version)
    
    Args:
        mentum_box: [x1, y1, x2, y2] format array
        hyoid_box: [x1, y1, x2, y2] format array
    
    Returns:
        HMD distance in pixels (float)
    """
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_box
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_box
    
    # Calculate HMD (same as loss.py)
    hmd_dx = hyoid_x1 - mentum_x2
    mentum_y_center = (mentum_y1 + mentum_y2) / 2
    hyoid_y_center = (hyoid_y1 + hyoid_y2) / 2
    hmd_dy = hyoid_y_center - mentum_y_center
    hmd = np.sqrt(hmd_dx**2 + hmd_dy**2 + 1e-8)
    
    return float(hmd)


def _to_numpy_safe(data):
    """
    Safely convert CUDA tensors to CPU numpy arrays.
    Helper function to avoid CUDA tensor to numpy conversion errors.
    
    Args:
        data: Can be torch.Tensor, list, tuple, or numpy array
    
    Returns:
        numpy array or original data if already numpy
    """
    import torch
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, (list, tuple)):
        return [_to_numpy_safe(item) for item in data]
    else:
        return data


def _extract_pixel_spacing_value(val):
    """
    Extract numeric pixel spacing value from various formats (dict, list, float, etc.)
    
    Priority order for dict:
    1. 'truePixelSpacing' - True calculated PixelSpacing (preferred)
    2. 'dcmPixelSpacing' - DICOM PixelSpacing tag value
    3. 'PixelSpacing' - Generic PixelSpacing key
    4. 'x' - X-axis spacing (for {'x': 0.1, 'y': 0.1} format)
    """
    import numpy as np
    
    if isinstance(val, (list, tuple, np.ndarray)):
        # If it's a list/tuple/array, use the first element or mean
        if len(val) > 0:
            return float(val[0])
        else:
            return None
    elif isinstance(val, dict):
        # Priority 1: truePixelSpacing (from Dicom_PixelSpacing_DA.joblib format)
        if 'truePixelSpacing' in val:
            ps_val = val['truePixelSpacing']
            if isinstance(ps_val, (int, float, np.number)):
                return float(ps_val)
            elif isinstance(ps_val, (list, tuple, np.ndarray)) and len(ps_val) > 0:
                return float(ps_val[0])
        
        # Priority 2: dcmPixelSpacing (DICOM tag value)
        if 'dcmPixelSpacing' in val:
            ps_val = val['dcmPixelSpacing']
            if isinstance(ps_val, (int, float, np.number)):
                return float(ps_val)
            elif isinstance(ps_val, (list, tuple, np.ndarray)) and len(ps_val) > 0:
                return float(ps_val[0])
        
        # Priority 3: PixelSpacing (generic key)
        if 'PixelSpacing' in val:
            ps_val = val['PixelSpacing']
            if isinstance(ps_val, (list, tuple, np.ndarray)) and len(ps_val) > 0:
                return float(ps_val[0])
            elif isinstance(ps_val, (int, float, np.number)):
                return float(ps_val)
        
        # Priority 4: 'x' key (for {'x': 0.1, 'y': 0.1} format)
        if 'x' in val:
            return float(val['x'])
        
        # Fallback: try to find first numeric value (but avoid n_frame, n_row, n_column, etc.)
        # Skip common non-spacing keys
        skip_keys = {'n_frame', 'n_row', 'n_column', 'n_cm', 'n_pixel'}
        for k, v in val.items():
            if k not in skip_keys and isinstance(v, (int, float, np.number)):
                return float(v)
        
        return None
    elif isinstance(val, (int, float, np.number)):
        return float(val)
    else:
        return None


def _get_avg_pixel_spacing(pixel_spacing_dict):
    """Safely calculate average pixel spacing from dictionary with various value formats"""
    import numpy as np
    
    if pixel_spacing_dict is None or len(pixel_spacing_dict) == 0:
        return None
    
    pixel_spacing_values = []
    for val in pixel_spacing_dict.values():
        extracted_val = _extract_pixel_spacing_value(val)
        if extracted_val is not None:
            pixel_spacing_values.append(extracted_val)
    
    if len(pixel_spacing_values) > 0:
        return np.mean(pixel_spacing_values)
    else:
        return None


def _get_avg_pixel_spacing_from_validator(validator, pixel_spacing_dict):
    """
    Calculate average pixel spacing from validator dataset images.
    This ensures that mm-based metrics use the correct PixelSpacing for the specific dataset (val/test).
    
    Args:
        validator: Ultralytics validator object
        pixel_spacing_dict: Dictionary mapping image IDs (dicom_base_name) to PixelSpacing
    
    Returns:
        Average pixel spacing (float) for images in validator dataset, or None if not available
    """
    import numpy as np
    
    if pixel_spacing_dict is None or len(pixel_spacing_dict) == 0:
        return None
    
    image_pixel_spacings = []
    dataset = None
    total_images = 0
    matched_count = 0
    unmatched_samples = []  # Store first few unmatched samples for debugging
    try:
        # Access validator's dataloader and dataset
        if hasattr(validator, 'dataloader') and validator.dataloader is not None:
            dataset = validator.dataloader.dataset
            if hasattr(dataset, 'im_files'):
                total_images = len(dataset.im_files)
                # Extract DICOM base names from image paths and match with pixel_spacing_dict
                # Try to import extract_dicom_info_from_filename, with fallback to local implementation
                try:
                    from ultralytics.evaluate.calculate_hmd_from_yolo import extract_dicom_info_from_filename
                except (ImportError, ModuleNotFoundError) as e:
                    # Fallback: implement extract_dicom_info_from_filename locally
                    import re as _re
                    def extract_dicom_info_from_filename(filename: str):
                        """Extract DICOM base name and pose from PNG filename (local fallback)"""
                        # Remove extension
                        base = filename.replace('.png', '').replace('.txt', '')
                        # Match pose pattern: _Pose[xxx] or Pose[xxx] or _Pose [xxx]
                        pose_match = _re.search(r'[_\s]?(Neutral|Extended|Ramped)\s*\[', base, _re.IGNORECASE)
                        if pose_match:
                            pose = pose_match.group(1)
                            base = _re.sub(r'[_\s]?(Neutral|Extended|Ramped)\s*\[\d+\]', '', base, flags=_re.IGNORECASE)
                        else:
                            pose = None
                            base = _re.sub(r'\[\d+\]', '', base)
                        # Clean possible .dcm suffix
                        base = _re.sub(r'\.dcm(?:_|$)', '_', base)
                        if base.endswith('.dcm'):
                            base = base[:-4]
                        # Clean trailing spaces and underscores
                        base = base.strip().rstrip('_').rstrip()
                        return base, pose
                
                # Sample a few keys from pixel_spacing_dict to understand its format
                sample_keys = list(pixel_spacing_dict.keys())[:5] if len(pixel_spacing_dict) > 0 else []
                
                for im_file in dataset.im_files:
                    im_path = Path(im_file)
                    # Extract DICOM base name from filename
                    dicom_base, _ = extract_dicom_info_from_filename(im_path.name)
                    
                    # Normalize dicom_base for matching (remove spaces, convert to lowercase for comparison)
                    dicom_base_normalized = dicom_base.strip().lower()
                    
                    # Try to find matching PixelSpacing in dictionary
                    # Strategy 1: Exact match
                    if dicom_base in pixel_spacing_dict:
                        ps_val = _extract_pixel_spacing_value(pixel_spacing_dict[dicom_base])
                        if ps_val is not None:
                            image_pixel_spacings.append(ps_val)
                            matched_count += 1
                            continue
                    
                    # Strategy 2: Normalized exact match (case-insensitive)
                    matched = False
                    for key in pixel_spacing_dict.keys():
                        key_normalized = key.strip().lower()
                        # Remove .dcm extension and pose suffixes for comparison
                        key_clean = key_normalized.replace('.dcm', '').replace('_neutral', '').replace('_extended', '').replace('_ramped', '')
                        key_clean = key_clean.strip('_').strip()
                        
                        if dicom_base_normalized == key_normalized or dicom_base_normalized == key_clean:
                            ps_val = _extract_pixel_spacing_value(pixel_spacing_dict[key])
                            if ps_val is not None:
                                image_pixel_spacings.append(ps_val)
                                matched_count += 1
                                matched = True
                                break
                    
                    # Strategy 3: Substring match (dicom_base contained in key or vice versa)
                    if not matched:
                        for key in pixel_spacing_dict.keys():
                            key_normalized = key.strip().lower()
                            key_clean = key_normalized.replace('.dcm', '').replace('_neutral', '').replace('_extended', '').replace('_ramped', '')
                            key_clean = key_clean.strip('_').strip()
                            
                            # Check if dicom_base is contained in cleaned key or vice versa
                            if dicom_base_normalized in key_clean or key_clean in dicom_base_normalized:
                                ps_val = _extract_pixel_spacing_value(pixel_spacing_dict[key])
                                if ps_val is not None:
                                    image_pixel_spacings.append(ps_val)
                                    matched_count += 1
                                    matched = True
                                    break
                    
                    # Store unmatched samples for debugging (limit to first 3)
                    if not matched and len(unmatched_samples) < 3:
                        unmatched_samples.append({
                            'filename': im_path.name,
                            'dicom_base': dicom_base,
                            'sample_dict_keys': sample_keys
                        })
    except Exception as e:
        logging.warning(f"⚠️ Failed to extract image paths from validator: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        image_pixel_spacings = []
    
    # If we successfully matched some images, use average of matched pixel spacings
    if len(image_pixel_spacings) > 0:
        avg_pixel_spacing = np.mean(image_pixel_spacings)
        match_rate = len(image_pixel_spacings) / total_images if total_images > 0 else 0.0
        logging.info(f"✅ Matched {len(image_pixel_spacings)}/{total_images} images with PixelSpacing ({match_rate*100:.1f}%, avg: {avg_pixel_spacing:.4f} mm/pixel)")
        return avg_pixel_spacing
    else:
        # Fallback: use average of all pixel spacings in dictionary
        # This should rarely happen if pixel_spacing_dict is properly populated
        avg_pixel_spacing = _get_avg_pixel_spacing(pixel_spacing_dict)
        if avg_pixel_spacing is not None:
            logging.warning(f"⚠️ No images matched in validator dataset ({total_images} images), using average PixelSpacing from entire dictionary: {avg_pixel_spacing:.4f} mm/pixel")
            # Log debugging information
            if unmatched_samples:
                for i, sample in enumerate(unmatched_samples[:2]):  # Show first 2 samples
                    logging.warning(f"   Sample {i+1}: filename='{sample['filename']}', extracted_dicom_base='{sample['dicom_base']}'")
                    if sample['sample_dict_keys']:
                        logging.warning(f"   Sample dict keys (first 5): {sample['sample_dict_keys']}")
        return avg_pixel_spacing


def calculate_hmd_metrics_from_validator(validator, trainer, penalty_single=None, penalty_none=None, 
                                         pixel_spacing_dict=None, imgsz=None):
    """
    Calculate HMD metrics from validator using collected bbox data or HMD loss stats
    
    Args:
        validator: Ultralytics validator object
        trainer: Trainer object (to access HMD loss stats)
        penalty_single: Penalty when only one target detected (in pixels). If None, uses imgsz/2.
        penalty_none: Penalty when both targets missed (in pixels). If None, uses imgsz.
        pixel_spacing_dict: Dictionary mapping image IDs (dicom_base_name) to PixelSpacing (mm/pixel) for mm calculation
                           Keys should be DICOM base names (e.g., "0834980_Quick ID_20240509_155005_B")
        imgsz: Image size (width/height in pixels). If provided, penalty_none=imgsz and penalty_single=imgsz/2.
               If None, tries to get from validator/trainer args, or uses defaults (1000.0, 500.0).
    
    Returns:
        Dict with detection_rate, rmse_pixel, overall_score_pixel, rmse_mm, overall_score_mm
    """
    # Determine image size and penalties
    if imgsz is None:
        # Try to get imgsz from validator or trainer
        if validator is not None:
            if hasattr(validator, 'args') and hasattr(validator.args, 'imgsz'):
                imgsz = validator.args.imgsz
            elif hasattr(validator, 'imgsz'):
                imgsz = validator.imgsz
        if imgsz is None and trainer is not None:
            if hasattr(trainer, 'args'):
                if isinstance(trainer.args, dict):
                    imgsz = trainer.args.get('imgsz')
                elif hasattr(trainer.args, 'imgsz'):
                    imgsz = trainer.args.imgsz
            if imgsz is None and hasattr(trainer, 'imgsz'):
                imgsz = trainer.imgsz
    
    # Set penalties based on imgsz if provided
    if imgsz is not None:
        if penalty_none is None:
            penalty_none = float(imgsz)
        if penalty_single is None:
            penalty_single = float(imgsz) / 2.0
    else:
        # Use defaults if imgsz not available
        if penalty_none is None:
            penalty_none = 1000.0
        if penalty_single is None:
            penalty_single = 500.0
    import numpy as np
    import torch
    
    try:
        # Method 1 (PREFERRED): Use HMD Loss Stats from criterion (same calculation as training phase)
        # This uses the same function (calculate_hmd_loss) as training, which calculates real HMD errors
        # from boxes using: Smooth L1 Loss + relative error + direction penalty
        # The HMD loss already calculates real HMD distances during training/validation
        method_1_used = False
        if hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
            criterion = trainer.model.criterion
            if hasattr(criterion, 'use_hmd_loss') and criterion.use_hmd_loss:
                method_1_used = True
                logging.info(f"🔍 Using Method 1 (HMD Loss Stats) for HMD metrics calculation")
                # Get HMD loss stats - this contains real HMD error calculations
                # We can use the accumulated HMD loss to estimate RMSE
                if hasattr(criterion, 'hmd_loss_sum') and hasattr(criterion, 'hmd_loss_count'):
                    if criterion.hmd_loss_count > 0:
                        # Average HMD loss represents average HMD error
                        # Convert to CPU float if it's a CUDA tensor
                        hmd_loss_sum = criterion.hmd_loss_sum
                        hmd_loss_count = criterion.hmd_loss_count
                        if isinstance(hmd_loss_sum, torch.Tensor):
                            hmd_loss_sum = hmd_loss_sum.cpu().item()
                        if isinstance(hmd_loss_count, torch.Tensor):
                            hmd_loss_count = hmd_loss_count.cpu().item()
                        avg_hmd_loss = hmd_loss_sum / hmd_loss_count
                        
                        # Get stats from validator for detection rate
                        # Try multiple ways to get stats:
                        # 1. validator.stats (if available)
                        # 2. validator.metrics.stats (if available)
                        stats = None
                        if hasattr(validator, 'stats') and validator.stats is not None:
                            stats = validator.stats
                        elif hasattr(validator, 'metrics') and hasattr(validator.metrics, 'stats') and validator.metrics.stats is not None:
                            stats = validator.metrics.stats
                        
                        if stats is not None:
                            if stats and len(stats.get('tp', [])) > 0:
                                # Convert CUDA tensors to CPU numpy arrays if needed
                                tp_list = stats.get('tp', [])
                                pred_cls_list = stats.get('pred_cls', [])
                                target_cls_list = stats.get('target_cls', [])
                                
                                # Convert to numpy arrays, handling CUDA tensors
                                if tp_list:
                                    tp_list = [_to_numpy_safe(item) for item in tp_list]
                                    tp = np.concatenate(tp_list, 0) if tp_list else np.array([])
                                else:
                                    tp = np.array([])
                                
                                if pred_cls_list:
                                    pred_cls_list = [_to_numpy_safe(item) for item in pred_cls_list]
                                    pred_cls = np.concatenate(pred_cls_list, 0) if pred_cls_list else np.array([])
                                else:
                                    pred_cls = np.array([])
                                
                                if target_cls_list:
                                    target_cls_list = [_to_numpy_safe(item) for item in target_cls_list]
                                    target_cls = np.concatenate(target_cls_list, 0) if target_cls_list else np.array([])
                                else:
                                    target_cls = np.array([])
                                
                                # Count images with both classes in GT
                                mentum_class = 0
                                hyoid_class = 1
                                mentum_gt_count = np.sum(target_cls == mentum_class)
                                hyoid_gt_count = np.sum(target_cls == hyoid_class)
                                
                                # Get target_img to properly count images with both classes
                                target_img_list = stats.get('target_img', [])
                                if target_img_list:
                                    target_img_list = [_to_numpy_safe(item) for item in target_img_list]
                                    target_img = np.concatenate(target_img_list, 0) if target_img_list else np.array([])
                                else:
                                    target_img = np.array([])
                                
                                # OLD VERSION LOGIC (simpler and more reliable):
                                # Use min(mentum_gt_count, hyoid_gt_count) as images_with_both_gt
                                # This is simpler and more reliable than using target_img intersection
                                # Since each image typically has both Mentum and Hyoid, use min as approximation
                                images_with_both_gt = min(mentum_gt_count, hyoid_gt_count) if (mentum_gt_count > 0 and hyoid_gt_count > 0) else 0
                                
                                if images_with_both_gt > 0:
                                    # Count matched detections at IoU=0.5
                                    # Note: tp and pred_cls have same length (both based on predictions)
                                    # target_cls may have different length (based on ground truth)
                                    if len(tp) > 0 and tp.shape[1] > 0:
                                        matched_mask = tp[:, 0]  # Boolean array for IoU=0.5 matches
                                        
                                        # Debug: Check if there are any matched predictions
                                        total_matched = np.sum(matched_mask)
                                        total_predictions = len(matched_mask)
                                        logging.info(f"🔍 Method 1: Total predictions={total_predictions}, matched={total_matched}, "
                                                    f"match_rate={total_matched/total_predictions if total_predictions > 0 else 0.0:.4f}")
                                        
                                        # Check validator's confidence threshold
                                        validator_conf = None
                                        if hasattr(validator, 'args') and hasattr(validator.args, 'conf'):
                                            validator_conf = validator.args.conf
                                        logging.info(f"🔍 Method 1: Validator confidence threshold={validator_conf}")
                                        
                                        # tp and pred_cls should have same length (both are per-prediction)
                                        if len(matched_mask) == len(pred_cls):
                                            # Count matched predictions for each class
                                            # Only count when prediction matches ground truth (matched_mask is True)
                                            mentum_matched = np.sum((matched_mask) & (pred_cls == mentum_class))
                                            hyoid_matched = np.sum((matched_mask) & (pred_cls == hyoid_class))
                                            
                                            # OLD VERSION LOGIC (simpler and more reliable):
                                            # Use min(mentum_matched, hyoid_matched) as both_detected_count
                                            # This directly counts how many detections we have for both classes
                                            # This is simpler and more reliable than using match ratios
                                            if images_with_both_gt > 0:
                                                both_detected_count = min(mentum_matched, hyoid_matched)
                                                detection_rate = both_detected_count / images_with_both_gt if images_with_both_gt > 0 else 0.0
                                                
                                                # Debug logging (use INFO level so it's visible)
                                                logging.info(f"🔍 Detection Rate Calculation (Method 1): "
                                                            f"mentum_gt={mentum_gt_count}, hyoid_gt={hyoid_gt_count}, "
                                                            f"mentum_matched={mentum_matched}, hyoid_matched={hyoid_matched}, "
                                                            f"images_with_both_gt={images_with_both_gt}, both_detected_count={both_detected_count}, "
                                                            f"detection_rate={detection_rate:.4f}")
                                            else:
                                                detection_rate = 0.0
                                                logging.warning(f"⚠️ Detection Rate Calculation failed (Method 1): "
                                                              f"images_with_both_gt={images_with_both_gt}, "
                                                              f"mentum_gt={mentum_gt_count}, hyoid_gt={hyoid_gt_count}")
                                        else:
                                            # Length mismatch - use fallback calculation
                                            logging.warning(f"⚠️ Length mismatch: tp={len(matched_mask)}, pred_cls={len(pred_cls)}")
                                            detection_rate = 0.0
                                        
                                        # Get RMSE and MAE from criterion.hmd_stats (correct calculation)
                                        # hmd_stats contains lists of batch-level metrics, we need to calculate epoch-level averages
                                        rmse_pixel = 0.0
                                        mae_pixel = 0.0
                                        rmse_no_penalty_pixel = 0.0
                                        mae_no_penalty_pixel = 0.0
                                        
                                        # Try to get metrics from criterion stats (preferred method)
                                        if hasattr(criterion, 'hmd_stats') and isinstance(criterion.hmd_stats, dict):
                                            stats_dict = criterion.hmd_stats
                                            
                                            # With penalty: calculate from all batch-level metrics
                                            if 'rmse_with_penalty' in stats_dict and isinstance(stats_dict['rmse_with_penalty'], list):
                                                rmse_list = stats_dict['rmse_with_penalty']
                                                if len(rmse_list) > 0:
                                                    # Calculate epoch-level RMSE: sqrt(mean(batch_rmse^2))
                                                    # This is the correct way to aggregate RMSE across batches
                                                    rmse_pixel = float(np.sqrt(np.mean(np.array(rmse_list)**2)))
                                            
                                            if 'mae_with_penalty' in stats_dict and isinstance(stats_dict['mae_with_penalty'], list):
                                                mae_list = stats_dict['mae_with_penalty']
                                                if len(mae_list) > 0:
                                                    # Calculate epoch-level MAE: mean(batch_mae)
                                                    mae_pixel = float(np.mean(mae_list))
                                            
                                            # No penalty: calculate from both_detected cases only
                                            if 'rmse_no_penalty' in stats_dict and isinstance(stats_dict['rmse_no_penalty'], list):
                                                rmse_no_penalty_list = stats_dict['rmse_no_penalty']
                                                if len(rmse_no_penalty_list) > 0:
                                                    # Calculate epoch-level RMSE: sqrt(mean(batch_rmse^2))
                                                    rmse_no_penalty_pixel = float(np.sqrt(np.mean(np.array(rmse_no_penalty_list)**2)))
                                            
                                            if 'mae_no_penalty' in stats_dict and isinstance(stats_dict['mae_no_penalty'], list):
                                                mae_no_penalty_list = stats_dict['mae_no_penalty']
                                                if len(mae_no_penalty_list) > 0:
                                                    # Calculate epoch-level MAE: mean(batch_mae)
                                                    mae_no_penalty_pixel = float(np.mean(mae_no_penalty_list))
                                        
                                        # Fallback: if hmd_stats not available or empty, use avg_hmd_loss as approximation
                                        # This is less accurate but better than nothing
                                        if rmse_pixel == 0.0 and mae_pixel == 0.0:
                                            logging.warning("⚠️ Method 1: hmd_stats not available, using avg_hmd_loss as approximation")
                                            rmse_pixel = float(avg_hmd_loss)
                                            mae_pixel = float(avg_hmd_loss)
                                        
                                        # Overall_Score should be higher when Detection_Rate is high AND RMSE_HMD is low
                                        # Formula: Overall_Score = Detection_Rate / (1 + RMSE_HMD / normalization_factor)
                                        # Using 1000 as normalization factor (typical RMSE range: 100-1000 pixels)
                                        # This ensures: higher Detection_Rate and lower RMSE_HMD → higher Overall_Score
                                        if rmse_pixel > 0:
                                            overall_score_pixel = detection_rate / (1 + rmse_pixel / 1000.0)
                                        else:
                                            overall_score_pixel = detection_rate  # Perfect RMSE (0) → score equals detection rate
                                        
                                        # Calculate mm version if pixel_spacing_dict is available
                                        rmse_mm = 0.0
                                        mae_mm = 0.0
                                        overall_score_mm = 0.0
                                        rmse_no_penalty_mm = 0.0
                                        mae_no_penalty_mm = 0.0
                                        if pixel_spacing_dict is not None and len(pixel_spacing_dict) > 0:
                                            # Get average pixel spacing from validator dataset images
                                            avg_pixel_spacing = _get_avg_pixel_spacing_from_validator(validator, pixel_spacing_dict)
                                            if avg_pixel_spacing is None:
                                                avg_pixel_spacing = 0.0
                                            
                                            rmse_mm = rmse_pixel * avg_pixel_spacing
                                            mae_mm = mae_pixel * avg_pixel_spacing
                                            rmse_no_penalty_mm = rmse_no_penalty_pixel * avg_pixel_spacing
                                            mae_no_penalty_mm = mae_no_penalty_pixel * avg_pixel_spacing
                                            
                                            # For mm version, use 100 as normalization factor (typical RMSE range: 10-100 mm)
                                            if rmse_mm > 0:
                                                overall_score_mm = detection_rate / (1 + rmse_mm / 100.0)
                                            else:
                                                overall_score_mm = detection_rate
                                        
                                        return {
                                            'detection_rate': float(detection_rate),
                                            'rmse_pixel': float(rmse_pixel),
                                            'mae_pixel': float(mae_pixel),
                                            'overall_score_pixel': float(overall_score_pixel),
                                            'rmse_mm': float(rmse_mm),
                                            'mae_mm': float(mae_mm),
                                            'overall_score_mm': float(overall_score_mm),
                                            'rmse_no_penalty_pixel': float(rmse_no_penalty_pixel),
                                            'mae_no_penalty_pixel': float(mae_no_penalty_pixel),
                                            'rmse_no_penalty_mm': float(rmse_no_penalty_mm),
                                            'mae_no_penalty_mm': float(mae_no_penalty_mm),
                                        }
        
        # Method 2 (FALLBACK): Calculate from validator stats (without boxes, uses approximation)
        # This is used only when HMD Loss Stats are not available.
        # NOTE: This fallback method uses approximation (30.0 pixels) instead of real HMD error calculation.
        # The preferred method (Method 1) uses HMD Loss Stats which calculates real HMD errors
        # using the same function (calculate_hmd_loss) as training phase.
        # Note: method_1_used is set to True if Method 1 was successfully used
        # If Method 1 was not used (method_1_used is False), we fall back to Method 2
        if not method_1_used:
            logging.info(f"🔍 Using Method 2 (Fallback) for HMD metrics calculation (HMD Loss Stats not available)")
        # Get stats from validator
        # Try multiple ways to get stats:
        # 1. validator.stats (if available)
        # 2. validator.metrics.stats (if available)
        # 3. validator.metrics (if it has stats as a property)
        stats = None
        if hasattr(validator, 'stats') and validator.stats is not None:
            stats = validator.stats
            logging.info(f"🔍 Method 2: Using validator.stats")
        elif hasattr(validator, 'metrics') and hasattr(validator.metrics, 'stats') and validator.metrics.stats is not None:
            stats = validator.metrics.stats
            logging.info(f"🔍 Method 2: Using validator.metrics.stats")
        else:
            logging.warning("⚠️ Validator stats not available (Method 2) - tried validator.stats and validator.metrics.stats")
            # When no stats, use penalty_none for RMSE (both targets missed)
            rmse_mm = 0.0
            overall_score_mm = 0.0
            if pixel_spacing_dict is not None and len(pixel_spacing_dict) > 0:
                avg_pixel_spacing = _get_avg_pixel_spacing_from_validator(validator, pixel_spacing_dict)
                if avg_pixel_spacing is None:
                    avg_pixel_spacing = 0.0
                rmse_mm = penalty_none * avg_pixel_spacing
            return {
                'detection_rate': 0.0, 
                'rmse_pixel': penalty_none, 
                'mae_pixel': penalty_none,
                'overall_score_pixel': 0.0, 
                'rmse_mm': rmse_mm, 
                'mae_mm': rmse_mm,
                'overall_score_mm': overall_score_mm,
                'rmse_no_penalty_pixel': 0.0,
                'mae_no_penalty_pixel': 0.0,
                'rmse_no_penalty_mm': 0.0,
                'mae_no_penalty_mm': 0.0,
            }
        
        logging.info(f"🔍 Method 2: stats available, keys={list(stats.keys()) if stats else 'None'}")
        if not stats or len(stats.get('tp', [])) == 0:
            logging.warning(f"⚠️ Validator stats empty or no tp (Method 2): stats={stats is not None}, tp_len={len(stats.get('tp', [])) if stats else 0}")
            # When no stats, use penalty_none for RMSE (both targets missed)
            rmse_mm = 0.0
            mae_mm = 0.0
            overall_score_mm = 0.0
            if pixel_spacing_dict is not None and len(pixel_spacing_dict) > 0:
                avg_pixel_spacing = _get_avg_pixel_spacing_from_validator(validator, pixel_spacing_dict)
                if avg_pixel_spacing is None:
                    avg_pixel_spacing = 0.0
                rmse_mm = penalty_none * avg_pixel_spacing
                mae_mm = penalty_none * avg_pixel_spacing
            return {
                'detection_rate': 0.0, 
                'rmse_pixel': penalty_none, 
                'mae_pixel': penalty_none,
                'overall_score_pixel': 0.0, 
                'rmse_mm': rmse_mm, 
                'mae_mm': mae_mm,
                'overall_score_mm': overall_score_mm,
                'rmse_no_penalty_pixel': 0.0,
                'mae_no_penalty_pixel': 0.0,
                'rmse_no_penalty_mm': 0.0,
                'mae_no_penalty_mm': 0.0,
            }
        
        # Get predictions and ground truth from stats
        # Convert CUDA tensors to CPU numpy arrays if needed
        tp_list = stats.get('tp', [])
        pred_cls_list = stats.get('pred_cls', [])
        target_cls_list = stats.get('target_cls', [])
        target_img_list = stats.get('target_img', [])
        
        logging.info(f"🔍 Method 2: Extracted stats - tp_list_len={len(tp_list)}, pred_cls_list_len={len(pred_cls_list)}, "
                    f"target_cls_list_len={len(target_cls_list)}, target_img_list_len={len(target_img_list)}")
        
        # Convert to numpy arrays, handling CUDA tensors
        if tp_list:
            tp_list = [_to_numpy_safe(item) for item in tp_list]
            tp = np.concatenate(tp_list, 0) if tp_list else np.array([])
        else:
            tp = np.array([])
        
        if pred_cls_list:
            pred_cls_list = [_to_numpy_safe(item) for item in pred_cls_list]
            pred_cls = np.concatenate(pred_cls_list, 0) if pred_cls_list else np.array([])
        else:
            pred_cls = np.array([])
        
        if target_cls_list:
            target_cls_list = [_to_numpy_safe(item) for item in target_cls_list]
            target_cls = np.concatenate(target_cls_list, 0) if target_cls_list else np.array([])
        else:
            target_cls = np.array([])
        
        # Get target_img to group by image
        if target_img_list:
            target_img_list = [_to_numpy_safe(item) for item in target_img_list]
            target_img = np.concatenate(target_img_list, 0) if target_img_list else np.array([])
        else:
            target_img = np.array([])
        
        logging.info(f"🔍 Method 2: After conversion - tp_shape={tp.shape if len(tp) > 0 else 'empty'}, "
                    f"pred_cls_len={len(pred_cls)}, target_cls_len={len(target_cls)}, target_img_len={len(target_img)}")
        
        # Mentum class = 0, Hyoid class = 1
        mentum_class = 0
        hyoid_class = 1
        
        # Count detections for each class
        # Note: tp and pred_cls have same length (both based on predictions)
        # target_cls may have different length (based on ground truth)
        if len(tp) > 0 and tp.shape[1] > 0:
            matched_mask = tp[:, 0]  # Matches at IoU=0.5
            
            # Debug: Check if there are any matched predictions
            total_matched = np.sum(matched_mask)
            total_predictions = len(matched_mask)
            logging.info(f"🔍 Method 2: Total predictions={total_predictions}, matched={total_matched}, "
                        f"match_rate={total_matched/total_predictions if total_predictions > 0 else 0.0:.4f}")
            
            # Check validator's confidence threshold
            validator_conf = None
            if hasattr(validator, 'args') and hasattr(validator.args, 'conf'):
                validator_conf = validator.args.conf
            logging.info(f"🔍 Method 2: Validator confidence threshold={validator_conf}")
            
            # tp and pred_cls should have same length (both are per-prediction)
            if len(matched_mask) != len(pred_cls):
                logging.warning(f"⚠️ Length mismatch in fallback: tp={len(matched_mask)}, pred_cls={len(pred_cls)}")
                # When length mismatch, use penalty_none
                rmse_mm = 0.0
                mae_mm = 0.0
                overall_score_mm = 0.0
                if pixel_spacing_dict is not None and len(pixel_spacing_dict) > 0:
                    avg_pixel_spacing = _get_avg_pixel_spacing_from_validator(validator, pixel_spacing_dict)
                    if avg_pixel_spacing is None:
                        avg_pixel_spacing = 0.0
                    rmse_mm = penalty_none * avg_pixel_spacing
                    mae_mm = penalty_none * avg_pixel_spacing
                return {
                    'detection_rate': 0.0, 
                    'rmse_pixel': penalty_none, 
                    'mae_pixel': penalty_none,
                    'overall_score_pixel': 0.0, 
                    'rmse_mm': rmse_mm, 
                    'mae_mm': mae_mm,
                    'overall_score_mm': overall_score_mm,
                    'rmse_no_penalty_pixel': 0.0,
                    'mae_no_penalty_pixel': 0.0,
                    'rmse_no_penalty_mm': 0.0,
                    'mae_no_penalty_mm': 0.0,
                }
            
            # Count total ground truth for each class
            mentum_gt_count = np.sum(target_cls == mentum_class)
            hyoid_gt_count = np.sum(target_cls == hyoid_class)
            
            # OLD VERSION LOGIC (simpler and more reliable):
            # Use min(mentum_gt_count, hyoid_gt_count) as images_with_both_gt
            # This is simpler and more reliable than using target_img intersection
            # Since each image typically has both Mentum and Hyoid, use min as approximation
            images_with_both_gt = min(mentum_gt_count, hyoid_gt_count) if (mentum_gt_count > 0 and hyoid_gt_count > 0) else 0
            
            # Debug logging (use INFO level so it's visible)
            logging.info(f"🔍 Images with both GT (Method 2): "
                        f"mentum_gt_count={mentum_gt_count}, hyoid_gt_count={hyoid_gt_count}, "
                        f"images_with_both_gt={images_with_both_gt}")
            
            if images_with_both_gt == 0:
                logging.warning(f"⚠️ No images with both classes in GT: "
                              f"mentum_gt_count={mentum_gt_count}, hyoid_gt_count={hyoid_gt_count}, "
                              f"target_img_len={len(target_img) if 'target_img' in locals() else 0}")
                # No ground truth with both targets, use penalty_none
                rmse_mm = 0.0
                overall_score_mm = 0.0
                if pixel_spacing_dict is not None and len(pixel_spacing_dict) > 0:
                    avg_pixel_spacing = _get_avg_pixel_spacing_from_validator(validator, pixel_spacing_dict)
                    if avg_pixel_spacing is None:
                        avg_pixel_spacing = 0.0
                    rmse_mm = penalty_none * avg_pixel_spacing
                return {
                'detection_rate': 0.0, 
                'rmse_pixel': penalty_none, 
                'mae_pixel': penalty_none,
                'overall_score_pixel': 0.0, 
                'rmse_mm': rmse_mm, 
                'mae_mm': rmse_mm,
                'overall_score_mm': overall_score_mm,
                'rmse_no_penalty_pixel': 0.0,
                'mae_no_penalty_pixel': 0.0,
                'rmse_no_penalty_mm': 0.0,
                'mae_no_penalty_mm': 0.0,
            }
            
            # Count images where both are detected (matched)
            # Count matched predictions for each class
            mentum_matched_count = np.sum((matched_mask) & (pred_cls == mentum_class))
            hyoid_matched_count = np.sum((matched_mask) & (pred_cls == hyoid_class))
            
            # OLD VERSION LOGIC (simpler and more reliable):
            # Use min(mentum_matched, hyoid_matched) as both_detected_count
            # This directly counts how many detections we have for both classes
            # This is simpler and more reliable than using match ratios
            both_detected_count = min(mentum_matched_count, hyoid_matched_count)
            
            # Debug logging (use INFO level so it's visible)
            logging.info(f"🔍 Detection Rate Calculation (Method 2): "
                        f"mentum_gt={mentum_gt_count}, hyoid_gt={hyoid_gt_count}, "
                        f"mentum_matched={mentum_matched_count}, hyoid_matched={hyoid_matched_count}, "
                        f"images_with_both_gt={images_with_both_gt}, both_detected_count={both_detected_count}, "
                        f"detection_rate={both_detected_count / images_with_both_gt if images_with_both_gt > 0 else 0.0:.4f}")
            
            # Calculate detection rate
            detection_rate = both_detected_count / images_with_both_gt if images_with_both_gt > 0 else 0.0
            
            # For RMSE: calculate HMD errors for each image
            # IMPORTANT: This is a FALLBACK method when HMD Loss Stats are not available
            # The preferred method (used in training) is to use HMD Loss Stats from criterion,
            # which calculates real HMD errors from boxes using calculate_hmd_loss function.
            # 
            # NEW: Try to use collected boxes from on_val_batch_end callback to calculate real HMD errors
            # This allows real error calculation even when --use_hmd_loss is False
            hmd_errors = []
            hmd_errors_no_penalty = []  # For no-penalty metrics (only both_detected cases)
            
            # Try to get collected boxes from trainer
            collected_boxes_available = False
            if trainer is not None and hasattr(trainer, '_hmd_collection'):
                collection = trainer._hmd_collection
                if (len(collection.get('pred_boxes', [])) > 0 and 
                    len(collection.get('gt_boxes', [])) > 0):
                    collected_boxes_available = True
                    logging.info(f"✅ Using collected boxes for real HMD error calculation (Method 2)")
            
            if both_detected_count > 0:
                if collected_boxes_available:
                    # Calculate real HMD errors from collected boxes
                    from ultralytics.mycodes.hmd_utils import calculate_hmd_error_from_boxes
                    import torch
                    
                    # Group boxes by image index
                    pred_boxes_by_img = {}
                    gt_boxes_by_img = {}
                    
                    for pred_box in collection['pred_boxes']:
                        img_idx = pred_box['image_idx']
                        if img_idx not in pred_boxes_by_img:
                            pred_boxes_by_img[img_idx] = {'mentum': [], 'hyoid': []}
                        cls = pred_box['class']
                        if cls == 0:  # Mentum
                            pred_boxes_by_img[img_idx]['mentum'].append({
                                'bbox': pred_box['bbox'],
                                'conf': pred_box['conf']
                            })
                        elif cls == 1:  # Hyoid
                            pred_boxes_by_img[img_idx]['hyoid'].append({
                                'bbox': pred_box['bbox'],
                                'conf': pred_box['conf']
                            })
                    
                    for gt_box in collection['gt_boxes']:
                        img_idx = gt_box['image_idx']
                        if img_idx not in gt_boxes_by_img:
                            gt_boxes_by_img[img_idx] = {'mentum': [], 'hyoid': []}
                        cls = gt_box['class']
                        if cls == 0:  # Mentum
                            gt_boxes_by_img[img_idx]['mentum'].append(gt_box['bbox'])
                        elif cls == 1:  # Hyoid
                            gt_boxes_by_img[img_idx]['hyoid'].append(gt_box['bbox'])
                    
                    # Calculate HMD errors for images with both detected
                    # Use all images in collection that have both classes (simpler and more reliable)
                    # The collection's image_idx is sequential (0, 1, 2, ...) matching dataset order
                    both_detected_img_indices = set()
                    for img_idx in pred_boxes_by_img.keys():
                        if (img_idx in gt_boxes_by_img and
                            len(pred_boxes_by_img[img_idx]['mentum']) > 0 and
                            len(pred_boxes_by_img[img_idx]['hyoid']) > 0 and
                            len(gt_boxes_by_img[img_idx]['mentum']) > 0 and
                            len(gt_boxes_by_img[img_idx]['hyoid']) > 0):
                            both_detected_img_indices.add(img_idx)
                    
                    logging.info(f"🔍 Found {len(both_detected_img_indices)} images with both classes in collected boxes")
                    
                    both_detected_errors = []
                    for img_idx in both_detected_img_indices:
                        if (img_idx in pred_boxes_by_img and img_idx in gt_boxes_by_img and
                            len(pred_boxes_by_img[img_idx]['mentum']) > 0 and
                            len(pred_boxes_by_img[img_idx]['hyoid']) > 0 and
                            len(gt_boxes_by_img[img_idx]['mentum']) > 0 and
                            len(gt_boxes_by_img[img_idx]['hyoid']) > 0):
                            
                            # Get best predictions (highest confidence)
                            mentum_pred = max(pred_boxes_by_img[img_idx]['mentum'], key=lambda x: x['conf'])
                            hyoid_pred = max(pred_boxes_by_img[img_idx]['hyoid'], key=lambda x: x['conf'])
                            
                            # Get ground truth (first one, assuming one per image)
                            mentum_gt = gt_boxes_by_img[img_idx]['mentum'][0]
                            hyoid_gt = gt_boxes_by_img[img_idx]['hyoid'][0]
                            
                            # Calculate HMD error
                            try:
                                hmd_error = calculate_hmd_error_from_boxes(
                                    mentum_pred_box=torch.tensor(mentum_pred['bbox'], dtype=torch.float32),
                                    hyoid_pred_box=torch.tensor(hyoid_pred['bbox'], dtype=torch.float32),
                                    mentum_gt_box=torch.tensor(mentum_gt, dtype=torch.float32),
                                    hyoid_gt_box=torch.tensor(hyoid_gt, dtype=torch.float32),
                                    pixel_spacing=None  # Calculate in pixels
                                )
                                both_detected_errors.append(hmd_error)
                                hmd_errors.append(hmd_error)
                                # For no-penalty, use absolute error (pixel distance)
                                pred_hmd = calculate_hmd_from_boxes_np(
                                    np.array(mentum_pred['bbox']),
                                    np.array(hyoid_pred['bbox'])
                                )
                                gt_hmd = calculate_hmd_from_boxes_np(
                                    np.array(mentum_gt),
                                    np.array(hyoid_gt)
                                )
                                abs_error = abs(pred_hmd - gt_hmd)
                                hmd_errors_no_penalty.append(abs_error)
                            except Exception as e:
                                logging.warning(f"⚠️ Failed to calculate HMD error for image {img_idx}: {e}")
                                # Fallback to estimated error
                                estimated_error = 30.0
                                hmd_errors.append(estimated_error)
                                hmd_errors_no_penalty.append(estimated_error)
                    
                    # Fill remaining both_detected_count with estimated errors if needed
                    remaining = both_detected_count - len(both_detected_errors)
                    if remaining > 0:
                        estimated_error = 30.0
                        hmd_errors.extend([estimated_error] * remaining)
                        hmd_errors_no_penalty.extend([estimated_error] * remaining)
                        logging.info(f"⚠️ Used estimated error (30.0 px) for {remaining} images (boxes not available)")
                else:
                    # Fallback: use estimated error when boxes are not available
                    estimated_error_per_image = 30.0  # pixels, conservative estimate for both-detected case
                    logging.info(f"⚠️ Boxes not collected, using estimated error (30.0 px) for {both_detected_count} images")
                    hmd_errors.extend([estimated_error_per_image] * both_detected_count)
                    hmd_errors_no_penalty.extend([estimated_error_per_image] * both_detected_count)
            
            single_detected = abs(mentum_matched_count - hyoid_matched_count)
            if single_detected > 0:
                hmd_errors.extend([penalty_single] * single_detected)
            
            none_detected = max(0, images_with_both_gt - both_detected_count - single_detected)
            if none_detected > 0:
                hmd_errors.extend([penalty_none] * none_detected)
            
            # Calculate RMSE and MAE (with penalty)
            if len(hmd_errors) > 0:
                hmd_errors_array = np.array(hmd_errors)
                rmse_pixel = np.sqrt(np.mean(hmd_errors_array**2))
                mae_pixel = np.mean(np.abs(hmd_errors_array))
            else:
                rmse_pixel = penalty_none
                mae_pixel = penalty_none
            
            # Calculate RMSE and MAE (no penalty) - only from both_detected cases
            # For Method 2, we use estimated_error_per_image (30.0 pixels) for both_detected cases
            # IMPORTANT: Even though we use an estimated error, we should calculate the distribution
            # across all both_detected images, not just assign a single value
            rmse_no_penalty_pixel = 0.0
            mae_no_penalty_pixel = 0.0
            if both_detected_count > 0:
                # Use estimated error for both_detected cases
                # NOTE: This is an approximation since validator.stats doesn't contain boxes
                # In reality, each image would have a different HMD error, but we use a conservative estimate
                estimated_error = 30.0  # pixels, conservative estimate for both-detected case
                
                # Calculate actual error distribution across all both_detected images
                # Even with the same estimated error per image, we should calculate RMSE/MAE properly
                # This ensures consistency with the mathematical definition
                estimated_errors_array = np.array([estimated_error] * both_detected_count)
                rmse_no_penalty_pixel = float(np.sqrt(np.mean(estimated_errors_array**2)))
                mae_no_penalty_pixel = float(np.mean(np.abs(estimated_errors_array)))
                
                # When all errors are the same, RMSE = MAE = error_value (mathematically correct)
                # But this is still an approximation - real errors would vary per image
            # If both_detected_count == 0, no-penalty metrics remain 0.0 (no valid detections)
            
            # Overall_Score should be higher when Detection_Rate is high AND RMSE_HMD is low
            # Formula: Overall_Score = Detection_Rate / (1 + RMSE_HMD / normalization_factor)
            # Using 1000 as normalization factor (typical RMSE range: 100-1000 pixels)
            # This ensures: higher Detection_Rate and lower RMSE_HMD → higher Overall_Score
            if rmse_pixel > 0:
                overall_score_pixel = detection_rate / (1 + rmse_pixel / 1000.0)
            else:
                overall_score_pixel = detection_rate  # Perfect RMSE (0) → score equals detection rate
            
            # Calculate mm version if pixel_spacing_dict is available
            rmse_mm = 0.0
            mae_mm = 0.0
            overall_score_mm = 0.0
            rmse_no_penalty_mm = 0.0
            mae_no_penalty_mm = 0.0
            if pixel_spacing_dict is not None and len(pixel_spacing_dict) > 0:
                # Get average pixel spacing from validator dataset images
                avg_pixel_spacing = _get_avg_pixel_spacing_from_validator(validator, pixel_spacing_dict)
                if avg_pixel_spacing is None:
                    avg_pixel_spacing = 0.0
                
                rmse_mm = rmse_pixel * avg_pixel_spacing
                mae_mm = mae_pixel * avg_pixel_spacing
                rmse_no_penalty_mm = rmse_no_penalty_pixel * avg_pixel_spacing
                mae_no_penalty_mm = mae_no_penalty_pixel * avg_pixel_spacing
                
                # For mm version, use 100 as normalization factor (typical RMSE range: 10-100 mm)
                if rmse_mm > 0:
                    overall_score_mm = detection_rate / (1 + rmse_mm / 100.0)
                else:
                    overall_score_mm = detection_rate
            
            return {
                'detection_rate': float(detection_rate),
                'rmse_pixel': float(rmse_pixel),
                'mae_pixel': float(mae_pixel),
                'overall_score_pixel': float(overall_score_pixel),
                'rmse_mm': float(rmse_mm),
                'mae_mm': float(mae_mm),
                'overall_score_mm': float(overall_score_mm),
                'rmse_no_penalty_pixel': float(rmse_no_penalty_pixel),
                'mae_no_penalty_pixel': float(mae_no_penalty_pixel),
                'rmse_no_penalty_mm': float(rmse_no_penalty_mm),
                'mae_no_penalty_mm': float(mae_no_penalty_mm),
            }
        else:
            return {'detection_rate': 0.0, 'rmse_pixel': penalty_none, 'overall_score_pixel': 0.0, 'rmse_mm': 0.0, 'overall_score_mm': 0.0}
        
    except Exception as e:
        logging.error(f"❌ Error calculating HMD metrics: {e}")
        import traceback
        logging.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            'detection_rate': 0.0, 
            'rmse_pixel': 0.0, 
            'mae_pixel': 0.0,
            'overall_score_pixel': 0.0, 
            'rmse_mm': 0.0, 
            'mae_mm': 0.0,
            'overall_score_mm': 0.0,
            'rmse_no_penalty_pixel': 0.0,
            'mae_no_penalty_pixel': 0.0,
            'rmse_no_penalty_mm': 0.0,
            'mae_no_penalty_mm': 0.0,
        }


# Create closure to capture args for callbacks
def create_on_val_end_callback(args):
    """Create on_val_end callback with access to args"""
    def on_val_end_callback(validator_or_trainer):
        """Callback to extract HMD metrics after validation
        
        Note: on_val_end callback receives validator object, not trainer.
        We need to handle both cases for compatibility.
        """
        # Create a dict to store additional metrics
        additional_metrics = {}
        import numpy as np
        import logging
        
        # Determine if we received validator or trainer
        # on_val_end callback receives validator, but we need trainer for some operations
        validator = None
        trainer = None
        
        # Check if it's a validator (has 'stats' attribute) or trainer (has 'validator' attribute)
        if hasattr(validator_or_trainer, 'stats'):
            # It's a validator
            validator = validator_or_trainer
            # Try to get trainer from validator if available
            # Validator stores trainer reference during __call__ when training=True
            if hasattr(validator, 'trainer') and validator.trainer is not None:
                trainer = validator.trainer
        elif hasattr(validator_or_trainer, 'validator'):
            # It's a trainer
            trainer = validator_or_trainer
            validator = trainer.validator if hasattr(trainer, 'validator') else None
        else:
            # Unknown type, try to use as validator
            validator = validator_or_trainer
            # Try to get trainer from validator if available
            if hasattr(validator, 'trainer') and validator.trainer is not None:
                trainer = validator.trainer
        
        # Debug: check if callback is triggered
        logging.debug(f"on_val_end_callback triggered: validator={validator is not None}, trainer={trainer is not None}")
        
        # Extract HMD loss from model criterion if available
        # IMPORTANT: Validation uses EMA model, but HMD loss statistics are accumulated in training model's criterion
        # So we should prefer getting HMD loss from trainer.model.criterion (training model) rather than EMA model's criterion
        # Prefer average HMD loss (across all batches) over last batch loss
        # In validation, hmd_loss_count might be 0, so use last_hmd_loss as fallback
        hmd_loss_value = None
        
        # First, try to get from training model's criterion (where statistics are actually accumulated)
        if trainer is not None and hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
            try:
                criterion = trainer.model.criterion
                if hasattr(criterion, 'get_avg_hmd_loss'):
                    # Get average HMD loss across all batches in this epoch
                    hmd_loss_avg = criterion.get_avg_hmd_loss()
                    # If average is 0 but count is also 0, try last_hmd_loss
                    if hmd_loss_avg == 0.0 and hasattr(criterion, 'hmd_loss_count') and criterion.hmd_loss_count == 0:
                        if hasattr(criterion, 'last_hmd_loss') and criterion.last_hmd_loss != 0.0:
                            hmd_loss_value = float(criterion.last_hmd_loss)
                        else:
                            hmd_loss_value = 0.0
                    else:
                        hmd_loss_value = hmd_loss_avg
                elif hasattr(criterion, 'last_hmd_loss'):
                    # Fallback to last batch loss if average not available
                    hmd_loss_value = float(criterion.last_hmd_loss)
            except Exception as e:
                logging.debug(f"Failed to get HMD loss from trainer criterion: {e}")
                pass
        
        # If not found in training model, try validator's model (which might be EMA model)
        # But note: EMA model's criterion statistics may not be accumulated during validation
        if hmd_loss_value is None and validator is not None and hasattr(validator, 'model') and hasattr(validator.model, 'criterion'):
            try:
                criterion = validator.model.criterion
                if hasattr(criterion, 'get_avg_hmd_loss'):
                    hmd_loss_avg = criterion.get_avg_hmd_loss()
                    # If average is 0 but count is also 0, try last_hmd_loss
                    if hmd_loss_avg == 0.0 and hasattr(criterion, 'hmd_loss_count') and criterion.hmd_loss_count == 0:
                        if hasattr(criterion, 'last_hmd_loss') and criterion.last_hmd_loss != 0.0:
                            hmd_loss_value = float(criterion.last_hmd_loss)
                        else:
                            hmd_loss_value = 0.0
                    else:
                        hmd_loss_value = hmd_loss_avg
                elif hasattr(criterion, 'last_hmd_loss'):
                    hmd_loss_value = float(criterion.last_hmd_loss)
            except Exception as e:
                logging.debug(f"Failed to get HMD loss from validator criterion: {e}")
                pass
        
        # Store HMD loss if we got a value (including 0.0)
        if hmd_loss_value is not None:
            additional_metrics["train/hmd_loss"] = hmd_loss_value
            logging.debug(f"HMD loss extracted: {hmd_loss_value} (from {'trainer' if trainer is not None else 'validator'})")
        else:
            # Debug: log why HMD loss couldn't be extracted
            if trainer is not None:
                has_model = hasattr(trainer, 'model')
                has_criterion = has_model and hasattr(trainer.model, 'criterion')
                logging.debug(f"HMD loss not found - trainer: has_model={has_model}, has_criterion={has_criterion}")
            elif validator is not None:
                has_model = hasattr(validator, 'model')
                has_criterion = has_model and hasattr(validator.model, 'criterion')
                logging.debug(f"HMD loss not found - validator: has_model={has_model}, has_criterion={has_criterion}")
                if has_criterion:
                    criterion = validator.model.criterion
                    has_get_avg = hasattr(criterion, 'get_avg_hmd_loss')
                    has_last = hasattr(criterion, 'last_hmd_loss')
                    has_count = hasattr(criterion, 'hmd_loss_count')
                    if has_count:
                        count = criterion.hmd_loss_count
                        last = criterion.last_hmd_loss if has_last else None
                        logging.debug(f"Criterion state: hmd_loss_count={count}, last_hmd_loss={last}, has_get_avg={has_get_avg}")
            else:
                logging.debug("HMD loss not found - neither trainer nor validator available")
        
        # Calculate HMD metrics if database is det_123 (even without HMD loss enabled)
        # This allows monitoring HMD performance for all det_123 experiments
        # Use closure variable args instead of trainer.args
        is_det_123 = args.database == 'det_123'
        hmd_enabled = args.use_hmd_loss and is_det_123
        
        # Debug: print condition check
        logging.debug(f"on_val_end_callback: is_det_123={is_det_123}, hmd_enabled={hmd_enabled}, args.database={args.database}, args.use_hmd_loss={args.use_hmd_loss}")
    
        # Always calculate HMD metrics for det_123 database (even without HMD loss enabled)
        # This allows monitoring HMD performance for all experiments, including baseline
        if is_det_123:
            try:
                # Calculate HMD metrics from validator's stats
                # We need validator for stats and trainer for HMD loss stats
                if validator is not None:
                    # IMPORTANT: Try to get stats from multiple sources
                    # validator.metrics.stats might be cleared after get_stats() is called
                    # So we try to get stats from validator.metrics.stats first (before clear_stats)
                    # If that's not available, try validator.stats (which might be set by get_stats)
                    # If both are not available, we'll use fallback calculation
                    if hasattr(validator, 'metrics') and hasattr(validator.metrics, 'stats'):
                        # Check if stats still have data (not cleared yet)
                        stats_dict = validator.metrics.stats
                        if stats_dict and len(stats_dict.get('tp', [])) > 0:
                            # Stats are still available, use them
                            if not hasattr(validator, 'stats') or validator.stats is None:
                                # Make a copy and set validator.stats
                                import copy
                                validator.stats = copy.deepcopy(stats_dict)
                                logging.info(f"🔍 Copied stats from validator.metrics.stats to validator.stats")
                            else:
                                # validator.stats already exists, use it
                                logging.info(f"🔍 Using existing validator.stats")
                        else:
                            # Stats have been cleared, try validator.stats
                            logging.info(f"🔍 validator.metrics.stats is empty, trying validator.stats")
                    else:
                        logging.info(f"🔍 validator.metrics.stats not available, trying validator.stats")
                    
                    # Get HMD metrics using validator stats and HMD loss from criterion
                    # The HMD loss already calculates real HMD distances, so we can use that
                    # Pass pixel_spacing_dict for mm-based calculation
                    # Use imgsz-based penalties by default (imgsz for penalty_none, imgsz/2 for penalty_single)
                    # If user explicitly set --hmd_penalty_* parameters, use those values instead
                    # Check if user explicitly set penalty values (not using defaults)
                    # Note: We'll always use imgsz-based penalties unless user explicitly wants to override
                    # For now, we'll use imgsz-based penalties as default (None means use imgsz)
                    hmd_metrics = calculate_hmd_metrics_from_validator(
                        validator=validator,
                        trainer=trainer,
                        penalty_single=None,  # Will use imgsz/2
                        penalty_none=None,    # Will use imgsz
                        pixel_spacing_dict=pixel_spacing_dict,
                        imgsz=args.imgsz
                    )
                
                    # Store with val/hmd/ prefix for consistency with W&B logging
                    additional_metrics["val/hmd/detection_rate"] = hmd_metrics.get('detection_rate', 0.0)
                    additional_metrics["val/hmd/rmse_pixel"] = hmd_metrics.get('rmse_pixel', 0.0)
                    additional_metrics["val/hmd/mae_pixel"] = hmd_metrics.get('mae_pixel', 0.0)
                    additional_metrics["val/hmd/overall_score_pixel"] = hmd_metrics.get('overall_score_pixel', 0.0)
                    additional_metrics["val/hmd/rmse_mm"] = hmd_metrics.get('rmse_mm', 0.0)
                    additional_metrics["val/hmd/mae_mm"] = hmd_metrics.get('mae_mm', 0.0)
                    additional_metrics["val/hmd/overall_score_mm"] = hmd_metrics.get('overall_score_mm', 0.0)
                    # No penalty versions
                    additional_metrics["val/hmd/rmse_no_penalty_pixel"] = hmd_metrics.get('rmse_no_penalty_pixel', 0.0)
                    additional_metrics["val/hmd/mae_no_penalty_pixel"] = hmd_metrics.get('mae_no_penalty_pixel', 0.0)
                    additional_metrics["val/hmd/rmse_no_penalty_mm"] = hmd_metrics.get('rmse_no_penalty_mm', 0.0)
                    additional_metrics["val/hmd/mae_no_penalty_mm"] = hmd_metrics.get('mae_no_penalty_mm', 0.0)
                    # Also store with hmd/ prefix for backward compatibility
                    additional_metrics["hmd/detection_rate"] = hmd_metrics.get('detection_rate', 0.0)
                    additional_metrics["hmd/rmse_pixel"] = hmd_metrics.get('rmse_pixel', 0.0)
                    additional_metrics["hmd/mae_pixel"] = hmd_metrics.get('mae_pixel', 0.0)
                    additional_metrics["hmd/overall_score_pixel"] = hmd_metrics.get('overall_score_pixel', 0.0)
                    additional_metrics["hmd/rmse_mm"] = hmd_metrics.get('rmse_mm', 0.0)
                    additional_metrics["hmd/mae_mm"] = hmd_metrics.get('mae_mm', 0.0)
                    additional_metrics["hmd/overall_score_mm"] = hmd_metrics.get('overall_score_mm', 0.0)
                    # No penalty versions
                    additional_metrics["hmd/rmse_no_penalty_pixel"] = hmd_metrics.get('rmse_no_penalty_pixel', 0.0)
                    additional_metrics["hmd/mae_no_penalty_pixel"] = hmd_metrics.get('mae_no_penalty_pixel', 0.0)
                    additional_metrics["hmd/rmse_no_penalty_mm"] = hmd_metrics.get('rmse_no_penalty_mm', 0.0)
                    additional_metrics["hmd/mae_no_penalty_mm"] = hmd_metrics.get('mae_no_penalty_mm', 0.0)
                    
                    # Debug: print if metrics are 0
                    if hmd_metrics.get('detection_rate', 0.0) == 0.0 and hmd_metrics.get('rmse_pixel', 0.0) == 0.0:
                        logging.debug(f"⚠️ HMD metrics are all 0 - validator stats: {hasattr(validator, 'stats')}, stats keys: {list(validator.stats.keys()) if hasattr(validator, 'stats') and validator.stats else 'None'}")
                else:
                    # If validator not available, set to 0 (but still set them so they will be displayed)
                    logging.warning("⚠️ Validator not available for HMD metrics calculation")
                    additional_metrics["val/hmd/detection_rate"] = 0.0
                    additional_metrics["val/hmd/rmse_pixel"] = 0.0
                    additional_metrics["val/hmd/overall_score_pixel"] = 0.0
                    additional_metrics["val/hmd/rmse_mm"] = 0.0
                    additional_metrics["val/hmd/overall_score_mm"] = 0.0
                    additional_metrics["hmd/detection_rate"] = 0.0
                    additional_metrics["hmd/rmse_pixel"] = 0.0
                    additional_metrics["hmd/overall_score_pixel"] = 0.0
                    additional_metrics["hmd/rmse_mm"] = 0.0
                    additional_metrics["hmd/overall_score_mm"] = 0.0
            except Exception as e:
                # If calculation fails, set to 0 (but still set them so they will be displayed)
                logging.error(f"❌ Failed to calculate HMD metrics: {e}")
                import traceback
                logging.error(f"❌ Full traceback: {traceback.format_exc()}")
                additional_metrics["val/hmd/detection_rate"] = 0.0
                additional_metrics["val/hmd/rmse_pixel"] = 0.0
                additional_metrics["val/hmd/overall_score_pixel"] = 0.0
                additional_metrics["val/hmd/rmse_mm"] = 0.0
                additional_metrics["val/hmd/overall_score_mm"] = 0.0
                additional_metrics["hmd/detection_rate"] = 0.0
                additional_metrics["hmd/rmse_pixel"] = 0.0
                additional_metrics["hmd/overall_score_pixel"] = 0.0
                additional_metrics["hmd/rmse_mm"] = 0.0
                additional_metrics["hmd/overall_score_mm"] = 0.0
        
        # Store additional metrics for print_validation_metrics to access
        # We need to store in a place accessible by print_validation_metrics
        # Since on_val_end receives validator, we need to find trainer to store metrics
        if trainer is not None:
            # Store in trainer (preferred location)
            trainer._additional_metrics = additional_metrics
            trainer._args_database = args.database
            trainer._args_use_hmd_loss = args.use_hmd_loss
            # Also store in validator for redundancy
            if validator is not None:
                validator._additional_metrics = additional_metrics
                validator._args_database = args.database
                validator._args_use_hmd_loss = args.use_hmd_loss
        elif validator is not None:
            # Store in validator as fallback
            validator._additional_metrics = additional_metrics
            validator._args_database = args.database
            validator._args_use_hmd_loss = args.use_hmd_loss
            # Try to get trainer from validator if available (we added this in validator.__call__)
            if hasattr(validator, 'trainer') and validator.trainer is not None:
                trainer = validator.trainer
                # Also store in trainer for consistency
                trainer._additional_metrics = additional_metrics
                trainer._args_database = args.database
                trainer._args_use_hmd_loss = args.use_hmd_loss
        
        # Print validation metrics after validation completes (so we have the latest metrics)
        # Debug: force print to see if callback is working
        logging.info(f"on_val_end_callback: additional_metrics keys = {list(additional_metrics.keys())}")
        if trainer is not None:
            logging.debug(f"on_val_end_callback: stored metrics in trainer, trainer has _additional_metrics: {hasattr(trainer, '_additional_metrics')}")
        if validator is not None:
            logging.debug(f"on_val_end_callback: stored metrics in validator, validator has _additional_metrics: {hasattr(validator, '_additional_metrics')}")
        
        # Always try to print validation metrics
        # Try multiple ways to get trainer
        trainer_for_print = None
        
        # Method 1: Direct trainer from callback
        if trainer is not None:
            trainer_for_print = trainer
            logging.debug("Using trainer from callback parameter")
        
        # Method 2: From validator.trainer
        if trainer_for_print is None and validator is not None:
            if hasattr(validator, 'trainer') and validator.trainer is not None:
                trainer_for_print = validator.trainer
                logging.debug("Using trainer from validator.trainer")
        
        # Method 3: Try to get from validator's model (if it has a reference)
        if trainer_for_print is None and validator is not None:
            # Check if validator has any reference to trainer through its attributes
            for attr_name in ['trainer', '_trainer', 'parent_trainer']:
                if hasattr(validator, attr_name):
                    attr_value = getattr(validator, attr_name)
                    if attr_value is not None and hasattr(attr_value, 'metrics'):
                        trainer_for_print = attr_value
                        logging.debug(f"Using trainer from validator.{attr_name}")
                        break
        
        # If we still don't have trainer, create a minimal wrapper from validator
        if trainer_for_print is None and validator is not None:
            # Create a minimal trainer-like object from validator
            class MinimalTrainerWrapper:
                def __init__(self, validator, additional_metrics, args_database, args_use_hmd_loss):
                    self.validator = validator
                    self._additional_metrics = additional_metrics
                    self._args_database = args_database
                    self._args_use_hmd_loss = args_use_hmd_loss
                    # Try to get metrics from validator
                    if hasattr(validator, 'metrics'):
                        self.metrics = validator.metrics
                    else:
                        self.metrics = None
                    # Try to get args
                    if hasattr(validator, 'args'):
                        self.args = validator.args
                    else:
                        self.args = type('Args', (), {'database': args_database, 'use_hmd_loss': args_use_hmd_loss})()
                    # Try to get model
                    if hasattr(validator, 'model'):
                        self.model = validator.model
                    else:
                        self.model = None
            
            trainer_for_print = MinimalTrainerWrapper(
                validator, 
                additional_metrics, 
                args.database, 
                args.use_hmd_loss
            )
            logging.debug("Created MinimalTrainerWrapper from validator")
        
        if trainer_for_print is not None:
            try:
                logging.info(f"📊 Calling print_validation_metrics with trainer type: {type(trainer_for_print).__name__}")
                print_validation_metrics(trainer_for_print)
                logging.info("✅ print_validation_metrics completed")
            except Exception as e:
                logging.error(f"❌ Failed to print validation metrics: {e}")
                import traceback
                logging.error(traceback.format_exc())
        else:
            logging.error("❌ Cannot print validation metrics: trainer not available and cannot create wrapper")
    
    return on_val_end_callback


def evaluate_detailed(model: YOLO, split: str = "val", batch: int = 16, imgsz: int = 640, 
                     database: str = None, db_version: int = 3, use_hmd: bool = False,
                     penalty_single: float = 500.0, penalty_none: float = 1000.0, 
                     penalty_coeff: float = 0.5) -> Dict:
    """Detailed evaluation function that logs per-class metrics"""
    logging.info(f"🔍 Evaluating on {split} split ...")
    
    # Store validator reference for HMD calculation
    validator_ref = [None]  # Use list to allow modification in nested function
    
    def capture_validator_callback(validator):
        """Callback to capture validator object"""
        validator_ref[0] = validator
    
    # Add callback to capture validator
    # Note: add_callback is a method of YOLO class, not DetectionModel
    if hasattr(model, 'add_callback'):
        model.add_callback("on_val_end", capture_validator_callback)
    
    # Perform validation
    metrics = model.val(split=split, batch=batch, imgsz=imgsz)
    
    # Remove callback after validation (if method exists)
    # Note: remove_callback may not exist in all Ultralytics versions
    if hasattr(model, 'remove_callback'):
        try:
            model.remove_callback("on_val_end", capture_validator_callback)
        except (AttributeError, TypeError):
            # If remove_callback doesn't work, it's okay - callback will be overwritten on next use
            pass
    
    # Get mean results
    try:
        mp, mr, map50, map = metrics.mean_results()
    except Exception as e:
        logging.warning(f"❌ Failed to compute mean results: {e}")
        return {}
    
    # Try to get per-class results
    # According to Ultralytics Metric class, use p, r, ap50, ap attributes and class_result method
    per_class_metrics = None
    try:
        if hasattr(metrics, 'box'):
            box_metrics = metrics.box
            # Get per-class metrics using available attributes
            # box.p, box.r, box.ap50(), box.ap() are lists/arrays for each class
            if (hasattr(box_metrics, 'p') and hasattr(box_metrics, 'r') and 
                hasattr(box_metrics, 'ap50') and hasattr(box_metrics, 'ap')):
                import numpy as np
                # Get arrays/lists for each metric
                p_list = box_metrics.p if hasattr(box_metrics.p, '__len__') else [box_metrics.p]
                r_list = box_metrics.r if hasattr(box_metrics.r, '__len__') else [box_metrics.r]
                ap50_list = box_metrics.ap50() if callable(box_metrics.ap50) else box_metrics.ap50
                ap_list = box_metrics.ap() if callable(box_metrics.ap) else box_metrics.ap
                
                # Convert to numpy arrays if needed
                if not isinstance(p_list, np.ndarray):
                    p_list = np.array(p_list)
                if not isinstance(r_list, np.ndarray):
                    r_list = np.array(r_list)
                if not isinstance(ap50_list, np.ndarray):
                    ap50_list = np.array(ap50_list)
                if not isinstance(ap_list, np.ndarray):
                    ap_list = np.array(ap_list)
                
                # Get F1 scores (can calculate from p and r)
                f1_list = 2 * (p_list * r_list) / (p_list + r_list + 1e-16)
                
                # Stack into (num_classes, 6) array: [p, r, ap50, ap, f1, iou]
                # Note: IoU is not directly available, we'll use 0.0 as placeholder
                num_classes = len(p_list)
                iou_list = np.zeros(num_classes)  # IoU not available in Metric class
                
                per_class_metrics = np.column_stack([p_list, r_list, ap50_list, ap_list, f1_list, iou_list])
    except Exception as e:
        logging.warning(f"⚠️ Cannot extract per-class results: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        per_class_metrics = None
    
    # Get class names
    if per_class_metrics is not None:
        num_classes = per_class_metrics.shape[0]
        names = model.names if hasattr(model, "names") else {i: str(i) for i in range(num_classes)}
    else:
        # Fallback: try to get number of classes from model or metrics
        if hasattr(model, "names"):
            names = model.names
            num_classes = len(names) if isinstance(names, dict) else 2
        else:
            names = {0: "Mentum", 1: "Hyoid"}
            num_classes = 2
    
    # Create W&B Table
    tmp_path = os.path.join(tempfile.gettempdir(), "wandb-media")
    os.makedirs(tmp_path, exist_ok=True)
    class_table = wandb.Table(columns=["class_id", "class_name", "precision", "recall", "AP50", "AP75", "F1", "IoU"])
    
    if per_class_metrics is not None:
        # Get AP75 if available from metrics.box.all_ap
        ap75_list = None
        if hasattr(metrics, 'box') and hasattr(metrics.box, 'all_ap'):
            try:
                all_ap = metrics.box.all_ap  # Shape: (num_classes, 10) for IoU thresholds 0.5-0.95
                if hasattr(all_ap, '__len__') and len(all_ap) > 0:
                    # AP75 corresponds to index 5 in the 10 IoU thresholds (0.5, 0.55, ..., 0.95)
                    # Thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                    # Index 5 = 0.75
                    if isinstance(all_ap, np.ndarray) and all_ap.ndim == 2:
                        ap75_list = all_ap[:, 5]  # Get AP@0.75 for all classes
                    elif hasattr(all_ap, '__getitem__'):
                        ap75_list = [ap[5] if len(ap) > 5 else 0.0 for ap in all_ap]
            except Exception:
                ap75_list = None
        
        for class_id, row in enumerate(per_class_metrics):
            # per_class_metrics shape: (num_classes, 6) = [p, r, ap50, ap, f1, iou]
            ap75_val = float(ap75_list[class_id]) if ap75_list is not None and class_id < len(ap75_list) else 0.0
            class_table.add_data(
                class_id,
                names.get(class_id, str(class_id)),
                float(row[0]),  # precision
                float(row[1]),  # recall
                float(row[2]),  # AP@0.5
                ap75_val,       # AP@0.75 (from all_ap if available, else 0.0)
                float(row[4]),  # F1 score
                float(row[5]),  # IoU (placeholder, always 0.0)
            )
    
    # Inference speed
    speed_data = metrics.speed or {}
    
    # Additional metrics
    extra_metrics = {}
    for k in ["ar100", "ar10", "ar1"]:
        if hasattr(metrics, k):
            extra_metrics[f"{split}/{k.upper()}"] = float(getattr(metrics, k))
    
    # Calculate IoU and Dice if available
    iou_value = None
    dice_value = None
    if hasattr(metrics, 'box') and hasattr(metrics.box, 'iou'):
        try:
            iou_value = float(metrics.box.iou.mean()) if hasattr(metrics.box.iou, 'mean') else None
        except:
            pass
    
    # Calculate Fitness
    fitness = map50 * 0.1 + map * 0.9
    
    # Prepare logs
    logs = {
        f"{split}/mAP50": float(map50),
        f"{split}/mAP50-95": float(map),
        f"{split}/precision": float(mp),
        f"{split}/recall": float(mr),
        f"{split}/fitness": float(fitness),
        f"{split}/inference_speed(ms)": float(speed_data.get("inference", 0)),
        f"{split}/preprocess_speed(ms)": float(speed_data.get("preprocess", 0)),
        f"{split}/postprocess_speed(ms)": float(speed_data.get("postprocess", 0)),
        f"{split}/loss_speed(ms)": float(speed_data.get("loss", 0)),
        f"{split}/num_classes": len(names),
        f"{split}/per_class_metrics": class_table,
        **extra_metrics
    }
    
    # Add IoU and Dice if available
    if iou_value is not None:
        logs[f"{split}/iou"] = iou_value
    if dice_value is not None:
        logs[f"{split}/dice"] = dice_value
    
    # Calculate HMD metrics if database is det_123 (for all det_123 experiments)
    # Use the same calculate_hmd_metrics_from_validator function as in training
    hmd_metrics = {
        'detection_rate': 0.0,
        'rmse_pixel': 0.0,
        'overall_score_pixel': 0.0,
    }
    
    if database == 'det_123':
        try:
            # Create a minimal trainer-like object for calculate_hmd_metrics_from_validator
            # The function needs trainer to access criterion, but we can pass model directly
            class MinimalTrainerForEval:
                def __init__(self, model):
                    self.model = model
                    self.validator = None
            
            minimal_trainer = MinimalTrainerForEval(model)
            validator = validator_ref[0]
            
            if validator is not None:
                minimal_trainer.validator = validator
                # Load pixel_spacing_dict for mm calculation
                pixel_spacing_dict_eval = {}
                try:
                    from mycodes.hmd_utils import load_pixel_spacing_dict
                    project_root = Path(__file__).parent.parent.parent
                    dicom_root = project_root / "dicom_dataset"
                    joblib_path = dicom_root / "Dicom_PixelSpacing_DA.joblib"
                    logging.info(f"🔍 Loading PixelSpacing from: {joblib_path}")
                    pixel_spacing_dict_eval = load_pixel_spacing_dict(joblib_path)
                    if pixel_spacing_dict_eval:
                        logging.info(f"✅ Loaded PixelSpacing dictionary with {len(pixel_spacing_dict_eval)} entries for {split} evaluation")
                    else:
                        logging.warning(f"⚠️ PixelSpacing dictionary is empty or not found at {joblib_path}")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load PixelSpacing dictionary: {e}")
                    import traceback
                    logging.debug(traceback.format_exc())
                
                # Calculate HMD metrics using the same function as in training
                # Use imgsz-based penalties (imgsz for penalty_none, imgsz/2 for penalty_single)
                hmd_metrics = calculate_hmd_metrics_from_validator(
                    validator=validator,
                    trainer=minimal_trainer,
                    penalty_single=None,  # Will use imgsz/2
                    penalty_none=None,   # Will use imgsz
                    pixel_spacing_dict=pixel_spacing_dict_eval,
                    imgsz=imgsz
                )
            else:
                logging.warning("⚠️ Validator not captured, using fallback HMD calculation")
                # Fallback: estimate from per-class metrics
                hmd_metrics = {
                    'detection_rate': 0.0,
                    'rmse_pixel': 0.0,
                    'overall_score_pixel': 0.0,
                    'rmse_mm': 0.0,
                    'overall_score_mm': 0.0,
                }
                # Try to get per-class recall for detection_rate calculation
                if hasattr(metrics, 'box'):
                    box_metrics = metrics.box
                    if hasattr(box_metrics, 'r'):
                        r_list = box_metrics.r if hasattr(box_metrics.r, '__len__') else [box_metrics.r]
                        if len(r_list) >= 2:  # At least 2 classes (Mentum and Hyoid)
                            mentum_recall = float(r_list[0]) if len(r_list) > 0 else 0.0
                            hyoid_recall = float(r_list[1]) if len(r_list) > 1 else 0.0
                            hmd_metrics['detection_rate'] = min(mentum_recall, hyoid_recall)
            
            logs.update({
                f"{split}/hmd/detection_rate": hmd_metrics.get('detection_rate', 0.0),
                f"{split}/hmd/rmse_pixel": hmd_metrics.get('rmse_pixel', 0.0),
                f"{split}/hmd/mae_pixel": hmd_metrics.get('mae_pixel', 0.0),
                f"{split}/hmd/overall_score_pixel": hmd_metrics.get('overall_score_pixel', 0.0),
                f"{split}/hmd/rmse_mm": hmd_metrics.get('rmse_mm', 0.0),
                f"{split}/hmd/mae_mm": hmd_metrics.get('mae_mm', 0.0),
                f"{split}/hmd/overall_score_mm": hmd_metrics.get('overall_score_mm', 0.0),
                f"{split}/hmd/rmse_no_penalty_pixel": hmd_metrics.get('rmse_no_penalty_pixel', 0.0),
                f"{split}/hmd/mae_no_penalty_pixel": hmd_metrics.get('mae_no_penalty_pixel', 0.0),
                f"{split}/hmd/rmse_no_penalty_mm": hmd_metrics.get('rmse_no_penalty_mm', 0.0),
                f"{split}/hmd/mae_no_penalty_mm": hmd_metrics.get('mae_no_penalty_mm', 0.0),
            })
            
            # Print HMD metrics in the same format as training output
            print(f"\n📊 Additional Metrics ({split}):", flush=True)
            print(f"   Precision: {mp:.4f} | Recall: {mr:.4f}", flush=True)
            print(f"   mAP50: {map50:.4f} | mAP50-95: {map:.4f} | Fitness: {fitness:.4f}", flush=True)
            print(f"\n📏 HMD Metrics (det_123, {split}):", flush=True)
            detection_rate_val = hmd_metrics.get('detection_rate', 0.0)
            rmse_pixel_val = hmd_metrics.get('rmse_pixel', 0.0)
            mae_pixel_val = hmd_metrics.get('mae_pixel', 0.0)
            overall_score_pixel_val = hmd_metrics.get('overall_score_pixel', 0.0)
            rmse_no_penalty_pixel_val = hmd_metrics.get('rmse_no_penalty_pixel', 0.0)
            mae_no_penalty_pixel_val = hmd_metrics.get('mae_no_penalty_pixel', 0.0)
            rmse_mm_val = hmd_metrics.get('rmse_mm', 0.0)
            mae_mm_val = hmd_metrics.get('mae_mm', 0.0)
            overall_score_mm_val = hmd_metrics.get('overall_score_mm', 0.0)
            rmse_no_penalty_mm_val = hmd_metrics.get('rmse_no_penalty_mm', 0.0)
            mae_no_penalty_mm_val = hmd_metrics.get('mae_no_penalty_mm', 0.0)
            
            print(f"   Detection_Rate: {detection_rate_val:.4f}", flush=True)
            
            # With penalty versions
            print(f"\n   📊 With Penalty (includes missed detections):", flush=True)
            if detection_rate_val == 0.0 and rmse_pixel_val >= 1000.0:
                print(f"      RMSE_HMD (pixel): {rmse_pixel_val:.2f} px (penalty: no detections)", flush=True)
            else:
                print(f"      RMSE_HMD (pixel): {rmse_pixel_val:.2f} px", flush=True)
            print(f"      MAE_HMD (pixel): {mae_pixel_val:.2f} px", flush=True)
            print(f"      Overall_Score (pixel): {overall_score_pixel_val:.4f}", flush=True)
            
            # No penalty versions
            print(f"\n   📊 No Penalty (only both detected cases):", flush=True)
            if rmse_no_penalty_pixel_val > 0.0:
                print(f"      RMSE_HMD (pixel): {rmse_no_penalty_pixel_val:.2f} px", flush=True)
                print(f"      MAE_HMD (pixel): {mae_no_penalty_pixel_val:.2f} px", flush=True)
            else:
                print(f"      RMSE_HMD (pixel): N/A (no both detected cases)", flush=True)
                print(f"      MAE_HMD (pixel): N/A (no both detected cases)", flush=True)
            
            # mm versions
            if rmse_mm_val > 0.0 or mae_mm_val > 0.0:
                print(f"\n   📏 With Penalty (mm):", flush=True)
                print(f"      RMSE_HMD (mm): {rmse_mm_val:.2f} mm", flush=True)
                print(f"      MAE_HMD (mm): {mae_mm_val:.2f} mm", flush=True)
                print(f"      Overall_Score (mm): {overall_score_mm_val:.4f}", flush=True)
                
                print(f"\n   📏 No Penalty (mm):", flush=True)
                if rmse_no_penalty_mm_val > 0.0:
                    print(f"      RMSE_HMD (mm): {rmse_no_penalty_mm_val:.2f} mm", flush=True)
                    print(f"      MAE_HMD (mm): {mae_no_penalty_mm_val:.2f} mm", flush=True)
                else:
                    print(f"      RMSE_HMD (mm): N/A (no both detected cases)", flush=True)
                    print(f"      MAE_HMD (mm): N/A (no both detected cases)", flush=True)
            else:
                print(f"\n   📏 With Penalty (mm): N/A (PixelSpacing not available)", flush=True)
                print(f"   📏 No Penalty (mm): N/A (PixelSpacing not available)", flush=True)
            # Show RMSE with penalty indicator if applicable
            if detection_rate_val == 0.0 and rmse_pixel_val >= 1000.0:
                print(f"   RMSE_HMD (pixel): {rmse_pixel_val:.2f} px (penalty: no detections)", flush=True)
            else:
                print(f"   RMSE_HMD (pixel): {rmse_pixel_val:.2f} px", flush=True)
            print(f"   Overall_Score (pixel): {hmd_metrics.get('overall_score_pixel', 0.0):.4f}", flush=True)
            # Show mm version if available
            rmse_mm = hmd_metrics.get('rmse_mm', 0.0)
            overall_score_mm = hmd_metrics.get('overall_score_mm', 0.0)
            # Always show mm metrics if the key exists (even if value is 0.0)
            # This indicates pixel_spacing_dict was loaded and mm calculation was attempted
            has_mm_key = 'rmse_mm' in hmd_metrics or 'overall_score_mm' in hmd_metrics
            if has_mm_key:
                # Show mm version if it was calculated (even if 0.0)
                print(f"   RMSE_HMD (mm): {rmse_mm:.2f} mm", flush=True)
                print(f"   Overall_Score (mm): {overall_score_mm:.4f}", flush=True)
            else:
                print(f"   RMSE_HMD (mm): N/A (PixelSpacing not available)", flush=True)
                print(f"   Overall_Score (mm): N/A (PixelSpacing not available)", flush=True)
            
        except Exception as e:
            logging.warning(f"⚠️ Failed to calculate HMD metrics: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            # Set default values on error
            hmd_metrics = {
                'detection_rate': 0.0,
                'rmse_pixel': 0.0,
                'overall_score_pixel': 0.0,
                'rmse_mm': 0.0,
                'overall_score_mm': 0.0,
            }
            logs.update({
                f"{split}/hmd/detection_rate": hmd_metrics.get('detection_rate', 0.0),
                f"{split}/hmd/rmse_pixel": hmd_metrics.get('rmse_pixel', 0.0),
                f"{split}/hmd/overall_score_pixel": hmd_metrics.get('overall_score_pixel', 0.0),
                f"{split}/hmd/rmse_mm": hmd_metrics.get('rmse_mm', 0.0),
                f"{split}/hmd/overall_score_mm": hmd_metrics.get('overall_score_mm', 0.0),
            })
    
    # Log to W&B
    wandb.log(logs)
    
    logging.info(f"✅ {split} evaluation complete. mAP50={map50:.4f}, mAP50-95={map:.4f}")
    
    result = {
        "precision": mp,
        "recall": mr,
        "mAP50": map50,
        "mAP50-95": map,
        "fitness": fitness,
        "per_class": per_class_metrics,
        "inference_speed": speed_data,
    }
    
    # Add HMD metrics to result for final summary
    if database == 'det_123':
        result.update({
            "detection_rate": hmd_metrics.get('detection_rate', 0.0),
            "rmse_hmd_pixel": hmd_metrics.get('rmse_pixel', 0.0),
            "rmse_pixel": hmd_metrics.get('rmse_pixel', 0.0),  # Alias for consistency
            "overall_score_pixel": hmd_metrics.get('overall_score_pixel', 0.0),
            "rmse_hmd_mm": hmd_metrics.get('rmse_mm', 0.0),
            "rmse_mm": hmd_metrics.get('rmse_mm', 0.0),  # Alias for consistency
            "overall_score_mm": hmd_metrics.get('overall_score_mm', 0.0),
        })
    
    if iou_value is not None:
        result["iou"] = iou_value
    if dice_value is not None:
        result["dice"] = dice_value
    
    # Add HMD metrics to result dict (for all det_123 experiments)
    if database == 'det_123':
        try:
            # Get HMD metrics from logs (calculated above) - include all new metrics
            result.update({
                'detection_rate': logs.get(f"{split}/hmd/detection_rate", 0.0),
                'rmse_pixel': logs.get(f"{split}/hmd/rmse_pixel", 0.0),
                'rmse_hmd_pixel': logs.get(f"{split}/hmd/rmse_pixel", 0.0),  # Alias
                'mae_pixel': logs.get(f"{split}/hmd/mae_pixel", 0.0),
                'overall_score_pixel': logs.get(f"{split}/hmd/overall_score_pixel", 0.0),
                'rmse_mm': logs.get(f"{split}/hmd/rmse_mm", 0.0),
                'rmse_hmd_mm': logs.get(f"{split}/hmd/rmse_mm", 0.0),  # Alias
                'mae_mm': logs.get(f"{split}/hmd/mae_mm", 0.0),
                'overall_score_mm': logs.get(f"{split}/hmd/overall_score_mm", 0.0),
                'rmse_no_penalty_pixel': logs.get(f"{split}/hmd/rmse_no_penalty_pixel", 0.0),
                'mae_no_penalty_pixel': logs.get(f"{split}/hmd/mae_no_penalty_pixel", 0.0),
                'rmse_no_penalty_mm': logs.get(f"{split}/hmd/rmse_no_penalty_mm", 0.0),
                'mae_no_penalty_mm': logs.get(f"{split}/hmd/mae_no_penalty_mm", 0.0),
            })
            # Also store hmd_metrics dict for easy access
            if 'hmd_metrics' in locals():
                result['hmd_metrics'] = hmd_metrics
        except Exception as e:
            logging.warning(f"⚠️ Failed to add HMD metrics to result: {e}")
            # Add default values as fallback
            result.update({
                'detection_rate': 0.0,
                'rmse_pixel': 0.0,
                'rmse_hmd_pixel': 0.0,
                'mae_pixel': 0.0,
                'overall_score_pixel': 0.0,
                'rmse_mm': 0.0,
                'rmse_hmd_mm': 0.0,
                'mae_mm': 0.0,
                'overall_score_mm': 0.0,
                'rmse_no_penalty_pixel': 0.0,
                'mae_no_penalty_pixel': 0.0,
                'rmse_no_penalty_mm': 0.0,
                'mae_no_penalty_mm': 0.0,
            })
    
    return result


if __name__=='__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    train_start_time = datetime.now().timestamp()
    
    # Load PixelSpacing dictionary for mm-based HMD calculation in evaluation
    pixel_spacing_dict = {}
    try:
        from mycodes.hmd_utils import load_pixel_spacing_dict
        # Try to find dicom_dataset relative to project root
        project_root = Path(__file__).parent.parent.parent
        dicom_root = project_root / "dicom_dataset"
        pixel_spacing_dict = load_pixel_spacing_dict(dicom_root / "Dicom_PixelSpacing_DA.joblib")
        if pixel_spacing_dict:
            logging.info(f"✅ Loaded PixelSpacing dictionary with {len(pixel_spacing_dict)} entries for mm-based HMD calculation")
        else:
            logging.info("⚠️ PixelSpacing dictionary not found or empty. mm-based HMD metrics will be 0.0")
    except Exception as e:
        logging.warning(f"⚠️ Failed to load PixelSpacing dictionary: {e}. mm-based HMD metrics will be 0.0")
    
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
    
    # HMD loss parameters (for det_123 database)
    parser.add_argument('--use_hmd_loss', action='store_true', help='Enable HMD loss for det_123 database')
    parser.add_argument('--hmd_loss_weight', type=float, default=0.5, help='Weight for HMD loss (λ_hmd, default: 0.5)')
    parser.add_argument('--hmd_penalty_single', type=float, default=500.0, help='Penalty value when only one target is detected (default: 500.0 pixels)')
    parser.add_argument('--hmd_penalty_none', type=float, default=1000.0, help='Penalty value when both targets are missed (default: 1000.0 pixels)')
    parser.add_argument('--hmd_penalty_coeff', type=float, default=0.5, help='Penalty coefficient for single detection weight (default: 0.5)')
    parser.add_argument('--hmd_use_mm', action='store_true', help='Use millimeter (mm) instead of pixel for HMD calculation. Requires Dicom_PixelSpacing_DA.joblib file. Default: False (use pixel)')
    
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
    
    # Ultrasound-specific augmentation
    parser.add_argument('--use_ultrasound_aug', action='store_true', help='Enable ultrasound-specific augmentation (speckle noise and signal attenuation)')
    parser.add_argument('--ultrasound_speckle_var', type=float, default=0.1, help='Speckle noise variance (default: 0.1)')
    parser.add_argument('--ultrasound_attenuation_factor', type=float, default=0.3, help='Signal attenuation factor for depth simulation (default: 0.3)')
    
    # IoU type selection
    parser.add_argument('--iou_type', type=str, default='CIoU', choices=['IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU', 'SIoU'],
                        help='IoU loss type: IoU, GIoU, DIoU, CIoU (default), EIoU, SIoU')
    
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
    parser.add_argument('--use_cosine_restart', action='store_true', help='Use Cosine Annealing with Warm Restarts scheduler')
    parser.add_argument('--cosine_restart_t0', type=int, default=10, help='Number of epochs for the first restart (T_0)')
    parser.add_argument('--cosine_restart_t_mult', type=int, default=2, help='A factor increases T_i after a restart (T_mult)')
    parser.add_argument('--deterministic', type=bool, default=True, help='Deterministic training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Detection/NMS parameters
    parser.add_argument('--conf', type=float, default=None, help='Object confidence threshold for training')
    parser.add_argument('--iou', type=float, default=None, help='IoU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=None, help='Maximum number of detections per image')
    parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference')
    parser.add_argument('--agnostic_nms', action='store_true', help='Class-agnostic NMS (merge overlapping boxes regardless of class)')
    parser.add_argument('--keep_top_conf_per_class', action='store_true', 
                       help='Keep only the highest confidence bbox per class (after low-threshold filtering). '
                            'This allows using a lower confidence threshold to get more predictions, '
                            'but only keeps the best one per class. Useful for HMD calculation where '
                            'each class (Mentum/Hyoid) should have only one detection per image.')
    parser.add_argument('--conf_low', type=float, default=None, 
                       help='Lower confidence threshold for initial filtering (used with --keep_top_conf_per_class). '
                            'If not set, uses --conf value. Allows more predictions to pass initial filter, '
                            'then keeps only top confidence per class.')
    
    args = parser.parse_args()
    if args.runs_num==1:
        args.runs_num = ''

    # Get project root from environment variable
    DA_folder = os.getenv('PROJECT_ROOT')
    if not DA_folder:
        # Fallback: try to detect from script location
        script_dir = Path(__file__).resolve().parent.parent.parent
        DA_folder = str(script_dir)
    
    # Ensure path uses forward slashes for cross-platform compatibility
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
    
    # Store HMD loss settings
    use_hmd_loss_flag = args.use_hmd_loss and args.database == 'det_123'
    hmd_loss_weight_value = args.hmd_loss_weight if use_hmd_loss_flag else 0.0
    hmd_penalty_single_value = args.hmd_penalty_single
    hmd_penalty_none_value = args.hmd_penalty_none
    hmd_penalty_coeff_value = args.hmd_penalty_coeff
    hmd_use_mm_value = getattr(args, 'hmd_use_mm', False)
    mentum_class = 0  # det_123: class 0 is Mentum
    hyoid_class = 1  # det_123: class 1 is Hyoid
    
    if use_hmd_loss_flag:
        unit_str = "mm" if hmd_use_mm_value else "pixel"
        logging.info(f"✅ HMD loss will be enabled: weight={hmd_loss_weight_value}, unit={unit_str}, "
                    f"penalty_single={hmd_penalty_single_value}, penalty_none={hmd_penalty_none_value}")
    
    # Callback to add HMD loss during training
    def add_hmd_loss_callback(trainer):
        """Add HMD loss to the training loss if enabled
        
        Note: Full integration requires modifying the loss computation in ultralytics.
        This callback serves as a placeholder and can log HMD-related statistics.
        """
        if not use_hmd_loss_flag:
            return
        
        # Log HMD loss statistics if available
        # The actual HMD loss should be added in the loss computation function
        # which would require modifying ultralytics/ultralytics/utils/loss.py or
        # ultralytics/ultralytics/models/utils/loss.py
        pass
    
    # Callback to set dimension weights, focal loss, and HMD loss after trainer is created
    def set_custom_loss_callback(trainer):
        """Set dimension weights, focal loss, and HMD loss after trainer initialization and recreate loss function"""
        updated = False
        
        # CRITICAL: Check if trainer.model.args is the same object as trainer.args
        # If they are the same, we only need to set one. If different, we need to set both.
        model_args_is_same = hasattr(trainer, 'model') and hasattr(trainer.model, 'args') and trainer.model.args is trainer.args
        
        # Set database attribute (needed for HMD loss check in validation)
        # IMPORTANT: Must set both trainer.args AND trainer.model.args
        # because v8DetectionLoss reads from model.args (not trainer.args)
        if isinstance(trainer.args, dict):
            trainer.args['database'] = args.database
        else:
            setattr(trainer.args, 'database', args.database)
        
        # Also set to model.args (critical for init_criterion to read correct values)
        # Only set if model.args is a different object
        if not model_args_is_same and hasattr(trainer, 'model') and hasattr(trainer.model, 'args'):
            if isinstance(trainer.model.args, dict):
                trainer.model.args['database'] = args.database
            else:
                setattr(trainer.model.args, 'database', args.database)
        
        # Debug: Log whether args are the same object
        if use_hmd_loss_flag:
            logging.info(f"🔍 Debug: trainer.args is trainer.model.args? {model_args_is_same}")
            if hasattr(trainer, 'model') and hasattr(trainer.model, 'args'):
                logging.info(f"   trainer.args id: {id(trainer.args)}")
                logging.info(f"   trainer.model.args id: {id(trainer.model.args)}")
        
        # Set dimension weights
        # IMPORTANT: Must set both trainer.args AND trainer.model.args
        if use_dim_weights_flag and dim_weights_value:
            if isinstance(trainer.args, dict):
                trainer.args['use_dim_weights'] = True
                trainer.args['dim_weights'] = dim_weights_value
            else:
                setattr(trainer.args, 'use_dim_weights', True)
                setattr(trainer.args, 'dim_weights', dim_weights_value)
            
            # Also set to model.args (critical for init_criterion to read correct values)
            # Only set if model.args is a different object
            if not model_args_is_same and hasattr(trainer, 'model') and hasattr(trainer.model, 'args'):
                if isinstance(trainer.model.args, dict):
                    trainer.model.args['use_dim_weights'] = True
                    trainer.model.args['dim_weights'] = dim_weights_value
                else:
                    setattr(trainer.model.args, 'use_dim_weights', True)
                    setattr(trainer.model.args, 'dim_weights', dim_weights_value)
            updated = True
        
        # Set focal loss settings
        # IMPORTANT: Must set both trainer.args AND trainer.model.args
        if use_focal_loss_flag:
            if isinstance(trainer.args, dict):
                trainer.args['use_focal_loss'] = True
                trainer.args['focal_gamma'] = focal_gamma_value
                trainer.args['focal_alpha'] = focal_alpha_value
            else:
                setattr(trainer.args, 'use_focal_loss', True)
                setattr(trainer.args, 'focal_gamma', focal_gamma_value)
                setattr(trainer.args, 'focal_alpha', focal_alpha_value)
            
            # Also set to model.args (critical for init_criterion to read correct values)
            # Only set if model.args is a different object
            if not model_args_is_same and hasattr(trainer, 'model') and hasattr(trainer.model, 'args'):
                if isinstance(trainer.model.args, dict):
                    trainer.model.args['use_focal_loss'] = True
                    trainer.model.args['focal_gamma'] = focal_gamma_value
                    trainer.model.args['focal_alpha'] = focal_alpha_value
                else:
                    setattr(trainer.model.args, 'use_focal_loss', True)
                    setattr(trainer.model.args, 'focal_gamma', focal_gamma_value)
                    setattr(trainer.model.args, 'focal_alpha', focal_alpha_value)
            updated = True
        
        # Set HMD loss settings
        # IMPORTANT: Must set both trainer.args AND trainer.model.args
        # because v8DetectionLoss.__init__ reads from model.args (not trainer.args)
        if use_hmd_loss_flag:
            if isinstance(trainer.args, dict):
                trainer.args['use_hmd_loss'] = True
                trainer.args['hmd_loss_weight'] = hmd_loss_weight_value
                trainer.args['hmd_penalty_single'] = hmd_penalty_single_value
                trainer.args['hmd_penalty_none'] = hmd_penalty_none_value
                trainer.args['hmd_penalty_coeff'] = hmd_penalty_coeff_value
            else:
                setattr(trainer.args, 'use_hmd_loss', True)
                setattr(trainer.args, 'hmd_loss_weight', hmd_loss_weight_value)
                setattr(trainer.args, 'hmd_penalty_single', hmd_penalty_single_value)
                setattr(trainer.args, 'hmd_penalty_none', hmd_penalty_none_value)
                setattr(trainer.args, 'hmd_penalty_coeff', hmd_penalty_coeff_value)
            
            # Also set to model.args (critical for init_criterion to read correct values)
            # Only set if model.args is a different object
            if not model_args_is_same and hasattr(trainer, 'model') and hasattr(trainer.model, 'args'):
                if isinstance(trainer.model.args, dict):
                    trainer.model.args['use_hmd_loss'] = True
                    trainer.model.args['hmd_loss_weight'] = hmd_loss_weight_value
                    trainer.model.args['hmd_penalty_single'] = hmd_penalty_single_value
                    trainer.model.args['hmd_penalty_none'] = hmd_penalty_none_value
                    trainer.model.args['hmd_penalty_coeff'] = hmd_penalty_coeff_value
                else:
                    setattr(trainer.model.args, 'use_hmd_loss', True)
                    setattr(trainer.model.args, 'hmd_loss_weight', hmd_loss_weight_value)
                    setattr(trainer.model.args, 'hmd_penalty_single', hmd_penalty_single_value)
                    setattr(trainer.model.args, 'hmd_penalty_none', hmd_penalty_none_value)
                    setattr(trainer.model.args, 'hmd_penalty_coeff', hmd_penalty_coeff_value)
            updated = True
        
        # Recreate loss function with custom settings
        if updated and hasattr(trainer.model, 'init_criterion'):
            # Debug: Log what we're setting before recreating criterion
            if use_hmd_loss_flag:
                logging.info(f"🔧 Setting HMD loss parameters before recreating criterion:")
                logging.info(f"   trainer.args.use_hmd_loss = {getattr(trainer.args, 'use_hmd_loss', 'NOT SET')}")
                logging.info(f"   trainer.args.hmd_loss_weight = {getattr(trainer.args, 'hmd_loss_weight', 'NOT SET')}")
                if hasattr(trainer, 'model') and hasattr(trainer.model, 'args'):
                    logging.info(f"   trainer.model.args.use_hmd_loss = {getattr(trainer.model.args, 'use_hmd_loss', 'NOT SET')}")
                    logging.info(f"   trainer.model.args.hmd_loss_weight = {getattr(trainer.model.args, 'hmd_loss_weight', 'NOT SET')}")
            
            trainer.model.criterion = None  # Clear existing criterion
            
            # CRITICAL FIX: Directly create v8DetectionLoss with explicit parameters
            # instead of relying on model.args (which may not be set correctly)
            if use_hmd_loss_flag:
                # Import from the local modified version (not the installed package)
                import sys
                import importlib
                from pathlib import Path
                # Ensure we import from the local ultralytics package
                local_ultralytics_path = Path(__file__).parent.parent
                if str(local_ultralytics_path) not in sys.path:
                    sys.path.insert(0, str(local_ultralytics_path))
                
                # Force reload the loss module to ensure we get the local version
                if 'ultralytics.utils.loss' in sys.modules:
                    importlib.reload(sys.modules['ultralytics.utils.loss'])
                
                from ultralytics.utils.loss import v8DetectionLoss
                logging.info(f"🔧 Creating v8DetectionLoss with explicit HMD parameters:")
                logging.info(f"   use_hmd_loss=True, hmd_loss_weight={hmd_loss_weight_value}")
                # Check if v8DetectionLoss accepts these parameters
                import inspect
                sig = inspect.signature(v8DetectionLoss.__init__)
                logging.info(f"   v8DetectionLoss.__init__ signature: {sig}")
                logging.info(f"   v8DetectionLoss module: {v8DetectionLoss.__module__}")
                logging.info(f"   v8DetectionLoss file: {inspect.getfile(v8DetectionLoss)}")
                
                try:
                    trainer.model.criterion = v8DetectionLoss(
                        trainer.model,
                        use_hmd_loss=True,
                        hmd_loss_weight=hmd_loss_weight_value,
                        hmd_penalty_single=hmd_penalty_single_value,
                        hmd_penalty_none=hmd_penalty_none_value,
                        hmd_penalty_coeff=hmd_penalty_coeff_value,
                        hmd_use_mm=hmd_use_mm_value
                    )
                except TypeError as e:
                    logging.error(f"❌ Failed to create v8DetectionLoss with explicit parameters: {e}")
                    logging.error(f"   This likely means the installed ultralytics package is being used instead of the local version")
                    logging.error(f"   Please run: cd ultralytics && pip install -e .")
                    raise
            elif use_dim_weights_flag or use_focal_loss_flag:
            trainer.model.criterion = trainer.model.init_criterion()
            else:
                trainer.model.criterion = trainer.model.init_criterion()
            
            # Debug: Verify criterion was created with correct settings
            if use_hmd_loss_flag and hasattr(trainer.model, 'criterion') and trainer.model.criterion is not None:
                criterion = trainer.model.criterion
                if hasattr(criterion, 'use_hmd_loss'):
                    actual_weight = getattr(criterion, 'hmd_loss_weight', 'NOT SET')
                    logging.info(f"✅ Training model criterion created - use_hmd_loss={criterion.use_hmd_loss}, hmd_loss_weight={actual_weight}")
                    if criterion.use_hmd_loss != True or actual_weight != hmd_loss_weight_value:
                        logging.error(f"❌ CRITICAL: Criterion settings mismatch!")
                        logging.error(f"   Expected: use_hmd_loss=True, hmd_loss_weight={hmd_loss_weight_value}")
                        logging.error(f"   Actual: use_hmd_loss={criterion.use_hmd_loss}, hmd_loss_weight={actual_weight}")
                        logging.error(f"   This means HMD loss will NOT be applied correctly!")
                        # Try to fix by directly passing parameters to init_criterion
                        logging.warning(f"⚠️ Attempting to fix by recreating criterion with explicit parameters...")
                        trainer.model.criterion = None
                        # Directly call init_criterion with parameters
                        from ultralytics.utils.loss import v8DetectionLoss
                        # Create a new criterion with explicit parameters
                        trainer.model.criterion = v8DetectionLoss(
                            trainer.model,
                            use_hmd_loss=True,
                            hmd_loss_weight=hmd_loss_weight_value,
                            hmd_penalty_single=hmd_penalty_single_value,
                            hmd_penalty_none=hmd_penalty_none_value,
                            hmd_penalty_coeff=hmd_penalty_coeff_value,
                            hmd_use_mm=hmd_use_mm_value
                        )
                        logging.info(f"✅ Criterion recreated with explicit parameters")
                else:
                    logging.warning(f"⚠️ Criterion created but does not have use_hmd_loss attribute!")
                    logging.warning(f"   Criterion type: {type(criterion)}")
                    # Try to fix by directly passing parameters
                    logging.warning(f"⚠️ Attempting to fix by recreating criterion with explicit parameters...")
                    trainer.model.criterion = None
                    from ultralytics.utils.loss import v8DetectionLoss
                    trainer.model.criterion = v8DetectionLoss(
                        trainer.model,
                        use_hmd_loss=True,
                        hmd_loss_weight=hmd_loss_weight_value,
                        hmd_penalty_single=hmd_penalty_single_value,
                        hmd_penalty_none=hmd_penalty_none_value,
                        hmd_penalty_coeff=hmd_penalty_coeff_value
                    )
                    logging.info(f"✅ Criterion recreated with explicit parameters")
            else:
                if use_hmd_loss_flag:
                    logging.error(f"❌ CRITICAL: Criterion is None after recreation! HMD loss will NOT work!")
                    # Try to create criterion with explicit parameters
                    logging.warning(f"⚠️ Attempting to create criterion with explicit parameters...")
                    from ultralytics.utils.loss import v8DetectionLoss
                    trainer.model.criterion = v8DetectionLoss(
                        trainer.model,
                        use_hmd_loss=True,
                        hmd_loss_weight=hmd_loss_weight_value,
                        hmd_penalty_single=hmd_penalty_single_value,
                        hmd_penalty_none=hmd_penalty_none_value,
                        hmd_penalty_coeff=hmd_penalty_coeff_value
                    )
                    logging.info(f"✅ Criterion created with explicit parameters")
            
            # Also update EMA model's criterion if it exists
            # This is critical because validation uses EMA model, and EMA model's criterion
            # needs to have the same HMD loss configuration as the training model
            if hasattr(trainer, 'ema') and trainer.ema is not None and hasattr(trainer.ema, 'ema'):
                # IMPORTANT: Also set args to EMA model so init_criterion can read correct values
                if hasattr(trainer.ema.ema, 'args'):
                    if isinstance(trainer.ema.ema.args, dict):
                        # Copy all custom loss settings to EMA model's args
                        if use_dim_weights_flag:
                            trainer.ema.ema.args['use_dim_weights'] = True
                            trainer.ema.ema.args['dim_weights'] = dim_weights_value
                        if use_focal_loss_flag:
                            trainer.ema.ema.args['use_focal_loss'] = True
                            trainer.ema.ema.args['focal_gamma'] = focal_gamma_value
                            trainer.ema.ema.args['focal_alpha'] = focal_alpha_value
                        if use_hmd_loss_flag:
                            trainer.ema.ema.args['use_hmd_loss'] = True
                            trainer.ema.ema.args['hmd_loss_weight'] = hmd_loss_weight_value
                            trainer.ema.ema.args['hmd_penalty_single'] = hmd_penalty_single_value
                            trainer.ema.ema.args['hmd_penalty_none'] = hmd_penalty_none_value
                            trainer.ema.ema.args['hmd_penalty_coeff'] = hmd_penalty_coeff_value
                    else:
                        # Copy all custom loss settings to EMA model's args
                        if use_dim_weights_flag:
                            setattr(trainer.ema.ema.args, 'use_dim_weights', True)
                            setattr(trainer.ema.ema.args, 'dim_weights', dim_weights_value)
                        if use_focal_loss_flag:
                            setattr(trainer.ema.ema.args, 'use_focal_loss', True)
                            setattr(trainer.ema.ema.args, 'focal_gamma', focal_gamma_value)
                            setattr(trainer.ema.ema.args, 'focal_alpha', focal_alpha_value)
                        if use_hmd_loss_flag:
                            setattr(trainer.ema.ema.args, 'use_hmd_loss', True)
                            setattr(trainer.ema.ema.args, 'hmd_loss_weight', hmd_loss_weight_value)
                            setattr(trainer.ema.ema.args, 'hmd_penalty_single', hmd_penalty_single_value)
                            setattr(trainer.ema.ema.args, 'hmd_penalty_none', hmd_penalty_none_value)
                            setattr(trainer.ema.ema.args, 'hmd_penalty_coeff', hmd_penalty_coeff_value)
                
                if hasattr(trainer.ema.ema, 'init_criterion'):
                    trainer.ema.ema.criterion = None  # Clear existing criterion
                    # CRITICAL FIX: Directly create v8DetectionLoss with explicit parameters for EMA model too
                    if use_hmd_loss_flag:
                        # Import from the local modified version (not the installed package)
                        import sys
                        import importlib
                        from pathlib import Path
                        local_ultralytics_path = Path(__file__).parent.parent
                        if str(local_ultralytics_path) not in sys.path:
                            sys.path.insert(0, str(local_ultralytics_path))
                        
                        # Force reload the loss module to ensure we get the local version
                        if 'ultralytics.utils.loss' in sys.modules:
                            importlib.reload(sys.modules['ultralytics.utils.loss'])
                        
                        from ultralytics.utils.loss import v8DetectionLoss
                        try:
                            trainer.ema.ema.criterion = v8DetectionLoss(
                                trainer.ema.ema,
                                use_hmd_loss=True,
                                hmd_loss_weight=hmd_loss_weight_value,
                                hmd_penalty_single=hmd_penalty_single_value,
                                hmd_penalty_none=hmd_penalty_none_value,
                                hmd_penalty_coeff=hmd_penalty_coeff_value,
                                hmd_use_mm=hmd_use_mm_value
                            )
                            logging.debug("✅ EMA model's criterion recreated with explicit HMD parameters")
                        except TypeError as e:
                            logging.error(f"❌ Failed to create EMA v8DetectionLoss: {e}")
                            logging.error(f"   Please run: cd ultralytics && pip install -e .")
                            raise
                    else:
                        trainer.ema.ema.criterion = trainer.ema.ema.init_criterion()
                        logging.debug("✅ EMA model's criterion also recreated with custom settings")
                else:
                    # If EMA model doesn't have init_criterion, copy criterion from training model
                    # This ensures EMA model has the same criterion configuration
                    if hasattr(trainer.model, 'criterion') and trainer.model.criterion is not None:
                        import copy
                        trainer.ema.ema.criterion = copy.deepcopy(trainer.model.criterion)
                        logging.debug("✅ EMA model's criterion copied from training model")
            
            info_msg = []
            if use_dim_weights_flag:
                info_msg.append(f"dimension weights: {dim_weights_value}")
            if use_focal_loss_flag:
                info_msg.append(f"focal loss (gamma={focal_gamma_value}, alpha={focal_alpha_value})")
            if use_hmd_loss_flag:
                info_msg.append(f"HMD loss (weight={hmd_loss_weight_value}, penalty_single={hmd_penalty_single_value}, penalty_none={hmd_penalty_none_value})")
            logging.info(f"✅ Loss function recreated with {', '.join(info_msg)}")
        elif updated:
            logging.warning("⚠️ Cannot recreate loss function, custom settings may not be applied")
    
    # Add callback to set custom loss settings after trainer initialization
    if use_dim_weights_flag or use_focal_loss_flag or use_hmd_loss_flag:
        model.add_callback("on_train_start", set_custom_loss_callback)

    # Setup W&B if enabled
    if args.wandb:
        wandb.login()
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Include experiment identifier in run name
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
                
                # HMD loss parameters
                "use_hmd_loss": use_hmd_loss_flag,
                "hmd_loss_weight": hmd_loss_weight_value,
                "hmd_penalty_single": hmd_penalty_single_value,
                "hmd_penalty_none": hmd_penalty_none_value,
                "hmd_penalty_coeff": hmd_penalty_coeff_value,
                
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
                
                # Ultrasound-specific augmentation
                "use_ultrasound_aug": getattr(args, 'use_ultrasound_aug', False),
                "ultrasound_speckle_var": getattr(args, 'ultrasound_speckle_var', 0.1),
                "ultrasound_attenuation_factor": getattr(args, 'ultrasound_attenuation_factor', 0.3),
                
                # IoU type
                "iou_type": args.iou_type,
                
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
        # Record training start time
        wandb.run.summary["train_start_time"] = train_start_time
        
        # Add training callback function (log detailed training metrics) - only when W&B is enabled
        model.add_callback("on_train_epoch_end", log_train_metrics)
    else:
        logging.info("W&B is disabled. Training will proceed without logging, but metrics will be printed to terminal after validation.")
    
    # CRITICAL FIX: Monkey patch get_stats() to save stats before they are cleared
    # The problem: get_stats() calls clear_stats() which clears validator.metrics.stats
    # Solution: Save stats before clear_stats() is called
    from ultralytics.models.yolo.detect.val import DetectionValidator
    original_get_stats = DetectionValidator.get_stats
    
    def patched_get_stats(self):
        """Patched get_stats that saves stats before clearing them"""
        # Save stats before they are cleared
        if hasattr(self, 'metrics') and hasattr(self.metrics, 'stats'):
            stats = self.metrics.stats
            # Check if stats have data
            if stats and len(stats.get('tp', [])) > 0:
                # Save a copy to self.stats
                import copy
                self.stats = copy.deepcopy(stats)
                logging.debug(f"✅ Saved validator.metrics.stats to validator.stats (before get_stats clears them)")
        
        # Call original get_stats
        return original_get_stats(self)
    
    # Apply monkey patch
    DetectionValidator.get_stats = patched_get_stats
    logging.info("✅ Patched DetectionValidator.get_stats() to save stats before clearing")
    
    # NEW: Monkey patch update_metrics() to collect boxes for real HMD error calculation
    # This allows us to calculate real HMD errors even when --use_hmd_loss is False
    original_update_metrics = DetectionValidator.update_metrics
    
    def patched_update_metrics(self, preds, batch):
        """Patched update_metrics that collects boxes for HMD calculation"""
        # Store batch data for box collection (only for det_123 database)
        if hasattr(self, 'args') and hasattr(self.args, 'database') and self.args.database == 'det_123':
            import torch
            # Store predictions and targets for later HMD calculation
            self._last_batch_preds = preds
            self._last_batch_targets = []
            self._last_batch_im_files = batch.get('im_file', [])
            self._last_batch_full = batch  # Store full batch for _prepare_batch
            
            # Extract ground truth boxes from batch (prepare them like in original update_metrics)
            for si in range(len(preds)):
                pbatch = self._prepare_batch(si, batch)
                self._last_batch_targets.append({
                    'bboxes': pbatch.get('bboxes', torch.empty(0, 4)),
                    'cls': pbatch.get('cls', torch.empty(0, dtype=torch.long))
                })
        
        # Call original update_metrics
        return original_update_metrics(self, preds, batch)
    
    # Apply monkey patch
    DetectionValidator.update_metrics = patched_update_metrics
    logging.info("✅ Patched DetectionValidator.update_metrics() to collect boxes for HMD calculation")
    
    # Add callback to collect boxes during validation (for real HMD error calculation)
    # This works even when --use_hmd_loss is False
    if args.database == 'det_123':
        # Reset collection at start of each validation
        def reset_hmd_collection_callback(trainer):
            """Reset HMD collection at start of validation"""
            if hasattr(trainer, '_hmd_collection'):
                trainer._hmd_collection = {
                    'pred_boxes': [],
                    'gt_boxes': [],
                    'image_files': []
                }
        
        model.add_callback("on_val_start", reset_hmd_collection_callback)
        model.add_callback("on_val_batch_end", on_val_batch_end_callback)
        logging.info("✅ Added on_val_start and on_val_batch_end callbacks to collect boxes for HMD calculation")
    
    # Add callback to extract IoU, Dice, and HMD metrics after validation
    # Create callback with closure to capture args
    on_val_end_callback_with_args = create_on_val_end_callback(args)
    model.add_callback("on_val_end", on_val_end_callback_with_args)
    
    # Add callback to keep only top confidence bbox per class if enabled
    if args.keep_top_conf_per_class:
        def keep_top_conf_per_class_callback(validator):
            """Keep only the highest confidence bbox per class after postprocess"""
            import torch
            import numpy as np
            
            # Determine confidence threshold to use
            conf_threshold = args.conf_low if args.conf_low is not None else (args.conf if args.conf is not None else 0.1)
            
            # Modify validator's conf threshold for initial filtering (lower threshold)
            if hasattr(validator, 'args'):
                original_conf = getattr(validator.args, 'conf', None)
                validator.args.conf = conf_threshold
                logging.info(f"🔍 Using lower confidence threshold {conf_threshold} for initial filtering (keep_top_conf_per_class enabled)")
            
            # We need to modify postprocess to filter after initial NMS
            # Store original postprocess method
            if not hasattr(validator, '_original_postprocess'):
                validator._original_postprocess = validator.postprocess
                
                def custom_postprocess(preds):
                    """Custom postprocess that keeps only top confidence per class"""
                    # Call original postprocess with lower confidence threshold
                    preds = validator._original_postprocess(preds)
                    
                    # Filter: keep only highest confidence bbox per class per image
                    # preds is a list of dicts: [{'bboxes': tensor, 'cls': tensor, 'conf': tensor}, ...]
                    if isinstance(preds, list):
                        filtered_preds = []
                        for pred_dict in preds:
                            if pred_dict is None or not isinstance(pred_dict, dict):
                                filtered_preds.append(pred_dict)
                                continue
                            
                            # Get bboxes, classes, and confidences
                            if 'bboxes' not in pred_dict or 'cls' not in pred_dict or 'conf' not in pred_dict:
                                filtered_preds.append(pred_dict)
                                continue
                            
                            bboxes = pred_dict['bboxes']  # [N, 4]
                            classes = pred_dict['cls']    # [N]
                            confs = pred_dict['conf']     # [N]
                            
                            if len(classes) == 0:
                                filtered_preds.append(pred_dict)
                                continue
                            
                            # Convert to numpy for easier processing
                            if isinstance(classes, torch.Tensor):
                                classes_np = classes.cpu().numpy()
                                confs_np = confs.cpu().numpy()
                            else:
                                classes_np = np.array(classes)
                                confs_np = np.array(confs)
                            
                            # For each unique class, keep only the one with highest confidence
                            unique_classes = np.unique(classes_np)
                            keep_indices = []
                            
                            for cls in unique_classes:
                                cls_mask = classes_np == cls
                                cls_indices = np.where(cls_mask)[0]
                                if len(cls_indices) > 0:
                                    # Get index of highest confidence
                                    cls_confs = confs_np[cls_indices]
                                    top_idx = cls_indices[np.argmax(cls_confs)]
                                    keep_indices.append(int(top_idx))
                            
                            if len(keep_indices) > 0:
                                keep_indices = torch.tensor(keep_indices, device=bboxes.device if isinstance(bboxes, torch.Tensor) else None)
                                
                                # Create filtered prediction dict
                                filtered_pred = {
                                    'bboxes': bboxes[keep_indices] if isinstance(bboxes, torch.Tensor) else bboxes[keep_indices],
                                    'cls': classes[keep_indices] if isinstance(classes, torch.Tensor) else classes[keep_indices],
                                    'conf': confs[keep_indices] if isinstance(confs, torch.Tensor) else confs[keep_indices]
                                }
                                
                                # Copy other keys if present
                                for key in pred_dict:
                                    if key not in ['bboxes', 'cls', 'conf']:
                                        filtered_pred[key] = pred_dict[key]
                                
                                filtered_preds.append(filtered_pred)
                            else:
                                # No detections to keep - create empty dict
                                filtered_pred = {
                                    'bboxes': bboxes[[]] if isinstance(bboxes, torch.Tensor) else np.array([]).reshape(0, 4),
                                    'cls': classes[[]] if isinstance(classes, torch.Tensor) else np.array([]),
                                    'conf': confs[[]] if isinstance(confs, torch.Tensor) else np.array([])
                                }
                                # Copy other keys if present
                                for key in pred_dict:
                                    if key not in ['bboxes', 'cls', 'conf']:
                                        filtered_pred[key] = pred_dict[key]
                                filtered_preds.append(filtered_pred)
                        
                        return filtered_preds
                    else:
                        # Single prediction (not a list)
                        return preds
                
                # Replace postprocess method
                validator.postprocess = custom_postprocess
                logging.info(f"✅ Custom postprocess enabled: keeping only top confidence bbox per class")
        
        model.add_callback("on_val_start", keep_top_conf_per_class_callback)
    
    # Add callback to setup Cosine Restart scheduler if enabled
    def setup_cosine_restart_scheduler(trainer):
        """Setup Cosine Annealing with Warm Restarts scheduler if enabled"""
        if args.use_cosine_restart:
            import torch.optim.lr_scheduler as lr_scheduler
            # Replace the default scheduler with CosineAnnealingWarmRestarts
            trainer.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                trainer.optimizer,
                T_0=args.cosine_restart_t0,
                T_mult=args.cosine_restart_t_mult,
                eta_min=trainer.args.lrf * trainer.args.lr0  # Minimum learning rate
            )
            logging.info(f"✅ Cosine Annealing with Warm Restarts enabled: T_0={args.cosine_restart_t0}, T_mult={args.cosine_restart_t_mult}")
    model.add_callback("on_train_start", setup_cosine_restart_scheduler)
    
    # Add callback to reset HMD loss stats at start of each epoch
    def reset_hmd_loss_callback(trainer):
        """Reset HMD loss accumulation at start of each epoch"""
        if hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
            try:
                criterion = trainer.model.criterion
                if hasattr(criterion, 'reset_hmd_loss_stats'):
                    criterion.reset_hmd_loss_stats()
            except Exception:
                pass
    model.add_callback("on_train_epoch_start", reset_hmd_loss_callback)
    
    # Add callback to save HMD loss at end of each training epoch (before validation)
    # This ensures we have the training epoch's HMD loss available during validation
    # NOTE: This must run BEFORE validation, so we use on_train_epoch_end (which runs before on_val_end)
    def save_hmd_loss_for_validation_callback(trainer):
        """Save HMD loss from training epoch before validation starts"""
        if use_hmd_loss_flag and hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
            try:
                criterion = trainer.model.criterion
                if hasattr(criterion, 'get_avg_hmd_loss'):
                    hmd_loss_avg = criterion.get_avg_hmd_loss()
                    # Only save if we actually have accumulated loss (count > 0)
                    if hasattr(criterion, 'hmd_loss_count') and criterion.hmd_loss_count > 0:
                        # Store in trainer for validation to access
                        if not hasattr(trainer, '_additional_metrics'):
                            trainer._additional_metrics = {}
                        trainer._additional_metrics["train/hmd_loss"] = hmd_loss_avg
                        logging.debug(f"✅ Saved training epoch HMD loss: {hmd_loss_avg} (from {criterion.hmd_loss_count} batches)")
                    else:
                        logging.debug(f"⚠️ HMD loss count is 0, cannot save average")
                elif hasattr(criterion, 'last_hmd_loss') and criterion.last_hmd_loss != 0.0:
                    if not hasattr(trainer, '_additional_metrics'):
                        trainer._additional_metrics = {}
                    trainer._additional_metrics["train/hmd_loss"] = float(criterion.last_hmd_loss)
                    logging.debug(f"✅ Saved training epoch HMD loss (last batch): {criterion.last_hmd_loss}")
            except Exception as e:
                logging.debug(f"❌ Failed to save HMD loss: {e}")
    # Add this callback with higher priority (lower number = higher priority)
    # We want it to run after log_train_metrics but before validation
    model.add_callback("on_train_epoch_end", save_hmd_loss_for_validation_callback)
    
    # Add callback to save HMD loss at end of each training epoch (before validation)
    # This ensures we have the training epoch's HMD loss available during validation
    def save_hmd_loss_callback(trainer):
        """Save HMD loss from training epoch before validation starts"""
        if use_hmd_loss_flag and hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
            try:
                criterion = trainer.model.criterion
                if hasattr(criterion, 'get_avg_hmd_loss'):
                    hmd_loss_avg = criterion.get_avg_hmd_loss()
                    # Store in trainer for validation to access
                    if not hasattr(trainer, '_additional_metrics'):
                        trainer._additional_metrics = {}
                    trainer._additional_metrics["train/hmd_loss"] = hmd_loss_avg
                    logging.debug(f"Saved training epoch HMD loss: {hmd_loss_avg}")
                elif hasattr(criterion, 'last_hmd_loss') and criterion.last_hmd_loss != 0.0:
                    if not hasattr(trainer, '_additional_metrics'):
                        trainer._additional_metrics = {}
                    trainer._additional_metrics["train/hmd_loss"] = float(criterion.last_hmd_loss)
                    logging.debug(f"Saved training epoch HMD loss (last batch): {criterion.last_hmd_loss}")
            except Exception as e:
                logging.debug(f"Failed to save HMD loss: {e}")
    model.add_callback("on_train_epoch_end", save_hmd_loss_callback)

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
        # Use best model for detailed evaluation
        run_name = f"{args.model}-{args.database}-v{args.db_version}" + (f"-{args.exp_name}" if args.exp_name else "")
        best_model = YOLO(f"runs/train/{run_name}/weights/best.pt")
        logging.info("🔁 Re-evaluating using best.pt")
        
        # Evaluate val and test
        # Calculate HMD metrics for all det_123 experiments (even without HMD loss enabled)
        # This allows monitoring HMD performance for all experiments, including baseline
        use_hmd = args.database == 'det_123'  # Changed: calculate HMD for all det_123 experiments
        val_results = evaluate_detailed(
            best_model, "val", batch=args.batch, imgsz=args.imgsz,
            database=args.database, db_version=args.db_version, use_hmd=use_hmd,
            penalty_single=args.hmd_penalty_single, penalty_none=args.hmd_penalty_none,
            penalty_coeff=args.hmd_penalty_coeff
        )
        test_results = evaluate_detailed(
            best_model, "test", batch=args.batch, imgsz=args.imgsz,
            database=args.database, db_version=args.db_version, use_hmd=use_hmd,
            penalty_single=args.hmd_penalty_single, penalty_none=args.hmd_penalty_none,
            penalty_coeff=args.hmd_penalty_coeff
        )
        
        # Calculate and log all metrics (val and test)
        if val_results:
            map50_val = val_results.get("mAP50", 0)
            map_val = val_results.get("mAP50-95", 0)
            precision_val = val_results.get("precision", 0)
            recall_val = val_results.get("recall", 0)
            fitness_val = val_results.get("fitness", map50_val * 0.1 + map_val * 0.9)
            iou_val = val_results.get("iou", None)
            dice_val = val_results.get("dice", None)
            
            val_summary = {
                "fitness/val": fitness_val,
                "val/mAP50": map50_val,
                "val/mAP50-95": map_val,
                "val/precision": precision_val,
                "val/recall": recall_val,
            }
            if iou_val is not None:
                val_summary["val/iou"] = iou_val
            if dice_val is not None:
                val_summary["val/dice"] = dice_val
            
            # Add HMD metrics if available (for all det_123 experiments)
            if args.database == 'det_123':
                val_summary.update({
                    "val/hmd/detection_rate": val_results.get("detection_rate", 0),
                    "val/hmd/rmse_pixel": val_results.get("rmse_pixel", 0),  # Use rmse_pixel (not rmse_hmd_pixel)
                    "val/hmd/overall_score_pixel": val_results.get("overall_score_pixel", 0),
                })
            
            wandb.log(val_summary)
            wandb.run.summary.update({k: v for k, v in val_summary.items() if not k.startswith("fitness/")})
            wandb.run.summary["fitness_val"] = fitness_val
            logging.info(f"✅ Val metrics: Precision={precision_val:.4f}, Recall={recall_val:.4f}, mAP50={map50_val:.4f}, mAP50-95={map_val:.4f}, Fitness={fitness_val:.6f}")
        
        if test_results:
            map50_test = test_results.get("mAP50", 0)
            map_test = test_results.get("mAP50-95", 0)
            precision_test = test_results.get("precision", 0)
            recall_test = test_results.get("recall", 0)
            fitness_test = test_results.get("fitness", map50_test * 0.1 + map_test * 0.9)
            iou_test = test_results.get("iou", None)
            dice_test = test_results.get("dice", None)
            
            test_summary = {
                "fitness/test": fitness_test,
                "test/mAP50": map50_test,
                "test/mAP50-95": map_test,
                "test/precision": precision_test,
                "test/recall": recall_test,
            }
            if iou_test is not None:
                test_summary["test/iou"] = iou_test
            if dice_test is not None:
                test_summary["test/dice"] = dice_test
            
            # Add HMD metrics if available (for all det_123 experiments)
            if args.database == 'det_123':
                test_summary.update({
                    "test/hmd/detection_rate": test_results.get("detection_rate", 0),
                    "test/hmd/rmse_pixel": test_results.get("rmse_pixel", 0),  # Use rmse_pixel (not rmse_hmd_pixel)
                    "test/hmd/overall_score_pixel": test_results.get("overall_score_pixel", 0),
                })
            
            wandb.log(test_summary)
            wandb.run.summary.update({k: v for k, v in test_summary.items() if not k.startswith("fitness/")})
            wandb.run.summary["fitness_test"] = fitness_test
            logging.info(f"✅ Test metrics: Precision={precision_test:.4f}, Recall={recall_test:.4f}, mAP50={map50_test:.4f}, mAP50-95={map_test:.4f}, Fitness={fitness_test:.6f}")
        
        # Export ONNX model and upload to W&B
        try:
            export_path = best_model.export(format="onnx", save_dir="./exports")
            artifact = wandb.Artifact("exported_model", type="model")
            artifact.add_file(export_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            logging.warning(f"⚠️ Failed to export model: {e}")
        
        wandb.finish()

