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



def print_validation_metrics(trainer):
    """Print additional validation metrics to terminal after each epoch"""
    # trainer.metrics is a DetMetrics object, not a dict
    # Extract metrics from DetMetrics object
    precision = 0.0
    recall = 0.0
    map50 = 0.0
    map50_95 = 0.0
    
    try:
        if hasattr(trainer, 'metrics') and trainer.metrics is not None:
            # Get mean results: [mp, mr, map50, map50-95]
            if hasattr(trainer.metrics, 'mean_results'):
                mp, mr, map50, map50_95 = trainer.metrics.mean_results()
                precision = float(mp)
                recall = float(mr)
                map50 = float(map50)
                map50_95 = float(map50_95)
    except Exception:
        pass
    
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
    if hmd_loss_value is None and hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
        try:
            criterion = trainer.model.criterion
            if hasattr(criterion, 'get_avg_hmd_loss'):
                # Get average HMD loss across all batches in this epoch
                hmd_loss_value = criterion.get_avg_hmd_loss()
            elif hasattr(criterion, 'last_hmd_loss'):
                # Fallback to last batch loss if average not available
                hmd_loss_value = float(criterion.last_hmd_loss)
        except Exception:
            pass
    
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
    import logging
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
    overall_score_pixel = 0.0
    
    # First try to get from trainer._additional_metrics
    if hasattr(trainer, '_additional_metrics') and trainer._additional_metrics is not None:
        additional_metrics = trainer._additional_metrics
        detection_rate = float(additional_metrics.get("hmd/detection_rate") or additional_metrics.get("val/hmd/detection_rate", 0))
        rmse_pixel = float(additional_metrics.get("hmd/rmse_pixel") or additional_metrics.get("val/hmd/rmse_pixel", 0))
        overall_score_pixel = float(additional_metrics.get("hmd/overall_score_pixel") or additional_metrics.get("val/hmd/overall_score_pixel", 0))
    # If not found in trainer, try to get from validator (if available)
    elif hasattr(trainer, 'validator') and trainer.validator is not None:
        validator = trainer.validator
        if hasattr(validator, '_additional_metrics') and validator._additional_metrics is not None:
            additional_metrics = validator._additional_metrics
            detection_rate = float(additional_metrics.get("hmd/detection_rate") or additional_metrics.get("val/hmd/detection_rate", 0))
            rmse_pixel = float(additional_metrics.get("hmd/rmse_pixel") or additional_metrics.get("val/hmd/rmse_pixel", 0))
            overall_score_pixel = float(additional_metrics.get("hmd/overall_score_pixel") or additional_metrics.get("val/hmd/overall_score_pixel", 0))
    
    # Always show HMD metrics section if database is det_123 (even without HMD loss enabled)
    # This allows monitoring HMD performance for all det_123 experiments
    if is_det_123:
        print(f"\n📏 HMD Metrics (det_123):", flush=True)
        print(f"   Detection_Rate: {detection_rate:.4f}", flush=True)
        print(f"   RMSE_HMD (pixel): {rmse_pixel:.2f} px", flush=True)
        print(f"   Overall_Score (pixel): {overall_score_pixel:.4f}", flush=True)


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
    # trainer.metrics is a DetMetrics object, extract metrics properly
    precision = 0.0
    recall = 0.0
    map50 = 0.0
    map50_95 = 0.0
    
    try:
        if hasattr(trainer, 'metrics') and trainer.metrics is not None:
            # Get mean results: [mp, mr, map50, map50-95]
            if hasattr(trainer.metrics, 'mean_results'):
                mp, mr, map50, map50_95 = trainer.metrics.mean_results()
                precision = float(mp)
                recall = float(mr)
                map50 = float(map50)
                map50_95 = float(map50_95)
    except Exception:
        pass
    
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
                hmd_loss_value = criterion.get_avg_hmd_loss()
            elif hasattr(criterion, 'last_hmd_loss'):
                hmd_loss_value = float(criterion.last_hmd_loss)
        except Exception:
            pass
    
    # Also check _additional_metrics as fallback (in case on_val_end was called first)
    if hmd_loss_value is None and hasattr(trainer, '_additional_metrics') and "train/hmd_loss" in trainer._additional_metrics:
        hmd_loss_value = trainer._additional_metrics["train/hmd_loss"]
    
    if hmd_loss_value is not None:
        logs["train/hmd_loss"] = float(hmd_loss_value)
    
    # HMD metrics (always log for det_123, even if 0) - pixel based only
    # Use val/hmd/ prefix for validation metrics in W&B
    if hasattr(trainer, 'args') and hasattr(trainer.args, 'database') and trainer.args.database == 'det_123':
        if hasattr(trainer, '_additional_metrics'):
            additional_metrics = trainer._additional_metrics
            # Try both naming conventions (hmd/... and val/hmd/...)
            detection_rate = float(additional_metrics.get("val/hmd/detection_rate") or additional_metrics.get("hmd/detection_rate", 0))
            rmse_pixel = float(additional_metrics.get("val/hmd/rmse_pixel") or additional_metrics.get("hmd/rmse_pixel", 0))
            overall_score_pixel = float(additional_metrics.get("val/hmd/overall_score_pixel") or additional_metrics.get("hmd/overall_score_pixel", 0))
            logs["val/hmd/detection_rate"] = detection_rate
            logs["val/hmd/rmse_pixel"] = rmse_pixel
            logs["val/hmd/overall_score_pixel"] = overall_score_pixel
        else:
            logs["val/hmd/detection_rate"] = 0.0
            logs["val/hmd/rmse_pixel"] = 0.0
            logs["val/hmd/overall_score_pixel"] = 0.0
    
    # 记录学习率
    for i, pg in enumerate(trainer.optimizer.param_groups):
        logs[f"lr/pg{i}"] = float(pg["lr"])
    
    wandb.log(logs, step=trainer.epoch)


def on_val_batch_end_callback(trainer):
    """Callback to collect predictions and ground truth bboxes for HMD calculation"""
    # Only collect if HMD loss is enabled and database is det_123
    if not (hasattr(trainer, 'args') and hasattr(trainer.args, 'database') and 
            trainer.args.database == 'det_123' and
            hasattr(trainer.args, 'use_hmd_loss') and trainer.args.use_hmd_loss):
        return
    
    # Initialize HMD data collection if not exists
    if not hasattr(trainer, '_hmd_collection'):
        trainer._hmd_collection = {
            'pred_boxes': [],  # List of (image_idx, class, bbox_xyxy, conf)
            'gt_boxes': [],    # List of (image_idx, class, bbox_xyxy)
            'image_files': []  # List of image file paths
        }
    
    # Get validator to access current batch
    if hasattr(trainer, 'validator') and trainer.validator is not None:
        validator = trainer.validator
        # Try to access batch data from validator
        # The validator processes batches in update_metrics, but we need to hook into that
        # For now, we'll collect from validator's jdict if save_json is enabled
        # Or we can access from validator's internal state
        pass


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


def calculate_hmd_metrics_from_validator(validator, trainer, penalty_single=500.0, penalty_none=1000.0):
    """
    Calculate HMD metrics from validator using collected bbox data or HMD loss stats
    
    Args:
        validator: Ultralytics validator object
        trainer: Trainer object (to access HMD loss stats)
        penalty_single: Penalty when only one target detected
        penalty_none: Penalty when both targets missed
    
    Returns:
        Dict with detection_rate, rmse_pixel, overall_score_pixel
    """
    import numpy as np
    import torch
    
    try:
        # First, try to use HMD loss statistics from criterion (most accurate)
        # The HMD loss already calculates real HMD distances during training/validation
        if hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
            criterion = trainer.model.criterion
            if hasattr(criterion, 'use_hmd_loss') and criterion.use_hmd_loss:
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
                        if hasattr(validator, 'stats') and validator.stats is not None:
                            stats = validator.stats
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
                                images_with_both_gt = min(mentum_gt_count, hyoid_gt_count) if (mentum_gt_count > 0 and hyoid_gt_count > 0) else 0
                                
                                if images_with_both_gt > 0:
                                    # Count matched detections at IoU=0.5
                                    # Note: tp and pred_cls have same length (both based on predictions)
                                    # target_cls may have different length (based on ground truth)
                                    if len(tp) > 0 and tp.shape[1] > 0:
                                        matched_mask = tp[:, 0]  # Boolean array for IoU=0.5 matches
                                        
                                        # tp and pred_cls should have same length (both are per-prediction)
                                        if len(matched_mask) == len(pred_cls):
                                            # Count matched predictions for each class
                                            # Only count when prediction matches ground truth (matched_mask is True)
                                            mentum_matched = np.sum((matched_mask) & (pred_cls == mentum_class))
                                            hyoid_matched = np.sum((matched_mask) & (pred_cls == hyoid_class))
                                            
                                            # For detection rate, we need to check if both classes are detected in the same images
                                            # This is an approximation: count how many images have both classes detected
                                            # We use the minimum of matched counts as a proxy
                                            both_detected_count = min(mentum_matched, hyoid_matched)
                                            detection_rate = both_detected_count / images_with_both_gt if images_with_both_gt > 0 else 0.0
                                        else:
                                            # Length mismatch - use fallback calculation
                                            logging.warning(f"⚠️ Length mismatch: tp={len(matched_mask)}, pred_cls={len(pred_cls)}")
                                            detection_rate = 0.0
                                        # Use average HMD loss as RMSE (it's already the average HMD error)
                                        rmse_pixel = float(avg_hmd_loss)
                                        overall_score_pixel = detection_rate * rmse_pixel if rmse_pixel > 0 else 1.0
                                        
                                        return {
                                            'detection_rate': float(detection_rate),
                                            'rmse_pixel': float(rmse_pixel),
                                            'overall_score_pixel': float(overall_score_pixel)
                                        }
        
        # Fallback: calculate from validator stats (without real HMD distance)
        # Get stats from validator
        if not hasattr(validator, 'stats') or validator.stats is None:
            logging.debug("⚠️ Validator stats not available")
            return {'detection_rate': 0.0, 'rmse_pixel': 0.0, 'overall_score_pixel': 0.0}
        
        stats = validator.stats
        if not stats or len(stats.get('tp', [])) == 0:
            logging.debug("⚠️ Validator stats empty or no tp")
            return {'detection_rate': 0.0, 'rmse_pixel': 0.0, 'overall_score_pixel': 0.0}
        
        # Get predictions and ground truth from stats
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
        
        # Mentum class = 0, Hyoid class = 1
        mentum_class = 0
        hyoid_class = 1
        
        # Count detections for each class
        # Note: tp and pred_cls have same length (both based on predictions)
        # target_cls may have different length (based on ground truth)
        if len(tp) > 0 and tp.shape[1] > 0:
            matched_mask = tp[:, 0]  # Matches at IoU=0.5
            
            # tp and pred_cls should have same length (both are per-prediction)
            if len(matched_mask) != len(pred_cls):
                logging.warning(f"⚠️ Length mismatch in fallback: tp={len(matched_mask)}, pred_cls={len(pred_cls)}")
                return {'detection_rate': 0.0, 'rmse_pixel': penalty_none, 'overall_score_pixel': 0.0}
            
            # Count matched predictions for each class
            # Only count when prediction matches ground truth (matched_mask is True)
            mentum_matched = np.sum((matched_mask) & (pred_cls == mentum_class))
            hyoid_matched = np.sum((matched_mask) & (pred_cls == hyoid_class))
            
            # Count total ground truth for each class
            mentum_gt_count = np.sum(target_cls == mentum_class)
            hyoid_gt_count = np.sum(target_cls == hyoid_class)
            
            # Estimate images with both classes in GT
            images_with_both_gt = min(mentum_gt_count, hyoid_gt_count) if (mentum_gt_count > 0 and hyoid_gt_count > 0) else 0
            
            if images_with_both_gt == 0:
                return {'detection_rate': 0.0, 'rmse_pixel': penalty_none, 'overall_score_pixel': 0.0}
            
            # Count images where both are detected (matched)
            both_detected_count = min(mentum_matched, hyoid_matched)
            
            # Calculate detection rate
            detection_rate = both_detected_count / images_with_both_gt if images_with_both_gt > 0 else 0.0
            
            # For RMSE: use penalty-based approximation (without bbox, can't calculate real HMD)
            hmd_errors = []
            if both_detected_count > 0:
                hmd_errors.extend([0.0] * both_detected_count)  # Approximation
            
            single_detected = abs(mentum_matched - hyoid_matched)
            if single_detected > 0:
                hmd_errors.extend([penalty_single] * single_detected)
            
            none_detected = max(0, images_with_both_gt - both_detected_count - single_detected)
            if none_detected > 0:
                hmd_errors.extend([penalty_none] * none_detected)
            
            # Calculate RMSE
            if len(hmd_errors) > 0:
                rmse_pixel = np.sqrt(np.mean(np.array(hmd_errors)**2))
            else:
                rmse_pixel = penalty_none
            
            overall_score_pixel = detection_rate * rmse_pixel if rmse_pixel > 0 else 1.0
            
            return {
                'detection_rate': float(detection_rate),
                'rmse_pixel': float(rmse_pixel),
                'overall_score_pixel': float(overall_score_pixel)
            }
        else:
            return {'detection_rate': 0.0, 'rmse_pixel': penalty_none, 'overall_score_pixel': 0.0}
        
    except Exception as e:
        logging.warning(f"⚠️ Error calculating HMD metrics: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return {'detection_rate': 0.0, 'rmse_pixel': 0.0, 'overall_score_pixel': 0.0}


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
        # Prefer average HMD loss (across all batches) over last batch loss
        # In validation, hmd_loss_count might be 0, so use last_hmd_loss as fallback
        hmd_loss_value = None
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
        elif validator is not None and hasattr(validator, 'model') and hasattr(validator.model, 'criterion'):
            # Try to get from validator's model directly
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
                    # Get HMD metrics using validator stats and HMD loss from criterion
                    # The HMD loss already calculates real HMD distances, so we can use that
                    hmd_metrics = calculate_hmd_metrics_from_validator(
                        validator=validator,
                        trainer=trainer,
                        penalty_single=args.hmd_penalty_single,
                        penalty_none=args.hmd_penalty_none
                    )
                
                    # Store with val/hmd/ prefix for consistency with W&B logging
                    additional_metrics["val/hmd/detection_rate"] = hmd_metrics.get('detection_rate', 0.0)
                    additional_metrics["val/hmd/rmse_pixel"] = hmd_metrics.get('rmse_pixel', 0.0)
                    additional_metrics["val/hmd/overall_score_pixel"] = hmd_metrics.get('overall_score_pixel', 0.0)
                    # Also store with hmd/ prefix for backward compatibility
                    additional_metrics["hmd/detection_rate"] = hmd_metrics.get('detection_rate', 0.0)
                    additional_metrics["hmd/rmse_pixel"] = hmd_metrics.get('rmse_pixel', 0.0)
                    additional_metrics["hmd/overall_score_pixel"] = hmd_metrics.get('overall_score_pixel', 0.0)
                    
                    # Debug: print if metrics are 0
                    if hmd_metrics.get('detection_rate', 0.0) == 0.0 and hmd_metrics.get('rmse_pixel', 0.0) == 0.0:
                        logging.debug(f"⚠️ HMD metrics are all 0 - validator stats: {hasattr(validator, 'stats')}, stats keys: {list(validator.stats.keys()) if hasattr(validator, 'stats') and validator.stats else 'None'}")
                else:
                    # If validator not available, set to 0 (but still set them so they will be displayed)
                    logging.warning("⚠️ Validator not available for HMD metrics calculation")
                    additional_metrics["val/hmd/detection_rate"] = 0.0
                    additional_metrics["val/hmd/rmse_pixel"] = 0.0
                    additional_metrics["val/hmd/overall_score_pixel"] = 0.0
                    additional_metrics["hmd/detection_rate"] = 0.0
                    additional_metrics["hmd/rmse_pixel"] = 0.0
                    additional_metrics["hmd/overall_score_pixel"] = 0.0
            except Exception as e:
                # If calculation fails, set to 0 (but still set them so they will be displayed)
                logging.warning(f"⚠️ Failed to calculate HMD metrics: {e}")
                import traceback
                logging.debug(traceback.format_exc())
                additional_metrics["val/hmd/detection_rate"] = 0.0
                additional_metrics["val/hmd/rmse_pixel"] = 0.0
                additional_metrics["val/hmd/overall_score_pixel"] = 0.0
                additional_metrics["hmd/detection_rate"] = 0.0
                additional_metrics["hmd/rmse_pixel"] = 0.0
                additional_metrics["hmd/overall_score_pixel"] = 0.0
        
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
    """详细的评估函数，记录per-class指标"""
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
    
    # 进行验证
    metrics = model.val(split=split, batch=batch, imgsz=imgsz)
    
    # Remove callback after validation (if method exists)
    # Note: remove_callback may not exist in all Ultralytics versions
    if hasattr(model, 'remove_callback'):
        try:
            model.remove_callback("on_val_end", capture_validator_callback)
        except (AttributeError, TypeError):
            # If remove_callback doesn't work, it's okay - callback will be overwritten on next use
            pass
    
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
                # Calculate HMD metrics using the same function as in training
                hmd_metrics = calculate_hmd_metrics_from_validator(
                    validator=validator,
                    trainer=minimal_trainer,
                    penalty_single=penalty_single,
                    penalty_none=penalty_none
                )
            else:
                logging.warning("⚠️ Validator not captured, using fallback HMD calculation")
                # Fallback: estimate from per-class metrics
                if hasattr(metrics, 'box') and hasattr(metrics.box, 'mean_class_results'):
                    per_class_metrics = metrics.box.mean_class_results
                    if per_class_metrics.shape[0] >= 2:  # At least 2 classes (Mentum and Hyoid)
                        mentum_recall = float(per_class_metrics[0, 1]) if per_class_metrics.shape[0] > 0 else 0.0
                        hyoid_recall = float(per_class_metrics[1, 1]) if per_class_metrics.shape[0] > 1 else 0.0
                        hmd_metrics['detection_rate'] = min(mentum_recall, hyoid_recall)
            
            logs.update({
                f"{split}/hmd/detection_rate": hmd_metrics['detection_rate'],
                f"{split}/hmd/rmse_pixel": hmd_metrics['rmse_pixel'],
                f"{split}/hmd/overall_score_pixel": hmd_metrics['overall_score_pixel'],
            })
            
            # Print HMD metrics in the same format as training output
            print(f"\n📊 Additional Metrics ({split}):", flush=True)
            print(f"   Precision: {mp:.4f} | Recall: {mr:.4f}", flush=True)
            print(f"   mAP50: {map50:.4f} | mAP50-95: {map:.4f} | Fitness: {fitness:.4f}", flush=True)
            print(f"\n📏 HMD Metrics (det_123, {split}):", flush=True)
            print(f"   Detection_Rate: {hmd_metrics['detection_rate']:.4f}", flush=True)
            print(f"   RMSE_HMD (pixel): {hmd_metrics['rmse_pixel']:.2f} px", flush=True)
            print(f"   Overall_Score (pixel): {hmd_metrics['overall_score_pixel']:.4f}", flush=True)
            
        except Exception as e:
            logging.warning(f"⚠️ Failed to calculate HMD metrics: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            # Set default values on error
            hmd_metrics = {
                'detection_rate': 0.0,
                'rmse_pixel': 0.0,
                'overall_score_pixel': 0.0,
            }
            logs.update({
                f"{split}/hmd/detection_rate": hmd_metrics['detection_rate'],
                f"{split}/hmd/rmse_pixel": hmd_metrics['rmse_pixel'],
                f"{split}/hmd/overall_score_pixel": hmd_metrics['overall_score_pixel'],
            })
    
    # log到W&B
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
            "detection_rate": hmd_metrics['detection_rate'],
            "rmse_hmd_pixel": hmd_metrics['rmse_pixel'],
            "rmse_pixel": hmd_metrics['rmse_pixel'],  # Alias for consistency
            "overall_score_pixel": hmd_metrics['overall_score_pixel'],
        })
    
    if iou_value is not None:
        result["iou"] = iou_value
    if dice_value is not None:
        result["dice"] = dice_value
    
    # Add HMD metrics to result dict (for all det_123 experiments)
    if database == 'det_123':
        try:
            # Get HMD metrics from logs (calculated above)
            result.update({
                'detection_rate': logs.get(f"{split}/hmd/detection_rate", 0.0),
                'rmse_hmd_pixel': logs.get(f"{split}/hmd/rmse_pixel", 0.0),
                'overall_score_pixel': logs.get(f"{split}/hmd/overall_score_pixel", 0.0),
            })
        except Exception as e:
            logging.warning(f"⚠️ Failed to add HMD metrics to result: {e}")
            # Add default values as fallback
            result.update({
                'detection_rate': 0.0,
                'rmse_hmd_pixel': 0.0,
                'overall_score_pixel': 0.0,
            })
    
    return result


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
    
    # HMD loss parameters (for det_123 database)
    parser.add_argument('--use_hmd_loss', action='store_true', help='Enable HMD loss for det_123 database')
    parser.add_argument('--hmd_loss_weight', type=float, default=0.1, help='Weight for HMD loss (λ_hmd, default: 0.1)')
    parser.add_argument('--hmd_penalty_single', type=float, default=500.0, help='Penalty value when only one target is detected (default: 500.0 pixels)')
    parser.add_argument('--hmd_penalty_none', type=float, default=1000.0, help='Penalty value when both targets are missed (default: 1000.0 pixels)')
    parser.add_argument('--hmd_penalty_coeff', type=float, default=0.5, help='Penalty coefficient for single detection weight (default: 0.5)')
    
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
    
    # Store HMD loss settings
    use_hmd_loss_flag = args.use_hmd_loss and args.database == 'det_123'
    hmd_loss_weight_value = args.hmd_loss_weight if use_hmd_loss_flag else 0.0
    hmd_penalty_single_value = args.hmd_penalty_single
    hmd_penalty_none_value = args.hmd_penalty_none
    hmd_penalty_coeff_value = args.hmd_penalty_coeff
    mentum_class = 0  # det_123: class 0 is Mentum
    hyoid_class = 1  # det_123: class 1 is Hyoid
    
    if use_hmd_loss_flag:
        logging.info(f"✅ HMD loss will be enabled: weight={hmd_loss_weight_value}, "
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
        
        # Set database attribute (needed for HMD loss check in validation)
        if isinstance(trainer.args, dict):
            trainer.args['database'] = args.database
        else:
            setattr(trainer.args, 'database', args.database)
        
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
        
        # Set HMD loss settings
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
        logging.info("W&B is disabled. Training will proceed without logging, but metrics will be printed to terminal after validation.")
    
    # Add callback to extract IoU, Dice, and HMD metrics after validation
    # Create callback with closure to capture args
    on_val_end_callback_with_args = create_on_val_end_callback(args)
    model.add_callback("on_val_end", on_val_end_callback_with_args)
    
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
        
        # 计算并记录所有指标 (val 和 test)
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
        
        # 导出ONNX模型并上传到W&B
        try:
            export_path = best_model.export(format="onnx", save_dir="./exports")
            artifact = wandb.Artifact("exported_model", type="model")
            artifact.add_file(export_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            logging.warning(f"⚠️ Failed to export model: {e}")
        
        wandb.finish()

