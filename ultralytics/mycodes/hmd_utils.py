"""
HMD (Hyomental Distance) utility functions for training and evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import joblib


def calculate_hmd_from_boxes(mentum_box: torch.Tensor, hyoid_box: torch.Tensor) -> torch.Tensor:
    """
    Calculate HMD from two bounding boxes in pixel coordinates
    
    Args:
        mentum_box: [x1, y1, x2, y2] format tensor
        hyoid_box: [x1, y1, x2, y2] format tensor
    
    Returns:
        HMD distance in pixels (scalar tensor)
    """
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_box
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_box
    
    # Calculate HMD
    hmd_dx = hyoid_x1 - mentum_x2
    mentum_y_center = (mentum_y1 + mentum_y2) / 2
    hyoid_y_center = (hyoid_y1 + hyoid_y2) / 2
    hmd_dy = hyoid_y_center - mentum_y_center
    hmd = torch.sqrt(hmd_dx**2 + hmd_dy**2)
    
    return hmd


def calculate_hmd_loss(pred_boxes: torch.Tensor, pred_conf: torch.Tensor, pred_cls: torch.Tensor,
                      target_boxes: torch.Tensor, target_cls: torch.Tensor,
                      mentum_class: int = 0, hyoid_class: int = 1,
                      penalty_single: float = 500.0, penalty_none: float = 1000.0,
                      penalty_coeff: float = 0.5) -> Tuple[torch.Tensor, Dict]:
    """
    Calculate HMD loss for a batch
    
    Args:
        pred_boxes: Predicted boxes [batch, num_pred, 4] in [x1, y1, x2, y2] format
        pred_conf: Predicted confidences [batch, num_pred]
        pred_cls: Predicted classes [batch, num_pred]
        target_boxes: Target boxes [batch, num_target, 4] in [x1, y1, x2, y2] format
        target_cls: Target classes [batch, num_target]
        mentum_class: Class ID for Mentum (default: 0)
        hyoid_class: Class ID for Hyoid (default: 1)
        penalty_single: Penalty when only one target detected
        penalty_none: Penalty when both targets missed
        penalty_coeff: Penalty coefficient for single detection
    
    Returns:
        Tuple of (hmd_loss, stats_dict)
    """
    batch_size = pred_boxes.shape[0]
    device = pred_boxes.device
    
    hmd_errors = []
    weights = []
    stats = {
        'both_detected': 0,
        'single_detected': 0,
        'none_detected': 0,
    }
    
    for b in range(batch_size):
        # Extract predictions for this image
        pred_mask = pred_conf[b] > 0.5  # Confidence threshold
        pred_boxes_b = pred_boxes[b][pred_mask]
        pred_conf_b = pred_conf[b][pred_mask]
        pred_cls_b = pred_cls[b][pred_mask]
        
        # Extract targets for this image
        target_boxes_b = target_boxes[b]
        target_cls_b = target_cls[b]
        
        # Find Mentum and Hyoid in predictions
        mentum_pred_mask = pred_cls_b == mentum_class
        hyoid_pred_mask = pred_cls_b == hyoid_class
        
        mentum_pred = pred_boxes_b[mentum_pred_mask]
        hyoid_pred = pred_boxes_b[hyoid_pred_mask]
        mentum_conf = pred_conf_b[mentum_pred_mask]
        hyoid_conf = pred_conf_b[hyoid_pred_mask]
        
        # Find Mentum and Hyoid in targets
        mentum_target_mask = target_cls_b == mentum_class
        hyoid_target_mask = target_cls_b == hyoid_class
        
        mentum_target = target_boxes_b[mentum_target_mask]
        hyoid_target = target_boxes_b[hyoid_target_mask]
        
        # Check detection status
        has_mentum_pred = len(mentum_pred) > 0
        has_hyoid_pred = len(hyoid_pred) > 0
        has_mentum_target = len(mentum_target) > 0
        has_hyoid_target = len(hyoid_target) > 0
        
        # Calculate HMD error and weight
        if has_mentum_pred and has_hyoid_pred and has_mentum_target and has_hyoid_target:
            # Both detected: calculate HMD error
            # Use highest confidence predictions
            mentum_idx = torch.argmax(mentum_conf) if len(mentum_conf) > 0 else 0
            hyoid_idx = torch.argmax(hyoid_conf) if len(hyoid_conf) > 0 else 0
            
            pred_hmd = calculate_hmd_from_boxes(mentum_pred[mentum_idx], hyoid_pred[hyoid_idx])
            gt_hmd = calculate_hmd_from_boxes(mentum_target[0], hyoid_target[0])
            
            hmd_error = torch.abs(pred_hmd - gt_hmd)
            weight = mentum_conf[mentum_idx] * hyoid_conf[hyoid_idx]
            
            hmd_errors.append(hmd_error)
            weights.append(weight)
            stats['both_detected'] += 1
            
        elif (has_mentum_pred or has_hyoid_pred) and (has_mentum_target and has_hyoid_target):
            # Single detected: use penalty
            detected_conf = mentum_conf[0] if has_mentum_pred else hyoid_conf[0]
            hmd_error = torch.tensor(penalty_single, device=device)
            weight = torch.min(mentum_conf[0] if has_mentum_pred else torch.tensor(0.0, device=device),
                             hyoid_conf[0] if has_hyoid_pred else torch.tensor(0.0, device=device)) * penalty_coeff
            
            hmd_errors.append(hmd_error)
            weights.append(weight)
            stats['single_detected'] += 1
            
        else:
            # None detected: use maximum penalty
            hmd_error = torch.tensor(penalty_none, device=device)
            weight = torch.tensor(1.0, device=device)
            
            hmd_errors.append(hmd_error)
            weights.append(weight)
            stats['none_detected'] += 1
    
    # Calculate weighted loss
    if len(hmd_errors) > 0:
        hmd_errors_tensor = torch.stack(hmd_errors)
        weights_tensor = torch.stack(weights)
        hmd_loss = (hmd_errors_tensor * weights_tensor).sum() / (weights_tensor.sum() + 1e-8)
    else:
        hmd_loss = torch.tensor(0.0, device=device)
    
    return hmd_loss, stats


def calculate_hmd_metrics_from_results(predictions_df: pd.DataFrame, 
                                      yolo_root: Path, dicom_root: Path,
                                      case_id: str, version: str,
                                      penalty_single: float = 500.0,
                                      penalty_none: float = 1000.0) -> Dict:
    """
    Calculate HMD metrics from prediction results (for validation/test)
    
    This function should be called with predictions from test_yolo.py output
    
    Args:
        predictions_df: DataFrame from predictions.joblib
        yolo_root: Root directory for yolo dataset
        dicom_root: Root directory for dicom dataset
        case_id: Dataset case ID (e.g., 'det_123')
        version: Dataset version (e.g., 'v3')
        penalty_single: Penalty for single detection
        penalty_none: Penalty for no detection
    
    Returns:
        Dictionary with HMD metrics
    """
    # This is a placeholder - actual implementation would:
    # 1. Load ground truth labels
    # 2. Match predictions with ground truth
    # 3. Calculate HMD for each image
    # 4. Compute statistics
    
    # For now, return placeholder values
    return {
        'detection_rate': 0.0,
        'rmse_hmd_pixel': 0.0,
        'rmse_hmd_mm': 0.0,
        'overall_score_pixel': 0.0,
        'overall_score_mm': 0.0,
    }

