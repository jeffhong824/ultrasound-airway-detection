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


def calculate_hmd_error_from_boxes(mentum_pred_box, hyoid_pred_box, 
                                    mentum_gt_box, hyoid_gt_box,
                                    pixel_spacing: Optional[float] = None) -> float:
    """
    Calculate HMD error from predicted and ground truth boxes.
    This function uses the same calculation logic as calculate_hmd_loss.
    
    Args:
        mentum_pred_box: Predicted Mentum box [x1, y1, x2, y2] (torch.Tensor or np.ndarray)
        hyoid_pred_box: Predicted Hyoid box [x1, y1, x2, y2] (torch.Tensor or np.ndarray)
        mentum_gt_box: Ground truth Mentum box [x1, y1, x2, y2] (torch.Tensor or np.ndarray)
        hyoid_gt_box: Ground truth Hyoid box [x1, y1, x2, y2] (torch.Tensor or np.ndarray)
        pixel_spacing: Optional pixel spacing (mm/pixel) for mm calculation
    
    Returns:
        HMD error (float) - same calculation as in calculate_hmd_loss
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    # Convert to torch tensors if needed
    if isinstance(mentum_pred_box, np.ndarray):
        mentum_pred_box = torch.from_numpy(mentum_pred_box).float()
    if isinstance(hyoid_pred_box, np.ndarray):
        hyoid_pred_box = torch.from_numpy(hyoid_pred_box).float()
    if isinstance(mentum_gt_box, np.ndarray):
        mentum_gt_box = torch.from_numpy(mentum_gt_box).float()
    if isinstance(hyoid_gt_box, np.ndarray):
        hyoid_gt_box = torch.from_numpy(hyoid_gt_box).float()
    
    # Calculate HMD from boxes
    pred_hmd = calculate_hmd_from_boxes(mentum_pred_box, hyoid_pred_box, pixel_spacing=pixel_spacing)
    gt_hmd = calculate_hmd_from_boxes(mentum_gt_box, hyoid_gt_box, pixel_spacing=pixel_spacing)
    
    # Use the same calculation as in calculate_hmd_loss
    eps = 1e-8
    # 1. Smooth L1 Loss
    hmd_error_smooth_l1 = F.smooth_l1_loss(pred_hmd, gt_hmd, reduction='none', beta=1.0)
    
    # 2. Scale-invariant loss (relative error)
    relative_error = torch.abs(pred_hmd - gt_hmd) / (gt_hmd + eps)
    
    # Combine Smooth L1 and relative error (same weights as in calculate_hmd_loss)
    hmd_error = 0.7 * hmd_error_smooth_l1 + 0.3 * relative_error * gt_hmd
    
    # 3. HMD direction constraint penalty
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_pred_box
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_pred_box
    direction_penalty = F.relu(mentum_x2 - hyoid_x1)  # Only penalize if wrong order
    direction_penalty_normalized = direction_penalty / (gt_hmd + eps) * 0.1  # 10% weight
    
    # Add direction penalty to HMD error
    hmd_error = hmd_error + direction_penalty_normalized
    
    # Convert to float
    if isinstance(hmd_error, torch.Tensor):
        hmd_error = hmd_error.item()
    
    return float(hmd_error)


def calculate_hmd_from_boxes(mentum_box: torch.Tensor, hyoid_box: torch.Tensor, 
                            pixel_spacing: Optional[float] = None) -> torch.Tensor:
    """
    Calculate HMD from two bounding boxes
    
    Args:
        mentum_box: [x1, y1, x2, y2] format tensor
        hyoid_box: [x1, y1, x2, y2] format tensor
        pixel_spacing: Optional pixel spacing (mm/pixel) to convert to mm. 
                      If None, returns pixel distance. If provided, returns mm distance.
    
    Returns:
        HMD distance in pixels (if pixel_spacing=None) or millimeters (if pixel_spacing provided)
    """
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_box
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_box
    
    # Calculate HMD in pixels
    hmd_dx = hyoid_x1 - mentum_x2
    mentum_y_center = (mentum_y1 + mentum_y2) / 2
    hyoid_y_center = (hyoid_y1 + hyoid_y2) / 2
    hmd_dy = hyoid_y_center - mentum_y_center
    hmd_pixel = torch.sqrt(hmd_dx**2 + hmd_dy**2)
    
    # Convert to mm if pixel_spacing is provided
    if pixel_spacing is not None:
        hmd = hmd_pixel * pixel_spacing
    else:
        hmd = hmd_pixel
    
    return hmd


def load_pixel_spacing_dict(joblib_path: Optional[Path] = None) -> Dict[str, float]:
    """
    Load PixelSpacing dictionary from Dicom_PixelSpacing_DA.joblib file
    
    Args:
        joblib_path: Path to Dicom_PixelSpacing_DA.joblib file. 
                    If None, tries to find it in dicom_dataset/ directory relative to project root.
    
    Returns:
        Dictionary mapping DICOM base name to PixelSpacing (mm/pixel)
        Returns empty dict if file not found or load fails
    """
    if joblib_path is None:
        # Try to find the file relative to project root
        # Assume we're in ultralytics/mycodes/, so go up 2 levels to project root
        project_root = Path(__file__).parent.parent.parent
        joblib_path = project_root / "dicom_dataset" / "Dicom_PixelSpacing_DA.joblib"
    
    if not joblib_path.exists():
        return {}
    
    try:
        data = joblib.load(joblib_path)
        # Convert to dict if it's not already
        if isinstance(data, dict):
            return data
        else:
            # If it's a DataFrame or other structure, convert appropriately
            return {}
    except Exception as e:
        print(f"⚠️  Failed to load PixelSpacing dictionary from {joblib_path}: {e}")
        return {}


def calculate_hmd_loss(pred_boxes: torch.Tensor, pred_conf: torch.Tensor, pred_cls: torch.Tensor,
                      target_boxes: torch.Tensor, target_cls: torch.Tensor,
                      mentum_class: int = 0, hyoid_class: int = 1,
                      penalty_single: float = 500.0, penalty_none: float = 1000.0,
                      penalty_coeff: float = 0.5,
                      pixel_spacing: Optional[float] = None) -> Tuple[torch.Tensor, Dict]:
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
    
    hmd_errors = []  # For loss calculation (with penalties)
    weights = []
    hmd_errors_no_penalty = []  # For metrics calculation (only both_detected cases, no penalties)
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
            
            pred_hmd = calculate_hmd_from_boxes(mentum_pred[mentum_idx], hyoid_pred[hyoid_idx], pixel_spacing=pixel_spacing)
            gt_hmd = calculate_hmd_from_boxes(mentum_target[0], hyoid_target[0], pixel_spacing=pixel_spacing)
            
            # 1. Use Smooth L1 Loss instead of absolute error for robustness to outliers
            # Smooth L1 is more robust to outliers than L1, and smoother than L2 near zero
            eps = 1e-8
            hmd_error_smooth_l1 = F.smooth_l1_loss(pred_hmd, gt_hmd, reduction='none', beta=1.0)
            
            # 2. Add scale-invariant loss (relative error)
            # Different patients may have different HMD ranges, so relative error is more meaningful
            relative_error = torch.abs(pred_hmd - gt_hmd) / (gt_hmd + eps)
            
            # Combine Smooth L1 and relative error (weighted combination)
            # Smooth L1 for absolute accuracy, relative error for scale-invariance
            hmd_error = 0.7 * hmd_error_smooth_l1 + 0.3 * relative_error * gt_hmd
            
            # 3. Add HMD direction constraint penalty
            # Hyoid should be to the right of Mentum (x direction: hyoid_x1 > mentum_x2)
            mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_pred[mentum_idx]
            hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_pred[hyoid_idx]
            # If order is wrong (mentum_x2 > hyoid_x1), apply penalty
            direction_penalty = F.relu(mentum_x2 - hyoid_x1)  # Only penalize if wrong order
            # Normalize direction penalty to be comparable with HMD error scale
            # Typical HMD is ~200-500 pixels, so normalize direction penalty accordingly
            direction_penalty_normalized = direction_penalty / (gt_hmd + eps) * 0.1  # 10% weight
            
            # Add direction penalty to HMD error
            hmd_error = hmd_error + direction_penalty_normalized
            
            # Calculate absolute error for MAE (without direction penalty for no-penalty version)
            abs_error = torch.abs(pred_hmd - gt_hmd)
            
            weight = mentum_conf[mentum_idx] * hyoid_conf[hyoid_idx]
            
            hmd_errors.append(hmd_error)
            weights.append(weight)
            # For no-penalty version, store absolute error (pixel/mm distance)
            hmd_errors_no_penalty.append(abs_error)
            stats['both_detected'] += 1
            
        elif (has_mentum_pred or has_hyoid_pred) and (has_mentum_target and has_hyoid_target):
            # Single detected: use penalty
            # CRITICAL: Make penalty depend on predictions to maintain gradient
            # penalty = base_penalty * (1.0 + min_conf) where min_conf depends on predictions
            # This ensures gradient flows through predictions while keeping penalty value reasonable
            if has_mentum_pred:
                mentum_conf_val = mentum_conf[0] if len(mentum_conf) > 0 else torch.tensor(0.0, device=device)
            else:
                mentum_conf_val = torch.tensor(0.0, device=device)
            
            if has_hyoid_pred:
                hyoid_conf_val = hyoid_conf[0] if len(hyoid_conf) > 0 else torch.tensor(0.0, device=device)
            else:
                hyoid_conf_val = torch.tensor(0.0, device=device)
            
            min_conf = torch.min(mentum_conf_val, hyoid_conf_val)
            # Use min_conf to create gradient: penalty scales with confidence
            # Higher confidence but missing one target = higher penalty (encourages detecting both)
            hmd_error = torch.tensor(penalty_single, device=device) * (1.0 + min_conf)
            weight = min_conf * penalty_coeff
            
            hmd_errors.append(hmd_error)
            weights.append(weight)
            stats['single_detected'] += 1
            
        else:
            # None detected: use maximum penalty
            # CRITICAL: Make penalty depend on predictions to maintain gradient
            # penalty = base_penalty * (1.0 + max_conf) where max_conf depends on predictions
            # This ensures gradient flows through predictions while keeping penalty value reasonable
            max_conf = pred_conf[b].max() if pred_conf[b].numel() > 0 else torch.tensor(0.0, device=device)
            # Use max_conf to create gradient: penalty scales with confidence
            # Higher confidence but no detection = higher penalty (encourages detection)
            hmd_error = torch.tensor(penalty_none, device=device) * (1.0 + max_conf)
            weight = torch.tensor(1.0, device=device, requires_grad=False)
            
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
    
    # Calculate metrics for stats
    # With penalty: includes all cases (both_detected, single_detected, none_detected)
    # Without penalty: only both_detected cases
    metrics = {
        'rmse_with_penalty': 0.0,
        'mae_with_penalty': 0.0,
        'rmse_no_penalty': 0.0,
        'mae_no_penalty': 0.0,
    }
    
    if len(hmd_errors) > 0:
        # With penalty: calculate from all errors (including penalties)
        hmd_errors_with_penalty = [e.item() if isinstance(e, torch.Tensor) else e for e in hmd_errors]
        if hmd_errors_with_penalty:
            errors_array = np.array(hmd_errors_with_penalty)
            metrics['rmse_with_penalty'] = float(np.sqrt(np.mean(errors_array**2)))
            metrics['mae_with_penalty'] = float(np.mean(np.abs(errors_array)))
    
    if len(hmd_errors_no_penalty) > 0:
        # Without penalty: only calculate from both_detected cases
        hmd_errors_no_penalty_list = [e.item() if isinstance(e, torch.Tensor) else e for e in hmd_errors_no_penalty]
        if hmd_errors_no_penalty_list:
            errors_array = np.array(hmd_errors_no_penalty_list)
            metrics['rmse_no_penalty'] = float(np.sqrt(np.mean(errors_array**2)))
            metrics['mae_no_penalty'] = float(np.mean(np.abs(errors_array)))
    
    # Add metrics to stats
    stats.update(metrics)
    
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

