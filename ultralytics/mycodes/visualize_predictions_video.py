"""
Generate video visualization of YOLO predictions on test dataset
Shows ground truth and predicted bounding boxes with HMD calculations
"""

import os
import cv2
import argparse
import logging
import numpy as np
import yaml
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from tqdm import tqdm
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Color definitions (BGR format for OpenCV)
COLORS = {
    'mentum_gt': (0, 255, 0),      # Green for GT Mentum
    'hyoid_gt': (0, 255, 255),     # Yellow for GT Hyoid
    'mentum_pred': (0, 165, 255),  # Orange for Pred Mentum
    'hyoid_pred': (255, 0, 255),   # Magenta for Pred Hyoid
    'hmd_line_gt': (0, 255, 0),   # Green line for GT HMD
    'hmd_line_pred': (255, 0, 0), # Blue line for Pred HMD
    'text': (255, 255, 255),       # White text
    'text_bg': (0, 0, 0),          # Black background for text
}


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Parse YOLO label file and return bbox list for each class
    
    Args:
        label_path: Path to YOLO label file
        img_width: Image width
        img_height: Image height
    
    Returns:
        Dict[class_id, List[(x1, y1, x2, y2)]]
    """
    boxes = {}
    
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Convert to (x1, y1, x2, y2) format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            if class_id not in boxes:
                boxes[class_id] = []
            boxes[class_id].append((x1, y1, x2, y2))
    
    return boxes


def calculate_hmd_from_boxes(mentum_box: np.ndarray, hyoid_box: np.ndarray, 
                             pixel_spacing: Optional[float] = None) -> float:
    """
    Calculate HMD distance from two bounding boxes
    
    Args:
        mentum_box: [x1, y1, x2, y2] format
        hyoid_box: [x1, y1, x2, y2] format
        pixel_spacing: Optional pixel spacing (mm/pixel) to convert to mm
    
    Returns:
        HMD distance in pixels (if pixel_spacing=None) or millimeters
    """
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_box
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_box
    
    # Calculate HMD in pixels
    hmd_dx = hyoid_x1 - mentum_x2
    mentum_y_center = (mentum_y1 + mentum_y2) / 2
    hyoid_y_center = (hyoid_y1 + hyoid_y2) / 2
    hmd_dy = hyoid_y_center - mentum_y_center
    hmd_pixel = np.sqrt(hmd_dx**2 + hmd_dy**2)
    
    # Convert to mm if pixel_spacing is provided
    if pixel_spacing is not None:
        return hmd_pixel * pixel_spacing
    return hmd_pixel


def draw_bbox_with_label(img: np.ndarray, bbox: Tuple[float, float, float, float], 
                         label: str, color: Tuple[int, int, int], 
                         is_gt: bool = True, conf: Optional[float] = None):
    """
    Draw bounding box with label on image
    
    Args:
        img: Image array (BGR format)
        bbox: (x1, y1, x2, y2) bounding box coordinates
        label: Label text
        color: BGR color tuple
        is_gt: Whether this is ground truth (affects line style)
        conf: Optional confidence score for predictions
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    line_thickness = 2 if is_gt else 2
    line_type = cv2.LINE_AA
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness, line_type)
    
    # Prepare label text
    if conf is not None:
        label_text = f"{label} {conf:.2f}"
    else:
        label_text = label
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # Draw text background
    text_x = x1
    text_y = y1 - 10 if y1 > 30 else y1 + text_height + 10
    cv2.rectangle(img, 
                 (text_x, text_y - text_height - 5), 
                 (text_x + text_width + 5, text_y + baseline + 5),
                 COLORS['text_bg'], -1)
    
    # Draw text
    cv2.putText(img, label_text, (text_x, text_y), font, font_scale, color, thickness, line_type)


def draw_hmd_line(img: np.ndarray, mentum_box: Tuple[float, float, float, float],
                 hyoid_box: Tuple[float, float, float, float], 
                 color: Tuple[int, int, int], label: str = ""):
    """
    Draw HMD line connecting Mentum and Hyoid boxes
    
    Args:
        img: Image array (BGR format)
        mentum_box: (x1, y1, x2, y2) Mentum bounding box
        hyoid_box: (x1, y1, x2, y2) Hyoid bounding box
        color: BGR color tuple for the line
        label: Optional label text to display
    """
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_box
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_box
    
    # Calculate connection points
    mentum_x_end = mentum_x2
    mentum_y_center = (mentum_y1 + mentum_y2) / 2
    hyoid_x_start = hyoid_x1
    hyoid_y_center = (hyoid_y1 + hyoid_y2) / 2
    
    # Draw line
    pt1 = (int(mentum_x_end), int(mentum_y_center))
    pt2 = (int(hyoid_x_start), int(hyoid_y_center))
    cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
    
    # Draw arrow at end
    arrow_length = 10
    angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    arrow_pt = (
        int(pt2[0] - arrow_length * np.cos(angle)),
        int(pt2[1] - arrow_length * np.sin(angle))
    )
    cv2.line(img, pt2, arrow_pt, color, 2, cv2.LINE_AA)
    
    # Draw label if provided
    if label:
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(img,
                     (mid_x - text_width // 2 - 3, mid_y - text_height - 5),
                     (mid_x + text_width // 2 + 3, mid_y + baseline + 5),
                     COLORS['text_bg'], -1)
        
        # Draw text
        cv2.putText(img, label, (mid_x - text_width // 2, mid_y), 
                   font, font_scale, color, thickness, cv2.LINE_AA)


def process_image(img_path: Path, model: YOLO, pixel_spacing_dict: Dict[str, float],
                 conf_threshold: float = 0.25) -> Tuple[np.ndarray, Dict]:
    """
    Process a single image: predict, load GT, calculate HMD, and draw visualization
    
    Returns:
        (annotated_image, metrics_dict)
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"⚠️ Failed to read image: {img_path}")
        return None, {}
    
    img_height, img_width = img.shape[:2]
    
    # Make a copy for annotation
    annotated_img = img.copy()
    
    # Load ground truth
    label_path = img_path.with_suffix('.txt')
    gt_boxes = parse_yolo_label(label_path, img_width, img_height)
    
    # Get predictions
    results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
    pred_boxes = {0: [], 1: []}  # 0: Mentum, 1: Hyoid
    
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes_xyxy, classes, confidences):
                if cls in [0, 1]:  # Only Mentum (0) and Hyoid (1)
                    x1, y1, x2, y2 = box
                    pred_boxes[cls].append((x1, y1, x2, y2, conf))
    
    # Get pixel spacing for this image
    pixel_spacing = None
    try:
        # Extract DICOM base name from filename
        from ultralytics.evaluate.calculate_hmd_from_yolo import extract_dicom_info_from_filename
        dicom_base, _ = extract_dicom_info_from_filename(img_path.name)
        
        # Try to find matching PixelSpacing in dictionary
        # Strategy 1: Exact match
        if dicom_base in pixel_spacing_dict:
            ps_val = pixel_spacing_dict[dicom_base]
            # Handle different formats (dict, list, float)
            if isinstance(ps_val, (dict, list)):
                from ultralytics.mycodes.train_yolo import _extract_pixel_spacing_value
                pixel_spacing = _extract_pixel_spacing_value(ps_val)
            else:
                pixel_spacing = float(ps_val) if ps_val is not None else None
        
        # Strategy 2: Normalized match (case-insensitive, remove extensions)
        if pixel_spacing is None:
            dicom_base_normalized = dicom_base.strip().lower()
            for key in pixel_spacing_dict.keys():
                key_normalized = key.strip().lower().replace('.dcm', '')
                key_clean = key_normalized.replace('_neutral', '').replace('_extended', '').replace('_ramped', '').strip('_').strip()
                dicom_base_clean = dicom_base_normalized.replace('_neutral', '').replace('_extended', '').replace('_ramped', '').strip('_').strip()
                
                if dicom_base_normalized == key_normalized or dicom_base_clean == key_clean:
                    ps_val = pixel_spacing_dict[key]
                    if isinstance(ps_val, (dict, list)):
                        from ultralytics.mycodes.train_yolo import _extract_pixel_spacing_value
                        pixel_spacing = _extract_pixel_spacing_value(ps_val)
                    else:
                        pixel_spacing = float(ps_val) if ps_val is not None else None
                    if pixel_spacing:
                        break
        
        # Strategy 3: Substring match
        if pixel_spacing is None:
            dicom_base_normalized = dicom_base.strip().lower()
            for key in pixel_spacing_dict.keys():
                key_normalized = key.strip().lower()
                if dicom_base_normalized in key_normalized or key_normalized in dicom_base_normalized:
                    ps_val = pixel_spacing_dict[key]
                    if isinstance(ps_val, (dict, list)):
                        from ultralytics.mycodes.train_yolo import _extract_pixel_spacing_value
                        pixel_spacing = _extract_pixel_spacing_value(ps_val)
                    else:
                        pixel_spacing = float(ps_val) if ps_val is not None else None
                    if pixel_spacing:
                        break
    except Exception as e:
        logging.debug(f"Failed to extract DICOM info: {e}")
    
    # If not found, try to get average
    if pixel_spacing is None and len(pixel_spacing_dict) > 0:
        try:
            from ultralytics.mycodes.train_yolo import _extract_pixel_spacing_value, _get_avg_pixel_spacing
            pixel_spacing = _get_avg_pixel_spacing(pixel_spacing_dict)
        except:
            # Fallback: simple average
            ps_values = []
            for val in pixel_spacing_dict.values():
                if isinstance(val, (int, float)):
                    ps_values.append(float(val))
                elif isinstance(val, dict) and 'truePixelSpacing' in val:
                    ps_values.append(float(val['truePixelSpacing']))
            if ps_values:
                pixel_spacing = np.mean(ps_values)
        
        if pixel_spacing:
            logging.debug(f"Using average pixel spacing: {pixel_spacing:.4f} mm/pixel")
    
    metrics = {
        'mentum_gt_count': len(gt_boxes.get(0, [])),
        'hyoid_gt_count': len(gt_boxes.get(1, [])),
        'mentum_pred_count': len(pred_boxes[0]),
        'hyoid_pred_count': len(pred_boxes[1]),
        'hmd_gt_pixel': None,
        'hmd_gt_mm': None,
        'hmd_pred_pixel': None,
        'hmd_pred_mm': None,
        'hmd_error_pixel': None,
        'hmd_error_mm': None,
    }
    
    # Draw ground truth boxes
    if 0 in gt_boxes and len(gt_boxes[0]) > 0:
        mentum_gt = gt_boxes[0][0]  # Take first one
        draw_bbox_with_label(annotated_img, mentum_gt, "Mentum (GT)", 
                           COLORS['mentum_gt'], is_gt=True)
    
    if 1 in gt_boxes and len(gt_boxes[1]) > 0:
        hyoid_gt = gt_boxes[1][0]  # Take first one
        draw_bbox_with_label(annotated_img, hyoid_gt, "Hyoid (GT)", 
                           COLORS['hyoid_gt'], is_gt=True)
    
    # Select best predictions (highest confidence) - one per class
    mentum_pred_best = None
    hyoid_pred_best = None
    
    if len(pred_boxes[0]) > 0:
        # Take highest confidence prediction for Mentum
        mentum_pred_best = max(pred_boxes[0], key=lambda x: x[4])
        draw_bbox_with_label(annotated_img, mentum_pred_best[:4], "Mentum (Pred)", 
                           COLORS['mentum_pred'], is_gt=False, conf=mentum_pred_best[4])
    
    if len(pred_boxes[1]) > 0:
        # Take highest confidence prediction for Hyoid
        hyoid_pred_best = max(pred_boxes[1], key=lambda x: x[4])
        draw_bbox_with_label(annotated_img, hyoid_pred_best[:4], "Hyoid (Pred)", 
                           COLORS['hyoid_pred'], is_gt=False, conf=hyoid_pred_best[4])
    
    # Calculate and draw HMD
    hmd_info_text = []
    
    # Ground truth HMD (one box per class)
    if 0 in gt_boxes and len(gt_boxes[0]) > 0 and 1 in gt_boxes and len(gt_boxes[1]) > 0:
        # Use first GT box for each class (YOLO format typically has one GT per class)
        mentum_gt_box = gt_boxes[0][0]  # Take first one
        hyoid_gt_box = gt_boxes[1][0]   # Take first one
        mentum_gt = np.array(mentum_gt_box)
        hyoid_gt = np.array(hyoid_gt_box)
        hmd_gt_pixel = calculate_hmd_from_boxes(mentum_gt, hyoid_gt)
        hmd_gt_mm = calculate_hmd_from_boxes(mentum_gt, hyoid_gt, pixel_spacing) if pixel_spacing else None
        
        metrics['hmd_gt_pixel'] = hmd_gt_pixel
        metrics['hmd_gt_mm'] = hmd_gt_mm
        
        # Draw GT HMD line
        draw_hmd_line(annotated_img, mentum_gt_box, hyoid_gt_box, 
                     COLORS['hmd_line_gt'], f"GT: {hmd_gt_pixel:.1f}px")
        
        hmd_info_text.append(f"GT HMD: {hmd_gt_pixel:.1f} px")
        if hmd_gt_mm:
            hmd_info_text.append(f"{hmd_gt_mm:.2f} mm")
    
    # Predicted HMD (one box per class - highest confidence)
    if mentum_pred_best is not None and hyoid_pred_best is not None:
        # Use the same best predictions that were drawn
        mentum_pred_box = mentum_pred_best[:4]  # (x1, y1, x2, y2)
        hyoid_pred_box = hyoid_pred_best[:4]
        mentum_pred = np.array(mentum_pred_box)
        hyoid_pred = np.array(hyoid_pred_box)
        hmd_pred_pixel = calculate_hmd_from_boxes(mentum_pred, hyoid_pred)
        hmd_pred_mm = calculate_hmd_from_boxes(mentum_pred, hyoid_pred, pixel_spacing) if pixel_spacing else None
        
        metrics['hmd_pred_pixel'] = hmd_pred_pixel
        metrics['hmd_pred_mm'] = hmd_pred_mm
        
        # Draw Pred HMD line
        draw_hmd_line(annotated_img, mentum_pred_box, hyoid_pred_box, 
                     COLORS['hmd_line_pred'], f"Pred: {hmd_pred_pixel:.1f}px")
        
        hmd_info_text.append(f"Pred HMD: {hmd_pred_pixel:.1f} px")
        if hmd_pred_mm:
            hmd_info_text.append(f"{hmd_pred_mm:.2f} mm")
        
        # Calculate error if both GT and Pred exist
        if metrics['hmd_gt_pixel'] is not None:
            hmd_error_pixel = abs(hmd_pred_pixel - hmd_gt_pixel)
            hmd_error_mm = abs(hmd_pred_mm - hmd_gt_mm) if (hmd_pred_mm and hmd_gt_mm) else None
            
            metrics['hmd_error_pixel'] = hmd_error_pixel
            metrics['hmd_error_mm'] = hmd_error_mm
            
            hmd_info_text.append(f"Error: {hmd_error_pixel:.1f} px")
            if hmd_error_mm:
                hmd_info_text.append(f"{hmd_error_mm:.2f} mm")
    
    # Draw HMD info text at top-left corner
    if hmd_info_text:
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        for i, text in enumerate(hmd_info_text):
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background
            cv2.rectangle(annotated_img,
                         (10, y_offset + i * (text_height + 10) - text_height - 5),
                         (10 + text_width + 10, y_offset + i * (text_height + 10) + baseline + 5),
                         COLORS['text_bg'], -1)
            
            # Draw text
            cv2.putText(annotated_img, text, (10, y_offset + i * (text_height + 10)),
                       font, font_scale, COLORS['text'], thickness, cv2.LINE_AA)
    
    return annotated_img, metrics


def calculate_and_print_statistics(model: YOLO, image_paths: List[Path], 
                                   all_metrics: List[Dict], pixel_spacing_dict: Dict,
                                   conf_threshold: float = 0.25):
    """
    Calculate and print comprehensive statistics including detection metrics and HMD metrics
    """
    from ultralytics.mycodes.hmd_utils import calculate_hmd_from_boxes
    
    # Use YOLO's validation functionality to calculate detection metrics
    # Create a temporary dataset file for validation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_txt = f.name
        for img_path in image_paths:
            f.write(f"{img_path}\n")
    
    try:
        # Run validation with the model
        # Note: We need to ensure each class only has one box (highest confidence)
        # This is already handled in process_image, but we need to do it for validation too
        
        # Collect all predictions and GT for manual calculation
        all_gt_boxes = {0: [], 1: []}  # class_id -> list of boxes per image
        all_pred_boxes = {0: [], 1: []}  # class_id -> list of (box, conf) per image
        all_hmd_gt_pixel = []
        all_hmd_pred_pixel = []
        all_hmd_gt_mm = []
        all_hmd_pred_mm = []
        
        for img_path, metrics in zip(image_paths, all_metrics):
            # Load GT boxes
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_height, img_width = img.shape[:2]
                    gt_boxes = parse_yolo_label(label_path, img_width, img_height)
                    
                    # Store GT boxes (one per class per image)
                    for cls in [0, 1]:
                        if cls in gt_boxes and len(gt_boxes[cls]) > 0:
                            all_gt_boxes[cls].append(gt_boxes[cls][0])  # Take first one
                        else:
                            all_gt_boxes[cls].append(None)
            
            # Get predictions (already filtered to highest confidence per class in process_image)
            # Re-run prediction to get all boxes, then filter
            results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
            pred_boxes_img = {0: None, 1: None}  # Store only highest confidence per class
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Group by class and take highest confidence
                    for cls in [0, 1]:
                        cls_boxes = [(box, conf) for box, c, conf in zip(boxes_xyxy, classes, confidences) 
                                    if c == cls]
                        if cls_boxes:
                            # Take highest confidence
                            best_box, best_conf = max(cls_boxes, key=lambda x: x[1])
                            pred_boxes_img[cls] = (best_box, best_conf)
            
            # Store predictions
            for cls in [0, 1]:
                all_pred_boxes[cls].append(pred_boxes_img[cls])
            
            # Store HMD values if available
            if metrics.get('hmd_gt_pixel') is not None:
                all_hmd_gt_pixel.append(metrics['hmd_gt_pixel'])
                if metrics.get('hmd_gt_mm') is not None:
                    all_hmd_gt_mm.append(metrics['hmd_gt_mm'])
            
            if metrics.get('hmd_pred_pixel') is not None:
                all_hmd_pred_pixel.append(metrics['hmd_pred_pixel'])
                if metrics.get('hmd_pred_mm') is not None:
                    all_hmd_pred_mm.append(metrics['hmd_pred_mm'])
        
        # Use YOLO's validation to get detection metrics
        # Create a dataset YAML file temporarily
        dataset_yaml = {
            'path': str(image_paths[0].parent.parent.parent.parent) if len(image_paths) > 0 else '.',
            'val': temp_txt,
            'names': {0: 'Mentum', 1: 'Hyoid'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(dataset_yaml, f)
            temp_yaml = f.name
        
        try:
            # Run validation with keep_top_conf_per_class if supported
            # Note: We need to ensure each class only has one box (highest confidence)
            # This is handled by filtering predictions in process_image, but we also need it for validation
            try:
                # Try to use keep_top_conf_per_class if the model supports it
                metrics_result = model.val(data=temp_yaml, conf=conf_threshold, verbose=False, plots=False,
                                         keep_top_conf_per_class=True, conf_low=0.0)
            except TypeError:
                # If keep_top_conf_per_class is not supported, use regular validation
                # The filtering will be done manually if needed
                metrics_result = model.val(data=temp_yaml, conf=conf_threshold, verbose=False, plots=False)
            
            # Extract metrics
            if hasattr(metrics_result, 'box'):
                box_metrics = metrics_result.box
                precision_all = float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0
                recall_all = float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0
                map50_all = float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0
                map50_95_all = float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0
                
                # Per-class metrics
                precision_per_class = {}
                recall_per_class = {}
                map50_per_class = {}
                map50_95_per_class = {}
                
                if hasattr(box_metrics, 'maps'):
                    maps = box_metrics.maps  # Per-class mAP50-95
                    if hasattr(box_metrics, 'map50'):
                        map50s = box_metrics.map50  # Per-class mAP50
                    else:
                        map50s = [0.0] * len(maps)
                    
                    for i, (map_val, map50_val) in enumerate(zip(maps, map50s)):
                        map50_95_per_class[i] = float(map_val)
                        map50_per_class[i] = float(map50_val)
                
                # Get per-class precision and recall if available
                if hasattr(box_metrics, 'p'):
                    p_per_class = box_metrics.p  # Per-class precision
                    for i, p_val in enumerate(p_per_class):
                        precision_per_class[i] = float(p_val)
                
                if hasattr(box_metrics, 'r'):
                    r_per_class = box_metrics.r  # Per-class recall
                    for i, r_val in enumerate(r_per_class):
                        recall_per_class[i] = float(r_val)
            else:
                precision_all = recall_all = map50_all = map50_95_all = 0.0
                precision_per_class = recall_per_class = map50_per_class = map50_95_per_class = {}
            
            # Calculate Fitness
            fitness = map50_all * 0.1 + map50_95_all * 0.9
            
            # Print detection metrics
            logging.info("\n" + "=" * 60)
            logging.info("📊 Detection Metrics:")
            logging.info("=" * 60)
            logging.info(f"                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)")
            logging.info(f"                   all      {len(image_paths):5d}      {len(image_paths) * 2:5d}      {precision_all:.3f}      {recall_all:.3f}     {map50_all:.3f}     {map50_95_all:.3f}")
            
            # Per-class metrics
            class_names = {0: 'Mentum', 1: 'Hyoid'}
            for cls_id in [0, 1]:
                cls_name = class_names[cls_id]
                p = precision_per_class.get(cls_id, 0.0)
                r = recall_per_class.get(cls_id, 0.0)
                m50 = map50_per_class.get(cls_id, 0.0)
                m50_95 = map50_95_per_class.get(cls_id, 0.0)
                logging.info(f"                {cls_name:8s}      {len(image_paths):5d}      {len(image_paths):5d}      {p:.3f}      {r:.3f}     {m50:.3f}     {m50_95:.3f}")
            
            logging.info("\n📊 Additional Metrics:")
            logging.info(f"   Precision: {precision_all:.4f} | Recall: {recall_all:.4f}")
            logging.info(f"   mAP50: {map50_all:.4f} | mAP50-95: {map50_95_all:.4f} | Fitness: {fitness:.4f}")
            
        finally:
            # Clean up temp files
            try:
                os.unlink(temp_yaml)
            except:
                pass
        
        # Calculate HMD metrics
        detection_rate = 0.0
        rmse_pixel = 0.0
        mae_pixel = 0.0
        overall_score_pixel = 0.0
        rmse_mm = 0.0
        mae_mm = 0.0
        overall_score_mm = 0.0
        
        # Count images with both GT and both Pred
        both_gt_and_pred_count = sum(1 for m in all_metrics 
                                    if m.get('mentum_gt_count', 0) > 0 and m.get('hyoid_gt_count', 0) > 0
                                    and m.get('mentum_pred_count', 0) > 0 and m.get('hyoid_pred_count', 0) > 0)
        total_images = len(all_metrics)
        
        if total_images > 0:
            detection_rate = both_gt_and_pred_count / total_images
        
        # Calculate HMD errors (only for images where both GT and both Pred exist)
        if all_hmd_gt_pixel and all_hmd_pred_pixel and len(all_hmd_gt_pixel) == len(all_hmd_pred_pixel):
            hmd_errors_pixel = [abs(gt - pred) for gt, pred in zip(all_hmd_gt_pixel, all_hmd_pred_pixel)]
            if hmd_errors_pixel:
                rmse_pixel = np.sqrt(np.mean(np.array(hmd_errors_pixel)**2))
                mae_pixel = np.mean(hmd_errors_pixel)
                overall_score_pixel = detection_rate / (1 + rmse_pixel / 1000.0)
        
        if all_hmd_gt_mm and all_hmd_pred_mm and len(all_hmd_gt_mm) == len(all_hmd_pred_mm):
            hmd_errors_mm = [abs(gt - pred) for gt, pred in zip(all_hmd_gt_mm, all_hmd_pred_mm)]
            if hmd_errors_mm:
                rmse_mm = np.sqrt(np.mean(np.array(hmd_errors_mm)**2))
                mae_mm = np.mean(hmd_errors_mm)
                overall_score_mm = detection_rate / (1 + rmse_mm / 100.0)
        
        # Print HMD metrics
        logging.info("\n📏 HMD Metrics (det_123):")
        logging.info(f"   Detection_Rate: {detection_rate:.4f}")
        logging.info("\n   📊 No Penalty (only both detected cases, all based on real boxes):")
        if rmse_pixel > 0:
            logging.info(f"      RMSE_HMD (pixel): {rmse_pixel:.2f} px")
            logging.info(f"      MAE_HMD (pixel): {mae_pixel:.2f} px")
            logging.info(f"      Overall_Score (pixel): {overall_score_pixel:.4f}")
        else:
            logging.info(f"      RMSE_HMD (pixel): N/A (no both detected cases)")
            logging.info(f"      MAE_HMD (pixel): N/A (no both detected cases)")
            logging.info(f"      Overall_Score (pixel): N/A")
        
        logging.info("\n   📏 No Penalty (mm):")
        if rmse_mm > 0:
            logging.info(f"      RMSE_HMD (mm): {rmse_mm:.2f} mm")
            logging.info(f"      MAE_HMD (mm): {mae_mm:.2f} mm")
            logging.info(f"      Overall_Score (mm): {overall_score_mm:.4f}")
        else:
            logging.info(f"      RMSE_HMD (mm): N/A (no both detected cases)")
            logging.info(f"      MAE_HMD (mm): N/A (no both detected cases)")
            logging.info(f"      Overall_Score (mm): N/A")
        
        logging.info("=" * 60)
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_txt)
        except:
            pass


def create_video_from_images(image_list: List[np.ndarray], output_path: Path, 
                            fps: float = 10.0):
    """
    Create video from list of images
    
    Args:
        image_list: List of image arrays (all must have same size)
        output_path: Output video file path
        fps: Frames per second
    """
    if not image_list:
        logging.error("No images to create video")
        return
    
    height, width = image_list[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        logging.error(f"Failed to create video writer: {output_path}")
        return
    
    for img in tqdm(image_list, desc="Writing video frames"):
        video_writer.write(img)
    
    video_writer.release()
    logging.info(f"✅ Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate video visualization of YOLO predictions")
    parser.add_argument('model_path', type=str, help='Path to trained model weights (best.pt)')
    parser.add_argument('--test_txt', type=str, 
                       default='yolo_dataset/det_123/v3/test_ES.txt',
                       help='Path to test dataset txt file')
    parser.add_argument('--output', type=str, default='runs/visualize/predictions_video.mp4',
                       help='Output video path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--fps', type=float, default=10.0, help='Video FPS')
    parser.add_argument('--max_images', type=int, default=None, 
                       help='Maximum number of images to process (None for all)')
    parser.add_argument('--pixel_spacing_path', type=str,
                       default='dicom_dataset/Dicom_PixelSpacing_DA.joblib',
                       help='Path to pixel spacing dictionary')
    
    args = parser.parse_args()
    
    # Load model
    logging.info(f"📦 Loading model from: {args.model_path}")
    model = YOLO(args.model_path)
    
    # Load pixel spacing dictionary
    pixel_spacing_dict = {}
    try:
        from mycodes.hmd_utils import load_pixel_spacing_dict
        pixel_spacing_path = Path(args.pixel_spacing_path)
        if pixel_spacing_path.is_absolute():
            pixel_spacing_dict = load_pixel_spacing_dict(pixel_spacing_path)
        else:
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            pixel_spacing_dict = load_pixel_spacing_dict(project_root / pixel_spacing_path)
        
        if pixel_spacing_dict:
            logging.info(f"✅ Loaded PixelSpacing dictionary with {len(pixel_spacing_dict)} entries")
        else:
            logging.warning("⚠️ PixelSpacing dictionary is empty or not found")
    except Exception as e:
        logging.warning(f"⚠️ Failed to load PixelSpacing dictionary: {e}")
    
    # Read test image paths
    test_txt_path = Path(args.test_txt)
    if not test_txt_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        test_txt_path = project_root / test_txt_path
    
    if not test_txt_path.exists():
        logging.error(f"Test txt file not found: {test_txt_path}")
        return
    
    with open(test_txt_path, 'r') as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    logging.info(f"📸 Processing {len(image_paths)} images")
    
    # Process images
    annotated_images = []
    all_metrics = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        if not img_path.exists():
            logging.warning(f"⚠️ Image not found: {img_path}")
            continue
        
        annotated_img, metrics = process_image(img_path, model, pixel_spacing_dict, args.conf)
        
        if annotated_img is not None:
            annotated_images.append(annotated_img)
            all_metrics.append(metrics)
    
    if not annotated_images:
        logging.error("No images were processed successfully")
        return
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video
    logging.info(f"🎬 Creating video with {len(annotated_images)} frames...")
    create_video_from_images(annotated_images, output_path, args.fps)
    
    # Calculate and print comprehensive statistics
    calculate_and_print_statistics(model, image_paths, all_metrics, pixel_spacing_dict, args.conf)
    
    logging.info(f"\n✅ Video visualization complete: {output_path}")


if __name__ == '__main__':
    main()

