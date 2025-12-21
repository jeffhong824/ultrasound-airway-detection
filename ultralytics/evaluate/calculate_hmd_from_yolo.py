# -*- coding: utf-8 -*-
"""
calculate_hmd_from_yolo.py

Calculate HMD (Hyomental Distance) from YOLO prediction results (or ground truth labels)
Outputs both pixel distance and millimeter distance (with PixelSpacing conversion)

Features:
1. Read bbox from yolo_dataset (supports ground truth or YOLO prediction results)
2. Find corresponding DICOM files in dicom_dataset and read PixelSpacing
3. Calculate HMD distance (grouped by pose: Neutral, Extended, Ramped)
4. Support batch processing for entire patient_data directory

Usage:
    # Single patient
    python calculate_hmd_from_yolo.py --case-id det_123 --patient-id 0834980
    
    # Batch processing
    python calculate_hmd_from_yolo.py --case-id det_123 --batch
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pydicom as dicom
import joblib
from tqdm import tqdm
import sys

# Add common_eval path
sys.path.append(str(Path(__file__).parent))
from common_eval import get_class_names


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Parse YOLO label file and return bbox list for each class
    
    Args:
        label_path: Path to YOLO label file
        img_width: Image width
        img_height: Image height
    
    Returns:
        Dict[class_id, List[(x1, y1, x2, y2)]]
        Returns empty list if class doesn't exist
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


def parse_predictions_joblib(joblib_path: Path, image_name: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Parse predictions from joblib file (output from test_yolo.py)
    
    Args:
        joblib_path: Path to predictions.joblib file
        image_name: Image filename (e.g., "0587648_Quick ID_20231213_140104_B[000].png")
    
    Returns:
        Dict[class_id, List[(x1, y1, x2, y2)]]
        Returns empty dict if not found
    """
    boxes = {}
    
    if not joblib_path.exists():
        return boxes
    
    try:
        # Load predictions DataFrame
        pred_df = joblib.load(joblib_path)
        
        # Extract image_id from filename (remove extension)
        # test_yolo.py uses: os.path.splitext(os.path.basename(result.path))[0]
        # result.path comes from test.txt which contains full paths like:
        # "D:/.../patient_data/0834980/0834980_Quick ID_20240509_155005_B.dcm_Neutral[034].png"
        # So image_id = basename without extension = "0834980_Quick ID_20240509_155005_B.dcm_Neutral[034]"
        image_id = Path(image_name).stem
        
        # Filter predictions for this image (exact match)
        image_preds = pred_df[pred_df['image_id'] == image_id]
        
        if len(image_preds) == 0:
            # Try alternative matching strategies
            # Strategy 1: Check if image_id is contained in any prediction image_id
            matching_preds = pred_df[pred_df['image_id'].str.contains(image_id, na=False, regex=False)]
            if len(matching_preds) > 0:
                image_preds = matching_preds
            else:
                # Strategy 2: Check if any prediction image_id is contained in our image_id
                matching_preds = pred_df[pred_df['image_id'].apply(lambda x: str(x) in image_id if pd.notna(x) else False)]
                if len(matching_preds) > 0:
                    image_preds = matching_preds
                else:
                    # Strategy 3: Try matching without .dcm part (some predictions might not have it)
                    image_id_no_dcm = image_id.replace('.dcm', '')
                    matching_preds = pred_df[pred_df['image_id'] == image_id_no_dcm]
                    if len(matching_preds) > 0:
                        image_preds = matching_preds
                    else:
                        # Strategy 4: Try matching with .dcm added (if prediction has it but we don't)
                        if '.dcm' not in image_id:
                            image_id_with_dcm = image_id.replace('_B ', '_B.dcm ').replace('_B[', '_B.dcm[')
                            matching_preds = pred_df[pred_df['image_id'] == image_id_with_dcm]
                            if len(matching_preds) > 0:
                                image_preds = matching_preds
                    
                    # If still not found, show debug info (only once per file)
                    if len(image_preds) == 0:
                        if not hasattr(parse_predictions_joblib, '_debug_printed'):
                            available_ids = pred_df['image_id'].unique()[:5]  # Show first 5
                            print(f"⚠️  Debug: Image ID '{image_id}' not found in predictions.joblib")
                            print(f"    Available image_ids (first 5): {list(available_ids)}")
                            print(f"    Total predictions in file: {len(pred_df)}")
                            print(f"    Sample image_id format: {available_ids[0] if len(available_ids) > 0 else 'N/A'}")
                            print(f"    Looking for: '{image_id}'")
                            parse_predictions_joblib._debug_printed = True
                        return boxes
        
        # Convert bbox from [x1, y1, x2, y2] format to tuple
        for _, row in image_preds.iterrows():
            class_id = int(row['category_id'])
            bbox = row['bbox']  # Already in [x1, y1, x2, y2] format
            
            if class_id not in boxes:
                boxes[class_id] = []
            boxes[class_id].append(tuple(bbox))
        
    except Exception as e:
        print(f"⚠️  Failed to load predictions from {joblib_path}: {e}")
        return boxes
    
    return boxes


def extract_dicom_info_from_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    Extract DICOM base name and pose from PNG filename
    
    Examples:
        "0834980_Quick ID_20240509_155005_B.dcm_Neutral[034].png" 
        -> ("0834980_Quick ID_20240509_155005_B", "Neutral")
        
        "0583808_Quick ID_20240805_113137_B _Neutral[003].png"
        -> ("0583808_Quick ID_20240805_113137_B", "Neutral")
    
    Returns:
        (dicom_base_name, pose)
        pose can be "Neutral", "Extended", "Ramped" or None
    """
    # Remove extension
    base = filename.replace('.png', '').replace('.txt', '')
    
    # Match pose pattern: _Pose[xxx] or Pose[xxx] or _Pose [xxx]
    pose_match = re.search(r'[_\s]?(Neutral|Extended|Ramped)\s*\[', base, re.IGNORECASE)
    if pose_match:
        pose = pose_match.group(1)
        # Remove pose[xxx] part (including preceding space or underscore)
        base = re.sub(r'[_\s]?(Neutral|Extended|Ramped)\s*\[\d+\]', '', base, flags=re.IGNORECASE)
    else:
        pose = None
        # Remove [xxx] part if exists
        base = re.sub(r'\[\d+\]', '', base)
    
    # Clean possible .dcm suffix (in filename, may be before _Neutral)
    # Example: "xxx_B.dcm_Neutral" -> "xxx_B"
    base = re.sub(r'\.dcm(?:_|$)', '_', base)
    if base.endswith('.dcm'):
        base = base[:-4]
    
    # Clean trailing spaces and underscores
    base = base.strip().rstrip('_').rstrip()
    
    return base, pose


def find_dicom_file(dicom_base: str, patient_id: str, dicom_root: Path) -> Optional[Path]:
    """
    Find corresponding DICOM file in dicom_dataset
    
    Search strategy:
    1. Exact match: {dicom_base}_Neutral.dcm, {dicom_base}_Extended.dcm, etc.
    2. Fuzzy match: Find files containing key parts of dicom_base
    
    Args:
        dicom_base: DICOM base name (without extension, may contain .dcm)
        patient_id: Patient ID
        dicom_root: dicom_dataset root directory
    
    Returns:
        DICOM file path, or None if not found
    """
    # Clean dicom_base (remove possible .dcm suffix)
    dicom_base_clean = dicom_base.replace('.dcm', '').strip()
    
    # Extract key parts for matching (e.g., 0834980_Quick ID_20240509_155005_B)
    # Remove possible spaces and underscores
    dicom_key_parts = [p for p in dicom_base_clean.split('_') if p and p.strip()]
    
    # Possible folder names
    folder_patterns = [
        f"{patient_id}_Quick ID",
        patient_id,  # Some may not have _Quick ID
    ]
    
    # Search in three category folders
    category_folders = ['內視鏡', '困難', '非困難']
    
    for category in category_folders:
        category_path = dicom_root / category
        if not category_path.exists():
            continue
        
        for folder_pattern in folder_patterns:
            patient_folder = category_path / folder_pattern
            if not patient_folder.exists():
                continue
            
            # Strategy 1: Exact match (if dicom_base contains complete info)
            # Try various combinations with .dcm suffix
            exact_patterns = [
                f"{dicom_base_clean}.dcm",
                f"{dicom_base_clean}_Neutral.dcm",
                f"{dicom_base_clean}_Extended.dcm",
                f"{dicom_base_clean}_Ramped.dcm",
            ]
            
            for pattern in exact_patterns:
                dicom_file = patient_folder / pattern
                if dicom_file.exists():
                    return dicom_file
            
            # Strategy 2: Fuzzy match - find files containing key parts
            # Use main parts of dicom_base for matching
            for dicom_file in patient_folder.glob("*.dcm"):
                file_stem = dicom_file.stem
                
                # Check if contains key parts of dicom_base
                # Example: dicom_base = "0834980_Quick ID_20240509_155005_B"
                # File might be "0834980_Quick ID_20240509_155005_B.dcm_Neutral.dcm"
                if len(dicom_key_parts) >= 3:
                    # Use first few key parts for matching
                    key_match = '_'.join(dicom_key_parts[:3])  # e.g., "0834980_Quick ID_20240509"
                    if key_match in file_stem:
                        return dicom_file
                
                # Or check if starts with patient_id and contains timestamp
                if file_stem.startswith(patient_id) and len(dicom_key_parts) > 1:
                    # Check if contains timestamp part
                    timestamp_part = None
                    for part in dicom_key_parts:
                        if len(part) == 8 and part.isdigit():  # Might be date YYYYMMDD
                            timestamp_part = part
                            break
                    
                    if timestamp_part and timestamp_part in file_stem:
                        return dicom_file
            
            # Strategy 3: If still not found, try matching any file containing patient_id and timestamp
            if len(dicom_key_parts) >= 2:
                for dicom_file in patient_folder.glob("*.dcm"):
                    file_stem = dicom_file.stem
                    # Check if contains patient_id and at least one other key part
                    if patient_id in file_stem:
                        # Check if contains other key parts (besides patient_id)
                        other_parts = [p for p in dicom_key_parts if p != patient_id]
                        for part in other_parts[:2]:  # Only check first two other parts
                            if part in file_stem:
                                return dicom_file
    
    return None


def get_pixel_spacing(dicom_path: Path) -> Optional[float]:
    """
    Read PixelSpacing from DICOM file
    
    Returns:
        PixelSpacing (mm/pixel), or None if read fails
    """
    try:
        ds = dicom.dcmread(str(dicom_path))
        
        # Try to read PixelSpacing
        if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing is not None:
            pixel_spacing = ds.PixelSpacing
            
            # Handle MultiValue type (special type in pydicom)
            if hasattr(pixel_spacing, '__iter__') and not isinstance(pixel_spacing, str):
                # If list/tuple/MultiValue, take first element
                try:
                    # Try direct indexing
                    value = pixel_spacing[0]
                    # If MultiValue, need to convert to numeric value
                    if hasattr(value, 'value'):
                        value = value.value
                    return float(value)
                except (IndexError, TypeError, AttributeError):
                    # If indexing fails, try converting to list
                    try:
                        pixel_list = list(pixel_spacing)
                        value = pixel_list[0]
                        if hasattr(value, 'value'):
                            value = value.value
                        return float(value)
                    except Exception:
                        pass
            else:
                # Single value
                if hasattr(pixel_spacing, 'value'):
                    pixel_spacing = pixel_spacing.value
                return float(pixel_spacing)
        
        # If no PixelSpacing, try other fields
        if hasattr(ds, 'ImagerPixelSpacing') and ds.ImagerPixelSpacing is not None:
            imager_spacing = ds.ImagerPixelSpacing
            if hasattr(imager_spacing, '__iter__') and not isinstance(imager_spacing, str):
                try:
                    value = imager_spacing[0]
                    if hasattr(value, 'value'):
                        value = value.value
                    return float(value)
                except (IndexError, TypeError, AttributeError):
                    try:
                        spacing_list = list(imager_spacing)
                        value = spacing_list[0]
                        if hasattr(value, 'value'):
                            value = value.value
                        return float(value)
                    except Exception:
                        pass
            else:
                if hasattr(imager_spacing, 'value'):
                    imager_spacing = imager_spacing.value
                return float(imager_spacing)
        
        return None
    except Exception as e:
        # Only print detailed error in debug mode to avoid too much output
        print(f"⚠️  Failed to read DICOM file {dicom_path.name}: {type(e).__name__}: {str(e)}")
        return None


def calculate_hmd(mentum_boxes: List[Tuple], hyoid_boxes: List[Tuple], 
                   pixel_spacing: Optional[float], img_width: int, img_height: int) -> Optional[Tuple[float, Optional[float]]]:
    """
    Calculate HMD (Hyomental Distance)
    
    According to DA_metrics.py logic:
    - HMD_dx = (Hyoid_xtl - Mentum_xbr) * PixelSpacing (if available)
    - HMD_dy = (Hyoid_y_center - Mentum_y_center) * PixelSpacing (if available)
    - HMD = sqrt(HMD_dx^2 + HMD_dy^2)
    
    Args:
        mentum_boxes: List of Mentum bboxes [(x1, y1, x2, y2), ...]
        hyoid_boxes: List of Hyoid bboxes [(x1, y1, x2, y2), ...]
        pixel_spacing: PixelSpacing (mm/pixel), or None if not available
        img_width: Image width (for validation)
        img_height: Image height (for validation)
    
    Returns:
        Tuple of (hmd_pixel, hmd_mm)
        - hmd_pixel: HMD distance in pixels (always calculated)
        - hmd_mm: HMD distance in millimeters (None if pixel_spacing is None)
        Returns None if calculation fails
    """
    if not mentum_boxes or not hyoid_boxes:
        return None
    
    # Take first bbox (if multiple, can be improved to take highest confidence)
    mentum = mentum_boxes[0]  # (x1, y1, x2, y2)
    hyoid = hyoid_boxes[0]    # (x1, y1, x2, y2)
    
    # Extract coordinates
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid
    
    # Calculate HMD in pixels (without unit conversion)
    hmd_dx_pixel = hyoid_x1 - mentum_x2
    mentum_y_center = (mentum_y1 + mentum_y2) / 2
    hyoid_y_center = (hyoid_y1 + hyoid_y2) / 2
    hmd_dy_pixel = hyoid_y_center - mentum_y_center
    hmd_pixel = np.sqrt(hmd_dx_pixel**2 + hmd_dy_pixel**2)
    
    # Calculate HMD in millimeters (with PixelSpacing conversion)
    hmd_mm = None
    if pixel_spacing is not None:
        hmd_dx_mm = hmd_dx_pixel * pixel_spacing
        hmd_dy_mm = hmd_dy_pixel * pixel_spacing
        hmd_mm = np.sqrt(hmd_dx_mm**2 + hmd_dy_mm**2)
    
    return (hmd_pixel, hmd_mm)


def get_patients_from_test_txt(case_id: str, yolo_root: Path, version: str = "v3") -> set:
    """
    Extract patient IDs from test.txt file
    
    Args:
        case_id: Dataset ID (e.g., "det_123")
        yolo_root: yolo_dataset root directory
        version: Dataset version (default "v3")
    
    Returns:
        Set of patient IDs found in test.txt
    """
    test_txt_path = yolo_root / case_id / version / "test.txt"
    if not test_txt_path.exists():
        print(f"⚠️  test.txt not found: {test_txt_path}")
        return set()
    
    patient_ids = set()
    with open(test_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract patient ID from path like:
            # D:/.../patient_data/0587648/0587648_Quick ID_20231213_140104_B[000].png
            # or: patient_data/0587648/0587648_Quick ID_20231213_140104_B[000].png
            parts = Path(line).parts
            # Find 'patient_data' in path and get next part (patient_id)
            try:
                patient_data_idx = [i for i, p in enumerate(parts) if 'patient_data' in p.lower()][0]
                if patient_data_idx + 1 < len(parts):
                    patient_id = parts[patient_data_idx + 1]
                    patient_ids.add(patient_id)
            except (IndexError, ValueError):
                continue
    
    return patient_ids


def process_patient(case_id: str, patient_id: str, yolo_root: Path, dicom_root: Path, 
                   version: str = "v3", use_pred: bool = False, pred_root: Optional[Path] = None,
                   pred_joblib: Optional[Path] = None, compare_gt: bool = False) -> pd.DataFrame:
    """
    Process single patient, calculate HMD for all images
    
    Args:
        case_id: Dataset ID (e.g., "det_123")
        patient_id: Patient ID
        yolo_root: yolo_dataset root directory
        dicom_root: dicom_dataset root directory
        version: Dataset version (default "v3")
        use_pred: Whether to use YOLO prediction results (False uses ground truth)
        pred_root: Prediction results root directory (if use_pred=True)
    
    Returns:
        DataFrame containing HMD calculation results for each image
    """
    class_names = get_class_names(case_id)
    # det_123: {0: "Mentum", 1: "Hyoid"}
    mentum_class = 0
    hyoid_class = 1
    
    patient_path = yolo_root / case_id / version / "patient_data" / patient_id
    if not patient_path.exists():
        print(f"❌ Patient directory does not exist: {patient_path}")
        return pd.DataFrame()
    
    # Get all PNG files
    png_files = sorted(patient_path.glob("*.png"))
    if not png_files:
        print(f"⚠️  No PNG files found for Patient {patient_id}")
        return pd.DataFrame()
    
    results = []
    pixel_spacing_cache = {}  # Cache PixelSpacing to avoid repeated reads
    
    print(f"📊 Processing Patient {patient_id}: {len(png_files)} images")
    
    for png_file in tqdm(png_files, desc=f"Patient {patient_id}"):
        # Extract DICOM info and pose
        dicom_base, pose = extract_dicom_info_from_filename(png_file.name)
        
        # Read image dimensions (needed for converting normalized coordinates)
        try:
            try:
                import cv2
            except ImportError:
                # If no cv2, try using PIL
                from PIL import Image
                img = Image.open(png_file)
                img_width, img_height = img.size
            else:
                img = cv2.imread(str(png_file))
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"⚠️  Failed to read image {png_file}: {e}")
            continue
        
        # Read bbox
        pred_boxes = {}
        gt_boxes = {}
        
        # Read ground truth
        gt_label_path = png_file.with_suffix('.txt')
        gt_boxes = parse_yolo_label(gt_label_path, img_width, img_height)
        
        # Read predictions
        if pred_joblib and pred_joblib.exists():
            # Read from predictions.joblib file
            pred_boxes = parse_predictions_joblib(pred_joblib, png_file.name)
            # Debug: check if predictions were found (only print first few)
            if not pred_boxes and not hasattr(process_patient, '_pred_debug_count'):
                process_patient._pred_debug_count = 0
            if not pred_boxes and process_patient._pred_debug_count < 3:
                print(f"⚠️  No predictions found for {png_file.name} in {pred_joblib.name}")
                process_patient._pred_debug_count += 1
        elif use_pred and pred_root:
            # Read from prediction results directory
            label_path = pred_root / patient_id / png_file.name.replace('.png', '.txt')
            pred_boxes = parse_yolo_label(label_path, img_width, img_height)
        else:
            # Use ground truth as prediction (for comparison mode)
            pred_boxes = gt_boxes.copy()
        
        # Determine which boxes to use
        if compare_gt:
            # Compare mode: use both pred and gt if available
            # If pred_boxes is empty, still calculate GT-only results
            boxes_to_use = pred_boxes if pred_boxes else gt_boxes
            boxes_gt = gt_boxes
            
            # Check if gt has required classes (always needed in compare mode)
            gt_has_both = (mentum_class in gt_boxes and hyoid_class in gt_boxes)
            if not gt_has_both:
                continue  # Skip if gt is missing required classes
            
            # Check if pred has required classes (if pred_boxes is not empty)
            if pred_boxes:
                pred_has_both = (mentum_class in pred_boxes and hyoid_class in pred_boxes)
                if not pred_has_both:
                    # If pred_boxes exists but missing classes, skip comparison but can still use GT
                    boxes_to_use = gt_boxes  # Use GT for calculation
            else:
                # No predictions available, use GT only
                boxes_to_use = gt_boxes
        else:
            # Normal mode: use pred if available, otherwise gt
            boxes_to_use = pred_boxes if pred_boxes else gt_boxes
            boxes_gt = None
            
            # Check if has Mentum and Hyoid
            if mentum_class not in boxes_to_use or hyoid_class not in boxes_to_use:
                continue
        
        mentum_boxes = boxes_to_use[mentum_class]
        hyoid_boxes = boxes_to_use[hyoid_class]
        
        # Get ground truth boxes for comparison if needed
        mentum_boxes_gt = None
        hyoid_boxes_gt = None
        if compare_gt and boxes_gt:
            if mentum_class in boxes_gt and hyoid_class in boxes_gt:
                mentum_boxes_gt = boxes_gt[mentum_class]
                hyoid_boxes_gt = boxes_gt[hyoid_class]
        
        # Get PixelSpacing
        pixel_spacing = None
        cache_key = f"{dicom_base}_{pose}"  # Use dicom_base + pose as cache key
        if cache_key not in pixel_spacing_cache:
            dicom_file = find_dicom_file(dicom_base, patient_id, dicom_root)
            if dicom_file is None:
                # Try adding .dcm suffix
                dicom_file = find_dicom_file(f"{dicom_base}.dcm", patient_id, dicom_root)
            
            if dicom_file is not None:
                pixel_spacing = get_pixel_spacing(dicom_file)
                if pixel_spacing is not None:
                    pixel_spacing_cache[cache_key] = pixel_spacing
        else:
            pixel_spacing = pixel_spacing_cache[cache_key]
        
        # Calculate HMD (both pixel and mm) for predictions
        hmd_result = calculate_hmd(mentum_boxes, hyoid_boxes, pixel_spacing, img_width, img_height)
        
        # Calculate HMD for ground truth if comparison mode
        hmd_result_gt = None
        if compare_gt:
            # If boxes_to_use is the same as gt_boxes (no predictions available), set GT = pred
            if boxes_to_use is gt_boxes:
                hmd_result_gt = hmd_result  # Same calculation, no difference
            elif mentum_boxes_gt and hyoid_boxes_gt:
                hmd_result_gt = calculate_hmd(mentum_boxes_gt, hyoid_boxes_gt, pixel_spacing, img_width, img_height)
        
        if hmd_result is not None:
            hmd_pixel, hmd_mm = hmd_result
            
            result_dict = {
                'patient_id': patient_id,
                'image_name': png_file.name,
                'dicom_base': dicom_base,
                'pose': pose or 'Unknown',
                'pixel_spacing': pixel_spacing,
                'hmd_pixel': hmd_pixel,
                'hmd_mm': hmd_mm,
            }
            
            # Add ground truth and difference if comparison mode
            if compare_gt and hmd_result_gt is not None:
                hmd_pixel_gt, hmd_mm_gt = hmd_result_gt
                result_dict.update({
                    'hmd_pixel_gt': hmd_pixel_gt,
                    'hmd_mm_gt': hmd_mm_gt,
                    'hmd_pixel_diff': hmd_pixel - hmd_pixel_gt,
                    'hmd_mm_diff': hmd_mm - hmd_mm_gt if hmd_mm is not None and hmd_mm_gt is not None else None,
                    'hmd_pixel_abs_diff': abs(hmd_pixel - hmd_pixel_gt),
                    'hmd_mm_abs_diff': abs(hmd_mm - hmd_mm_gt) if hmd_mm is not None and hmd_mm_gt is not None else None,
                })
            
            results.append(result_dict)
    
    if not results:
        print(f"⚠️  No successful calculation results for Patient {patient_id}")
        if pred_boxes is None or len(pred_boxes) == 0:
            print(f"   Note: No predictions found in predictions.joblib for this patient.")
            print(f"   This patient may not be in the test set used to generate predictions.")
            print(f"   You can still calculate HMD from ground truth by omitting --pred-joblib.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Statistics grouped by pose
    if 'pose' in df.columns:
        print(f"\n📈 Patient {patient_id} HMD Statistics (by pose):")
        for pose in df['pose'].unique():
            pose_df = df[df['pose'] == pose]
            if len(pose_df) > 0:
                # Pixel distance statistics
                median_pixel = pose_df['hmd_pixel'].median()
                mean_pixel = pose_df['hmd_pixel'].mean()
                std_pixel = pose_df['hmd_pixel'].std()
                
                # Millimeter distance statistics (only for rows with valid pixel_spacing)
                mm_df = pose_df[pose_df['hmd_mm'].notna()]
                if len(mm_df) > 0:
                    median_mm = mm_df['hmd_mm'].median()
                    mean_mm = mm_df['hmd_mm'].mean()
                    std_mm = mm_df['hmd_mm'].std()
                    print(f"  {pose}: Pixel - median={median_pixel:.2f}, mean={mean_pixel:.2f}, std={std_pixel:.2f} | "
                          f"MM - median={median_mm:.2f}mm, mean={mean_mm:.2f}mm, std={std_mm:.2f}mm (n={len(pose_df)}, mm_n={len(mm_df)})")
                else:
                    print(f"  {pose}: Pixel - median={median_pixel:.2f}, mean={mean_pixel:.2f}, std={std_pixel:.2f} | "
                          f"MM - N/A (no PixelSpacing) (n={len(pose_df)})")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Calculate HMD distance from YOLO bbox')
    parser.add_argument('--case-id', type=str, required=True, help='Dataset ID (e.g., det_123)')
    parser.add_argument('--patient-id', type=str, default=None, help='Patient ID (e.g., 0834980), if not specified then batch processing')
    parser.add_argument('--batch', action='store_true', help='Batch process all patients')
    parser.add_argument('--yolo-root', type=Path, default=None, help='yolo_dataset root directory (default: auto-detect from project root)')
    parser.add_argument('--dicom-root', type=Path, default=None, help='dicom_dataset root directory (default: auto-detect from project root)')
    parser.add_argument('--version', type=str, default='v3', help='Dataset version')
    parser.add_argument('--use-pred', action='store_true', help='Use YOLO prediction results (instead of ground truth)')
    parser.add_argument('--pred-root', type=Path, default=None, help='Prediction results root directory (if using predictions)')
    parser.add_argument('--pred-joblib', type=Path, default=None, help='Path to predictions.joblib file (output from test_yolo.py). Can be relative to project root or current directory.')
    parser.add_argument('--compare-gt', action='store_true', help='Compare predictions with ground truth and calculate differences')
    parser.add_argument('--test-only', action='store_true', help='Only process patients that are in test.txt (useful when using --pred-joblib)')
    parser.add_argument('--output', type=Path, default=None, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Auto-detect project root and set default paths if not provided
    script_dir = Path(__file__).resolve().parent
    # Try to find project root (go up from ultralytics/evaluate to project root)
    project_root = script_dir.parent.parent
    
    # Set default paths if not provided
    if args.yolo_root is None:
        # Try multiple possible locations
        possible_yolo_roots = [
            project_root / 'yolo_dataset',
            Path('yolo_dataset'),
            Path('../../yolo_dataset'),
        ]
        for yolo_root in possible_yolo_roots:
            if yolo_root.exists():
                args.yolo_root = yolo_root.resolve()
                break
        else:
            args.yolo_root = project_root / 'yolo_dataset'  # Default to project root
    
    if args.dicom_root is None:
        # Try multiple possible locations
        possible_dicom_roots = [
            project_root / 'dicom_dataset',
            Path('dicom_dataset'),
            Path('../../dicom_dataset'),
        ]
        for dicom_root in possible_dicom_roots:
            if dicom_root.exists():
                args.dicom_root = dicom_root.resolve()
                break
        else:
            args.dicom_root = project_root / 'dicom_dataset'  # Default to project root
    
    # Convert to Path if string
    if isinstance(args.yolo_root, str):
        args.yolo_root = Path(args.yolo_root)
    if isinstance(args.dicom_root, str):
        args.dicom_root = Path(args.dicom_root)
    
    # Resolve relative paths
    if not args.yolo_root.is_absolute():
        args.yolo_root = (Path.cwd() / args.yolo_root).resolve()
    if not args.dicom_root.is_absolute():
        args.dicom_root = (Path.cwd() / args.dicom_root).resolve()
    
    # Validate paths
    yolo_path = args.yolo_root / args.case_id / args.version / "patient_data"
    if not yolo_path.exists():
        print(f"❌ YOLO dataset path does not exist: {yolo_path}")
        print(f"   Checked yolo_root: {args.yolo_root}")
        return
    
    if not args.dicom_root.exists():
        print(f"❌ DICOM dataset path does not exist: {args.dicom_root}")
        return
    
    print(f"📁 Using yolo_root: {args.yolo_root}")
    print(f"📁 Using dicom_root: {args.dicom_root}")
    
    # Resolve pred_joblib path if provided
    if args.pred_joblib:
        if not args.pred_joblib.is_absolute():
            # Try relative to project root first
            project_root = Path(__file__).resolve().parent.parent.parent
            possible_paths = [
                project_root / args.pred_joblib,
                Path.cwd() / args.pred_joblib,
                args.pred_joblib,
            ]
            for path in possible_paths:
                if path.exists():
                    args.pred_joblib = path.resolve()
                    break
            else:
                args.pred_joblib = (Path.cwd() / args.pred_joblib).resolve()
        if not args.pred_joblib.exists():
            print(f"⚠️  Warning: predictions.joblib file not found: {args.pred_joblib}")
            print(f"   Continuing without predictions (will use ground truth only)")
            args.pred_joblib = None
    
    # Determine patient list to process
    if args.patient_id:
        patient_ids = [args.patient_id]
        # If test-only is specified, check if patient is in test.txt
        if args.test_only:
            test_patients = get_patients_from_test_txt(args.case_id, args.yolo_root, args.version)
            if args.patient_id not in test_patients:
                print(f"⚠️  Patient {args.patient_id} is not in test.txt")
                print(f"   Available patients in test.txt: {len(test_patients)}")
                if len(test_patients) > 0:
                    print(f"   Sample patients: {sorted(list(test_patients))[:5]}")
                return
    elif args.batch:
        all_patients = sorted([d.name for d in yolo_path.iterdir() if d.is_dir()])
        
        # If test-only is specified, filter to only patients in test.txt
        if args.test_only:
            test_patients = get_patients_from_test_txt(args.case_id, args.yolo_root, args.version)
            patient_ids = sorted([p for p in all_patients if p in test_patients])
            print(f"📋 Batch processing {len(patient_ids)} patients (filtered from {len(all_patients)} total, {len(test_patients)} in test.txt)")
            if len(patient_ids) == 0:
                print(f"⚠️  No patients found in both patient_data and test.txt")
                print(f"   Total patients in patient_data: {len(all_patients)}")
                print(f"   Total patients in test.txt: {len(test_patients)}")
                return
        else:
            patient_ids = all_patients
            print(f"📋 Batch processing {len(patient_ids)} patients")
    else:
        print("❌ Please specify --patient-id or --batch")
        return
    
    # Process each patient
    all_results = []
    for patient_id in patient_ids:
        df = process_patient(
            case_id=args.case_id,
            patient_id=patient_id,
            yolo_root=args.yolo_root,
            dicom_root=args.dicom_root,
            version=args.version,
            use_pred=args.use_pred,
            pred_root=args.pred_root,
            pred_joblib=args.pred_joblib,
            compare_gt=args.compare_gt
        )
        if not df.empty:
            all_results.append(df)
    
    if not all_results:
        print("❌ No successful processing results")
        return
    
    # Merge results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Output results
    if args.output:
        final_df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n✅ Results saved to: {args.output}")
    else:
        output_path = Path(f"hmd_results_{args.case_id}.csv")
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ Results saved to: {output_path}")
    
    # Print overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"  Total images: {len(final_df)}")
    print(f"  Total patients: {final_df['patient_id'].nunique()}")
    
    # Pixel distance statistics
    print(f"\n  Pixel Distance Statistics:")
    print(f"    Median: {final_df['hmd_pixel'].median():.2f} pixels")
    print(f"    Mean: {final_df['hmd_pixel'].mean():.2f} pixels")
    print(f"    Std: {final_df['hmd_pixel'].std():.2f} pixels")
    
    # Millimeter distance statistics (only for rows with valid pixel_spacing)
    mm_df = final_df[final_df['hmd_mm'].notna()]
    if len(mm_df) > 0:
        print(f"\n  Millimeter Distance Statistics (n={len(mm_df)}):")
        print(f"    Median: {mm_df['hmd_mm'].median():.2f} mm")
        print(f"    Mean: {mm_df['hmd_mm'].mean():.2f} mm")
        print(f"    Std: {mm_df['hmd_mm'].std():.2f} mm")
    else:
        print(f"\n  Millimeter Distance Statistics: N/A (no PixelSpacing available)")
    
    if 'pose' in final_df.columns:
        print(f"\n  Statistics by Pose:")
        for pose in sorted(final_df['pose'].unique()):
            pose_df = final_df[final_df['pose'] == pose]
            mm_pose_df = pose_df[pose_df['hmd_mm'].notna()]
            
            # Calculate pixel statistics
            pixel_median = pose_df['hmd_pixel'].median()
            pixel_mean = pose_df['hmd_pixel'].mean()
            
            # Calculate mm statistics (if available)
            if len(mm_pose_df) > 0:
                mm_median = mm_pose_df['hmd_mm'].median()
                mm_mean = mm_pose_df['hmd_mm'].mean()
                mm_str = f"MM - median={mm_median:.2f}mm, mean={mm_mean:.2f}mm"
            else:
                mm_str = "MM - N/A"
            
            print(f"    {pose}: Pixel - median={pixel_median:.2f}, mean={pixel_mean:.2f} | "
                  f"{mm_str} (n={len(pose_df)})")
    
    # Comparison statistics if compare_gt mode
    if args.compare_gt and 'hmd_pixel_diff' in final_df.columns:
        print(f"\n  Prediction vs Ground Truth Comparison:")
        diff_df = final_df[final_df['hmd_pixel_diff'].notna()]
        if len(diff_df) > 0:
            print(f"    Pixel Distance Difference (n={len(diff_df)}):")
            print(f"      Mean Error: {diff_df['hmd_pixel_diff'].mean():.2f} pixels")
            print(f"      Mean Absolute Error: {diff_df['hmd_pixel_abs_diff'].mean():.2f} pixels")
            print(f"      RMSE: {np.sqrt((diff_df['hmd_pixel_diff']**2).mean()):.2f} pixels")
            
            mm_diff_df = final_df[final_df['hmd_mm_diff'].notna()]
            if len(mm_diff_df) > 0:
                print(f"    Millimeter Distance Difference (n={len(mm_diff_df)}):")
                print(f"      Mean Error: {mm_diff_df['hmd_mm_diff'].mean():.2f} mm")
                print(f"      Mean Absolute Error: {mm_diff_df['hmd_mm_abs_diff'].mean():.2f} mm")
                print(f"      RMSE: {np.sqrt((mm_diff_df['hmd_mm_diff']**2).mean()):.2f} mm")


if __name__ == '__main__':
    main()
