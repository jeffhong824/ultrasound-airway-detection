"""
Record Experiment Results to Excel
==================================
This module records the best epoch and metrics (val & test) for each experiment to an Excel file.
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# Fix import path for ultralytics
# Add parent directories to path to ensure correct import
_mycodes_dir = Path(__file__).parent
_ultralytics_dir = _mycodes_dir.parent
_project_root = _ultralytics_dir.parent

# Add to path if not already there
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_ultralytics_dir) not in sys.path:
    sys.path.insert(0, str(_ultralytics_dir))

# Delay YOLO import to function level to avoid import issues
# YOLO will be imported when needed in functions


def find_best_epoch_from_csv(csv_path: str) -> Tuple[int, float]:
    """
    Find the best epoch from results.csv based on fitness
    
    Args:
        csv_path: Path to results.csv file
        
    Returns:
        Tuple of (best_epoch, best_fitness)
    """
    try:
        results = pd.read_csv(csv_path)
        results.columns = results.columns.str.strip()
        
        # Calculate fitness
        if "metrics/mAP50(B)" in results.columns and "metrics/mAP50-95(B)" in results.columns:
            box_fitness = results["metrics/mAP50(B)"] * 0.1 + results["metrics/mAP50-95(B)"] * 0.9
            results["fitness"] = box_fitness
        else:
            # Fallback: use mAP50-95 if available
            if "metrics/mAP50-95(B)" in results.columns:
                results["fitness"] = results["metrics/mAP50-95(B)"]
            else:
                logging.warning(f"⚠️ Cannot calculate fitness from {csv_path}")
                return (1, 0.0)
        
        # Find the epoch with the highest fitness
        idx = results['fitness'].idxmax()
        best_epoch = int(results.loc[idx, 'epoch']) if 'epoch' in results.columns else idx + 1
        best_fitness = float(results.loc[idx, 'fitness'])
        
        return (best_epoch, best_fitness)
    
    except Exception as e:
        logging.error(f"❌ Error reading {csv_path}: {e}")
        return (1, 0.0)


def find_best_epoch_by_hmd_metrics(
    run_dir: Path,
    project: str,
    exp_name: str,
    database: str = "det_123",
    db_version: int = 3,
    use_mm: bool = False,
    fallback_to_fitness: bool = True
) -> Tuple[int, float, str]:
    """
    Find the best epoch based on HMD Overall_Score metrics
    
    This function tries to get HMD metrics from wandb API first.
    If wandb is not available or metrics are not found, falls back to fitness-based selection.
    
    Args:
        run_dir: Path to run directory (e.g., runs/train/run_name)
        project: Wandb project name
        exp_name: Experiment name
        database: Database name (must be 'det_123' for HMD metrics)
        use_mm: If True, use Overall_Score_mm; otherwise use Overall_Score_pixel
        fallback_to_fitness: If True, fall back to fitness-based selection if HMD metrics not available
        
    Returns:
        Tuple of (best_epoch, best_score, method_used)
        method_used: "hmd_overall_score_pixel", "hmd_overall_score_mm", or "fitness"
    """
    if database != 'det_123':
        # HMD metrics only available for det_123
        if fallback_to_fitness:
            csv_path = run_dir / "results.csv"
            if csv_path.exists():
                best_epoch, best_fitness = find_best_epoch_from_csv(str(csv_path))
                return (best_epoch, best_fitness, "fitness")
        return (1, 0.0, "fitness")
    
    # Try to get HMD metrics from wandb
    try:
        import wandb
        api = wandb.Api()
        
        # Construct run name (must match the format used in train_yolo.py)
        model_name = "yolo11n"
        run_name = f"{model_name}-{database}-v{db_version}" + (f"-{exp_name}" if exp_name else "")
        
        # Try to find the run in the project
        try:
            run = api.run(f"{project}/{run_name}")
            
            # Get history (all epochs)
            history = run.history()
            
            # Determine which metric to use
            if use_mm:
                metric_key = "val/hmd/overall_score_mm"
                fallback_metric_key = "val/hmd/overall_score_pixel"
                method_name = "hmd_overall_score_mm"
            else:
                metric_key = "val/hmd/overall_score_pixel"
                fallback_metric_key = None
                method_name = "hmd_overall_score_pixel"
            
            # Check if metric exists in history
            if metric_key in history.columns:
                # Find epoch with highest Overall_Score
                # Handle NaN values
                valid_scores = history[metric_key].dropna()
                if len(valid_scores) > 0:
                    # Get all rows with the maximum score (in case of ties)
                    max_score = valid_scores.max()
                    best_candidates = valid_scores[valid_scores == max_score]
                    
                    # If there are ties, use val_fitness as tiebreaker
                    if len(best_candidates) > 1 and 'val/fitness' in history.columns:
                        # Get fitness values for tied candidates
                        fitness_values = history.loc[best_candidates.index, 'val/fitness']
                        # Find the one with highest fitness
                        best_idx = fitness_values.idxmax()
                        best_score = float(max_score)
                        logging.info(f"✅ Found best epoch by {method_name} (tie broken by val_fitness): score={best_score:.4f}, fitness={float(fitness_values.loc[best_idx]):.4f}")
                    else:
                        # No ties, or no fitness available, use the first one with max score
                        best_idx = valid_scores.idxmax()
                        best_score = float(valid_scores.loc[best_idx])
                    
                    # Get epoch number from history
                    # wandb history uses '_step' for step number (which is epoch number)
                    # If '_step' is not available, try 'epoch' column, otherwise use index + 1
                    if '_step' in history.columns:
                        best_epoch = int(history.loc[best_idx, '_step'])
                    elif 'epoch' in history.columns:
                        best_epoch = int(history.loc[best_idx, 'epoch'])
                    else:
                        # Index-based (0-based, so add 1)
                        best_epoch = int(best_idx) + 1
                    logging.info(f"✅ Found best epoch {best_epoch} by {method_name}: {best_score:.4f}")
                    return (best_epoch, best_score, method_name)
                elif fallback_metric_key and fallback_metric_key in history.columns:
                    # Try fallback metric
                    valid_scores = history[fallback_metric_key].dropna()
                    if len(valid_scores) > 0:
                        best_idx = valid_scores.idxmax()
                        best_score = float(valid_scores.loc[best_idx])
                        # Get epoch number from history
                        if '_step' in history.columns:
                            best_epoch = int(history.loc[best_idx, '_step'])
                        elif 'epoch' in history.columns:
                            best_epoch = int(history.loc[best_idx, 'epoch'])
                        else:
                            best_epoch = int(best_idx) + 1
                        logging.info(f"✅ Found best epoch {best_epoch} by {fallback_metric_key}: {best_score:.4f}")
                        return (best_epoch, best_score, "hmd_overall_score_pixel")
            
            logging.warning(f"⚠️ HMD metrics not found in wandb history for {run_name}")
            
        except Exception as e:
            logging.warning(f"⚠️ Could not find wandb run {project}/{run_name}: {e}")
    
    except ImportError:
        logging.warning("⚠️ wandb not installed, cannot get HMD metrics from wandb")
    except Exception as e:
        logging.warning(f"⚠️ Error getting HMD metrics from wandb: {e}")
    
    # Fallback to fitness-based selection
    if fallback_to_fitness:
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            best_epoch, best_fitness = find_best_epoch_from_csv(str(csv_path))
            logging.info(f"⚠️ Falling back to fitness-based selection: epoch {best_epoch}, fitness {best_fitness:.4f}")
            return (best_epoch, best_fitness, "fitness")
    
    # Last resort: return epoch 1
    logging.warning(f"⚠️ Could not determine best epoch, using epoch 1")
    return (1, 0.0, "fitness")


def get_best_epoch_metrics_from_csv(csv_path: str, best_epoch: int) -> Dict:
    """
    Get metrics for the best epoch from results.csv
    
    Args:
        csv_path: Path to results.csv file
        best_epoch: Best epoch number
        
    Returns:
        Dictionary with metrics from the best epoch
    """
    try:
        results = pd.read_csv(csv_path)
        results.columns = results.columns.str.strip()
        
        # Find row with best epoch
        if 'epoch' in results.columns:
            epoch_row = results[results['epoch'] == best_epoch]
        else:
            # If no epoch column, use index (0-based, so best_epoch - 1)
            epoch_row = results.iloc[[best_epoch - 1]]
        
        if len(epoch_row) == 0:
            logging.warning(f"⚠️ Best epoch {best_epoch} not found in {csv_path}")
            return {}
        
        row = epoch_row.iloc[0]
        
        metrics = {
            'epoch': best_epoch,
            'precision': float(row.get('metrics/precision(B)', 0.0)) if pd.notna(row.get('metrics/precision(B)', 0.0)) else 0.0,
            'recall': float(row.get('metrics/recall(B)', 0.0)) if pd.notna(row.get('metrics/recall(B)', 0.0)) else 0.0,
            'mAP50': float(row.get('metrics/mAP50(B)', 0.0)) if pd.notna(row.get('metrics/mAP50(B)', 0.0)) else 0.0,
            'mAP50-95': float(row.get('metrics/mAP50-95(B)', 0.0)) if pd.notna(row.get('metrics/mAP50-95(B)', 0.0)) else 0.0,
        }
        
        # Calculate fitness
        metrics['fitness'] = metrics['mAP50'] * 0.1 + metrics['mAP50-95'] * 0.9
        
        return metrics
    
    except Exception as e:
        logging.error(f"❌ Error reading metrics from {csv_path}: {e}")
        return {}


def evaluate_best_model_on_test(
    model_path: str,
    database: str = "det_123",
    db_version: int = 3,
    batch: int = 16,
    imgsz: int = 640,
    hmd_penalty_single: Optional[float] = None,
    hmd_penalty_none: Optional[float] = None,
) -> Dict:
    """
    Evaluate the best model on test set and return all metrics including HMD metrics
    
    Args:
        model_path: Path to best.pt model file
        database: Database name (e.g., 'det_123')
        db_version: Database version
        batch: Batch size for evaluation
        imgsz: Image size
        hmd_penalty_single: HMD penalty for single detection (if None, uses imgsz/2)
        hmd_penalty_none: HMD penalty for no detection (if None, uses imgsz)
        
    Returns:
        Dictionary with all test metrics including HMD metrics
    """
    try:
        # Import here to avoid circular imports
        from train_yolo import evaluate_detailed
        
        # Load best model - import YOLO here with error handling
        try:
            from ultralytics import YOLO
        except ImportError as e:
            # Try to fix import path
            import sys
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            if str(project_root / "ultralytics") not in sys.path:
                sys.path.insert(0, str(project_root / "ultralytics"))
            from ultralytics import YOLO
        model = YOLO(model_path)
        
        # Determine penalties
        if hmd_penalty_single is None:
            hmd_penalty_single = imgsz / 2.0
        if hmd_penalty_none is None:
            hmd_penalty_none = float(imgsz)
        
        # Evaluate on test set
        test_results = evaluate_detailed(
            model, "test", batch=batch, imgsz=imgsz,
            database=database, db_version=db_version, use_hmd=(database == 'det_123'),
            penalty_single=hmd_penalty_single, penalty_none=hmd_penalty_none,
            penalty_coeff=0.5
        )
        
        return test_results
    
    except Exception as e:
        logging.error(f"❌ Error evaluating best model on test set: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {}


def record_experiment_to_excel(
    exp_name: str,
    project: str,
    config: str,
    excel_path: Optional[str] = None,
    database: str = "det_123",
    db_version: int = 3,
    batch: int = 16,
    imgsz: int = 640,
) -> bool:
    """
    Record experiment results (best epoch and metrics) to Excel file
    
    Args:
        exp_name: Experiment name (e.g., "exp0 baseline")
        project: Wandb project name (e.g., "ultrasound-det_123_ES-v3-4090")
        config: Configuration name ("4090" or "h200")
        excel_path: Path to Excel file (if None, auto-generate)
        database: Database name
        db_version: Database version
        batch: Batch size
        imgsz: Image size
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine project root - try multiple methods
        # Method 1: From __file__
        project_root = Path(__file__).parent.parent.parent
        
        # Method 2: Try to get from current working directory
        # Since train_yolo.py uses project="./runs/train", results are saved relative to CWD
        # Check both locations: runs/train (relative to CWD) and ultralytics/runs/train (absolute)
        cwd = Path.cwd()
        runs_dir_relative = cwd / "runs" / "train"
        runs_dir_absolute = project_root / "ultralytics" / "runs" / "train"
        
        # Construct run name
        model_name = "yolo11n"
        run_name = f"{model_name}-{database}-v{db_version}" + (f"-{exp_name}" if exp_name else "")
        
        # Try to find the run directory in both possible locations
        run_dir_relative = runs_dir_relative / run_name
        run_dir_absolute = runs_dir_absolute / run_name
        
        # Determine which path exists
        if run_dir_relative.exists():
            run_dir = run_dir_relative
            runs_dir = runs_dir_relative.parent
        elif run_dir_absolute.exists():
            run_dir = run_dir_absolute
            runs_dir = runs_dir_absolute.parent
        else:
            # Default to absolute path for error message
            run_dir = run_dir_absolute
            runs_dir = runs_dir_absolute.parent
        
        # Paths
        csv_path = run_dir / "results.csv"
        best_model_path = run_dir / "weights" / "best.pt"
        
        # Check if files exist
        if not csv_path.exists():
            logging.warning(f"⚠️ results.csv not found: {csv_path}")
            return False
        
        if not best_model_path.exists():
            logging.warning(f"⚠️ best.pt not found: {best_model_path}")
            return False
        
        # Find best epoch - prioritize HMD Overall_Score over fitness
        # For det_123 database, use HMD metrics; otherwise use fitness
        if database == 'det_123':
            best_epoch, best_score, method = find_best_epoch_by_hmd_metrics(
                run_dir=run_dir,
                project=project,
                exp_name=exp_name,
                database=database,
                db_version=db_version,
                use_mm=False,  # Use pixel version by default
                fallback_to_fitness=True
            )
            logging.info(f"✅ Best epoch for {exp_name}: {best_epoch} ({method}={best_score:.6f})")
        else:
            best_epoch, best_fitness = find_best_epoch_from_csv(str(csv_path))
            logging.info(f"✅ Best epoch for {exp_name}: {best_epoch} (fitness={best_fitness:.6f})")
        
        # Get best epoch metrics from CSV (validation metrics)
        val_metrics = get_best_epoch_metrics_from_csv(str(csv_path), best_epoch)
        
        # Ensure best_fitness is set (for det_123, get it from val_metrics or CSV)
        if database == 'det_123':
            # Get fitness from val_metrics if available, otherwise from CSV
            if 'fitness' in val_metrics:
                best_fitness = val_metrics['fitness']
            else:
                # Fallback: get fitness from CSV for the best epoch
                _, best_fitness = find_best_epoch_from_csv(str(csv_path))
        
        # Evaluate best model on test set
        logging.info(f"🔍 Evaluating best model on test set for {exp_name}...")
        test_results = evaluate_best_model_on_test(
            str(best_model_path),
            database=database,
            db_version=db_version,
            batch=batch,
            imgsz=imgsz,
        )
        
        # Prepare Excel file path
        if excel_path is None:
            excel_path = project_root / f"experiments_results_{config}.xlsx"
        
        excel_path = Path(excel_path)
        
        # Evaluate best model on validation set for HMD metrics
        logging.info(f"🔍 Evaluating best model on validation set for {exp_name}...")
        val_results_detailed = evaluate_best_model_on_val(
            str(best_model_path),
            database=database,
            db_version=db_version,
            batch=batch,
            imgsz=imgsz,
        )
        
        # Determine device from config
        if config == "4090":
            device = "cuda:0"
        elif config == "h200":
            device = "0,1"
        else:
            device = "unknown"
        
        # Prepare data row - only essential fields (Experiment will be used as index)
        # Column order: device, batch-size, best_epoch, then all val columns, then all test columns
        row_data = {
            'device': device,
            'batch-size': batch,
            'best_epoch': best_epoch,  # Record which epoch was selected as best
        }
        
        # Validation metrics (from detailed evaluation, more accurate) - ALL VAL COLUMNS FIRST
        if val_results_detailed:
            row_data.update({
                'val_mAP50': val_results_detailed.get('mAP50', val_metrics.get('mAP50', 0.0)),
                'val_mAP50-95': val_results_detailed.get('mAP50-95', val_metrics.get('mAP50-95', 0.0)),
                'val_fitness': val_results_detailed.get('fitness', val_metrics.get('fitness', 0.0)),
            })
        else:
            row_data.update({
                'val_mAP50': val_metrics.get('mAP50', 0.0),
                'val_mAP50-95': val_metrics.get('mAP50-95', 0.0),
                'val_fitness': val_metrics.get('fitness', 0.0),
            })
        
        # Validation HMD metrics (if available) - continue with val columns
        if database == 'det_123':
            # Validation HMD metrics - get from val_results_detailed
            # evaluate_detailed returns direct keys: detection_rate, rmse_no_penalty_pixel, mae_no_penalty_pixel, overall_score_pixel
            row_data.update({
                'val_Detection_Rate': val_results_detailed.get('detection_rate', 0.0),
                'val_RMSE_HMD': val_results_detailed.get('rmse_no_penalty_pixel', 0.0),
                'val_MAE_HMD': val_results_detailed.get('mae_no_penalty_pixel', 0.0),
                'val_HMD_score': val_results_detailed.get('overall_score_pixel', 0.0),
            })
        else:
            # For non-det_123 databases, set HMD metrics to 0
            row_data.update({
                'val_Detection_Rate': 0.0,
                'val_RMSE_HMD': 0.0,
                'val_MAE_HMD': 0.0,
                'val_HMD_score': 0.0,
            })
        
        # Test metrics (general) - ALL TEST COLUMNS AFTER VAL COLUMNS
        row_data.update({
            'test_mAP50': test_results.get('mAP50', 0.0),
            'test_mAP50-95': test_results.get('mAP50-95', 0.0),
            'test_fitness': test_results.get('fitness', 0.0),
        })
        
        # Test HMD metrics - continue with test columns
        if database == 'det_123':
            # Test HMD metrics - get from test_results
            # evaluate_detailed returns direct keys: detection_rate, rmse_no_penalty_pixel, mae_no_penalty_pixel, overall_score_pixel
            row_data.update({
                'test_Detection_Rate': test_results.get('detection_rate', 0.0),
                'test_RMSE_HMD': test_results.get('rmse_no_penalty_pixel', 0.0),
                'test_MAE_HMD': test_results.get('mae_no_penalty_pixel', 0.0),
                'test_HMD_score': test_results.get('overall_score_pixel', 0.0),
            })
        else:
            # For non-det_123 databases, set HMD metrics to 0
            row_data.update({
                'test_Detection_Rate': 0.0,
                'test_RMSE_HMD': 0.0,
                'test_MAE_HMD': 0.0,
                'test_HMD_score': 0.0,
            })
        
        # Load or create Excel file
        if excel_path.exists():
            df = pd.read_excel(excel_path, index_col=0)  # Read with first column as index
            # Check if experiment already exists (using index)
            if exp_name in df.index:
                # Update existing row
                for col, val in row_data.items():
                    if col in df.columns:
                        df.loc[exp_name, col] = val
                    else:
                        # Add new column if it doesn't exist
                        df[col] = None
                        df.loc[exp_name, col] = val
                logging.info(f"✅ Updated existing row for {exp_name} in {excel_path}")
            else:
                # Append new row with exp_name as index
                new_row = pd.DataFrame([row_data], index=[exp_name])
                df = pd.concat([df, new_row])
                logging.info(f"✅ Added new row for {exp_name} to {excel_path}")
        else:
            # Create new DataFrame with exp_name as index
            df = pd.DataFrame([row_data], index=[exp_name])
        
        # Ensure index has a name
        df.index.name = 'Experiment'
        
        # Define column order: device, batch-size, best_epoch, then all val columns, then all test columns
        column_order = [
            'device', 'batch-size', 'best_epoch',
            'val_mAP50', 'val_mAP50-95', 'val_fitness',
            'val_Detection_Rate', 'val_RMSE_HMD', 'val_MAE_HMD', 'val_HMD_score',
            'test_mAP50', 'test_mAP50-95', 'test_fitness',
            'test_Detection_Rate', 'test_RMSE_HMD', 'test_MAE_HMD', 'test_HMD_score'
        ]
        
        # Reorder columns (only include columns that exist in df)
        existing_columns = [col for col in column_order if col in df.columns]
        # Add any additional columns that are not in the predefined order
        additional_columns = [col for col in df.columns if col not in column_order]
        df = df[existing_columns + additional_columns]
        
        # Save to Excel with index=True so Experiment appears as first column
        df.to_excel(excel_path, index=True)
        logging.info(f"✅ Saved results to {excel_path}")
        
        return True
    
    except Exception as e:
        logging.error(f"❌ Error recording experiment to Excel: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def evaluate_best_model_on_val(
    model_path: str,
    database: str = "det_123",
    db_version: int = 3,
    batch: int = 16,
    imgsz: int = 640,
    hmd_penalty_single: Optional[float] = None,
    hmd_penalty_none: Optional[float] = None,
) -> Dict:
    """
    Evaluate the best model on validation set and return all metrics including HMD metrics
    
    Args:
        model_path: Path to best.pt model file
        database: Database name (e.g., 'det_123')
        db_version: Database version
        batch: Batch size for evaluation
        imgsz: Image size
        hmd_penalty_single: HMD penalty for single detection (if None, uses imgsz/2)
        hmd_penalty_none: HMD penalty for no detection (if None, uses imgsz)
        
    Returns:
        Dictionary with all validation metrics including HMD metrics
    """
    try:
        # Import here to avoid circular imports
        try:
            from train_yolo import evaluate_detailed
        except ImportError:
            # Try alternative import path
            train_yolo_path = Path(__file__).parent / "train_yolo.py"
            if train_yolo_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("train_yolo", train_yolo_path)
                train_yolo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_yolo)
                evaluate_detailed = train_yolo.evaluate_detailed
            else:
                raise ImportError("Cannot import evaluate_detailed from train_yolo")
        
        # Load best model - import YOLO here with error handling
        try:
            from ultralytics import YOLO
        except ImportError as e:
            # Try to fix import path
            import sys
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            if str(project_root / "ultralytics") not in sys.path:
                sys.path.insert(0, str(project_root / "ultralytics"))
            from ultralytics import YOLO
        model = YOLO(model_path)
        
        # Determine penalties
        if hmd_penalty_single is None:
            hmd_penalty_single = imgsz / 2.0
        if hmd_penalty_none is None:
            hmd_penalty_none = float(imgsz)
        
        # Evaluate on validation set
        val_results = evaluate_detailed(
            model, "val", batch=batch, imgsz=imgsz,
            database=database, db_version=db_version, use_hmd=(database == 'det_123'),
            penalty_single=hmd_penalty_single, penalty_none=hmd_penalty_none,
            penalty_coeff=0.5
        )
        
        return val_results
    
    except Exception as e:
        logging.error(f"❌ Error evaluating best model on validation set: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {}


