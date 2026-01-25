#!/bin/bash

# =============================================================================
# Run All Experiments Script
# =============================================================================
# This script runs all experiments from README.md sequentially
# Usage: bash run_all_experiments.sh [--config 4090|h200] [--skip-failed]
# 
# Options:
#   --config 4090|h200    : Select configuration (default: 4090)
#   --skip-failed         : Continue to next experiment if current one fails
#   --start-from <name>   : Start from a specific experiment name
#   --stop-at <name>      : Stop at a specific experiment name
# =============================================================================

set -e  # Exit on error (unless --skip-failed is used)

# Parse arguments
CONFIG="4090"
SKIP_FAILED=false
START_FROM=""
STOP_AT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --skip-failed)
            SKIP_FAILED=true
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --stop-at)
            STOP_AT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_all_experiments.sh [--config 4090|h200] [--skip-failed] [--start-from <name>] [--stop-at <name>]"
            exit 1
            ;;
    esac
done

# Configuration
if [ "$CONFIG" == "4090" ]; then
    BATCH=16
    DEVICE="cuda:0"
    PROJECT="ultrasound-det_123_ES-v3-4090"
elif [ "$CONFIG" == "h200" ]; then
    BATCH=256
    DEVICE="0,1"
    PROJECT="ultrasound-det_123_ES-v3-h200"
else
    echo "Error: Invalid config. Use '4090' or 'h200'"
    exit 1
fi

# Common parameters
MODEL="yolo11n"
DATABASE="det_123"
DB_VERSION=3
EPOCHS=10
SEED=42

# Log file
LOG_FILE="experiments_${CONFIG}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

# Function to run experiment
run_experiment() {
    local exp_name="$1"
    shift
    local extra_args="$@"
    
    # Check if we should start from this experiment
    if [ -n "$START_FROM" ] && [ "$exp_name" != "$START_FROM" ] && [ "$STARTED" != "true" ]; then
        echo "Skipping $exp_name (waiting for --start-from $START_FROM)"
        return 0
    fi
    STARTED="true"
    
    # Check if we should stop at this experiment
    if [ -n "$STOP_AT" ] && [ "$exp_name" == "$STOP_AT" ]; then
        echo "Stopping at $exp_name (--stop-at reached)"
        exit 0
    fi
    
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Running: $exp_name" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    if [ "$SKIP_FAILED" == "true" ]; then
        set +e  # Don't exit on error
    fi
    
    python ultralytics/mycodes/train_yolo.py "$MODEL" "$DATABASE" \
        --db_version="$DB_VERSION" \
        --es \
        --batch="$BATCH" \
        --epochs="$EPOCHS" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --wandb \
        --project="$PROJECT" \
        --exp_name="$exp_name" \
        $extra_args 2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ "$SKIP_FAILED" == "true" ]; then
        set -e  # Re-enable exit on error
        if [ $exit_code -ne 0 ]; then
            echo "WARNING: Experiment $exp_name failed with exit code $exit_code, continuing..." | tee -a "$LOG_FILE"
        else
            echo "SUCCESS: Experiment $exp_name completed successfully" | tee -a "$LOG_FILE"
        fi
    else
        if [ $exit_code -ne 0 ]; then
            echo "ERROR: Experiment $exp_name failed with exit code $exit_code" | tee -a "$LOG_FILE"
            exit $exit_code
        fi
    fi
    
    echo "" | tee -a "$LOG_FILE"
}

# =============================================================================
# RTX 4090 Configuration Experiments
# =============================================================================

if [ "$CONFIG" == "4090" ]; then
    # exp0 baseline
    run_experiment "exp0 baseline"
    
    # exp0 baseline+keep_top_conf_per_class
    run_experiment "exp0 baseline+keep_top_conf_per_class" \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp1-1 data_aug
    run_experiment "exp1-1 data_aug" \
        --scale 0.7 \
        --translate 0.15 \
        --hsv_s 0.8 \
        --hsv_v 0.5 \
        --hsv_h 0.0
    
    # exp1-1 data_aug+keep_top_conf_per_class
    run_experiment "exp1-1 data_aug+keep_top_conf_per_class" \
        --scale 0.7 \
        --translate 0.15 \
        --hsv_s 0.8 \
        --hsv_v 0.5 \
        --hsv_h 0.0 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp1-2 ultrasound_aug
    run_experiment "exp1-2 ultrasound_aug" \
        --use_ultrasound_aug \
        --ultrasound_speckle_var 0.1 \
        --ultrasound_attenuation_factor 0.3
    
    # exp1-2 ultrasound_aug+keep_top_conf_per_class
    run_experiment "exp1-2 ultrasound_aug+keep_top_conf_per_class" \
        --use_ultrasound_aug \
        --ultrasound_speckle_var 0.1 \
        --ultrasound_attenuation_factor 0.3 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp2 loss_weights
    run_experiment "exp2 loss_weights" \
        --box 8.5 \
        --dfl 2.0 \
        --cls 0.6
    
    # exp2 loss_weights+keep_top_conf_per_class
    run_experiment "exp2 loss_weights+keep_top_conf_per_class" \
        --box 8.5 \
        --dfl 2.0 \
        --cls 0.6 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp3 focal_loss
    run_experiment "exp3 focal_loss" \
        --use_focal_loss \
        --focal_gamma 1.5 \
        --focal_alpha 0.25
    
    # exp3 focal_loss+keep_top_conf_per_class
    run_experiment "exp3 focal_loss+keep_top_conf_per_class" \
        --use_focal_loss \
        --focal_gamma 1.5 \
        --focal_alpha 0.25 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp4 dim_weights
    run_experiment "exp4 dim_weights" \
        --use_dim_weights \
        --dim_weights 5.0 1.0 5.0 1.0
    
    # exp4 dim_weights+keep_top_conf_per_class
    run_experiment "exp4 dim_weights+keep_top_conf_per_class" \
        --use_dim_weights \
        --dim_weights 5.0 1.0 5.0 1.0 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp5-1 hmd_loss_pixel
    run_experiment "exp5-1 hmd_loss_pixel" \
        --use_hmd_loss \
        --hmd_loss_weight 0.5 \
        --hmd_penalty_coeff 0.5
    
    # exp5-1 hmd_loss_pixel+keep_top_conf_per_class
    run_experiment "exp5-1 hmd_loss_pixel+keep_top_conf_per_class" \
        --use_hmd_loss \
        --hmd_loss_weight 0.5 \
        --hmd_penalty_coeff 0.5 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp5-2 hmd_loss_mm
    run_experiment "exp5-2 hmd_loss_mm" \
        --use_hmd_loss \
        --hmd_use_mm \
        --hmd_loss_weight 0.5 \
        --hmd_penalty_coeff 0.5
    
    # exp5-2 hmd_loss_mm+keep_top_conf_per_class
    run_experiment "exp5-2 hmd_loss_mm+keep_top_conf_per_class" \
        --use_hmd_loss \
        --hmd_use_mm \
        --hmd_loss_weight 0.5 \
        --hmd_penalty_coeff 0.5 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp6-1 warmup_optimized
    run_experiment "exp6-1 warmup_optimized" \
        --warmup_epochs 5.0 \
        --warmup_momentum 0.9 \
        --warmup_bias_lr 0.05
    
    # exp6-1 warmup_optimized+keep_top_conf_per_class
    run_experiment "exp6-1 warmup_optimized+keep_top_conf_per_class" \
        --warmup_epochs 5.0 \
        --warmup_momentum 0.9 \
        --warmup_bias_lr 0.05 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp6-2 warmup_cosine_restart
    run_experiment "exp6-2 warmup_cosine_restart" \
        --use_cosine_restart \
        --cosine_restart_t0 10 \
        --cosine_restart_t_mult 2 \
        --warmup_epochs 5.0
    
    # exp6-2 warmup_cosine_restart+keep_top_conf_per_class
    run_experiment "exp6-2 warmup_cosine_restart+keep_top_conf_per_class" \
        --use_cosine_restart \
        --cosine_restart_t0 10 \
        --cosine_restart_t_mult 2 \
        --warmup_epochs 5.0 \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp7-1 siou
    run_experiment "exp7-1 siou" \
        --iou_type SIoU
    
    # exp7-1 siou+keep_top_conf_per_class
    run_experiment "exp7-1 siou+keep_top_conf_per_class" \
        --iou_type SIoU \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp7-2 eiou
    run_experiment "exp7-2 eiou" \
        --iou_type EIoU
    
    # exp7-2 eiou+keep_top_conf_per_class
    run_experiment "exp7-2 eiou+keep_top_conf_per_class" \
        --iou_type EIoU \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp7-3 diou
    run_experiment "exp7-3 diou" \
        --iou_type DIoU
    
    # exp7-3 diou+keep_top_conf_per_class
    run_experiment "exp7-3 diou+keep_top_conf_per_class" \
        --iou_type DIoU \
        --keep_top_conf_per_class \
        --conf_low 0.1

# =============================================================================
# H200 Configuration Experiments
# =============================================================================

elif [ "$CONFIG" == "h200" ]; then
    # exp0 baseline
    run_experiment "exp0 baseline" \
        --keep_top_conf_per_class \
        --conf_low 0.1
    
    # exp1-1 data_aug
    run_experiment "exp1-1 data_aug" \
        --scale 0.7 \
        --translate 0.15 \
        --hsv_s 0.8 \
        --hsv_v 0.5 \
        --hsv_h 0.0
    
    # exp1-2 ultrasound_aug
    run_experiment "exp1-2 ultrasound_aug" \
        --use_ultrasound_aug \
        --ultrasound_speckle_var 0.1 \
        --ultrasound_attenuation_factor 0.3
    
    # exp2 loss_weights
    run_experiment "exp2 loss_weights" \
        --box 8.5 \
        --dfl 2.0 \
        --cls 0.6
    
    # exp3 focal_loss
    run_experiment "exp3 focal_loss" \
        --use_focal_loss \
        --focal_gamma 1.5 \
        --focal_alpha 0.25
    
    # exp4 dim_weights
    run_experiment "exp4 dim_weights" \
        --use_dim_weights \
        --dim_weights 5.0 1.0 5.0 1.0
    
    # exp5-1 hmd_loss_pixel
    run_experiment "exp5-1 hmd_loss_pixel" \
        --use_hmd_loss \
        --hmd_loss_weight 0.5 \
        --hmd_penalty_coeff 0.5
    
    # exp5-2 hmd_loss_mm
    run_experiment "exp5-2 hmd_loss_mm" \
        --use_hmd_loss \
        --hmd_use_mm \
        --hmd_loss_weight 0.5 \
        --hmd_penalty_coeff 0.5
    
    # exp6-1 warmup_optimized
    run_experiment "exp6-1 warmup_optimized" \
        --warmup_epochs 5.0 \
        --warmup_momentum 0.9 \
        --warmup_bias_lr 0.05
    
    # exp6-2 warmup_cosine_restart
    run_experiment "exp6-2 warmup_cosine_restart" \
        --use_cosine_restart \
        --cosine_restart_t0 10 \
        --cosine_restart_t_mult 2 \
        --warmup_epochs 5.0
    
    # exp7-1 siou
    run_experiment "exp7-1 siou" \
        --iou_type SIoU
    
    # exp7-2 eiou
    run_experiment "exp7-2 eiou" \
        --iou_type EIoU
    
    # exp7-3 diou
    run_experiment "exp7-3 diou" \
        --iou_type DIoU
fi

echo "==========================================" | tee -a "$LOG_FILE"
echo "All experiments completed!" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"


