#!/usr/bin/env python3
"""
Run All Experiments Script
==========================
This script runs all experiments from README.md sequentially.

Usage:
    python run_all_experiments.py [--config 4090|h200] [--skip-failed] [--start-from <name>] [--stop-at <name>]

Options:
    --config 4090|h200    : Select configuration (default: 4090)
    --skip-failed         : Continue to next experiment if current one fails
    --start-from <name>   : Start from a specific experiment name
    --stop-at <name>      : Stop at a specific experiment name
"""

import subprocess
import sys
import argparse
import datetime
import os
import logging
import signal
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_experiment(exp_name, extra_args, config, log_file, skip_failed=False, batch=None, project=None):
    """Run a single experiment"""
    # Configuration
    if config == "4090":
        batch = 16
        device = "cuda:0"
        project = "ultrasound-det_123_ES-v3-4090"
    elif config == "h200":
        batch = 256
        device = "0,1"
        project = "ultrasound-det_123_ES-v3-h200"
    else:
        raise ValueError(f"Invalid config: {config}. Use '4090' or 'h200'")
    
    # Common parameters
    model = "yolo11n"
    database = "det_123"
    db_version = 3
    epochs = 10
    seed = 42
    
    # Build command
    cmd = [
        "python", "ultralytics/mycodes/train_yolo.py", model, database,
        f"--db_version={db_version}",
        "--es",
        f"--batch={batch}",
        f"--epochs={epochs}",
        "--device", device,
        f"--seed={seed}",
        "--wandb",
        f"--project={project}",
        f"--exp_name={exp_name}",
    ] + extra_args
    
    # Log
    print("=" * 60)
    print(f"Running: {exp_name}")
    print(f"Time: {datetime.datetime.now()}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Running: {exp_name}\n")
        f.write(f"Time: {datetime.datetime.now()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 60 + "\n")
        f.flush()
    
    # Run command with proper signal handling for Ctrl+C
    process = None
    try:
        # Use Popen for better signal handling
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Set up signal handler to forward SIGINT to child process
        # Note: On Windows, signal handling works differently, but KeyboardInterrupt will still be caught
        def signal_handler(sig, frame):
            if process is not None:
                print("\n⚠️ Interrupted by user (Ctrl+C). Terminating subprocess...")
                process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("⚠️ Process did not terminate, forcing kill...")
                    process.kill()
                    process.wait()
                sys.exit(130)  # Standard exit code for SIGINT
        
        # Register signal handler (works on Unix/Linux/Mac, KeyboardInterrupt handles Windows)
        try:
            signal.signal(signal.SIGINT, signal_handler)
        except (ValueError, OSError):
            # On Windows, SIGINT may not be available, but KeyboardInterrupt will still work
            pass
        
        # Real-time output and logging
        with open(log_file, "a", encoding="utf-8") as log_f:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output, end='', flush=True)
                    log_f.write(output)
                    log_f.flush()
        
        # Get return code
        returncode = process.poll()
        if returncode is None:
            returncode = process.wait()
        
        if returncode == 0:
            print(f"SUCCESS: Experiment {exp_name} completed successfully")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"SUCCESS: Experiment {exp_name} completed successfully\n\n")
        else:
            print(f"WARNING: Experiment {exp_name} failed with exit code {returncode}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"WARNING: Experiment {exp_name} failed with exit code {returncode}\n\n")
            if not skip_failed:
                sys.exit(returncode)
        
        return returncode == 0
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user (Ctrl+C)")
        if process is not None:
            print("Terminating subprocess...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n⚠️ Interrupted by user (Ctrl+C)\n\n")
        sys.exit(130)
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment {exp_name} failed")
        print(f"Exit code: {e.returncode}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ERROR: Experiment {exp_name} failed\n")
            f.write(f"Exit code: {e.returncode}\n\n")
        if not skip_failed:
            sys.exit(e.returncode)
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error running {exp_name}: {e}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ERROR: Unexpected error running {exp_name}: {e}\n\n")
        if not skip_failed:
            sys.exit(1)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments from README.md sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        choices=["4090", "h200"],
        default="4090",
        help="Select configuration (default: 4090)"
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Continue to next experiment if current one fails"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="",
        help="Start from a specific experiment name"
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        default="",
        help="Stop at a specific experiment name"
    )
    
    args = parser.parse_args()
    
    # Log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments_{args.config}_{timestamp}.log"
    
    print(f"Configuration: {args.config}")
    print(f"Logging to: {log_file}")
    print(f"Skip failed: {args.skip_failed}")
    if args.start_from:
        print(f"Start from: {args.start_from}")
    if args.stop_at:
        print(f"Stop at: {args.stop_at}")
    print()
    
    # Set configuration variables
    if args.config == "4090":
        BATCH = 16
        PROJECT = "ultrasound-det_123_ES-v3-4090"
    elif args.config == "h200":
        BATCH = 256
        PROJECT = "ultrasound-det_123_ES-v3-h200"
    else:
        raise ValueError(f"Invalid config: {args.config}. Use '4090' or 'h200'")
    
    started = False
    
    # =============================================================================
    # RTX 4090 Configuration Experiments
    # =============================================================================
    
    if args.config == "4090":
        experiments = [
            # exp0 baseline
            # ("exp0 baseline", []),
            
            # exp0 baseline+keep_top_conf_per_class
            ("exp0 baseline+keep_top_conf_per_class", [
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp1-1 data_aug
            # ("exp1-1 data_aug", [
            #     "--scale", "0.7",
            #     "--translate", "0.15",
            #     "--hsv_s", "0.8",
            #     "--hsv_v", "0.5",
            #     "--hsv_h", "0.0"
            # ]),
            
            # exp1-1 data_aug+keep_top_conf_per_class
            ("exp1-1 data_aug+keep_top_conf_per_class", [
                "--scale", "0.7",
                "--translate", "0.15",
                "--hsv_s", "0.8",
                "--hsv_v", "0.5",
                "--hsv_h", "0.0",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp1-2 ultrasound_aug
            # ("exp1-2 ultrasound_aug", [
            #     "--use_ultrasound_aug",
            #     "--ultrasound_speckle_var", "0.1",
            #     "--ultrasound_attenuation_factor", "0.3"
            # ]),
            
            # exp1-2 ultrasound_aug+keep_top_conf_per_class
            ("exp1-2 ultrasound_aug+keep_top_conf_per_class", [
                "--use_ultrasound_aug",
                "--ultrasound_speckle_var", "0.1",
                "--ultrasound_attenuation_factor", "0.3",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp2 loss_weights
            # ("exp2 loss_weights", [
            #     "--box", "8.5",
            #     "--dfl", "2.0",
            #     "--cls", "0.6"
            # ]),
            
            # exp2 loss_weights+keep_top_conf_per_class
            ("exp2 loss_weights+keep_top_conf_per_class", [
                "--box", "8.5",
                "--dfl", "2.0",
                "--cls", "0.6",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp3 focal_loss
            # ("exp3 focal_loss", [
            #     "--use_focal_loss",
            #     "--focal_gamma", "1.5",
            #     "--focal_alpha", "0.25"
            # ]),
            
            # exp3 focal_loss+keep_top_conf_per_class
            ("exp3 focal_loss+keep_top_conf_per_class", [
                "--use_focal_loss",
                "--focal_gamma", "1.5",
                "--focal_alpha", "0.25",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp4 dim_weights
            # ("exp4 dim_weights", [
            #     "--use_dim_weights",
            #     "--dim_weights", "5.0", "1.0", "5.0", "1.0"
            # ]),
            
            # exp4 dim_weights+keep_top_conf_per_class
            ("exp4 dim_weights+keep_top_conf_per_class", [
                "--use_dim_weights",
                "--dim_weights", "5.0", "1.0", "5.0", "1.0",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp5-1 hmd_loss_pixel
            # ("exp5-1 hmd_loss_pixel", [
            #     "--use_hmd_loss",
            #     "--hmd_loss_weight", "0.5",
            #     "--hmd_penalty_coeff", "0.5"
            # ]),
            
            # exp5-1 hmd_loss_pixel+keep_top_conf_per_class
            ("exp5-1 hmd_loss_pixel+keep_top_conf_per_class", [
                "--use_hmd_loss",
                "--hmd_loss_weight", "0.5",
                "--hmd_penalty_coeff", "0.5",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp5-2 hmd_loss_mm
            # ("exp5-2 hmd_loss_mm", [
            #     "--use_hmd_loss",
            #     "--hmd_use_mm",
            #     "--hmd_loss_weight", "0.5",
            #     "--hmd_penalty_coeff", "0.5"
            # ]),
            
            # # exp5-2 hmd_loss_mm+keep_top_conf_per_class
            # ("exp5-2 hmd_loss_mm+keep_top_conf_per_class", [
            #     "--use_hmd_loss",
            #     "--hmd_use_mm",
            #     "--hmd_loss_weight", "0.5",
            #     "--hmd_penalty_coeff", "0.5",
            #     "--keep_top_conf_per_class",
            #     "--conf_low", "0.1"
            # ]),
            
            # exp6-1 warmup_optimized
            # ("exp6-1 warmup_optimized", [
            #     "--warmup_epochs", "5.0",
            #     "--warmup_momentum", "0.9",
            #     "--warmup_bias_lr", "0.05"
            # ]),
            
            # exp6-1 warmup_optimized+keep_top_conf_per_class
            ("exp6-1 warmup_optimized+keep_top_conf_per_class", [
                "--warmup_epochs", "5.0",
                "--warmup_momentum", "0.9",
                "--warmup_bias_lr", "0.05",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp6-2 warmup_cosine_restart
            # ("exp6-2 warmup_cosine_restart", [
            #     "--use_cosine_restart",
            #     "--cosine_restart_t0", "10",
            #     "--cosine_restart_t_mult", "2",
            #     "--warmup_epochs", "5.0"
            # ]),
            
            # exp6-2 warmup_cosine_restart+keep_top_conf_per_class
            ("exp6-2 warmup_cosine_restart+keep_top_conf_per_class", [
                "--use_cosine_restart",
                "--cosine_restart_t0", "10",
                "--cosine_restart_t_mult", "2",
                "--warmup_epochs", "5.0",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp7-1 siou
            # ("exp7-1 siou", [
            #     "--iou_type", "SIoU"
            # ]),
            
            # exp7-1 siou+keep_top_conf_per_class
            ("exp7-1 siou+keep_top_conf_per_class", [
                "--iou_type", "SIoU",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp7-2 eiou
            # ("exp7-2 eiou", [
            #     "--iou_type", "EIoU"
            # ]),
            
            # exp7-2 eiou+keep_top_conf_per_class
            ("exp7-2 eiou+keep_top_conf_per_class", [
                "--iou_type", "EIoU",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp7-3 diou
            # ("exp7-3 diou", [
            #     "--iou_type", "DIoU"
            # ]),
            
            # exp7-3 diou+keep_top_conf_per_class
            ("exp7-3 diou+keep_top_conf_per_class", [
                "--iou_type", "DIoU",
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
        ]
    
    # =============================================================================
    # H200 Configuration Experiments
    # =============================================================================
    
    elif args.config == "h200":
        experiments = [
            # exp0 baseline
            ("exp0 baseline", [
                "--keep_top_conf_per_class",
                "--conf_low", "0"
            ]),
            
            # exp1-1 data_aug
            # ("exp1-1 data_aug", [
            #     "--scale", "0.7",
            #     "--translate", "0.15",
            #     "--hsv_s", "0.8",
            #     "--hsv_v", "0.5",
            #     "--hsv_h", "0.0"
            # ]),
            
            # exp1-2 ultrasound_aug
            # ("exp1-2 ultrasound_aug", [
            #     "--use_ultrasound_aug",
            #     "--ultrasound_speckle_var", "0.1",
            #     "--ultrasound_attenuation_factor", "0.3"
            # ]),
            
            # exp2 loss_weights
            # ("exp2 loss_weights", [
            #     "--box", "8.5",
            #     "--dfl", "2.0",
            #     "--cls", "0.6"
            # ]),
            
            # exp3 focal_loss
            # ("exp3 focal_loss", [
            #     "--use_focal_loss",
            #     "--focal_gamma", "1.5",
            #     "--focal_alpha", "0.25"
            # ]),
            
            # exp4 dim_weights
            # ("exp4 dim_weights", [
            #     "--use_dim_weights",
            #     "--dim_weights", "5.0", "1.0", "5.0", "1.0"
            # ]),
            
            # exp5-1 hmd_loss_pixel
            # ("exp5-1 hmd_loss_pixel", [
            #     "--use_hmd_loss",
            #     "--hmd_loss_weight", "0.5",
            #     "--hmd_penalty_coeff", "0.5"
            # ]),
            
            # exp5-2 hmd_loss_mm
            # ("exp5-2 hmd_loss_mm", [
            #     "--use_hmd_loss",
            #     "--hmd_use_mm",
            #     "--hmd_loss_weight", "0.5",
            #     "--hmd_penalty_coeff", "0.5"
            # ]),
            
            # exp6-1 warmup_optimized
            # ("exp6-1 warmup_optimized", [
            #     "--warmup_epochs", "5.0",
            #     "--warmup_momentum", "0.9",
            #     "--warmup_bias_lr", "0.05"
            # ]),
            
            # exp6-2 warmup_cosine_restart
            # ("exp6-2 warmup_cosine_restart", [
            #     "--use_cosine_restart",
            #     "--cosine_restart_t0", "10",
            #     "--cosine_restart_t_mult", "2",
            #     "--warmup_epochs", "5.0"
            # ]),
            
            # exp7-1 siou
            # ("exp7-1 siou", [
            #     "--iou_type", "SIoU"
            # ]),
            
            # exp7-2 eiou
            # ("exp7-2 eiou", [
            #     "--iou_type", "EIoU"
            # ]),
            
            # exp7-3 diou
            # ("exp7-3 diou", [
            #     "--iou_type", "DIoU"
            # ]),
        ]
    
    # Run experiments
    for exp_name, extra_args in experiments:
        # Check if we should start from this experiment
        if args.start_from and not started:
            if exp_name != args.start_from:
                print(f"Skipping {exp_name} (waiting for --start-from {args.start_from})")
                continue
            started = True
        
        # Check if we should stop at this experiment
        if args.stop_at and exp_name == args.stop_at:
            print(f"Stopping at {exp_name} (--stop-at reached)")
            break
        
        # Run experiment
        success = run_experiment(exp_name, extra_args, args.config, log_file, args.skip_failed, batch=BATCH, project=PROJECT)
        
        if not success and not args.skip_failed:
            print(f"Experiment {exp_name} failed. Exiting.")
            sys.exit(1)
    
    # Final message
    print("=" * 60)
    print("All experiments completed!")
    print(f"Time: {datetime.datetime.now()}")
    print(f"Log file: {log_file}")
    print("=" * 60)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("All experiments completed!\n")
        f.write(f"Time: {datetime.datetime.now()}\n")
        f.write(f"Log file: {log_file}\n")
        f.write("=" * 60 + "\n")


if __name__ == "__main__":
    main()

