#!/usr/bin/env python3
"""
Test script to verify record_experiment_results.py works correctly
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))
sys.path.insert(0, str(project_root / "ultralytics" / "mycodes"))

try:
    from record_experiment_results import record_experiment_to_excel
    
    # Test with a known experiment
    success = record_experiment_to_excel(
        exp_name="exp0 baseline",
        project="ultrasound-det_123_ES-v3-4090",
        config="4090",
        database="det_123",
        db_version=3,
        batch=16,
        imgsz=640,
    )
    
    if success:
        print("✅ Test passed! Results recorded successfully.")
    else:
        print("❌ Test failed! Could not record results.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


