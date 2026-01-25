#!/usr/bin/env python3
# Setup script for new machines / 新電腦設置腳本
# Updates all paths in dataset files to match current machine / 更新所有資料集檔案中的路徑以匹配當前電腦

import os
import sys
from pathlib import Path
import re

# Get current script directory / 獲取當前腳本目錄
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DATASET_DIR = PROJECT_ROOT / 'yolo_dataset'

print("==========================================")
print("Setting up paths for new machine")
print("更新路徑以適配新電腦")
print("==========================================")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset dir: {DATASET_DIR}")
print()

# Check if dataset directory exists / 檢查資料集目錄是否存在
if not DATASET_DIR.exists():
    print(f"❌ Dataset directory not found: {DATASET_DIR}")
    print("Please download the dataset first. / 請先下載資料集")
    sys.exit(1)

# Update .env file PROJECT_ROOT / 更新 .env 檔案中的 PROJECT_ROOT
ENV_FILE = PROJECT_ROOT / 'ultralytics' / '.env'
ENV_EXAMPLE = PROJECT_ROOT / 'ultralytics' / '.env.example'

print("Updating .env file PROJECT_ROOT...")
# Convert path to use forward slashes for .env file (works on both Windows and Linux)
# 將路徑轉換為正斜線以用於 .env 檔案（在 Windows 和 Linux 上都有效）
env_path = str(PROJECT_ROOT.resolve()).replace("\\", "/")

# Update or create .env file / 更新或創建 .env 檔案
if ENV_FILE.exists():
    # Read existing .env file / 讀取現有的 .env 檔案
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace PROJECT_ROOT line if exists, otherwise add it / 如果存在則替換 PROJECT_ROOT 行，否則添加
    if re.search(r'^PROJECT_ROOT=', content, re.MULTILINE):
        content = re.sub(r'^PROJECT_ROOT=.*$', f'PROJECT_ROOT={env_path}', content, flags=re.MULTILINE)
        print(f"✅ Updated PROJECT_ROOT in .env to: {env_path}")
    else:
        # Add PROJECT_ROOT if not found / 如果未找到則添加 PROJECT_ROOT
        if not content.endswith('\n'):
            content += '\n'
        content += f'\n# Project Root Path\nPROJECT_ROOT={env_path}\n'
        print(f"✅ Added PROJECT_ROOT to .env: {env_path}")
    
    with open(ENV_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
else:
    # Create .env from .env.example if it exists / 如果存在則從 .env.example 創建 .env
    if ENV_EXAMPLE.exists():
        with open(ENV_EXAMPLE, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Replace PROJECT_ROOT in example / 替換範例中的 PROJECT_ROOT
        content = re.sub(r'^PROJECT_ROOT=.*$', f'PROJECT_ROOT={env_path}', content, flags=re.MULTILINE)
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created .env from .env.example with PROJECT_ROOT: {env_path}")
    else:
        # Create basic .env file / 創建基本的 .env 檔案
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# Project Root Path\nPROJECT_ROOT={env_path}\n")
        print(f"✅ Created .env with PROJECT_ROOT: {env_path}")

print()

# Process all datasets / 處理所有資料集
DATASETS = ["det_123", "det_678", "seg_45"]
VERSIONS = ["v1", "v2", "v3"]
SPLIT_FILES = ["train.txt", "val.txt", "test.txt", "train_ES.txt", "val_ES.txt", "test_ES.txt"]

# Common old path patterns that might need replacement / 可能需要替換的常見舊路徑模式
OLD_PREFIXES = [
    "/root/ultrasound/DifficultAirway/",
    "D:/workplace/project_management/github_project/ultrasound-airway-detection/",
    "D:/workplace/project_management/github_project/ultrasound-airway-detection2/",
]

for dataset in DATASETS:
    for version in VERSIONS:
        dataset_version_dir = DATASET_DIR / dataset / version
        
        if not dataset_version_dir.exists():
            continue  # Skip if directory doesn't exist / 目錄不存在則跳過
        
        print(f"Processing: {dataset}/{version}")
        print(f"處理: {dataset}/{version}")
        
        # Process split files / 處理分割檔案
        for split_file in SPLIT_FILES:
            split_file_path = dataset_version_dir / split_file
            
            if not split_file_path.exists():
                continue  # Skip if file doesn't exist / 檔案不存在則跳過
            
            print(f"  Processing: {split_file}")
            
            # Read file
            with open(split_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = []
            changed = False

            for line in lines:
                original_line = line.rstrip('\n\r')
                new_line = original_line
                
                # Check if line contains any old prefix / 檢查行是否包含舊前綴
                for old_prefix in OLD_PREFIXES:
                    if old_prefix in new_line:
                        # Find yolo_dataset in the path / 在路徑中找到 yolo_dataset
                        if "yolo_dataset" in new_line:
                            # Extract relative path from yolo_dataset onwards / 從 yolo_dataset 開始提取相對路徑
                            idx = new_line.find("yolo_dataset")
                            relative_part = new_line[idx:]
                            # Build new path using project root / 使用專案根目錄構建新路徑
                            new_line = str(PROJECT_ROOT / relative_part.replace("\\", "/"))
                            changed = True
                            break
                
                new_lines.append(new_line)

            if changed:
                # Write back / 寫回檔案
                with open(split_file_path, 'w', encoding='utf-8') as f:
                    for line in new_lines:
                        f.write(line + "\n")
                print(f"  ✅ Updated: {split_file}")
            else:
                print(f"  ℹ️  No changes needed: {split_file}")
        
        # Process YAML files / 處理 YAML 檔案
        for yaml_file in dataset_version_dir.glob("*.yaml"):
            yaml_name = yaml_file.name
            print(f"  Processing YAML: {yaml_name}")
            
            # Read file
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace path: line / 替換 path: 行
            # Match pattern like "path: D:/path/to/yolo_dataset/det_123/v3"
            # Extract dataset/version part and rebuild path / 提取資料集/版本部分並重建路徑
            pattern = r'^path:\s*.*?yolo_dataset/([^\s]+)$'

            def replace_path(match):
                dataset_version = match.group(1)
                new_path = PROJECT_ROOT / "yolo_dataset" / dataset_version
                # Use forward slashes for YAML compatibility / 使用正斜線以兼容 YAML
                return f'path: {new_path.as_posix()}'

            original_content = content
            content = re.sub(pattern, replace_path, content, flags=re.MULTILINE)

            if content != original_content:
                # Write back / 寫回檔案
                with open(yaml_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Updated YAML: {yaml_file.name}")
            else:
                print(f"  ℹ️  No changes needed in YAML: {yaml_file.name}")
        print()

print("==========================================")
print("✅ Path setup complete!")
print("路徑設置完成！")
print("==========================================")



