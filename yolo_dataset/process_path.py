from pathlib import Path
import os

# Load environment variables from .env file / 從 .env 檔案載入環境變數
try:
    from dotenv import load_dotenv
    # Load .env file from ultralytics directory / 從 ultralytics 目錄載入 .env 檔案
    env_path = Path(__file__).parent.parent / 'ultralytics' / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, skip / 未安裝 python-dotenv，跳過
    pass

# Get project root from environment variable / 從環境變數獲取專案根目錄
PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if not PROJECT_ROOT:
    # Fallback: try to detect from script location / 備選：嘗試從腳本位置偵測
    PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

# 設定參數
dataset_dir = Path("./seg_45/v1/")  # 含 train/val/test.txt 的資料夾
old_prefix = "/root/ultrasound/DifficultAirway/"
new_prefix = str(Path(PROJECT_ROOT).resolve()) + "/"  # Use PROJECT_ROOT from .env / 使用 .env 中的 PROJECT_ROOT

# 可處理的檔案列表
split_files = ["train.txt", "val.txt", "test.txt"]

for split in split_files:
    file_path = dataset_dir / split
    if not file_path.exists():
        print(f"❌ {file_path} 不存在，略過")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(old_prefix):
            new_line = line.replace(old_prefix, new_prefix)
            new_lines.append(new_line)
        else:
            new_lines.append(line)  # 保留沒改的行

    # 你可以選擇：是否要覆蓋原檔
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"✅ 已處理並更新: {file_path}")
