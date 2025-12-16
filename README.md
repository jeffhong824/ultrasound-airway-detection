# Ultrasound Airway Detection / 超音波呼吸道偵測

Object detection and segmentation for difficult airway ultrasound imaging in clinical settings.

用於臨床困難呼吸道超音波影像的物件偵測與分割。

---

## 🚀 Quick Start / 快速開始

### Install / 安裝

```bash
git clone https://github.com/jeffhong824/ultrasound-airway-detection.git
cd ultrasound-airway-detection/ultralytics

# Create virtual environment
conda create -n ultrasound-yolo python=3.10
conda activate ultrasound-yolo
# or: python -m venv venv && source venv/bin/activate

# Install
pip install -e .
```

### Train / 訓練

```bash
python mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=15 \
  --device 0,1 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-small-obj" \
  --exp_name="exp10-small-obj-optimized"
```

### Test Example / 測試範例

Quick test with minimal epochs / 快速測試（最少輪數）：

```bash
python mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=128 \
  --epochs=3 \
  --device 0 \
  --wandb \
  --project="test-project" \
  --exp_name="test-exp"
```

### Find Best Epoch / 查找最佳 Epoch

```bash
# For production training / 正式訓練
python mycodes/best_epoch.py detect 1 \
  --run_name="yolo11n-det_123-v3-exp10-small-obj-optimized"

# For test training / 測試訓練
python mycodes/best_epoch.py detect 1 \
  --run_name="yolo11n-det_123-v3-test-exp"
```

---

## 📖 Usage / 使用說明

### Basic Command / 基本命令

```bash
python mycodes/train_yolo.py <model> <database> [options]
```

**Required / 必需參數：**
- `model`: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` (or `-seg` variants)
- `database`: `det_123`, `seg_45`, `det_678`

**Common Options / 常用選項：**

| Parameter | Default | Description / 說明 |
|-----------|---------|-------------------|
| `--db_version` | `1` | Dataset version: `1`, `2`, `3` |
| `--es` | - | Use Endoscopy dataset suffix |
| `--epochs` | `50` | Training epochs |
| `--batch` | `16` | Batch size (adjust based on GPU memory / 根據 GPU 記憶體調整) |
| `--device` | `cuda:0` | Device(s): `0`, `0,1`, `0,1,2,3` for multi-GPU / 多 GPU |
| `--imgsz` | `640` | Image size: `640`, `1280`, etc. |
| `--wandb` | - | Enable Wandb logging |
| `--project` | auto | Wandb project name |
| `--exp_name` | - | Experiment identifier |
| `--optimizer` | `AdamW` | `SGD`, `Adam`, `AdamW` |
| `--lr0` | `0.01` | Initial learning rate |
| `--box` | `7.5` | Box loss weight (8-12 for small objects) |
| `--cls` | `0.5` | Class loss weight (0.7-1.0 for imbalance) |
| `--dfl` | `1.5` | DFL loss weight (1.5-2.0 recommended) |
| `--use_focal_loss` | - | Enable Focal Loss for small objects |
| `--use_dim_weights` | - | Enable dimension-specific weights |
| `--dim_weights` | - | `W_L W_T W_R W_B` (e.g., `5.0 1.0 5.0 1.0`) |

**Hardware Configuration / 硬體配置：**
- **Multi-GPU Training / 多 GPU 訓練**:
  - Use `--device 0,1` for 2 GPUs / 使用 2 個 GPU
  - Use `--device 0,1,2,3` for 4 GPUs / 使用 4 個 GPU
  - Batch size will be distributed across GPUs / Batch size 會分散到各 GPU
- **Batch Size / 批次大小**:
  - Adjust `--batch` based on GPU memory / 根據 GPU 記憶體調整
  - Example: `--batch=256` for large GPU memory / 大 GPU 記憶體範例
  - With multi-GPU, effective batch size = `--batch × num_GPUs` / 多 GPU 時，有效批次大小 = `--batch × GPU 數量`

**Ultrasound-specific / 超音波專用設定：**
- `--hsv_h=0` (grayscale images / 灰階影像)
- `--degrees=0 --shear=0 --perspective=0` (no rotation / 無旋轉)

See [mycodes/README.md](ultralytics/mycodes/README.md) for detailed documentation.

詳細文件請參考 [mycodes/README.md](ultralytics/mycodes/README.md)。

---

## 📁 Project Structure / 專案結構

```
ultrasound-airway-detection/
├── ultralytics/
│   ├── mycodes/           # Training scripts / 訓練腳本
│   │   ├── train_yolo.py  # Main training script / 主要訓練腳本
│   │   └── best_epoch.py  # Find best epoch / 查找最佳 epoch
│   ├── loss_docs/         # Loss function docs / Loss 函數文件
│   ├── weights/           # Pretrained models (gitignored) / 預訓練模型
│   └── runs/              # Training outputs (gitignored) / 訓練輸出
└── yolo_dataset/          # Dataset (gitignored, 106 GB) / 資料集
```

---

## 📥 Dataset Download / 資料集下載

Datasets are not included in the repository. Download from Google Drive:

資料集不包含在倉庫中。從 Google Drive 下載：

```bash
# Install dependencies
pip install gdown tqdm

# Option 1: Download complete dataset / 下載完整資料集
gdown 1Y8Ow9JHqeASeB7Mg4QbAQQPL0RYB8iJB -O yolo_dataset.zip --fuzzy
python -c "from tqdm import tqdm; import zipfile; z=zipfile.ZipFile('yolo_dataset.zip'); z.extractall('.', members=tqdm(z.namelist(), desc='Extracting', unit='files'))"

# Option 2: Download individual datasets / 下載個別資料集
mkdir -p yolo_dataset
cd yolo_dataset

# Download det_123 (with progress bar / 顯示進度條)
gdown 1zKJuabh1PygMH9H3eYq4djTYu3kk7KaP -O det_123.zip --fuzzy
python -c "from tqdm import tqdm; import zipfile; z=zipfile.ZipFile('det_123.zip'); z.extractall('.', members=tqdm(z.namelist(), desc='Extracting det_123', unit='files'))"

# Download det_678 (with progress bar / 顯示進度條)
gdown 1Le-DAEpLFSQpcPHn7bdvbLYYe1-4TV-C -O det_678.zip --fuzzy
python -c "from tqdm import tqdm; import zipfile; z=zipfile.ZipFile('det_678.zip'); z.extractall('.', members=tqdm(z.namelist(), desc='Extracting det_678', unit='files'))"

# Verify structure
ls
# Should see: det_123/, det_678/, seg_45/ (if you downloaded complete dataset)
```

**Links / 連結：**
- Complete dataset / 完整資料集: https://drive.google.com/file/d/1Y8Ow9JHqeASeB7Mg4QbAQQPL0RYB8iJB/view
- det_123.zip: https://drive.google.com/file/d/1zKJuabh1PygMH9H3eYq4djTYu3kk7KaP/view
- det_678.zip: https://drive.google.com/file/d/1Le-DAEpLFSQpcPHn7bdvbLYYe1-4TV-C/view

### Download Model Weights / 下載模型權重

```bash
# Download yolo11n.pt pretrained weights
gdown 1f8tmI2Jo9rMTPMl0X4cYcVSzHguckAs8 -O ultralytics/weights/yolo11n.pt --fuzzy

# Other weights (yolo11s, yolo11m, etc.) can be downloaded from Ultralytics official releases
# 其他權重（yolo11s, yolo11m 等）可從 Ultralytics 官方版本下載
```

**Weights link / 權重連結：**
- yolo11n.pt: https://drive.google.com/file/d/1f8tmI2Jo9rMTPMl0X4cYcVSzHguckAs8/view

**Note / 注意：**
- `--fuzzy` required for files >100MB / 大檔案需要 `--fuzzy` 參數
- Extraction uses Python + tqdm for progress bar (quiet, no verbose logs) / 解壓使用 Python + tqdm 顯示進度條（安靜模式，無冗長日誌）
- If tqdm not installed: `pip install tqdm` / 若未安裝 tqdm：`pip install tqdm`
- Alternative: use `unzip -q file.zip` for quiet extraction without progress / 替代方案：使用 `unzip -q file.zip` 安靜解壓（無進度條）
- Ensure sufficient disk space / 確保有足夠的磁碟空間

### Setup Paths for New Machine / 新電腦路徑設置

After downloading datasets and weights, update all paths for your machine:

下載資料集和權重後，更新路徑以適配您的電腦：

```bash
# Run setup script to update all paths
bash setup_paths.sh
```

This script automatically updates:
此腳本會自動更新：

- ✅ `.env` file `PROJECT_ROOT` variable / `.env` 檔案中的 `PROJECT_ROOT` 變數
- ✅ All YAML files `path:` field / 所有 YAML 檔案的 `path:` 欄位
- ✅ All split files (train.txt, val.txt, test.txt, train_ES.txt, val_ES.txt, test_ES.txt) / 所有分割檔案
- ✅ Handles all datasets: det_123, det_678, seg_45 / 處理所有資料集
- ✅ Processes all versions: v1, v2, v3 / 處理所有版本

The script detects the current project root directory and:
腳本會自動偵測當前專案根目錄並：

- Updates `PROJECT_ROOT` in `.env` (used by `train_yolo.py` and `process_path.py`) / 更新 `.env` 中的 `PROJECT_ROOT`（由 `train_yolo.py` 和 `process_path.py` 使用）
- Replaces old paths in split files and YAML files / 替換分割檔案和 YAML 檔案中的舊路徑

---

## 🔧 Configuration / 設定

### Environment Variables / 環境變數

Copy and edit `.env.example`:

```bash
cp ultralytics/.env.example ultralytics/.env
# Edit .env and set:
# - PROJECT_ROOT: your project root directory path
# - WANDB_API_KEY: your Wandb API key
```

**Required variables / 必須變數：**
- `PROJECT_ROOT`: Project root directory path / 專案根目錄路徑
  - Used by `train_yolo.py` and `process_path.py` / 由 `train_yolo.py` 和 `process_path.py` 使用
  - Example: `PROJECT_ROOT=D:/workplace/project_management/github_project/ultrasound-airway-detection2`
- `WANDB_API_KEY`: Wandb API key (get from https://wandb.ai/authorize)

**Note / 注意：**
- The `setup_paths.sh` script will automatically update `PROJECT_ROOT` in `.env` / `setup_paths.sh` 腳本會自動更新 `.env` 中的 `PROJECT_ROOT`
- Install `python-dotenv` if not already installed: `pip install python-dotenv` / 如果未安裝請安裝：`pip install python-dotenv`

---

## 📚 Documentation / 文件

- **Training Guide / 訓練指南**: [ultralytics/mycodes/README.md](ultralytics/mycodes/README.md)
- **Loss Functions / Loss 函數**: [ultralytics/loss_docs/README.md](ultralytics/loss_docs/README.md)

---

## ⚠️ Notes / 注意事項

1. **Dataset Path / 資料集路徑**: Ensure YAML files exist in `yolo_dataset/{database}/v{version}/`
2. **ES Dataset / ES 資料集**: Using `--es` requires `{database}_ES.yaml` file
3. **GPU Memory / GPU 記憶體**: Reduce `--batch` or `--imgsz` if OOM errors occur
4. **Large Files / 大檔案**: Dataset (106 GB) and model weights are gitignored

---

## 🏷️ Version / 版本

Current version: **v0.0.1**

```bash
git fetch --tags
git tag --sort=-creatordate
```

---

## 📝 License / 授權

Based on Ultralytics YOLO. See [LICENSE](ultralytics/LICENSE).

基於 Ultralytics YOLO。詳見 [LICENSE](ultralytics/LICENSE)。

---

## 🙏 Acknowledgments / 致謝

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
