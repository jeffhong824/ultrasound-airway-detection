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
  --epochs=15 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-small-obj" \
  --exp_name="exp10-small-obj-optimized"
```

### Find Best Epoch / 查找最佳 Epoch

```bash
python mycodes/best_epoch.py detect 1 \
  --run_name="yolo11n-det_123-v3-exp10-small-obj-optimized"
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
| `--batch` | `16` | Batch size |
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
# Install gdown
pip install gdown

# Option 1: Download complete dataset / 下載完整資料集
gdown 1Y8Ow9JHqeASeB7Mg4QbAQQPL0RYB8iJB -O yolo_dataset.zip --fuzzy
unzip yolo_dataset.zip -d .

# Option 2: Download individual datasets / 下載個別資料集
mkdir -p yolo_dataset
cd yolo_dataset

# Download det_123
gdown 1zKJuabh1PygMH9H3eYq4djTYu3kk7KaP -O det_123.zip --fuzzy
unzip det_123.zip

# Download det_678
gdown 1Le-DAEpLFSQpcPHn7bdvbLYYe1-4TV-C -O det_678.zip --fuzzy
unzip det_678.zip

# Verify structure
ls
# Should see: det_123/, det_678/, seg_45/ (if you downloaded complete dataset)
```

**Links / 連結：**
- Complete dataset / 完整資料集: https://drive.google.com/file/d/1Y8Ow9JHqeASeB7Mg4QbAQQPL0RYB8iJB/view
- det_123.zip: https://drive.google.com/file/d/1zKJuabh1PygMH9H3eYq4djTYu3kk7KaP/view
- det_678.zip: https://drive.google.com/file/d/1Le-DAEpLFSQpcPHn7bdvbLYYe1-4TV-C/view

**Note / 注意：**
- `--fuzzy` required for files >100MB / 大檔案需要 `--fuzzy` 參數
- Ensure sufficient disk space / 確保有足夠的磁碟空間

---

## 🔧 Configuration / 設定

### Environment Variables / 環境變數

Copy and edit `.env.example`:

```bash
cp ultralytics/.env.example ultralytics/.env
# Edit .env and add your Wandb API key
```

Get Wandb API key: https://wandb.ai/authorize

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
