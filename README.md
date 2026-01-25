# Ultrasound Airway Detection / 超音波呼吸道偵測

Object detection and segmentation for difficult airway ultrasound imaging in clinical settings.

用於臨床困難呼吸道超音波影像的物件偵測與分割。

**Current Version**: v0.1.1

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

#### 實驗設計 / Experiment Design

**重要提示 / Important Note**: 
為了確保實驗結果的可比較性，不同硬件配置（RTX 4090 和 H200）使用**不同的 project 名稱**，避免因 batch size 差異導致的訓練動態不同影響比較。

**Project 命名規則 / Project Naming Convention**:
- RTX 4090 (batch=16): `ultrasound-det_123_ES-v3-4090`
- H200 (batch=256): `ultrasound-det_123_ES-v3-h200`

**實驗規劃 / Experiment Plan**:

每個 project 包含以下實驗，所有實驗均使用 `--seed 42` 確保可重現性：

**基礎實驗（每個實驗都有兩個版本）**：
- **exp0 baseline**: 基準實驗，使用所有默認參數，作為對照組（**注意**：即使未啟用 HMD Loss，也會自動計算 HMD 評估指標）
  - **exp0 baseline**: 原始版本
  - **exp0 baseline+keep_top_conf_per_class**: 使用 `--keep_top_conf_per_class --conf_low 0.1` 參數（提高 HMD Detection_Rate）
- **exp1-1 data_aug**: 相對於 exp0，優化 Data Augmentation 參數（針對小物件）
  - **exp1-1 data_aug**: 原始版本
  - **exp1-1 data_aug+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp1-2 ultrasound_aug**: 相對於 exp0，啟用超音波專用數據增強（斑點雜訊、深度衰減）
  - **exp1-2 ultrasound_aug**: 原始版本
  - **exp1-2 ultrasound_aug+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp2 loss_weights**: 相對於 exp0，調整 Loss 權重參數（定位優先）
  - **exp2 loss_weights**: 原始版本
  - **exp2 loss_weights+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp3 focal_loss**: 相對於 exp0，啟用 Focal Loss（處理類別不平衡）
  - **exp3 focal_loss**: 原始版本
  - **exp3 focal_loss+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp4 dim_weights**: 相對於 exp0，啟用水平方向維度權重（HMD 優化）
  - **exp4 dim_weights**: 原始版本
  - **exp4 dim_weights+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp5-1 hmd_loss_pixel**: 相對於 exp0，啟用 HMD Loss（像素級別）
  - **exp5-1 hmd_loss_pixel**: 原始版本
  - **exp5-1 hmd_loss_pixel+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp5-2 hmd_loss_mm**: 相對於 exp0，啟用 HMD Loss（毫米級別，使用真實尺寸）
  - **exp5-2 hmd_loss_mm**: 原始版本
  - **exp5-2 hmd_loss_mm+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp6-1 warmup_optimized**: 相對於 exp0，優化 Warmup 參數（針對超音波小物件）
  - **exp6-1 warmup_optimized**: 原始版本
  - **exp6-1 warmup_optimized+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp6-2 warmup_cosine_restart**: 相對於 exp0，使用 Cosine Annealing with Warm Restarts 學習率調度
  - **exp6-2 warmup_cosine_restart**: 原始版本
  - **exp6-2 warmup_cosine_restart+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp7-1 siou**: 相對於 exp0，使用 SIoU Loss（對角度敏感，適合細長目標）
  - **exp7-1 siou**: 原始版本
  - **exp7-1 siou+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp7-2 eiou**: 相對於 exp0，使用 EIoU Loss（直接優化長寬邊長，適合細長目標）
  - **exp7-2 eiou**: 原始版本
  - **exp7-2 eiou+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數
- **exp7-3 diou**: 相對於 exp0，使用 DIoU Loss（考慮中心點距離，對 HMD 計算有幫助）
  - **exp7-3 diou**: 原始版本
  - **exp7-3 diou+keep_top_conf_per_class**: 帶 `--keep_top_conf_per_class --conf_low 0.1` 參數

##### exp0 baseline 默認參數說明

**Loss 權重**（默認值）：
- `--box`: 7.5
- `--cls`: 0.5
- `--dfl`: 1.5

**Data Augmentation**（默認值）：
- `--scale`: 0.5
- `--translate`: 0.1
- `--hsv_h`: 0.0
- `--hsv_s`: 0.7
- `--hsv_v`: 0.4

**Warmup 參數**（默認值）：
- `--warmup_epochs`: 3.0
- `--warmup_momentum`: 0.8
- `--warmup_bias_lr`: 0.1

**學習率調度**（默認值）：
- `--cos_lr`: False（使用線性衰減）
- `--lr0`: 0.01（初始學習率）
- `--lrf`: 0.01（最終學習率）
- `--use_cosine_restart`: False（未啟用 Cosine Restart）
- `--cosine_restart_t0`: 10（第一個週期 epoch 數，僅當 `--use_cosine_restart` 啟用時有效）
- `--cosine_restart_t_mult`: 2（週期倍增因子，僅當 `--use_cosine_restart` 啟用時有效）

**其他參數**（默認值）：
- `--use_focal_loss`: False（未啟用）
- `--use_dim_weights`: False（未啟用）
- `--use_hmd_loss`: False（未啟用）

#### RTX 4090 配置 (Single GPU / 單 GPU)

**exp0 baseline: 基準實驗（所有默認參數）**

**注意**：
- 即使未啟用 HMD Loss（`--use_hmd_loss=False`），所有 `det_123` 資料庫的實驗（包括 baseline）都會自動計算 HMD 評估指標（Detection_Rate、RMSE_HMD、Overall_Score），以便監控和比較所有實驗的 HMD 性能。
- 如果 Detection_Rate 為 0，可能是 confidence 閾值過高導致預測被過濾。可以嘗試：
  - 降低 `--conf` 參數（例如從 0.25 降到 0.1）
  - 或使用 `--keep_top_conf_per_class` 參數：使用較低的 confidence 閾值進行初始過濾，但每個類別只保留 confidence 最高的 bbox（適合 HMD 計算，因為每個類別應該只有一個檢測）

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp0 baseline"
```

**exp0 baseline+keep_top_conf_per_class: 基準實驗（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp0 baseline+keep_top_conf_per_class" \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp1-1 data_aug: Data Augmentation 優化（針對小物件）**

相對於 exp0 的改動：
- `--scale`: 0.5 → **0.7**（增加尺寸多樣性，讓小目標在縮放後仍可被模型辨識）
- `--translate`: 0.1 → **0.15**（增加位置變異，提升模型在不同掃描位置的穩定性）
- `--hsv_s`: 0.7 → **0.8**（強化亮度變化，使小病灶在高噪音背景中更突出）
- `--hsv_v`: 0.4 → **0.5**（強化對比變化）
- `--hsv_h`: 0.0（保持不變，超音波為黑白影像，不需色調遷移）

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp1-1 data_aug" \
  --scale 0.7 \
  --translate 0.15 \
  --hsv_s 0.8 \
  --hsv_v 0.5 \
  --hsv_h 0.0
```

**exp1-1 data_aug+keep_top_conf_per_class: Data Augmentation 優化（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp1-1 data_aug+keep_top_conf_per_class" \
  --scale 0.7 \
  --translate 0.15 \
  --hsv_s 0.8 \
  --hsv_v 0.5 \
  --hsv_h 0.0 \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp2 loss_weights: Loss 權重調整（定位優先）**

相對於 exp0 的改動：
- `--box`: 7.5 → **8.5**（+13%，更強調定位誤差，適合小範圍、細長結構）
- `--dfl`: 1.5 → **2.0**（+33%，直接提高邊界框細緻回歸精度，改善線段邊緣定位）
- `--cls`: 0.5 → **0.6**（+20%，提高分類損失權重）

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp2 loss_weights" \
  --box 8.5 \
  --dfl 2.0 \
  --cls 0.6
```

**exp2 loss_weights+keep_top_conf_per_class: Loss 權重調整（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp2 loss_weights+keep_top_conf_per_class" \
  --box 8.5 \
  --dfl 2.0 \
  --cls 0.6 \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp3 focal_loss: Focal Loss（處理類別不平衡）**

相對於 exp0 的改動：
- `--use_focal_loss`: False → **True**（啟用 Focal Loss）
- `--focal_gamma`: **1.5**（減少 easy-negative 的干擾）
- `--focal_alpha`: **0.25**（提高稀少正樣本（超音波病灶）的學習權重）

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp3 focal_loss" \
  --use_focal_loss \
  --focal_gamma 1.5 \
  --focal_alpha 0.25
```

**exp3 focal_loss+keep_top_conf_per_class: Focal Loss（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp3 focal_loss+keep_top_conf_per_class" \
  --use_focal_loss \
  --focal_gamma 1.5 \
  --focal_alpha 0.25 \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp4 dim_weights: 水平方向維度權重（HMD 優化）**

相對於 exp0 的改動：
- `--use_dim_weights`: False → **True**（啟用維度權重）
- `--dim_weights`: [1.0, 1.0, 1.0, 1.0] → **[5.0, 1.0, 5.0, 1.0]**（加強水平定位（Δx）的敏感度，適用目標呈現「水平細長」特性）

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp4 dim_weights" \
  --use_dim_weights \
  --dim_weights 5.0 1.0 5.0 1.0
```

**exp4 dim_weights+keep_top_conf_per_class: 水平方向維度權重（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp4 dim_weights+keep_top_conf_per_class" \
  --use_dim_weights \
  --dim_weights 5.0 1.0 5.0 1.0 \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp1-2 ultrasound_aug: 超音波專用數據增強**

相對於 exp0 的改動：
- `--use_ultrasound_aug`: False → **True**（啟用超音波專用數據增強）
- `--ultrasound_speckle_var`: **0.1**（斑點雜訊變異數）
- `--ultrasound_attenuation_factor`: **0.3**（深度信號衰減因子）

**設計理念**：
- **斑點雜訊（Speckle Noise）**：超音波影像的固有特性，由聲波干涉產生，會降低影像解析度和對比度
- **深度信號衰減（Signal Attenuation）**：模擬超音波在組織中傳播時的深度相關衰減，底部（深層）信號較弱
- 這兩種增強技術模擬真實超音波影像的物理特性，提高模型對實際臨床環境的適應性

**參考文獻**：
1. **Despeckling of Medical Ultrasound Images** (Michailovich & Tannenbaum, 2006)
   - 概述：研究超音波影像中斑點雜訊的統計特性，提出使用乘性模型描述斑點雜訊的形成過程。論文分析了對數轉換後斑點雜訊的特性，並評估了多種非線性濾波器（小波去噪、總變分濾波、各向異性擴散）在去斑點處理中的性能。研究指出，斑點雜訊會降低影像對比度、模糊細節，從而影響診斷價值。
   - 連結：https://pmc.ncbi.nlm.nih.gov/articles/PMC3639001/
   - 關鍵發現：斑點雜訊是超音波影像的固有特性，通過乘性模型可以更好地描述其統計特性；適當的預處理可以將對數轉換後的雜訊轉換為接近白高斯雜訊，從而提高濾波效果。

2. **Speckle Noise Reduction in Ultrasound Images** (Rajabi et al., ISPRS)
   - 概述：評估多種斑點雜訊去除濾波器在超音波影像上的效果與性能。研究比較了不同濾波方法的優缺點，為超音波影像處理提供了實用的參考。
   - 連結：https://www.isprs.org/proceedings/xxxvi/1-W41/makaleler/Rajabi_Specle_Noise.pdf
   - 關鍵發現：不同濾波方法對超音波影像的處理效果各有優劣，需要根據具體應用場景選擇合適的方法。

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp1-2 ultrasound_aug" \
  --use_ultrasound_aug \
  --ultrasound_speckle_var 0.1 \
  --ultrasound_attenuation_factor 0.3
```

**exp5-1 hmd_loss_pixel: HMD Loss（像素級別）**

相對於 exp0 的改動：
- `--use_hmd_loss`: False → **True**（啟用 HMD Loss）
- `--hmd_loss_weight`: **0.5**（HMD loss 的權重係數）
- `--hmd_penalty_coeff`: **0.5**（單個檢測時的權重係數）

**注意**：`--hmd_penalty_single` 和 `--hmd_penalty_none` 會根據 `--imgsz` 自動計算（預設 `imgsz=640`）：
- `penalty_none = imgsz`（預設 640.0 像素）
- `penalty_single = imgsz / 2`（預設 320.0 像素）
- 如需自訂，可明確指定這些參數

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp5-1 hmd_loss_pixel" \
  --use_hmd_loss \
  --hmd_loss_weight 0.5 \
  --hmd_penalty_coeff 0.5
```

**exp5-2 hmd_loss_mm: HMD Loss（毫米級別，真實尺寸）**

相對於 exp0 的改動：
- `--use_hmd_loss`: False → **True**（啟用 HMD Loss）
- `--hmd_use_mm`: False → **True**（使用毫米而非像素）
- `--hmd_loss_weight`: **0.5**（HMD loss 的權重係數）
- `--hmd_penalty_coeff`: **0.5**（單個檢測時的權重係數）

**注意**：
- `--hmd_penalty_single` 和 `--hmd_penalty_none` 會根據 `--imgsz` 自動計算（預設 `imgsz=640`）：
  - `penalty_none = imgsz`（預設 640.0 像素）
  - `penalty_single = imgsz / 2`（預設 320.0 像素）
- 使用 mm 模式時，penalty 值會自動轉換為毫米（根據每個圖像的 PixelSpacing）
- 如需自訂 penalty 值（像素），可明確指定 `--hmd_penalty_single` 和 `--hmd_penalty_none`

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp5-2 hmd_loss_mm" \
  --use_hmd_loss \
  --hmd_use_mm \
  --hmd_loss_weight 0.5 \
  --hmd_penalty_coeff 0.5
```

**exp6-1 warmup_optimized: Warmup 參數優化（針對超音波小物件）**

相對於 exp0 的改動：
- `--warmup_epochs`: 3.0 → **5.0**（增加 warmup 週期，讓模型更穩定地適應超音波數據）
- `--warmup_momentum`: 0.8 → **0.9**（提高初始 momentum，加速收斂）
- `--warmup_bias_lr`: 0.1 → **0.05**（降低 bias 初始學習率，避免過度調整）

**設計理念**：
- 超音波影像具有高噪音特性，需要更長的 warmup 週期讓模型適應
- 小物件檢測需要更穩定的訓練初期，避免梯度爆炸

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp6-1 warmup_optimized" \
  --warmup_epochs 5.0 \
  --warmup_momentum 0.9 \
  --warmup_bias_lr 0.05
```

**exp6-2 warmup_cosine_restart: Cosine Annealing with Warm Restarts**

相對於 exp0 的改動：
- `--use_cosine_restart`: False → **True**（啟用 Cosine Annealing with Warm Restarts）
- `--cosine_restart_t0`: **10**（第一個週期的 epoch 數）
- `--cosine_restart_t_mult`: **2**（每個週期長度的倍增因子）
- `--warmup_epochs`: 3.0 → **5.0**（配合 cosine restart 的 warmup）

**設計理念**：
- Cosine Annealing with Warm Restarts 適合超音波數據的週期性特徵
- 通過週期性重啟學習率，幫助模型跳出局部最優，探索更好的解
- 適合處理超音波影像中不同深度、不同角度的多樣性

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp6-2 warmup_cosine_restart" \
  --use_cosine_restart \
  --cosine_restart_t0 10 \
  --cosine_restart_t_mult 2 \
  --warmup_epochs 5.0
```

**exp6-2 warmup_cosine_restart+keep_top_conf_per_class: Cosine Annealing with Warm Restarts（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp6-2 warmup_cosine_restart+keep_top_conf_per_class" \
  --use_cosine_restart \
  --cosine_restart_t0 10 \
  --cosine_restart_t_mult 2 \
  --warmup_epochs 5.0 \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

#### H200 配置 (Multi-GPU / 多 GPU)

**exp0 baseline: 基準實驗（所有默認參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp0 baseline" \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp1-1 data_aug: Data Augmentation 優化（針對小物件）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp1-1 data_aug" \
  --scale 0.7 \
  --translate 0.15 \
  --hsv_s 0.8 \
  --hsv_v 0.5 \
  --hsv_h 0.0
```

**exp1-2 ultrasound_aug: 超音波專用數據增強**

**設計理念**：
- **斑點雜訊（Speckle Noise）**：超音波影像的固有特性，由聲波干涉產生，會降低影像解析度和對比度
- **深度信號衰減（Signal Attenuation）**：模擬超音波在組織中傳播時的深度相關衰減，底部（深層）信號較弱
- 這兩種增強技術模擬真實超音波影像的物理特性，提高模型對實際臨床環境的適應性

**參考文獻**：
1. **Despeckling of Medical Ultrasound Images** (Michailovich & Tannenbaum, 2006)
   - 概述：研究超音波影像中斑點雜訊的統計特性，提出使用乘性模型描述斑點雜訊的形成過程。論文分析了對數轉換後斑點雜訊的特性，並評估了多種非線性濾波器（小波去噪、總變分濾波、各向異性擴散）在去斑點處理中的性能。研究指出，斑點雜訊會降低影像對比度、模糊細節，從而影響診斷價值。
   - 連結：https://pmc.ncbi.nlm.nih.gov/articles/PMC3639001/
   - 關鍵發現：斑點雜訊是超音波影像的固有特性，通過乘性模型可以更好地描述其統計特性；適當的預處理可以將對數轉換後的雜訊轉換為接近白高斯雜訊，從而提高濾波效果。

2. **Speckle Noise Reduction in Ultrasound Images** (Rajabi et al., ISPRS)
   - 概述：評估多種斑點雜訊去除濾波器在超音波影像上的效果與性能。研究比較了不同濾波方法的優缺點，為超音波影像處理提供了實用的參考。
   - 連結：https://www.isprs.org/proceedings/xxxvi/1-W41/makaleler/Rajabi_Specle_Noise.pdf
   - 關鍵發現：不同濾波方法對超音波影像的處理效果各有優劣，需要根據具體應用場景選擇合適的方法。

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp1-2 ultrasound_aug" \
  --use_ultrasound_aug \
  --ultrasound_speckle_var 0.1 \
  --ultrasound_attenuation_factor 0.3
```

**exp2 loss_weights: Loss 權重調整（定位優先）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp2 loss_weights" \
  --box 8.5 \
  --dfl 2.0 \
  --cls 0.6
```

**exp3 focal_loss: Focal Loss（處理類別不平衡）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp3 focal_loss" \
  --use_focal_loss \
  --focal_gamma 1.5 \
  --focal_alpha 0.25
```

**exp4 dim_weights: 水平方向維度權重（HMD 優化）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp4 dim_weights" \
  --use_dim_weights \
  --dim_weights 5.0 1.0 5.0 1.0
```

**exp5-1 hmd_loss_pixel: HMD Loss（像素級別）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp1-2 ultrasound_aug" \
  --use_ultrasound_aug \
  --ultrasound_speckle_var 0.1 \
  --ultrasound_attenuation_factor 0.3
```

**exp5-1 hmd_loss_pixel: HMD Loss（像素級別）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp5-1 hmd_loss_pixel" \
  --use_hmd_loss \
  --hmd_loss_weight 0.5 \
  --hmd_penalty_coeff 0.5
```

**exp5-2 hmd_loss_mm: HMD Loss（毫米級別，真實尺寸）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp5-2 hmd_loss_mm" \
  --use_hmd_loss \
  --hmd_use_mm \
  --hmd_loss_weight 0.5 \
  --hmd_penalty_coeff 0.5
```

**exp6-1 warmup_optimized: Warmup 參數優化（針對超音波小物件）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp6-1 warmup_optimized" \
  --warmup_epochs 5.0 \
  --warmup_momentum 0.9 \
  --warmup_bias_lr 0.05
```

**exp6-2 warmup_cosine_restart: Cosine Annealing with Warm Restarts**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp6-2 warmup_cosine_restart" \
  --use_cosine_restart \
  --cosine_restart_t0 10 \
  --cosine_restart_t_mult 2 \
  --warmup_epochs 5.0
```

**exp7-1 siou: SIoU Loss（對角度敏感，適合細長目標）**

相對於 exp0 的改動：
- `--iou_type`: CIoU → **SIoU**（使用 SIoU Loss）

**設計理念**：
- **SIoU (Scylla IoU)** 考慮了角度成本、距離成本和形狀成本
- **對角度敏感**：通過角度成本項，對細長目標的旋轉角度變化更敏感，適合超音波影像中可能出現的角度偏差
- **適合細長目標**：形狀成本直接優化長寬差異，對 Mentum 和 Hyoid 這類細長結構特別有效
- **距離成本**：考慮中心點距離，對 HMD 計算有幫助

**參考文獻**：[SIoU Loss: More Powerful Learning for Bounding Box Regression](https://arxiv.org/abs/2205.12740)

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp7-1 siou" \
  --iou_type SIoU
```

**exp7-1 siou+keep_top_conf_per_class: SIoU Loss（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp7-1 siou+keep_top_conf_per_class" \
  --iou_type SIoU \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp7-2 eiou: EIoU Loss（直接優化長寬邊長，適合細長目標）**

相對於 exp0 的改動：
- `--iou_type`: CIoU → **EIoU**（使用 EIoU Loss）

**設計理念**：
- **EIoU (Efficient IoU)** 直接優化長寬邊長的真實差異，而非縱橫比
- **適合細長目標**：直接最小化寬度和高度的差異，對 Mentum 和 Hyoid 這類細長結構特別有效
- **解決 CIOU 的模糊定義**：CIoU 使用縱橫比，但相同縱橫比可能對應不同的長寬組合；EIoU 直接優化長寬，更精確
- **中心點距離**：考慮中心點距離，對 HMD 計算有幫助

**參考文獻**：[Focal and Efficient IOU Loss for Accurate Bounding Box Regression](https://arxiv.org/abs/2101.08158)

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp7-2 eiou" \
  --iou_type EIoU
```

**exp7-2 eiou+keep_top_conf_per_class: EIoU Loss（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp7-2 eiou+keep_top_conf_per_class" \
  --iou_type EIoU \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

**exp7-3 diou: DIoU Loss（考慮中心點距離，對 HMD 計算有幫助）**

相對於 exp0 的改動：
- `--iou_type`: CIoU → **DIoU**（使用 DIoU Loss）

**設計理念**：
- **DIoU (Distance IoU)** 考慮重疊面積和中心點距離
- **對 HMD 計算有幫助**：HMD 是 Mentum 和 Hyoid 之間的距離，DIoU 直接優化中心點距離，與 HMD 計算高度相關
- **收斂速度快**：相比 GIoU，DIoU 收斂更快，適合訓練週期較短的場景
- **簡單有效**：相比 CIoU，DIoU 不考慮縱橫比，計算更簡單，但對細長目標仍然有效

**參考文獻**：[Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp7-3 diou" \
  --iou_type DIoU
```

**exp7-3 diou+keep_top_conf_per_class: DIoU Loss（帶 keep_top_conf_per_class 參數）**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=10 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp7-3 diou+keep_top_conf_per_class" \
  --iou_type DIoU \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

#### H200 配置 (Multi-GPU / 多 GPU)

**exp7-1 siou: SIoU Loss**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp7-1 siou" \
  --iou_type SIoU
```

**exp7-2 eiou: EIoU Loss**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp7-2 eiou" \
  --iou_type EIoU
```

**exp7-3 diou: DIoU Loss**

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=10 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-h200" \
  --exp_name="exp7-3 diou" \
  --iou_type DIoU
```

### IoU Loss 方法選擇分析 / IoU Loss Selection Analysis

根據[深度學習筆記：IOU、GIOU、DIOU、CIOU、EIOU、Focal EIOU、alpha IOU、SIOU、WIOU損失函數分析](https://developer.aliyun.com/article/1625721)，邊界框回歸的三大幾何因素為：**重疊面積、中心點距離、縱橫比**。

#### 本專案（det_123 超音波檢測）的特點：
1. **細長目標**：Mentum 和 Hyoid 都是細長結構
2. **HMD 計算**：需要計算兩個目標之間的水平距離，中心點距離很重要
3. **小目標檢測**：目標尺寸較小，需要精確的定位
4. **高噪音環境**：超音波影像具有高噪音特性

#### 適合的 IoU 方法分析：

| IoU 方法 | 考慮因素 | 適合原因 | 實驗編號 |
|---------|---------|---------|---------|
| **SIoU** | 重疊面積 + 角度成本 + 距離成本 + 形狀成本 | **對角度敏感**，適合細長目標；形狀成本直接優化長寬差異 | exp7-1 |
| **EIoU** | 重疊面積 + 中心點距離 + 長寬邊長真實差 | **直接優化長寬邊長**，適合細長目標；解決 CIOU 的模糊定義 | exp7-2 |
| **DIoU** | 重疊面積 + 中心點距離 | **考慮中心點距離**，對 HMD 計算有幫助；收斂速度快 | exp7-3 |
| **CIoU** | 重疊面積 + 中心點距離 + 縱橫比 | 默認方法，作為對照組 | exp0 |
| **GIoU** | 重疊面積 + 最小外接框 | 解決不相交時 loss=0 的問題，但收斂較慢 | - |
| **IoU** | 僅重疊面積 | 最簡單，但不相交時無梯度 | - |

#### 推薦順序（針對本專案）：
1. **SIoU (exp7-1)**：最適合細長目標，對角度敏感，形狀成本直接優化長寬差異
2. **EIoU (exp7-2)**：直接優化長寬邊長，適合細長目標，計算效率高
3. **DIoU (exp7-3)**：考慮中心點距離，對 HMD 計算有幫助，收斂速度快

### HMD Loss 設計說明 / HMD Loss Design

#### 1. HMD (Hyomental Distance) 定義

HMD 是超音波影像中用於評估困難呼吸道的重要指標，計算 Mentum（下頜骨）和 Hyoid（舌骨）兩個解剖結構之間的距離。

**計算公式**：
```python
# 從兩個 bounding box 計算 HMD
def calculate_hmd(mentum_box, hyoid_box):
    # mentum_box 和 hyoid_box 格式: [x1, y1, x2, y2] (像素座標)
    mentum_x1, mentum_y1, mentum_x2, mentum_y2 = mentum_box
    hyoid_x1, hyoid_y1, hyoid_x2, hyoid_y2 = hyoid_box
    
    # X 方向距離：Hyoid 左邊界 - Mentum 右邊界
    hmd_dx = hyoid_x1 - mentum_x2
    
    # Y 方向距離：兩個 box 中心點的 Y 座標差
    mentum_y_center = (mentum_y1 + mentum_y2) / 2
    hyoid_y_center = (hyoid_y1 + hyoid_y2) / 2
    hmd_dy = hyoid_y_center - mentum_y_center
    
    # 歐幾里得距離
    hmd = sqrt(hmd_dx² + hmd_dy²)
    return hmd
```

#### 2. 維度權重 (Dimension Weights) 原理與應用

##### 2.1 維度權重的基本概念

`--dim_weights` 參數允許對 bounding box 的四個邊界（左、上、右、下）應用不同的損失權重，從而控制模型對不同方向定位精度的重視程度。

**參數格式**：
```bash
--use_dim_weights --dim_weights <left> <top> <right> <bottom>
```

**工作原理**：
- 在 DFL (Distribution Focal Loss) 計算中，對每個維度的損失應用對應的權重
- 權重越大，該維度的定位誤差在總損失中的貢獻越大
- 模型會更重視高權重維度的定位精度

**程式碼實作**（`ultralytics/utils/loss.py` 第 184-206 行）：
```python
if self.use_dim_weights:
    # 分別計算每個維度 [l, t, r, b] 的 DFL loss
    loss_dfl_per_dim = []
    for dim_idx in range(4):  # [l, t, r, b]
        dim_loss = self.dfl_loss(...)  # 計算該維度的損失
        dim_loss = dim_loss * self.dim_weights[dim_idx]  # 應用權重
        loss_dfl_per_dim.append(dim_loss)
    # 合併所有維度的損失
    loss_dfl = torch.cat(loss_dfl_per_dim, dim=1)
```

##### 2.2 對 det_123 資料庫的 HMD 計算應用

**HMD 計算的關鍵依賴**：

從 HMD 計算公式可以看出：
1. **水平方向（X）**：`hmd_dx = hyoid_x1 - mentum_x2`
   - 直接依賴於 **Mentum 的右邊界（right）** 和 **Hyoid 的左邊界（left）**
   - 這兩個邊界的定位精度直接影響 HMD 的準確性
   
2. **垂直方向（Y）**：`hmd_dy = hyoid_y_center - mentum_y_center`
   - 依賴於兩個 box 的中心點 Y 座標
   - 中心點 = (top + bottom) / 2，因此上下邊界（top, bottom）的精度也會影響 HMD

**推薦的權重設定**：

對於 `det_123` 資料庫的 HMD 計算，建議使用以下設定：

```bash
--use_dim_weights --dim_weights 5.0 1.0 5.0 1.0
```

**權重解釋**：
- **Left (5.0)**：Hyoid 的左邊界（`hyoid_x1`）權重高，因為直接用於計算 `hmd_dx`
- **Top (1.0)**：上邊界權重較低，因為只間接影響中心點計算
- **Right (5.0)**：Mentum 的右邊界（`mentum_x2`）權重高，因為直接用於計算 `hmd_dx`
- **Bottom (1.0)**：下邊界權重較低，因為只間接影響中心點計算

**為什麼這樣設定？**

1. **水平方向優先**：HMD 的水平分量（`hmd_dx`）直接依賴於左右邊界，因此需要更高的定位精度
2. **垂直方向次要**：HMD 的垂直分量（`hmd_dy`）使用中心點，對上下邊界的精度要求相對較低
3. **權重比例**：5:1 的比例可以讓模型在訓練時更重視水平方向的定位，同時不忽略垂直方向

**其他可能的設定**：

如果垂直方向的定位也很重要，可以考慮：
```bash
--use_dim_weights --dim_weights 5.0 2.0 5.0 2.0
```

這樣可以同時提高水平和垂直方向的定位精度，但可能會降低模型對水平方向的專注度。

#### 3. HMD Loss 設計原理

HMD Loss 是一個輔助損失函數，旨在優化模型對 HMD 距離的預測準確性。它與標準檢測損失（box loss, cls loss, dfl loss）結合使用：

```
總損失 = 標準檢測損失 + λ_hmd × HMD_loss
```

其中：
- `標準檢測損失` = box_loss + cls_loss + dfl_loss
- `λ_hmd` = `--hmd_loss_weight`（預設 0.5）
- `HMD_loss` = 加權平均的 HMD 誤差

#### 4. HMD Loss 計算邏輯

HMD Loss 針對每張影像的三種情況進行處理：

##### 情況 1：兩個目標都檢測到（最佳情況）

當模型同時檢測到 Mentum 和 Hyoid，且 Ground Truth 中也存在這兩個目標時：

**HMD Loss 計算改進**（v0.1.1+）：

1. **Smooth L1 Loss 替代絕對誤差**：
   - 使用 `F.smooth_l1_loss(pred_hmd, gt_hmd)` 替代 `torch.abs(pred_hmd - gt_hmd)`
   - **原因**：Smooth L1 Loss 對異常值更穩健，在小誤差時表現類似 L2（平滑），在大誤差時表現類似 L1（對異常值不敏感）
   - 這對於超音波影像中的異常檢測結果特別重要，可以減少極端錯誤對訓練的影響

2. **Scale-Invariant Loss（相對誤差）**：
   - 計算相對誤差：`relative_error = |pred_hmd - gt_hmd| / (gt_hmd + eps)`
   - **原因**：不同患者的 HMD 範圍可能不同（例如：成人 vs. 兒童），絕對誤差可能無法公平地評估不同尺度的預測
   - 相對誤差確保模型在不同 HMD 範圍下都能得到公平的訓練信號
   - 最終誤差 = `0.7 × Smooth_L1 + 0.3 × relative_error × gt_hmd`

3. **HMD 方向約束**：
   - 添加方向懲罰：`direction_penalty = F.relu(mentum_x2 - hyoid_x1)`
   - **原因**：在正常解剖結構中，Hyoid 應該在 Mentum 的右邊（x 方向：`hyoid_x1 > mentum_x2`）
   - 如果順序錯誤（`mentum_x2 > hyoid_x1`），則施加懲罰
   - 方向懲罰標準化為 HMD 誤差的 10%，確保不會過度影響主要誤差項
   - 這有助於模型學習正確的解剖結構順序，提高預測的臨床合理性

**程式碼實作**（`ultralytics/mycodes/hmd_utils.py` 和 `ultralytics/utils/loss.py`）：
```python
# 1. Smooth L1 Loss
hmd_error_smooth_l1 = F.smooth_l1_loss(pred_hmd, gt_hmd, reduction='none', beta=1.0)

# 2. Scale-invariant loss (relative error)
relative_error = torch.abs(pred_hmd - gt_hmd) / (gt_hmd + eps)
hmd_error = 0.7 * hmd_error_smooth_l1 + 0.3 * relative_error * gt_hmd

# 3. HMD direction constraint
direction_penalty = F.relu(mentum_x2 - hyoid_x1)  # Only penalize if wrong order
direction_penalty_normalized = direction_penalty / (gt_hmd + eps) * 0.1  # 10% weight
hmd_error = hmd_error + direction_penalty_normalized
```

```python
# 計算預測的 HMD 和 Ground Truth 的 HMD
pred_hmd = calculate_hmd(pred_mentum_box, pred_hyoid_box)
gt_hmd = calculate_hmd(gt_mentum_box, gt_hyoid_box)

# HMD 誤差 = |預測 HMD - 真實 HMD|
hmd_error = abs(pred_hmd - gt_hmd)

# 權重 = Mentum 置信度 × Hyoid 置信度
weight = confidence_mentum × confidence_hyoid
```

**程式碼實作**（`ultralytics/utils/loss.py`）：
```python
if has_mentum_pred and has_hyoid_pred and has_mentum_target and has_hyoid_target:
    # 選擇置信度最高的預測框
    mentum_idx = argmax(mentum_confidences)
    hyoid_idx = argmax(hyoid_confidences)
    
    # 計算 HMD
    pred_hmd = self._calculate_hmd_from_boxes(
        pred_boxes_fg[mentum_idx], pred_boxes_fg[hyoid_idx]
    )
    gt_hmd = self._calculate_hmd_from_boxes(
        target_boxes_fg[mentum_target_idx], target_boxes_fg[hyoid_target_idx]
    )
    
    # 誤差和權重
    hmd_error = abs(pred_hmd - gt_hmd)
    weight = pred_conf_fg[mentum_idx, mentum_class] * pred_conf_fg[hyoid_idx, hyoid_class]
```

##### 情況 2：只檢測到一個目標（部分漏檢）

當模型只檢測到 Mentum 或 Hyoid 其中一個時：

```python
# 使用固定懲罰值
hmd_error = penalty_single  # 預設 500.0 像素

# 權重 = min(mentum_conf, hyoid_conf) × penalty_coeff
# 如果只檢測到一個，另一個置信度為 0
weight = min(confidence_mentum, confidence_hyoid) × penalty_coeff
```

**程式碼實作**：
```python
elif (has_mentum_pred or has_hyoid_pred) and (has_mentum_target and has_hyoid_target):
    # 單個檢測：使用懲罰值
    hmd_error = torch.tensor(self.hmd_penalty_single, device=device)  # 500.0
    
    # 獲取已檢測目標的置信度，未檢測的為 0
    mentum_conf = max(mentum_confidences) if has_mentum_pred else 0.0
    hyoid_conf = max(hyoid_confidences) if has_hyoid_pred else 0.0
    
    # 權重 = 較小置信度 × 懲罰係數
    weight = min(mentum_conf, hyoid_conf) * self.hmd_penalty_coeff  # 0.5
```

##### 情況 3：兩個目標都漏檢（最差情況）

當模型完全沒有檢測到 Mentum 和 Hyoid 時：

```python
# 使用最大懲罰值
hmd_error = penalty_none  # 預設 1000.0 像素

# 權重固定為 1.0
weight = 1.0
```

**程式碼實作**：
```python
else:
    # 都漏檢：使用最大懲罰值
    hmd_error = torch.tensor(self.hmd_penalty_none, device=device)  # 1000.0
    weight = torch.tensor(1.0, device=device)
```

#### 5. 批次級別的 HMD Loss 計算

對於一個 batch 中的多張影像，HMD Loss 計算加權平均：

```python
# 對 batch 中每張影像計算 hmd_error 和 weight
hmd_errors = [error_1, error_2, ..., error_N]
weights = [weight_1, weight_2, ..., weight_N]

# 加權平均 HMD Loss
hmd_loss = sum(hmd_errors × weights) / sum(weights)
```

**程式碼實作**：
```python
# 收集所有影像的誤差和權重
hmd_errors_tensor = torch.stack(hmd_errors)
weights_tensor = torch.stack(weights)

# 加權平均
hmd_loss = (hmd_errors_tensor * weights_tensor).sum() / (weights_tensor.sum() + 1e-8)
```

#### 6. 整合到總損失函數

HMD Loss 被加權後添加到 box loss 中：

```python
# 在 v8DetectionLoss.__call__ 中
if self.use_hmd_loss and fg_mask.sum() > 0:
    hmd_loss_value = self._calculate_hmd_loss(...)
    
    # 累積用於記錄（計算 epoch 平均）
    self.hmd_loss_sum += hmd_loss_value
    self.hmd_loss_count += 1
    
    # 添加到 box loss（加權）
    loss[0] = loss[0] + self.hmd_loss_weight * hmd_loss_value
```

#### 7. HMD Loss 計算原理與實現

##### 7.1 核心計算邏輯

HMD Loss 的核心是計算**預測 HMD 與 Ground Truth HMD 的絕對差值**，並將其作為損失函數的一部分：

```python
# 在 ultralytics/ultralytics/utils/loss.py 的 v8DetectionLoss._calculate_hmd_loss 中

# 情況 1：兩個目標都檢測到
if has_mentum_pred and has_hyoid_pred and has_mentum_target and has_hyoid_target:
    # 計算預測 HMD
    pred_hmd = self._calculate_hmd_from_boxes(
        pred_boxes_fg[mentum_idx], pred_boxes_fg[hyoid_idx]
    )
    
    # 計算 Ground Truth HMD
    gt_hmd = self._calculate_hmd_from_boxes(
        target_boxes_fg[mentum_target_idx], target_boxes_fg[hyoid_target_idx]
    )
    
    # HMD 誤差 = |預測 HMD - 真實 HMD|
    hmd_error = torch.abs(pred_hmd - gt_hmd)
    
    # 權重 = Mentum 置信度 × Hyoid 置信度
    weight = pred_conf_fg[mentum_idx, self.mentum_class] * pred_conf_fg[hyoid_idx, self.hyoid_class]
```

**關鍵點**：
- **絕對差值**：使用 `torch.abs(pred_hmd - gt_hmd)` 確保誤差為正值
- **置信度加權**：使用兩個目標的置信度乘積作為權重，高置信度預測對損失貢獻更大
- **像素級計算**：HMD 距離以像素為單位計算，不依賴 DICOM PixelSpacing（訓練階段）

##### 7.2 v8DetectionLoss 類實現位置

HMD Loss 的實現位於 `ultralytics/ultralytics/utils/loss.py` 中的 `v8DetectionLoss` 類：

**類定義**（第 274-293 行）：
```python
class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""
    
    def __init__(
        self, 
        model, 
        use_hmd_loss: Optional[bool] = None,
        hmd_loss_weight: Optional[float] = None,
        hmd_penalty_single: Optional[float] = None,
        hmd_penalty_none: Optional[float] = None,
        hmd_penalty_coeff: Optional[float] = None,
        mentum_class: int = 0,
        hyoid_class: int = 1,
    ):
        # 初始化 HMD loss 參數
        self.use_hmd_loss = use_hmd_loss
        self.hmd_loss_weight = hmd_loss_weight
        # ... 其他參數
```

**損失計算入口**（第 419-494 行）：
```python
def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    # ... 標準檢測損失計算（box, cls, dfl）
    
    # HMD loss calculation (if enabled)
    if self.use_hmd_loss and fg_mask.sum() > 0:
        hmd_loss_value = self._calculate_hmd_loss(
            pred_bboxes, pred_scores, target_bboxes, gt_labels, fg_mask, stride_tensor
        )
        # 累積用於記錄（計算 epoch 平均）
        self.hmd_loss_sum += hmd_loss_value
        self.hmd_loss_count += 1
        # 添加到 box loss（加權）
        loss[0] = loss[0] + self.hmd_loss_weight * hmd_loss_value
```

**HMD Loss 計算方法**（第 536-759 行）：
```python
def _calculate_hmd_loss(
    self,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    fg_mask: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate HMD loss for the batch
    
    返回加權平均的 HMD 誤差（像素單位）
    """
    # 實現細節見上述 6.1 節
```

**HMD 距離計算方法**（第 508-534 行）：
```python
def _calculate_hmd_from_boxes(self, mentum_box: torch.Tensor, hyoid_box: torch.Tensor) -> torch.Tensor:
    """
    Calculate HMD from two bounding boxes in pixel coordinates
    
    計算公式：
    - hmd_dx = hyoid_x1 - mentum_x2
    - hmd_dy = (hyoid_y1 + hyoid_y2) / 2 - (mentum_y1 + mentum_y2) / 2
    - hmd = sqrt(hmd_dx² + hmd_dy²)
    """
    # 優先使用 hmd_utils.calculate_hmd_from_boxes（如果可用）
    if _HMD_UTILS_AVAILABLE:
        return calculate_hmd_from_boxes(mentum_box, hyoid_box)
    
    # 回退到本地實現
    # ...
```

##### 7.3 與 hmd_utils.py 的整合

`v8DetectionLoss` 類會優先使用 `ultralytics/mycodes/hmd_utils.py` 中的函數（如果可用）：

```python
# 在 loss.py 頂部（第 20-31 行）
try:
    _mycodes_path = Path(__file__).parent.parent.parent / "mycodes"
    if _mycodes_path.exists():
        sys.path.insert(0, str(_mycodes_path.parent))
        from mycodes.hmd_utils import calculate_hmd_from_boxes, calculate_hmd_loss
        _HMD_UTILS_AVAILABLE = True
except ImportError:
    _HMD_UTILS_AVAILABLE = False
```

這樣設計的好處：
- **代碼復用**：避免重複實現相同的 HMD 計算邏輯
- **易於維護**：HMD 計算邏輯集中在 `hmd_utils.py` 中
- **向後兼容**：如果 `hmd_utils.py` 不可用，會回退到本地實現

##### 7.4 HMD Loss 梯度傳播機制（關鍵修復）

**⚠️ 重要：Penalty 的梯度問題與修復**

在 HMD Loss 的實現中，有一個**關鍵的梯度傳播問題**，這會導致即使設置了很大的 `hmd_loss_weight`（如 10000），HMD Loss 也可能沒有實際效果。

###### 7.4.1 問題根源：Penalty 沒有梯度

**原始實現的問題**：

當使用 penalty（漏檢懲罰）時，原始代碼使用了常量 tensor：

```python
# ❌ 錯誤：常量 tensor 沒有梯度
hmd_error = torch.tensor(self.hmd_penalty_none, device=device)  # 1000.0
```

**為什麼沒有梯度？**

1. **常量 tensor 的特性**：
   - `torch.tensor(1000.0, device=device)` 創建的是一個**常量值**
   - 這個值不依賴於模型的任何預測輸出
   - 在反向傳播時，梯度無法通過常量傳播到模型參數

2. **梯度傳播鏈斷裂**：
   ```
   模型預測 → HMD Loss 計算 → Penalty（常量）→ 總損失
                    ↑                    ↑
                有梯度              無梯度（斷裂）
   ```
   - 即使 `hmd_loss_weight=10000`，penalty 部分也不會產生梯度
   - 模型無法從 penalty 中學習，HMD Loss 實際上沒有效果

3. **為什麼 HMD 誤差有梯度？**

當兩個目標都檢測到時，HMD 誤差是這樣計算的：

```python
# ✅ 正確：依賴於預測，有梯度
pred_hmd = self._calculate_hmd_from_boxes(
    pred_boxes_fg[mentum_idx], pred_boxes_fg[hyoid_idx]  # 依賴於 pred_bboxes
)
gt_hmd = self._calculate_hmd_from_boxes(
    target_boxes_fg[mentum_target_idx], target_boxes_fg[hyoid_target_idx]  # 常量
)
hmd_error = torch.abs(pred_hmd - gt_hmd)  # 依賴於 pred_hmd，有梯度
```

**關鍵點**：
- `pred_hmd` 是從 `pred_boxes_fg`（模型預測的 bbox）計算出來的
- `pred_boxes_fg` 依賴於模型的輸出，因此有梯度
- `hmd_error = |pred_hmd - gt_hmd|` 依賴於 `pred_hmd`，因此也有梯度
- 梯度可以通過 `pred_hmd` → `pred_boxes_fg` → 模型參數 傳播

###### 7.4.2 修復方案：讓 Penalty 依賴於預測

**修復後的實現**：

讓 penalty 依賴於預測的置信度，確保梯度能正確傳播：

```python
# ✅ 正確：依賴於預測置信度，有梯度
# 情況 1：沒有檢測到任何目標
max_conf = pred_conf[b].max() if pred_conf[b].numel() > 0 else torch.tensor(0.0, device=device)
hmd_error = torch.tensor(self.hmd_penalty_none, device=device) * (1.0 + max_conf)
# max_conf 依賴於 pred_conf，pred_conf 依賴於模型輸出，因此有梯度

# 情況 2：只檢測到一個目標
min_conf = torch.min(mentum_conf, hyoid_conf)
hmd_error = torch.tensor(self.hmd_penalty_single, device=device) * (1.0 + min_conf)
# min_conf 依賴於預測置信度，因此有梯度
```

**為什麼預測置信度有梯度？**

1. **置信度的來源**：
   ```python
   pred_conf = pred_scores.sigmoid()  # pred_scores 是模型的分類輸出
   ```
   - `pred_scores` 是模型的分類頭（classification head）的輸出
   - `pred_scores` 依賴於模型的權重參數，因此有梯度

2. **梯度傳播鏈**：
   ```
   模型參數 → pred_scores → pred_conf → penalty → HMD Loss → 總損失
      ↑           ↑            ↑          ↑          ↑
   有梯度      有梯度       有梯度      有梯度     有梯度
   ```
   - 現在梯度可以完整地從總損失傳播回模型參數

3. **Penalty 值的變化**：
   - 原始：`penalty = 1000.0`（固定值，無梯度）
   - 修復後：`penalty = 1000.0 * (1.0 + max_conf)`
     - 如果 `max_conf = 0.0`（完全沒有預測），`penalty = 1000.0`
     - 如果 `max_conf = 0.5`（中等置信度），`penalty = 1500.0`
     - 如果 `max_conf = 0.9`（高置信度），`penalty = 1900.0`

###### 7.4.3 為什麼高置信度但仍然漏檢應該受到更大懲罰？

**設計理念**：

1. **高置信度但漏檢 = 模型過度自信但錯誤**：
   - 如果模型對某個位置有很高的置信度（如 0.9），但實際上沒有檢測到目標
   - 這意味著模型**過度自信**，認為某個位置有目標，但實際上沒有
   - 這種情況應該受到**更大的懲罰**，因為模型需要學習"不要過度自信"

2. **低置信度但漏檢 = 模型不確定**：
   - 如果模型對所有位置都有很低的置信度（如 0.1），沒有檢測到目標
   - 這意味著模型**不確定**，不知道目標在哪裡
   - 這種情況的懲罰相對較小，因為模型至少知道自己不確定

3. **數學表達**：
   ```
   penalty = base_penalty × (1.0 + max_conf)
   
   情況 A：max_conf = 0.0（完全沒有預測）
   → penalty = 1000.0 × (1.0 + 0.0) = 1000.0
   
   情況 B：max_conf = 0.5（中等置信度但漏檢）
   → penalty = 1000.0 × (1.0 + 0.5) = 1500.0  （+50%）
   
   情況 C：max_conf = 0.9（高置信度但漏檢）
   → penalty = 1000.0 × (1.0 + 0.9) = 1900.0  （+90%）
   ```

4. **訓練效果**：
   - 模型會學習到：**高置信度但漏檢會受到更大懲罰**
   - 這鼓勵模型：
     - 要麼提高檢測率（減少漏檢）
     - 要麼降低不確定的預測的置信度（避免過度自信）

**實際影響**：

- **修復前**：即使 `hmd_loss_weight=10000`，penalty 部分也沒有梯度，HMD Loss 沒有實際效果
- **修復後**：penalty 有梯度，模型可以從 penalty 中學習，HMD Loss 能真正影響訓練

**程式碼位置**：

- 修復實現：`ultralytics/ultralytics/utils/loss.py` 第 671-759 行
- 關鍵修復點：
  - 第 671-677 行：沒有檢測到任何目標時的 penalty
  - 第 733-751 行：只檢測到一個目標時的 penalty
  - 第 753-759 行：兩個目標都漏檢時的 penalty

##### 7.5 為什麼修改代碼後不需要重新安裝？

**重要說明**：如果 ultralytics 包是以**可編輯模式（editable mode）**安裝的，修改代碼後**不需要重新安裝**。

###### 7.5.1 可編輯模式安裝（`pip install -e .`）

**什麼是可編輯模式？**

當使用 `pip install -e .` 安裝包時，Python 會：
1. **創建一個鏈接**（而不是複製文件）到源代碼目錄
2. **直接從源代碼目錄導入模塊**，而不是從 `site-packages`
3. **修改源代碼立即生效**，不需要重新安裝

**安裝方式**（在 `ultralytics` 目錄下）：
```bash
cd ultralytics
pip install -e .
```

**如何確認是否是可編輯模式？**

1. **檢查安裝信息**：
   ```bash
   pip show ultralytics
   ```
   如果看到 `Location: D:\workplace\project_management\github_project\ultrasound-airway-detection2\ultralytics`，說明是可編輯模式。

2. **檢查 Python 導入路徑**：
   ```python
   import ultralytics
   import inspect
   print(inspect.getfile(ultralytics))
   ```
   如果路徑指向項目目錄（而不是 `site-packages`），說明是可編輯模式。

###### 7.5.2 為什麼代碼中還需要強制重新加載模塊？

雖然可編輯模式安裝後修改會立即生效，但在某些情況下，Python 可能已經**緩存了舊版本的模塊**：

1. **模塊已被導入**：如果 `ultralytics.utils.loss` 已經被導入過，Python 會使用緩存版本
2. **多個導入路徑**：如果同時存在已安裝的包和本地修改的包，Python 可能優先使用已安裝的版本

**解決方案**：在 `train_yolo.py` 中，我們添加了強制重新加載模塊的邏輯：

```python
# 確保導入本地修改的版本
local_ultralytics_path = Path(__file__).parent.parent
if str(local_ultralytics_path) not in sys.path:
    sys.path.insert(0, str(local_ultralytics_path))

# 強制重新加載模塊（清除緩存）
if 'ultralytics.utils.loss' in sys.modules:
    importlib.reload(sys.modules['ultralytics.utils.loss'])

from ultralytics.utils.loss import v8DetectionLoss
```

**這樣做的好處**：
- ✅ 確保使用本地修改的版本，而不是已安裝的版本
- ✅ 清除 Python 的模塊緩存，強制重新加載
- ✅ 即使包沒有以可編輯模式安裝，也能正常工作（通過 `sys.path.insert`）

###### 7.5.3 什麼時候需要重新安裝？

**需要重新安裝的情況**：

1. **包沒有以可編輯模式安裝**：
   ```bash
   # 如果之前是這樣安裝的（錯誤）
   pip install ultralytics
   
   # 需要改為可編輯模式（正確）
   cd ultralytics
   pip install -e .
   ```

2. **修改了 `pyproject.toml` 或 `setup.py`**：
   - 添加了新的依賴項
   - 修改了包結構
   - 需要重新安裝以應用這些更改

3. **Python 環境問題**：
   - 切換了 conda/virtualenv 環境
   - 需要在新環境中重新安裝

**不需要重新安裝的情況**：

1. ✅ **只修改了 `.py` 源代碼文件**（如 `loss.py`、`train_yolo.py`）
2. ✅ **包已經以可編輯模式安裝**（`pip install -e .`）
3. ✅ **代碼中已經有強制重新加載邏輯**（如 `train_yolo.py` 中的實現）

###### 7.5.4 如何確認修改是否生效？

**方法 1：檢查導入路徑**（在訓練腳本中添加）：
```python
import ultralytics.utils.loss
import inspect
print(f"loss.py 路徑: {inspect.getfile(ultralytics.utils.loss)}")
# 應該顯示：D:\workplace\project_management\github_project\ultrasound-airway-detection2\ultralytics\ultralytics\utils\loss.py
```

**方法 2：檢查函數簽名**（已在代碼中實現）：
```python
import inspect
sig = inspect.signature(v8DetectionLoss.__init__)
print(f"v8DetectionLoss.__init__ signature: {sig}")
# 應該包含 use_hmd_loss 參數
```

**方法 3：查看訓練日誌**：
- 如果看到 `v8DetectionLoss: HMD Loss enabled - weight=10000.0`，說明修改已生效
- 如果看到 `TypeError: ... got an unexpected keyword argument 'use_hmd_loss'`，說明仍在使用舊版本

##### 7.6 EMA 模型與 Criterion 配置

**重要說明**：在 Ultralytics YOLO 訓練中，驗證階段使用的是 **EMA（Exponential Moving Average，指數移動平均）模型**，而不是訓練模型本身。這意味著任何自定義 loss 配置都必須同時應用到訓練模型和 EMA 模型。

**EMA 模型是什麼？**

EMA 模型是訓練模型的平滑版本，通過對歷史權重進行指數移動平均來維護：

```python
# EMA 更新公式（每次訓練步驟後）
EMA_weight = 0.9999 × EMA_weight + 0.0001 × current_weight
```

**為什麼使用 EMA 模型？**
- **更穩定**：平滑權重變化，減少訓練波動
- **更好的驗證性能**：平滑後的權重在驗證集上通常表現更好
- **減少過擬合**：對訓練噪聲更不敏感

**驗證階段使用 EMA 模型**：

在 `ultralytics/ultralytics/engine/validator.py` 第 151 行：
```python
model = trainer.ema.ema or trainer.model  # 優先使用 EMA 模型
```

**問題：EMA 模型的 Criterion 需要同步配置**

當我們修改 loss 函數（如添加 HMD loss、Focal Loss、Dimension Weights）時，必須確保：
1. **訓練模型的 criterion** 被正確配置
2. **EMA 模型的 criterion** 也被正確配置（因為驗證階段使用 EMA 模型）

**解決方案：`set_custom_loss_callback`**

所有 loss 函數的修改都必須通過 `set_custom_loss_callback` 回調函數來實現，這個回調會在 `on_train_start` 時觸發：

**實現位置**：`ultralytics/mycodes/train_yolo.py` 第 1197-1262 行

```python
def set_custom_loss_callback(trainer):
    """Set dimension weights, focal loss, and HMD loss after trainer initialization"""
    # 1. 設置參數到 trainer.args
    if use_hmd_loss_flag:
        setattr(trainer.args, 'use_hmd_loss', True)
        setattr(trainer.args, 'hmd_loss_weight', hmd_loss_weight_value)
        # ... 其他參數
    
    # 2. 重新創建訓練模型的 criterion
    if updated and hasattr(trainer.model, 'init_criterion'):
        trainer.model.criterion = None
        trainer.model.criterion = trainer.model.init_criterion()
    
    # 3. ⚠️ 關鍵：同時更新 EMA 模型的 criterion
    if hasattr(trainer, 'ema') and trainer.ema is not None:
        if hasattr(trainer.ema.ema, 'init_criterion'):
            trainer.ema.ema.criterion = None
            trainer.ema.ema.criterion = trainer.ema.ema.init_criterion()
        else:
            # 如果 EMA 模型沒有 init_criterion，則從訓練模型複製
            import copy
            trainer.ema.ema.criterion = copy.deepcopy(trainer.model.criterion)
```

**為什麼必須同時更新 EMA 模型的 Criterion？**

1. **驗證階段使用 EMA 模型**：
   - 驗證階段會調用 `model.loss()` 來計算損失
   - 如果 EMA 模型的 criterion 沒有 HMD loss 配置，驗證階段的 loss 計算就不會包含 HMD loss
   - 這會導致啟用和未啟用 `--use_hmd_loss` 的結果相同

2. **HMD Loss 統計信息的獲取**：
   - HMD loss 的統計信息（`hmd_loss_sum`, `hmd_loss_count`）在訓練過程中累積在**訓練模型的 criterion** 中
   - 驗證階段雖然使用 EMA 模型，但 HMD loss 值應該從**訓練模型的 criterion** 中獲取（因為統計信息在那裡）
   - 因此，`on_val_end_callback` 會優先從 `trainer.model.criterion` 獲取 HMD loss 統計信息

**所有 Loss 修改都必須經過這一層**

無論是以下哪種 loss 修改，都必須通過 `set_custom_loss_callback` 來實現：
- ✅ **HMD Loss**：`--use_hmd_loss`, `--hmd_loss_weight` 等
- ✅ **Focal Loss**：`--use_focal_loss`, `--focal_gamma`, `--focal_alpha`
- ✅ **Dimension Weights**：`--use_dim_weights`, `--dim_weights`
- ✅ **Loss 權重調整**：`--box`, `--cls`, `--dfl`

**註冊回調**：

```python
# 在 train_yolo.py 中（第 1261-1262 行）
if use_dim_weights_flag or use_focal_loss_flag or use_hmd_loss_flag:
    model.add_callback("on_train_start", set_custom_loss_callback)
```

**重要提醒**：

⚠️ **如果直接修改 `trainer.model.criterion` 而不通過 `set_custom_loss_callback`，會導致以下問題**：
- EMA 模型的 criterion 沒有被更新
- 驗證階段的 loss 計算不包含自定義 loss
- 自定義 loss 的效果無法在驗證階段體現
- 啟用和未啟用自定義 loss 的結果可能相同

**程式碼位置**：
- `set_custom_loss_callback`：`ultralytics/mycodes/train_yolo.py` 第 1197-1262 行
- EMA 模型使用：`ultralytics/ultralytics/engine/validator.py` 第 151 行
- HMD loss 獲取：`ultralytics/mycodes/train_yolo.py` 第 567-609 行（`on_val_end_callback`）

#### 8. 訓練監控指標

##### 8.0 HMD 指標顯示時間

**顯示時機**：
- **每個 validation epoch 結束後**：HMD 指標會在每個驗證階段（validation）結束後立即顯示
- **適用於所有 det_123 實驗**：無論是否啟用 HMD Loss（`--use_hmd_loss`），只要資料庫是 `det_123`，都會計算並顯示 HMD 指標
- **顯示位置**：終端輸出中，緊接在標準檢測指標（Precision, Recall, mAP50, mAP50-95）之後

**顯示格式**：
```
📊 Additional Metrics:
   Precision: 0.6258 | Recall: 0.5744
   mAP50: 0.5248 | mAP50-95: 0.1559 | Fitness: 0.1928
   HMD_loss: 123.4567  (僅在啟用 --use_hmd_loss 時顯示)

📏 HMD Metrics (det_123):
   Detection_Rate: 0.8500
   RMSE_HMD (pixel): 45.67 px
   Overall_Score (pixel): 0.82
   RMSE_HMD (mm): 3.45 mm  (如果 PixelSpacing 可用)
   Overall_Score (mm): 0.81  (如果 PixelSpacing 可用)
```

**重要說明**：
- **exp0 baseline**：即使未啟用 HMD Loss，也會顯示 HMD 指標（從驗證集的預測結果計算）
- **exp1-exp5**：所有實驗都會顯示 HMD 指標，方便比較不同實驗配置對 HMD 性能的影響
- **HMD_loss 值**：僅在啟用 `--use_hmd_loss` 時顯示，因為它需要從訓練過程中的 HMD Loss 統計中獲取

##### 8.1 指標列表與解釋

在訓練過程中，系統會在**每個 validation epoch 結束後**顯示以下 HMD 相關指標：

**1. HMD_loss（HMD 損失值）**
- **定義**：每個 epoch 的平均 HMD loss（跨所有 batch 的平均值）
- **計算方式**：`hmd_loss_sum / hmd_loss_count`（在 `v8DetectionLoss.get_avg_hmd_loss()` 中計算）
- **單位**：像素（pixels）
- **意義**：
  - 反映模型預測 HMD 與真實 HMD 的平均誤差
  - 值越小表示 HMD 預測越準確
  - 包含三種情況的加權平均：完全檢測、部分檢測、完全漏檢
- **顯示位置**：終端輸出中的 `📊 Additional Metrics` 區塊
- **程式碼位置**：`ultralytics/mycodes/train_yolo.py` 第 71-77 行

**2. Detection_Rate（檢測率）**
- **定義**：同時檢測到 Mentum 和 Hyoid 兩個目標的影像比例
- **計算公式**：`Detection_Rate = (同時檢測到兩個目標的圖片數) / (總圖片數)`
- **範圍**：0.0 到 1.0
- **意義**：
  - 反映模型同時檢測兩個目標的能力
  - 值越接近 1.0 表示模型漏檢率越低
  - 是評估模型完整性的重要指標
- **顯示位置**：終端輸出中的 `📏 HMD Metrics (det_123)` 區塊
- **程式碼位置**：`ultralytics/mycodes/train_yolo.py` 第 95-96 行

**3. RMSE_HMD (pixel)（HMD 均方根誤差）**
- **定義**：HMD 預測的均方根誤差（Root Mean Squared Error）
- **計算公式**：`RMSE_HMD = sqrt(mean((pred_HMD - GT_HMD)²))`
- **單位**：像素（pixels）
- **意義**：
  - 反映 HMD 預測的整體準確性
  - 對大誤差更敏感（因為平方操作）
  - 值越小表示 HMD 預測越準確
  - **注意**：此指標基於 HMD loss 中累積的真實 HMD 誤差計算，而非僅使用懲罰值
- **顯示位置**：終端輸出中的 `📏 HMD Metrics (det_123)` 區塊
- **程式碼位置**：`ultralytics/mycodes/train_yolo.py` 第 97 行

**4. Overall_Score (pixel)（綜合評分）**
- **定義**：綜合評分，結合檢測率和 HMD 誤差
- **計算公式**：`Overall_Score = Detection_Rate / (1 + RMSE_HMD / 1000)`
  - 使用 1000 作為歸一化因子（典型 RMSE 範圍：100-1000 像素）
  - 當 RMSE_HMD = 0 時，Overall_Score = Detection_Rate（完美情況）
- **單位**：無單位（0-1 之間的分數）
- **意義**：
  - 同時考慮檢測完整性和預測準確性
  - **值越大表示整體性能越好**（與 Detection_Rate 和 RMSE_HMD 的改進方向一致）
  - 當 Detection_Rate 高且 RMSE_HMD 低時，Overall_Score 會接近 1.0
  - 當 Detection_Rate 低或 RMSE_HMD 高時，Overall_Score 會相應降低
- **範例**：
  - Detection_Rate = 1.0, RMSE_HMD = 0 → Overall_Score = 1.0（最佳）
  - Detection_Rate = 1.0, RMSE_HMD = 1000 → Overall_Score = 0.5
  - Detection_Rate = 0.5, RMSE_HMD = 0 → Overall_Score = 0.5
  - Detection_Rate = 0.5, RMSE_HMD = 1000 → Overall_Score = 0.25
- **顯示位置**：終端輸出中的 `📏 HMD Metrics (det_123)` 區塊
- **程式碼位置**：`ultralytics/mycodes/train_yolo.py` 第 396、495 行

##### 8.2 指標計算流程

**訓練階段（每個 batch）**：
1. 在 `v8DetectionLoss.__call__` 中計算 HMD loss（僅在啟用 `--use_hmd_loss` 時）
2. 累積 `hmd_loss_sum` 和 `hmd_loss_count`
3. 將加權 HMD loss 添加到總損失中

**驗證階段（每個 epoch 結束後）**：
1. **驗證完成**：模型在驗證集上完成所有 batch 的驗證
2. **觸發回調**：`on_val_end_callback` 被觸發（`ultralytics/mycodes/train_yolo.py` 第 511 行）
3. **計算 HMD 指標**：
   - 如果啟用 HMD Loss：從 `criterion.get_avg_hmd_loss()` 獲取平均 HMD loss，並從 validator stats 計算 Detection_Rate
   - 如果未啟用 HMD Loss：僅從 validator stats 計算 Detection_Rate 和 RMSE_HMD（基於預測與 Ground Truth 的匹配情況）
4. **計算綜合指標**：計算 Overall_Score = Detection_Rate / (1 + RMSE_HMD / 1000)
5. **顯示指標**：調用 `print_validation_metrics` 在終端顯示所有指標（包括 HMD 指標）

**顯示時間點**：
- **即時顯示**：每個 validation epoch 結束後立即顯示，無需等待訓練完成
- **每個 epoch**：訓練過程中的每個 epoch 都會顯示一次 HMD 指標
- **最終評估**：訓練結束後，可以使用 `test_yolo.py` 和 `calculate_hmd_from_yolo.py` 進行更詳細的 HMD 評估

**程式碼位置**（`ultralytics/mycodes/train_yolo.py`）：
```python
# 在 on_val_end_callback 中提取平均 HMD loss（第 392-406 行）
if hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
    criterion = trainer.model.criterion
    if hasattr(criterion, 'get_avg_hmd_loss'):
        hmd_loss_avg = criterion.get_avg_hmd_loss()  # 整個 epoch 的平均值
        additional_metrics["train/hmd_loss"] = hmd_loss_avg

# 在 calculate_hmd_metrics_from_validator 中計算其他指標（第 243-383 行）
hmd_metrics = calculate_hmd_metrics_from_validator(
    validator=validator,
    trainer=trainer,
    penalty_single=getattr(trainer.args, 'hmd_penalty_single', 500.0),
    penalty_none=getattr(trainer.args, 'hmd_penalty_none', 1000.0)
)
# 返回：detection_rate, rmse_pixel, overall_score_pixel
```

##### 8.3 W&B 記錄指標說明

**訓練過程中（每個 epoch）**：通過 `log_train_metrics` 函數記錄到 W&B

**記錄的指標**：

| 指標類別 | 指標名稱 | 說明 | 記錄條件 |
|---------|---------|------|---------|
| **訓練損失** | `train/box_loss` | Box loss（定位損失） | 每個 epoch |
| | `train/cls_loss` | Classification loss（分類損失） | 每個 epoch |
| | `train/dfl_loss` | DFL loss（分布損失） | 每個 epoch |
| | `train/hmd_loss` | HMD loss（HMD 損失） | 僅在啟用 `--use_hmd_loss` 時 |
| **檢測指標** | `metrics/precision` | Precision（精確率） | 每個 epoch |
| | `metrics/recall` | Recall（召回率） | 每個 epoch |
| | `metrics/mAP50` | mAP@0.5 | 每個 epoch |
| | `metrics/mAP50-95` | mAP@0.5:0.95 | 每個 epoch |
| | `metrics/fitness` | Fitness (0.1×mAP50 + 0.9×mAP50-95) | 每個 epoch |
| **HMD 指標** | `hmd/detection_rate` | HMD 檢測率 | 每個 epoch（僅 det_123） |
| | `hmd/rmse_pixel` | HMD RMSE（像素） | 每個 epoch（僅 det_123） |
| | `hmd/overall_score_pixel` | HMD 綜合評分（像素） | 每個 epoch（僅 det_123） |
| | `hmd/rmse_mm` | HMD RMSE（毫米） | 每個 epoch（僅 det_123，需要 PixelSpacing） |
| | `hmd/overall_score_mm` | HMD 綜合評分（毫米） | 每個 epoch（僅 det_123，需要 PixelSpacing） |
| **學習率** | `lr/pg0` | 學習率（參數組 0） | 每個 epoch |
| | `lr/pg1` | 學習率（參數組 1，如果存在） | 每個 epoch |
| **其他** | `epoch` | 當前 epoch 編號 | 每個 epoch |
| | `time` | 訓練經過時間（秒） | 每個 epoch |

**最終評估階段（val & test）**：通過 `evaluate_detailed` 函數記錄到 W&B

**HMD 指標的 mm 版本**（v0.1.1+）：

在評估階段（validation 和 test），除了像素級別的 HMD 指標外，還會自動計算毫米（mm）版本的指標：

- **RMSE_HMD (mm)**：使用 `PixelSpacing` 將像素級別的 RMSE 轉換為毫米
- **Overall_Score (mm)**：基於 mm 版本的 RMSE 計算的綜合分數

**計算方式**（v0.1.1+ 改進為按 patient/image 匹配）：

```python
# 從 Dicom_PixelSpacing_DA.joblib 載入 PixelSpacing 字典
# 字典鍵值為 DICOM base name（例如："0834980_Quick ID_20240509_155005_B"）
pixel_spacing_dict = load_pixel_spacing_dict(dicom_root / "Dicom_PixelSpacing_DA.joblib")

# 從 validator 的 dataset 中提取所有圖片路徑
dataset = validator.dataloader.dataset
for im_file in dataset.im_files:
    # 從圖片檔名提取 DICOM base name
    dicom_base, _ = extract_dicom_info_from_filename(Path(im_file).name)
    
    # 在 pixel_spacing_dict 中匹配對應的 PixelSpacing
    if dicom_base in pixel_spacing_dict:
        image_pixel_spacings.append(pixel_spacing_dict[dicom_base])
    else:
        # 模糊匹配：檢查是否包含或部分匹配
        # ...

# 使用匹配到的 PixelSpacing 的平均值（而非整個字典的平均值）
avg_pixel_spacing = np.mean(image_pixel_spacings)
rmse_mm = rmse_pixel * avg_pixel_spacing

# Overall_Score (mm) 使用 100 作為標準化因子（典型 RMSE 範圍：10-100 mm）
overall_score_mm = detection_rate / (1 + rmse_mm / 100.0)
```

**PixelSpacing 提取邏輯**：

`Dicom_PixelSpacing_DA.joblib` 文件中的值為字典格式，包含多個字段。提取時按以下優先級順序：

1. **`truePixelSpacing`**：真正的計算 PixelSpacing（優先使用，約 0.086-0.192 mm/pixel）
2. **`dcmPixelSpacing`**：DICOM 文件中的原始 PixelSpacing 標籤值
3. **`PixelSpacing`**：通用 PixelSpacing 鍵值
4. **`x`**：X 軸間距（用於 `{'x': 0.1, 'y': 0.1}` 格式）

**注意**：會自動跳過非 PixelSpacing 字段（如 `n_frame`, `n_row`, `n_column`, `n_cm`, `n_pixel`），避免誤提取。

**改進說明**：
- **v0.1.1 之前**：使用整個 `pixel_spacing_dict` 的平均值，可能包含不在當前驗證集中的圖片
- **v0.1.1+**：從 validator 的 dataset 中提取實際使用的圖片路徑，為每張圖片匹配對應的 PixelSpacing，只計算當前驗證集中圖片的 PixelSpacing 平均值
- **匹配策略**：
  1. 精確匹配：直接查找 DICOM base name
  2. 規範化匹配：移除 `.dcm` 擴展名和 pose 信息（`_Neutral`, `_Extended`, `_Ramped`）後進行匹配
  3. 子串匹配：檢查 DICOM base name 是否包含在字典鍵值中或反之
  4. 回退機制：如果無法匹配任何圖片，回退到使用整個字典的平均值

**顯示位置**：
- **終端輸出**：每個 epoch 的 validation 結果中會顯示 mm 版本的指標（如果 PixelSpacing 字典可用）
- **W&B 日誌**：記錄為 `val/hmd/rmse_mm` 和 `val/hmd/overall_score_mm`
- **最終評估**：在 `evaluate_detailed` 函數的結果中也包含 mm 版本指標

**注意事項**：
- 如果 `Dicom_PixelSpacing_DA.joblib` 文件不存在或無法載入，mm 版本的指標將顯示為 `0.0`
- mm 版本的指標僅在評估階段計算，訓練階段的 HMD Loss 仍使用像素級別（除非啟用 `--hmd_use_mm`）

**Val 評估記錄的指標**：

| 指標類別 | 指標名稱 | 說明 |
|---------|---------|------|
| **檢測指標** | `val/mAP50` | Val mAP@0.5 |
| | `val/mAP50-95` | Val mAP@0.5:0.95 |
| | `val/precision` | Val Precision |
| | `val/recall` | Val Recall |
| | `val/fitness` | Val Fitness |
| **HMD 指標** | `val/hmd/detection_rate` | Val HMD 檢測率（僅 det_123） |
| | `val/hmd/rmse_pixel` | Val HMD RMSE（像素，僅 det_123） |
| | `val/hmd/overall_score_pixel` | Val HMD 綜合評分（像素，僅 det_123） |
| | `val/hmd/rmse_mm` | Val HMD RMSE（毫米，僅 det_123，需要 PixelSpacing） |
| | `val/hmd/overall_score_mm` | Val HMD 綜合評分（毫米，僅 det_123，需要 PixelSpacing） |
| **速度指標** | `val/inference_speed(ms)` | 推理速度（毫秒） |
| | `val/preprocess_speed(ms)` | 預處理速度（毫秒） |
| | `val/postprocess_speed(ms)` | 後處理速度（毫秒） |
| | `val/loss_speed(ms)` | Loss 計算速度（毫秒） |
| **其他** | `val/num_classes` | 類別數量 |
| | `val/per_class_metrics` | Per-class 指標表格（W&B Table） |
| | `val/AR100`, `val/AR10`, `val/AR1` | Average Recall 指標（如果可用） |
| | `val/iou` | IoU（如果可用） |
| | `val/dice` | Dice 係數（如果可用） |
| **Summary** | `fitness/val` | Val Fitness（記錄到 summary） |
| | `fitness_val` | Val Fitness（記錄到 summary） |

**Test 評估記錄的指標**：

| 指標類別 | 指標名稱 | 說明 |
|---------|---------|------|
| **檢測指標** | `test/mAP50` | Test mAP@0.5 |
| | `test/mAP50-95` | Test mAP@0.5:0.95 |
| | `test/precision` | Test Precision |
| | `test/recall` | Test Recall |
| | `test/fitness` | Test Fitness |
| **HMD 指標** | `test/hmd/detection_rate` | Test HMD 檢測率（僅 det_123） |
| | `test/hmd/rmse_pixel` | Test HMD RMSE（像素，僅 det_123） |
| | `test/hmd/overall_score_pixel` | Test HMD 綜合評分（像素，僅 det_123） |
| | `test/hmd/rmse_mm` | Test HMD RMSE（毫米，僅 det_123，需要 PixelSpacing） |
| | `test/hmd/overall_score_mm` | Test HMD 綜合評分（毫米，僅 det_123，需要 PixelSpacing） |
| **速度指標** | `test/inference_speed(ms)` | 推理速度（毫秒） |
| | `test/preprocess_speed(ms)` | 預處理速度（毫秒） |
| | `test/postprocess_speed(ms)` | 後處理速度（毫秒） |
| | `test/loss_speed(ms)` | Loss 計算速度（毫秒） |
| **其他** | `test/num_classes` | 類別數量 |
| | `test/per_class_metrics` | Per-class 指標表格（W&B Table） |
| | `test/AR100`, `test/AR10`, `test/AR1` | Average Recall 指標（如果可用） |
| | `test/iou` | IoU（如果可用） |
| | `test/dice` | Dice 係數（如果可用） |
| **Summary** | `fitness/test` | Test Fitness（記錄到 summary） |
| | `fitness_test` | Test Fitness（記錄到 summary） |

**重要說明**：
- **訓練過程指標**：每個 epoch 記錄一次，用於追蹤訓練進度
- **最終評估指標**：訓練結束後記錄一次，使用最佳模型（best.pt）進行評估
- **HMD 指標**：所有 det_123 實驗都會記錄（包括 exp0 baseline），無需啟用 `--use_hmd_loss`
- **Summary 指標**：最終評估的指標會同時記錄到 `wandb.run.summary`，方便在 W&B 界面查看最終結果

##### 8.4 終端輸出範例

訓練時，每個 validation epoch 結束後會看到類似輸出：

```
📊 Additional Metrics:
   Precision: 0.7770 | Recall: 0.7160
   mAP50: 0.7028 | mAP50-95: 0.2495 | Fitness: 0.2948
   HMD_loss: 123.4567  (僅在啟用 --use_hmd_loss 時顯示)

📏 HMD Metrics (det_123):
   Detection_Rate: 0.8500
   RMSE_HMD (pixel): 45.67 px
   Overall_Score (pixel): 0.82
   RMSE_HMD (mm): 3.45 mm  (如果 PixelSpacing 可用)
   Overall_Score (mm): 0.81  (如果 PixelSpacing 可用)
```

**說明**：
- `HMD_loss: 123.4567` 表示該 epoch 的平均 HMD 損失為 123.46 像素
- `Detection_Rate: 0.8500` 表示 85% 的影像同時檢測到兩個目標
- `RMSE_HMD (pixel): 45.67 px` 表示 HMD 預測的均方根誤差為 45.67 像素
- `Overall_Score (pixel): 0.78` 表示綜合評分為 0.78（0.85 / (1 + 45.67 / 1000) ≈ 0.78）
  - 注意：Overall_Score 現在是 0-1 之間的分數，值越大越好

#### 9. 類別映射

HMD Loss 僅適用於 `det_123` 資料庫，類別映射如下：

```python
mentum_class = 0  # det_123: class 0 是 Mentum（下頜骨）
hyoid_class = 1   # det_123: class 1 是 Hyoid（舌骨）
```

**啟用條件檢查**（`ultralytics/mycodes/train_yolo.py`）：
```python
use_hmd_loss_flag = args.use_hmd_loss and args.database == 'det_123'
```

只有當 `--use_hmd_loss` 被指定且 `database == 'det_123'` 時，HMD Loss 才會被啟用。

#### 10. 資料集 HMD 分布分析

根據對 `det_123` 資料集的實際分析，所有 Ground Truth 標註都包含完整的 Mentum 和 Hyoid 兩個目標：

##### det_123.yaml（標準資料集）

| Split | 總圖像數 | 情況1（兩個都有） | 情況2（只有一個） | 情況3（都沒有） |
|-------|---------|-----------------|----------------|---------------|
| train | 74,107 | 74,107 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| val   | 16,074 | 16,074 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| test  | 15,369 | 15,369 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| **總計** | **105,550** | **105,550 (100.00%)** | **0 (0.00%)** | **0 (0.00%)** |

##### det_123_ES.yaml（內視鏡資料集）

| Split | 總圖像數 | 情況1（兩個都有） | 情況2（只有一個） | 情況3（都沒有） |
|-------|---------|-----------------|----------------|---------------|
| train | 54,053 | 54,053 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| val   | 11,532 | 11,532 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| test  | 11,600 | 11,600 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| **總計** | **77,185** | **77,185 (100.00%)** | **0 (0.00%)** | **0 (0.00%)** |

##### 重要發現

1. **完整的標註品質**：所有 Ground Truth 標註都包含 Mentum 和 Hyoid 兩個目標（情況1：100%），沒有部分標註或缺失標註的情況。
2. **資料集品質優良**：標註完整且一致，非常適合訓練 HMD Loss。
3. **訓練階段影響**：
   - 在訓練階段，所有樣本都屬於**情況1**，HMD Loss 會直接計算 `|pred_hmd - gt_hmd|` 的誤差。
   - 情況2和情況3的懲罰機制主要用於處理模型在訓練過程中可能產生的漏檢情況。
4. **驗證/測試階段**：如果模型在驗證或測試時出現漏檢，會觸發情況2或情況3的懲罰機制，幫助模型學習同時檢測兩個目標。

##### 分析工具

可以使用以下命令重新分析資料集分布：

```bash
python ultralytics/mycodes/analyze_hmd_distribution.py --yaml-dir yolo_dataset/det_123/v3
```

#### 10. 參數調優建議

- **`--hmd_loss_weight` (λ_hmd)**：
  - 預設值：`0.5`
  - 建議範圍：`0.2 - 1.0`
  - 過大可能影響標準檢測性能，過小可能無法有效優化 HMD
  
  **詳細分析**：
  
  **當前情況分析**：
  - **Box loss（縮放後）**：
    - Box loss (raw) ≈ 0.5-2.0
    - Box loss (scaled by `box=7.5`) ≈ **3.75-15.0**
  
  - **HMD loss（未加權）**：
    - 兩個都檢測到：HMD error ≈ 50-500 像素
    - 只檢測到一個：penalty = 500.0 像素
    - 都沒檢測到：penalty = 1000.0 像素
    - Batch 級別加權平均後：約 **100-800 像素**
  
  - **當前預設值 0.5 的影響**：
    - 加權後的 HMD loss = 0.5 × (100-800) = **50-400**
    - 對比 Box loss (3.75-15.0)，HMD loss 的影響很大（約為 box loss 的 10-25 倍）
  
  **建議設置**：
  
  如果想要不同的影響程度，可以參考以下設置：
  
  | 權重值 | 影響程度 | 加權後 HMD Loss | 說明 |
  |--------|---------|----------------|------|
  | **0.1** | 中等 | 10-80 | HMD loss 約為 box loss 的 2-5 倍 |
  | **0.2-0.3** | 較大 | 20-240 | HMD loss 約為 box loss 的 5-15 倍，推薦 |
  | **0.5** (預設) | 很大 | 50-400 | HMD loss 約為 box loss 的 10-25 倍，需謹慎 |
  | **1.0** | 極大 | 100-800 | HMD loss 約為 box loss 的 20-50 倍，可能過度優化 |
  
  **推薦設置**：
  - **預設值 0.5**：適合大多數情況，對 HMD 優化有較大影響
  - **0.2-0.3**：如果想要更平衡的優化（HMD 和一般檢測目標）
  - **0.1**：如果想要較小的影響，保持對一般檢測目標的關注
  - **1.0**：僅在 HMD 優化是唯一目標時使用，需謹慎
  
  **注意事項**：
  - 如果權重過大（>0.5），可能：
    - 過度優化 HMD 精度，忽略其他檢測目標
    - 導致訓練不穩定
    - 降低整體檢測性能
  - 建議：
    - 先用預設值 0.5 訓練，觀察效果
    - 如果 HMD 指標仍不夠好，可以嘗試增加到 0.7-1.0
    - 如果整體檢測性能下降，可以降低到 0.2-0.3
    - 監控訓練過程中的 loss 曲線，確保穩定

- **`--hmd_penalty_single`**：
  - **自動計算**：根據 `--imgsz` 自動設定為 `imgsz / 2`（預設 `imgsz=640`，因此預設值為 `320.0` 像素）
  - 可選：如需自訂，可明確指定此參數
  - **設定原則**：此值應設定為影像中可能出現的最大 HMD 距離的一半左右。對於 640×640 影像，自動計算為 320.0 像素

- **`--hmd_penalty_none`**：
  - **自動計算**：根據 `--imgsz` 自動設定為 `imgsz`（預設 `imgsz=640`，因此預設值為 `640.0` 像素）
  - 可選：如需自訂，可明確指定此參數
  - **設定原則**：此值應設定為影像寬度或更大，以確保完全漏檢時有足夠的懲罰。對於 640×640 影像，自動計算為 640.0 像素

- **`--hmd_penalty_coeff`**：
  - 預設值：`0.5`
  - 建議範圍：`0.3 - 0.7`
  - 控制單個檢測情況下的權重衰減

**HMD Loss Parameters / HMD Loss 參數說明**:
- `--use_hmd_loss`: 啟用 HMD loss（必需參數）
- `--hmd_loss_weight`: HMD loss 權重（λ_hmd，預設：0.5）
- `--hmd_penalty_single`: 只檢測到一個目標時的懲罰值（自動計算：`imgsz / 2`，預設 `imgsz=640` 時為 `320.0` 像素）
- `--hmd_penalty_none`: 兩個目標都漏檢時的懲罰值（自動計算：`imgsz`，預設 `imgsz=640` 時為 `640.0` 像素）
- `--hmd_penalty_coeff`: 單個檢測情況下的權重係數（預設：0.5）

**注意**：`--hmd_penalty_single` 和 `--hmd_penalty_none` 會根據 `--imgsz` 自動計算，通常不需要手動指定。如需自訂，可明確指定這些參數。

**Note / 注意**: HMD loss 僅適用於 `det_123` 資料庫。損失函數會自動檢查 `args.database == 'det_123'`，只有在此條件滿足時才會應用 HMD loss。

### Test Example / 測試範例

Quick test with minimal epochs / 快速測試（最少輪數）：

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=128 \
  --epochs=3 \
  --device 0 \
  --seed 42 \
  --wandb \
  --project="test-project" \
  --exp_name="test-exp"
```

### Find Best Epoch / 查找最佳 Epoch

```bash
# For production training / 正式訓練
python ultralytics/mycodes/best_epoch.py detect 1 \
  --run_name="ultrasound-det_123_ES-v3-4090/exp0"

# For test training / 測試訓練
python ultralytics/mycodes/best_epoch.py detect 1 \
  --run_name="yolo11n-det_123-v3-test-exp"
```

---

## 📖 Usage / 使用說明

### Basic Command / 基本命令

```bash
python ultralytics/mycodes/train_yolo.py <model> <database> [options]
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
| `--use_hmd_loss` | - | Enable HMD loss for `det_123` database only |
| `--hmd_loss_weight` | `0.5` | HMD loss weight (λ_hmd) |
| `--hmd_penalty_single` | `imgsz / 2` (default: `320.0` when `imgsz=640`) | Penalty when only one target detected (pixels, auto-calculated from `imgsz`) |
| `--hmd_penalty_none` | `imgsz` (default: `640.0` when `imgsz=640`) | Penalty when both targets missed (pixels, auto-calculated from `imgsz`) |
| `--hmd_penalty_coeff` | `0.5` | Penalty coefficient for single detection |

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

### Download DICOM PixelSpacing Dictionary / 下載 DICOM PixelSpacing 字典

Required for HMD metrics calculation in millimeters. Required for video visualization with HMD metrics.

用於計算毫米級別的 HMD 指標。視頻可視化中顯示 HMD 指標時需要此文件。

```bash
# Create dicom_dataset directory if it doesn't exist
mkdir -p dicom_dataset

# Download Dicom_PixelSpacing_DA.joblib (with progress bar / 顯示進度條)
gdown 11N-QGw_7IdIlA4RpMvpl7LTxoWGm0bZC -O dicom_dataset/Dicom_PixelSpacing_DA.joblib --fuzzy
```

**PixelSpacing link / PixelSpacing 連結：**
- Dicom_PixelSpacing_DA.joblib: https://drive.google.com/file/d/11N-QGw_7IdIlA4RpMvpl7LTxoWGm0bZC/view?usp=sharing

**Note / 注意：**
- This file is required for HMD metrics calculation in millimeters (mm) / 此文件用於計算毫米級別的 HMD 指標
- Used by `train_yolo.py` for validation/test HMD metrics / 由 `train_yolo.py` 用於驗證/測試 HMD 指標
- Used by `visualize_predictions_video.py` for video visualization / 由 `visualize_predictions_video.py` 用於視頻可視化
- The file is a joblib format dictionary, no extraction needed / 文件為 joblib 格式字典，無需解壓

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
- **HMD Distance Calculation / HMD 距離計算**: [ultralytics/evaluate/README_HMD.md](ultralytics/evaluate/README_HMD.md)

---

## 🔬 Model Evaluation & HMD Calculation / 模型評估與 HMD 計算

### Complete Workflow / 完整工作流程

#### Step 1: Train Model / 訓練模型

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --epochs=10 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-4090" \
  --exp_name="exp0" \
  --keep_top_conf_per_class \
  --conf_low 0.1
```

#### Step 2: Test Model on Test Set / 在測試集上測試模型

```bash
python ultralytics/mycodes/test_yolo.py detect "" det_123 \
  --db_version 3 \
  --weights ultralytics/runs/train/ultrasound-det_123_ES-v3-4090/exp0/weights/best.pt \
  --dev cuda:0 \
  --batch_size 4 \
  --output-name test_exp0
```

**Output / 輸出**: `ultralytics/runs/detect/test_exp0/predictions.joblib`

**Note / 注意**: 
- Use `--output-name` to specify custom output folder name (e.g., `test_exp1` instead of `test2`)
- If not specified, uses default format: `test{runs_num}` (e.g., `test`, `test2`, `test3`)
- 使用 `--output-name` 指定自定義輸出資料夾名稱（例如 `test_exp1` 而不是 `test2`）
- 如果不指定，使用默認格式：`test{runs_num}`（例如 `test`、`test2`、`test3`）

#### Step 3: Calculate HMD from Predictions / 從預測結果計算 HMD

**Single Patient / 單個患者**:

```bash
# From project root directory
python evaluate/calculate_hmd_from_yolo.py \
    --case-id det_123 \
    --patient-id 0587648 \
    --pred-joblib ultralytics/runs/detect/test_exp0/predictions.joblib \
    --compare-gt \
    --version v3 \
    --output hmd_comparison_0587648.csv
```

**Note / 注意**: 
- Paths are auto-detected from project root. You can also specify manually:
- 路徑會自動從項目根目錄檢測。也可以手動指定：
- `--yolo-root yolo_dataset` (default: auto-detect)
- `--dicom-root dicom_dataset` (default: auto-detect)

**Batch Processing / 批量處理**:

```bash
# From project root directory
# Only process patients in test.txt (recommended when using --pred-joblib)
python evaluate/calculate_hmd_from_yolo.py \
    --case-id det_123 \
    --batch \
    --test-only \
    --pred-joblib ultralytics/runs/detect/test_exp0/predictions.joblib \
    --compare-gt \
    --version v3 \
    --output hmd_comparison_all.csv
```

**Note / 注意**: 
- Use `--test-only` to only process patients in `test.txt` (recommended when using `--pred-joblib`)
- Without `--test-only`, all patients in `patient_data` will be processed
- 使用 `--test-only` 只處理 `test.txt` 中的患者（使用 `--pred-joblib` 時建議使用）
- 不使用 `--test-only` 時，會處理 `patient_data` 中的所有患者

**Output Columns / 輸出列** (with `--compare-gt`):
- `hmd_pixel`: Predicted pixel distance
- `hmd_mm`: Predicted millimeter distance
- `hmd_pixel_gt`: Ground truth pixel distance
- `hmd_mm_gt`: Ground truth millimeter distance
- `hmd_pixel_diff`: Pixel distance difference (pred - gt)
- `hmd_mm_diff`: Millimeter distance difference (pred - gt)
- `hmd_pixel_abs_diff`: Absolute pixel difference
- `hmd_mm_abs_diff`: Absolute millimeter difference

**Statistics / 統計指標**:
- Mean Error (ME)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

#### Step 4: Generate Video Visualization / 生成視頻可視化

Generate a video showing ground truth and predicted bounding boxes with HMD metrics:

生成視頻顯示 Ground Truth 和預測的邊界框及 HMD 指標：

```bash
# From project root directory
python ultralytics/mycodes/visualize_predictions_video.py \
    ultralytics/runs/train/yolo11n-det_123-v3-exp0-reference-baseline/weights/best.pt \
    --test_txt yolo_dataset/det_123/v3/test_ES.txt \
    --output runs/visualize/predictions_video.mp4 \
    --conf 0.25 \
    --fps 10.0
```

**Parameters / 參數**:
- `model_path`: Path to trained model weights (best.pt) / 訓練好的模型權重路徑
- `--test_txt`: Path to test dataset txt file (default: `yolo_dataset/det_123/v3/test_ES.txt`) / 測試集 txt 文件路徑
- `--output`: Output video path (default: `runs/visualize/predictions_video.mp4`) / 輸出視頻路徑
- `--conf`: Confidence threshold (default: 0.25) / 置信度閾值
- `--fps`: Video FPS (default: 10.0) / 視頻幀率
- `--max_images`: Maximum number of images to process (None for all) / 處理的最大圖像數量（None 表示全部）
- `--pixel_spacing_path`: Path to pixel spacing dictionary (default: `dicom_dataset/Dicom_PixelSpacing_DA.joblib`) / PixelSpacing 字典路徑

**Output / 輸出**: 
- **Video file / 視頻文件**: `runs/visualize/predictions_video.mp4` (default) / 默認路徑
- The script automatically creates the output directory if it doesn't exist / 腳本會自動創建輸出目錄（如果不存在）

**Video Features / 視頻特性**:
- **Ground Truth Boxes / Ground Truth 邊界框**:
  - Mentum (GT): Green box / 綠色框
  - Hyoid (GT): Yellow box / 黃色框
- **Predicted Boxes / 預測邊界框**:
  - Mentum (Pred): Orange box with confidence score / 橙色框（帶置信度）
  - Hyoid (Pred): Magenta box with confidence score / 洋紅色框（帶置信度）
- **HMD Visualization / HMD 可視化**:
  - GT HMD line: Green line / 綠色線
  - Pred HMD line: Blue line / 藍色線
  - HMD values displayed in pixels and millimeters / HMD 值以像素和毫米顯示
- **Text Overlay / 文字疊加**:
  - HMD error (pixel and mm) / HMD 誤差（像素和毫米）
  - Class labels (Mentum, Hyoid) / 類別標籤
  - Confidence scores for predictions / 預測的置信度分數

**Note / 注意**:
- Each frame shows at most one bounding box per class (highest confidence for predictions, first box for GT) / 每個幀每個類別最多顯示一個邊界框（預測使用最高置信度，GT 使用第一個框）
- The script automatically matches pixel spacing from DICOM metadata / 腳本會自動從 DICOM 元數據匹配像素間距
- If pixel spacing is not found, only pixel-based HMD is displayed / 如果找不到像素間距，僅顯示基於像素的 HMD

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
