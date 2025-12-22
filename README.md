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

#### 實驗設計 / Experiment Design

本專案使用 `ultrasound-det_123_ES-v3` 作為實驗專案名稱，包含以下實驗：

- **實驗 0 (exp0)**: 基準實驗，不使用 HMD Loss
- **實驗 1 (exp1)**: 使用 HMD Loss 進行訓練

所有實驗均使用 `--seed 42` 確保可重現性。

#### RTX 4070 配置 (Single GPU / 單 GPU)

**實驗 0: 基準訓練 (Baseline Training / 不使用 HMD Loss)**:

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=15 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="exp0"
```

**實驗 1: 使用 HMD Loss (With HMD Loss)**:

**Simplified / 簡化版** (using default values / 使用預設值):

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=15 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="exp1" \
  --use_hmd_loss
```

**Full Command / 完整命令** (with all parameters / 包含所有參數):

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=16 \
  --epochs=15 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="exp1" \
  --use_hmd_loss \
  --hmd_loss_weight 0.1 \
  --hmd_penalty_single 500.0 \
  --hmd_penalty_none 1000.0 \
  --hmd_penalty_coeff 0.5
```

#### H200 配置 (Multi-GPU / 多 GPU)

**實驗 0: 基準訓練 (Baseline Training / 不使用 HMD Loss)**:

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=15 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="exp0"
```

**實驗 1: 使用 HMD Loss (With HMD Loss)**:

**Simplified / 簡化版** (using default values / 使用預設值):

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=15 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="exp1" \
  --use_hmd_loss
```

**Full Command / 完整命令** (with all parameters / 包含所有參數):

```bash
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --batch=256 \
  --epochs=15 \
  --device 0,1 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="exp1" \
  --use_hmd_loss \
  --hmd_loss_weight 0.1 \
  --hmd_penalty_single 500.0 \
  --hmd_penalty_none 1000.0 \
  --hmd_penalty_coeff 0.5
```

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

#### 2. HMD Loss 設計原理

HMD Loss 是一個輔助損失函數，旨在優化模型對 HMD 距離的預測準確性。它與標準檢測損失（box loss, cls loss, dfl loss）結合使用：

```
總損失 = 標準檢測損失 + λ_hmd × HMD_loss
```

其中：
- `標準檢測損失` = box_loss + cls_loss + dfl_loss
- `λ_hmd` = `--hmd_loss_weight`（預設 0.1）
- `HMD_loss` = 加權平均的 HMD 誤差

#### 3. HMD Loss 計算邏輯

HMD Loss 針對每張影像的三種情況進行處理：

##### 情況 1：兩個目標都檢測到（最佳情況）

當模型同時檢測到 Mentum 和 Hyoid，且 Ground Truth 中也存在這兩個目標時：

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

#### 4. 批次級別的 HMD Loss 計算

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

#### 5. 整合到總損失函數

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

#### 6. HMD Loss 計算原理與實現

##### 6.1 核心計算邏輯

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

##### 6.2 v8DetectionLoss 類實現位置

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

##### 6.3 與 hmd_utils.py 的整合

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

#### 7. 訓練監控指標

在訓練過程中，系統會在**每個 validation epoch 結束後**顯示以下 HMD 相關指標：

##### 7.1 指標列表與解釋

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
- **計算公式**：`Overall_Score = Detection_Rate × RMSE_HMD`
- **單位**：像素（pixels）
- **意義**：
  - 同時考慮檢測完整性和預測準確性
  - 值越小表示整體性能越好
  - 當 Detection_Rate 接近 1.0 時，Overall_Score 主要反映 RMSE_HMD
  - 當 Detection_Rate 較低時，Overall_Score 會相應降低，反映漏檢的影響
- **顯示位置**：終端輸出中的 `📏 HMD Metrics (det_123)` 區塊
- **程式碼位置**：`ultralytics/mycodes/train_yolo.py` 第 98 行

##### 7.2 指標計算流程

**訓練階段（每個 batch）**：
1. 在 `v8DetectionLoss.__call__` 中計算 HMD loss
2. 累積 `hmd_loss_sum` 和 `hmd_loss_count`
3. 將加權 HMD loss 添加到總損失中

**驗證階段（每個 epoch 結束後）**：
1. `on_val_end_callback` 被觸發（`ultralytics/mycodes/train_yolo.py` 第 386 行）
2. 從 `criterion.get_avg_hmd_loss()` 獲取平均 HMD loss
3. 從 validator stats 計算 Detection_Rate
4. 使用 HMD loss 統計計算 RMSE_HMD（基於真實 HMD 誤差）
5. 計算 Overall_Score
6. 調用 `print_validation_metrics` 顯示所有指標

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

##### 7.3 終端輸出範例

訓練時，每個 validation epoch 結束後會看到類似輸出：

```
📊 Additional Metrics:
   Precision: 0.7770 | Recall: 0.7160
   mAP50: 0.7028 | mAP50-95: 0.2495 | Fitness: 0.2948
   HMD_loss: 123.4567

📏 HMD Metrics (det_123):
   Detection_Rate: 0.8500
   RMSE_HMD (pixel): 45.67 px
   Overall_Score (pixel): 38.82
```

**說明**：
- `HMD_loss: 123.4567` 表示該 epoch 的平均 HMD 損失為 123.46 像素
- `Detection_Rate: 0.8500` 表示 85% 的影像同時檢測到兩個目標
- `RMSE_HMD (pixel): 45.67 px` 表示 HMD 預測的均方根誤差為 45.67 像素
- `Overall_Score (pixel): 38.82` 表示綜合評分為 38.82（0.85 × 45.67 ≈ 38.82）

#### 8. 類別映射

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

#### 9. 資料集 HMD 分布分析

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
  - 預設值：`0.1`
  - 建議範圍：`0.05 - 0.2`
  - 過大可能影響標準檢測性能，過小可能無法有效優化 HMD

- **`--hmd_penalty_single`**：
  - 預設值：`500.0` 像素
  - 建議範圍：`300.0 - 800.0`
  - 應根據影像解析度調整（640×640 影像建議 500.0）
  - **設定原則**：此值應設定為影像中可能出現的最大 HMD 距離的一半左右。對於 640×640 影像，影像對角線長度為 √(640² + 640²) ≈ 905 像素，因此 `penalty_single` 設定為 500.0 像素是合理的（約為對角線長度的 55%）

- **`--hmd_penalty_none`**：
  - 預設值：`1000.0` 像素
  - 建議範圍：`800.0 - 1500.0`
  - 應大於 `penalty_single`，通常為其 2 倍
  - **設定原則**：此值應設定為影像對角線長度或更大，以確保完全漏檢時有足夠的懲罰。對於 640×640 影像，影像對角線長度為 √(640² + 640²) ≈ 905 像素，因此 `penalty_none` 設定為 1000.0 像素是合理的（略大於對角線長度，確保懲罰足夠）

- **`--hmd_penalty_coeff`**：
  - 預設值：`0.5`
  - 建議範圍：`0.3 - 0.7`
  - 控制單個檢測情況下的權重衰減

**HMD Loss Parameters / HMD Loss 參數說明**:
- `--use_hmd_loss`: 啟用 HMD loss（必需參數）
- `--hmd_loss_weight`: HMD loss 權重（λ_hmd，預設：0.1）
- `--hmd_penalty_single`: 只檢測到一個目標時的懲罰值（預設：500.0 像素）
- `--hmd_penalty_none`: 兩個目標都漏檢時的懲罰值（預設：1000.0 像素）
- `--hmd_penalty_coeff`: 單個檢測情況下的權重係數（預設：0.5）

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
  --run_name="ultrasound-det_123_ES-v3/exp0"

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
| `--hmd_loss_weight` | `0.1` | HMD loss weight (λ_hmd) |
| `--hmd_penalty_single` | `500.0` | Penalty when only one target detected (pixels) |
| `--hmd_penalty_none` | `1000.0` | Penalty when both targets missed (pixels) |
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
  --epochs=15 \
  --seed 42 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="exp0"
```

#### Step 2: Test Model on Test Set / 在測試集上測試模型

```bash
python ultralytics/mycodes/test_yolo.py detect "" det_123 \
  --db_version 3 \
  --weights ultralytics/runs/train/ultrasound-det_123_ES-v3/exp0/weights/best.pt \
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
