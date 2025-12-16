# 小目标与背景噪音 Loss 参数调整指南

## 概述

针对超音波小目标检测任务，这些参数属于**超参数调整**，可以通过 `train_yolo.py` 直接调整。

## 当前 train_yolo.py 支持的超参数

### ✅ 已支持（可直接使用）

1. **Loss 权重调整**：
   - `--box`: Box loss gain（默认 7.5）
   - `--cls`: Class loss gain（默认 0.5）
   - `--dfl`: DFL loss gain（默认 1.5）

2. **Focal Loss**（新增）：
   - `--use_focal_loss`: 启用 Focal Loss
   - `--focal_gamma`: Focal Loss gamma 参数（默认 1.5）
   - `--focal_alpha`: Focal Loss alpha 参数（默认 0.25）

3. **维度权重**（自定义功能）：
   - `--use_dim_weights`: 启用维度权重
   - `--dim_weights`: 设置维度权重 [left, top, right, bottom]

## 针对小目标与背景噪音的参数建议

### 1. Loss 权重调整（box, cls, dfl）

#### 典型值（YOLOv5～YOLOv11）

| Loss 名 | 典型权重 | 说明 |
|---------|---------|------|
| box | 7.5 | 定位比分類更重要 |
| cls | 0.5 | 避免分類過強干擾 |
| dfl | 1.5 | 細緻邊界調整 |

#### 针对小目标 + 背景模糊（超音波）的调整

**推荐配置**：
```bash
--box=10.0      # 提高定位精度（8-12）
--cls=0.7       # 适度提高分类权重（0.7-1.0）
--dfl=2.0       # 提高边界回归精度（1.5-2.0）
```

**理由**：
- **提高 box 权重**：小目标定位困难，需要更强调定位精度
- **适度提高 cls 权重**：背景噪音多，需要更好的分类能力
- **提高 dfl 权重**：小目标的边界需要更精确的回归

### 2. Focal Loss（推荐用于小目标）

#### 为什么使用 Focal Loss？

- ✅ **自动关注难分类样本**：小目标通常是难样本
- ✅ **减少简单样本权重**：背景区域权重降低
- ✅ **处理类别不平衡**：超音波中目标 vs 背景不平衡

#### 使用方法

```bash
--use_focal_loss \
--focal_gamma=2.0 \
--focal_alpha=0.25
```

**参数说明**：
- `focal_gamma`: 控制难样本的关注程度
  - `1.0-1.5`: 轻度关注难样本
  - `1.5-2.0`: **推荐值**，适合小目标
  - `2.0-2.5`: 强烈关注难样本（可能过于激进）
- `focal_alpha`: 类别平衡参数
  - `0.25`: 默认值，适合大多数情况

#### 完整示例

```bash
python mycodes/train_yolo.py yolo11n det_123 --db_version=3 --es \
  --box=10.0 \
  --cls=0.7 \
  --dfl=2.0 \
  --use_focal_loss \
  --focal_gamma=2.0 \
  --focal_alpha=0.25 \
  # ... 其他参数
```

### 3. 维度权重（针对特定方向）

**针对 det_123（w 和 x 重要）**：
```bash
--use_dim_weights \
--dim_weights 2.0 1.0 2.0 1.0
```

**针对 det_456（h 和 y 重要）**：
```bash
--use_dim_weights \
--dim_weights 1.0 2.0 1.0 2.0
```

## 组合策略

### 策略 1: 保守调整（推荐先试）

```bash
--box=10.0 \
--cls=0.7 \
--dfl=2.0 \
--use_focal_loss \
--focal_gamma=1.5 \
--focal_alpha=0.25
```

**特点**：
- 适度提高所有权重
- Focal Loss 使用中等 gamma
- 风险较低，适合初次尝试

### 策略 2: 激进调整（小目标特别困难时）

```bash
--box=12.0 \
--cls=1.0 \
--dfl=2.5 \
--use_focal_loss \
--focal_gamma=2.0 \
--focal_alpha=0.25 \
--use_dim_weights \
--dim_weights 2.0 1.0 2.0 1.0  # 根据任务调整
```

**特点**：
- 大幅提高定位权重
- Focal Loss 强烈关注难样本
- 结合维度权重
- 可能过拟合，需要更多数据

### 策略 3: 平衡调整（推荐）

```bash
--box=9.0 \
--cls=0.6 \
--dfl=1.8 \
--use_focal_loss \
--focal_gamma=1.8 \
--focal_alpha=0.25
```

**特点**：
- 平衡各项权重
- 适合大多数小目标检测任务

## 实验建议

### 实验顺序

1. **Baseline**：使用默认参数
   ```bash
   --box=7.5 --cls=0.5 --dfl=1.5
   ```

2. **调整 Loss 权重**：
   ```bash
   --box=10.0 --cls=0.7 --dfl=2.0
   ```

3. **添加 Focal Loss**：
   ```bash
   --box=10.0 --cls=0.7 --dfl=2.0 \
   --use_focal_loss --focal_gamma=2.0
   ```

4. **组合所有优化**：
   ```bash
   --box=10.0 --cls=0.7 --dfl=2.0 \
   --use_focal_loss --focal_gamma=2.0 \
   --use_dim_weights --dim_weights ...
   ```

### 评估指标

对比以下指标：
- **mAP50**: 整体检测精度
- **mAP50-95**: 精确定位精度
- **小目标召回率**: 特别关注
- **背景误检率**: 背景噪音处理能力

## 参数对比表

| 参数 | 默认值 | 小目标推荐 | 说明 |
|------|--------|-----------|------|
| `--box` | 7.5 | 8-12 | 定位精度权重 |
| `--cls` | 0.5 | 0.7-1.0 | 分类精度权重 |
| `--dfl` | 1.5 | 1.5-2.0 | 边界回归权重 |
| `--use_focal_loss` | False | True | 启用 Focal Loss |
| `--focal_gamma` | 1.5 | 1.5-2.0 | 难样本关注度 |
| `--focal_alpha` | 0.25 | 0.25 | 类别平衡参数 |

## 注意事项

1. **不要同时大幅调整所有参数**：可能导致训练不稳定
2. **逐步调整**：一次调整一个参数，观察效果
3. **记录实验**：使用 W&B 记录所有实验，便于对比
4. **验证集监控**：关注验证集性能，避免过拟合
5. **早停机制**：使用 `--patience` 避免过训练

## 完整命令示例

### 针对小目标 + 背景噪音的完整配置

```bash
python mycodes/train_yolo.py yolo11n det_123 --db_version=3 --es \
  --epochs=45 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-small-obj" \
  --exp_name="exp-small-obj-optimized" \
  --batch=6 \
  --imgsz=1280 \
  --lr0=0.0005 \
  --lrf=0.005 \
  --optimizer="AdamW" \
  --box=10.0 \
  --cls=0.7 \
  --dfl=2.0 \
  --use_focal_loss \
  --focal_gamma=2.0 \
  --focal_alpha=0.25 \
  --conf=0.3 \
  --iou=0.85 \
  --max_det=50 \
  --agnostic_nms \
  --cos_lr \
  --mosaic=1.0 \
  --mixup=0.2 \
  --hsv_h=0 \
  --hsv_s=1.0 \
  --hsv_v=0.6 \
  --scale=1.0 \
  --translate=0.2 \
  --degrees=0.0 \
  --shear=0.0 \
  --perspective=0.0 \
  --fliplr=0.5 \
  --copy_paste=0.2 \
  --close_mosaic=8 \
  --patience=3
```



