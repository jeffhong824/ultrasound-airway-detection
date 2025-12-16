# 维度权重 Loss Function 使用说明

## 概述

为了适应超音波小物件检测的特殊需求，我们在 `BboxLoss` 中添加了维度权重功能。这个功能允许你针对不同的检测任务（如 det_123 和 det_456）设置不同的维度权重。

## 功能说明

### 维度权重含义

- **det_123**: w（宽度）和 x（水平位置）重要
  - 权重设置：`[2.0, 1.0, 2.0, 1.0]` (对应 [left, top, right, bottom])
  - 这意味着左右距离（l, r）的 loss 权重更高

- **det_456**: h（高度）和 y（垂直位置）重要
  - 权重设置：`[1.0, 2.0, 1.0, 2.0]` (对应 [left, top, right, bottom])
  - 这意味着上下距离（t, b）的 loss 权重更高

### 开关控制

- **`use_dim_weights`**: 布尔值，控制是否启用维度权重
  - `False` (默认): 使用原始实现，所有维度权重相等
  - `True`: 启用维度权重功能

## 使用方法

### 方法 1: 通过训练脚本参数（推荐）

在训练脚本中，可以通过设置 `model.args` 来启用维度权重：

```python
from ultralytics import YOLO

# 创建模型
model = YOLO('yolo11n.pt')

# 设置维度权重参数
model.args.use_dim_weights = True
model.args.dim_weights = [2.0, 1.0, 2.0, 1.0]  # det_123: w 和 x 重要

# 或者
# model.args.dim_weights = [1.0, 2.0, 1.0, 2.0]  # det_456: h 和 y 重要

# 开始训练
model.train(data='your_dataset.yaml', epochs=100)
```

### 方法 2: 通过配置文件

在训练配置文件中添加：

```yaml
# 在你的训练配置中
use_dim_weights: True
dim_weights: [2.0, 1.0, 2.0, 1.0]  # [left, top, right, bottom]
```

### 方法 3: 直接传递参数（高级用法）

如果需要直接创建 loss 对象：

```python
from ultralytics.utils.loss import v8DetectionLoss

# 创建 loss 对象
loss_fn = v8DetectionLoss(
    model=your_model,
    use_dim_weights=True,
    dim_weights=[2.0, 1.0, 2.0, 1.0]  # det_123
)
```

## 实验建议

### det_123 实验（w 和 x 重要）

```python
model.args.use_dim_weights = True
model.args.dim_weights = [2.0, 1.0, 2.0, 1.0]  # 强调左右距离
```

### det_456 实验（h 和 y 重要）

```python
model.args.use_dim_weights = True
model.args.dim_weights = [1.0, 2.0, 1.0, 2.0]  # 强调上下距离
```

### 关闭维度权重（对比实验）

```python
model.args.use_dim_weights = False  # 或直接不设置
```

## 代码位置

- **主要修改文件**: `ultralytics/ultralytics/utils/loss.py`
- **修改的类**: 
  - `BboxLoss`: 添加了维度权重支持
  - `v8DetectionLoss`: 添加了参数传递支持

## 注意事项

1. **默认行为**: 如果不设置 `use_dim_weights` 或设置为 `False`，代码行为与原始实现完全相同
2. **权重格式**: `dim_weights` 必须是包含 4 个浮点数的列表，对应 [left, top, right, bottom]
3. **实验对比**: 建议同时运行开启和关闭维度权重的实验，对比效果
4. **权重值**: 建议从 1.0-2.0 之间开始实验，根据效果调整

## 技术细节

- 维度权重只影响 **DFL (Distribution Focal Loss)** 的计算
- IoU loss 保持不变
- 不影响模型架构，只改变 loss 计算方式
- 权重通过 `register_buffer` 注册，会自动移动到正确的设备（CPU/GPU）



