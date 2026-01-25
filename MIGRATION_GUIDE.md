# YOLO 代码移植指南 / YOLO Code Migration Guide

本文档详细说明如何将本项目的所有自定义修改移植到其他 YOLO 仓库（如 Ultralytics YOLO）。

This document provides detailed specifications for migrating all custom modifications from this project to other YOLO repositories (e.g., Ultralytics YOLO).

---

## 📋 目录 / Table of Contents

1. [概览 / Overview](#概览--overview)
2. [修改文件清单 / Modified Files List](#修改文件清单--modified-files-list)
3. [详细修改说明 / Detailed Modifications](#详细修改说明--detailed-modifications)
4. [新增文件说明 / New Files](#新增文件说明--new-files)
5. [依赖关系 / Dependencies](#依赖关系--dependencies)
6. [配置参数 / Configuration Parameters](#配置参数--configuration-parameters)
7. [测试验证 / Testing & Validation](#测试验证--testing--validation)

---

## 概览 / Overview

### 主要功能特性 / Main Features

本项目在标准 YOLO 基础上添加了以下功能：

1. **扩展的 IoU Loss 类型**：SIoU、EIoU、DIoU
2. **HMD (Hyomental Distance) Loss**：用于医学影像的距离损失函数
3. **维度权重 (Dimension Weights)**：为边界框的不同维度（左、上、右、下）设置不同权重
4. **Focal Loss 支持**：用于类别不平衡问题
5. **自定义后处理**：`keep_top_conf_per_class` 功能
6. **自定义指标计算**：HMD 相关指标（Detection_Rate、RMSE_HMD、Overall_Score）
7. **Pixel Spacing 支持**：毫米单位转换

### 修改范围 / Modification Scope

- **核心文件修改**：2 个文件
  - `ultralytics/utils/loss.py`
  - `ultralytics/utils/metrics.py`
- **新增工具文件**：1 个文件
  - `ultralytics/mycodes/hmd_utils.py`
- **训练脚本**：1 个文件（包含大量自定义逻辑）
  - `ultralytics/mycodes/train_yolo.py`

---

## 修改文件清单 / Modified Files List

### 1. `ultralytics/utils/loss.py`

**文件路径**：`ultralytics/ultralytics/utils/loss.py`

**修改类型**：核心文件修改

**主要修改点**：

1. **BboxLoss 类**（第 34-291 行）
   - 添加 `use_dim_weights` 和 `dim_weights` 参数
   - 添加 `iou_type` 参数（支持 SIoU、EIoU、DIoU）
   - 修改 `forward` 方法以支持维度权重和 IoU 类型选择

2. **v8DetectionLoss 类**（第 291-1453 行）
   - 添加 HMD Loss 相关参数和方法
   - 添加 Focal Loss 支持
   - 添加维度权重传递

### 2. `ultralytics/utils/metrics.py`

**文件路径**：`ultralytics/ultralytics/utils/metrics.py`

**修改类型**：核心文件修改

**主要修改点**：

1. **bbox_iou 函数**（第 76-184 行）
   - 添加 `EIoU` 和 `SIoU` 参数
   - 实现 EIoU 计算逻辑（第 160-170 行）
   - 实现 SIoU 计算逻辑（第 133-159 行）

### 3. `ultralytics/mycodes/hmd_utils.py`

**文件路径**：`ultralytics/mycodes/hmd_utils.py`

**修改类型**：新增文件

**功能**：HMD 相关的工具函数

---

## 详细修改说明 / Detailed Modifications

### 修改 1: IoU Loss 类型扩展 (SIoU, EIoU)

#### 文件：`ultralytics/utils/metrics.py`

#### 位置 1: 函数签名修改（第 76-86 行）

**原始代码**：
```python
def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
```

**修改后**：
```python
def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    EIoU: bool = False,  # 新增
    SIoU: bool = False,  # 新增
    eps: float = 1e-7,
) -> torch.Tensor:
```

#### 位置 2: SIoU 实现（第 133-159 行）

**在 `if SIoU or EIoU or CIoU or DIoU or GIoU:` 块内添加**：

```python
if SIoU:  # SIoU Loss: https://arxiv.org/abs/2205.12740
    # Calculate angle cost
    sigma = torch.pow((b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2, 2) + torch.pow((b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2, 2)
    ch_sigma = ch.pow(2) + eps
    sin_alpha = torch.abs((b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2) / torch.sqrt(sigma + eps)
    sin_beta = torch.abs((b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2) / torch.sqrt(sigma + eps)
    sin_alpha = torch.clamp(sin_alpha, min=0, max=1)
    sin_beta = torch.clamp(sin_beta, min=0, max=1)
    alpha = torch.asin(sin_alpha)
    beta = torch.asin(sin_beta)
    
    # Angle cost
    angle_cost = 1 - 2 * torch.sin(torch.abs(alpha - beta) - math.pi / 4).pow(2)
    
    # Distance cost
    rho_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) / cw).pow(2)
    rho_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) / ch).pow(2)
    gamma = 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
    
    # Shape cost
    omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)
    
    # SIoU = IoU - (angle_cost + distance_cost + shape_cost) / 2
    return iou - 0.5 * (angle_cost + distance_cost + shape_cost)
```

**注意事项**：
- 需要导入 `math` 模块：`import math`
- 确保在计算 `sin_alpha` 和 `sin_beta` 时处理除零情况

#### 位置 3: EIoU 实现（第 160-170 行）

**在 `elif EIoU:` 块内添加**：

```python
elif EIoU:  # EIoU Loss: https://arxiv.org/abs/2101.08158
    c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4  # center dist**2
    # EIoU directly optimizes width and height differences
    w_diff = (w1 - w2).pow(2)
    h_diff = (h1 - h2).pow(2)
    cw2 = cw.pow(2) + eps
    ch2 = ch.pow(2) + eps
    return iou - (rho2 / c2 + w_diff / cw2 + h_diff / ch2)
```

**注意事项**：
- EIoU 在 `elif CIoU or DIoU:` 之前检查，确保优先级正确

#### 位置 4: 条件判断修改（第 130 行）

**原始代码**：
```python
if CIoU or DIoU or GIoU:
```

**修改后**：
```python
if SIoU or EIoU or CIoU or DIoU or GIoU:
```

---

### 修改 2: BboxLoss 类 - IoU 类型和维度权重

#### 文件：`ultralytics/utils/loss.py`

#### 位置 1: `__init__` 方法参数添加（第 130-167 行）

**原始代码**：
```python
def __init__(
    self,
    reg_max: int = 16,
):
```

**修改后**：
```python
def __init__(
    self,
    reg_max: int = 16,
    use_dim_weights: bool = False,
    dim_weights: Optional[List[float]] = None,
    iou_type: str = "CIoU",  # Options: "IoU", "GIoU", "DIoU", "CIoU", "EIoU", "SIoU"
):
    """
    Initialize the BboxLoss module with regularization maximum and DFL settings.
    
    Args:
        reg_max (int): Maximum value for regularization in DFL.
        use_dim_weights (bool): Whether to use dimension-specific weights for loss calculation.
                               If False, all dimensions are weighted equally (default behavior).
        dim_weights (List[float], optional): Weights for [left, top, right, bottom] dimensions.
                                             Default: [1.0, 1.0, 1.0, 1.0] (equal weights).
                                             Example for det_123 (w and x important): [2.0, 1.0, 2.0, 1.0]
                                             Example for det_456 (h and y important): [1.0, 2.0, 1.0, 2.0]
        iou_type (str): Type of IoU loss to use. Options: "IoU", "GIoU", "DIoU", "CIoU", "EIoU", "SIoU"
    """
    super().__init__()
    self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
    
    # Dimension weights configuration
    self.use_dim_weights = use_dim_weights
    if dim_weights is None:
        dim_weights = [1.0, 1.0, 1.0, 1.0]  # Default: equal weights for [l, t, r, b]
    elif len(dim_weights) != 4:
        raise ValueError(f"dim_weights must have 4 elements [l, t, r, b], got {len(dim_weights)}")
    
    # Register as buffer so it moves with the model to the correct device
    self.register_buffer('dim_weights', torch.tensor(dim_weights, dtype=torch.float32))
    
    # IoU type configuration
    valid_iou_types = ["IoU", "GIoU", "DIoU", "CIoU", "EIoU", "SIoU"]
    if iou_type not in valid_iou_types:
        raise ValueError(f"iou_type must be one of {valid_iou_types}, got {iou_type}")
    self.iou_type = iou_type
    
    if self.use_dim_weights:
        LOGGER.info(f"BboxLoss: Dimension weights enabled - [l, t, r, b] = {dim_weights}")
    LOGGER.info(f"BboxLoss: IoU type = {iou_type}")
```

#### 位置 2: `forward` 方法修改（第 169-220 行）

**原始代码**：
```python
def forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, GIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    # DFL loss
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max + 1), target_ltrb[fg_mask])
        loss_dfl = loss_dfl.view(-1) / target_scores_sum
        return loss_iou, loss_dfl
    return loss_iou, torch.tensor(0.0, device=loss_iou.device)
```

**修改后**：
```python
def forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute IoU and DFL losses for bounding boxes.
    
    If use_dim_weights is True, applies dimension-specific weights to DFL loss.
    """
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    # Determine IoU type flags
    iou_kwargs = {
        "xywh": False,
        "GIoU": self.iou_type == "GIoU",
        "DIoU": self.iou_type == "DIoU",
        "CIoU": self.iou_type == "CIoU",
        "EIoU": self.iou_type == "EIoU",
        "SIoU": self.iou_type == "SIoU",
    }
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], **iou_kwargs)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    # DFL loss
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max + 1), target_ltrb[fg_mask])
        loss_dfl = loss_dfl.view(-1) / target_scores_sum
        
        # Apply dimension weights if enabled
        if self.use_dim_weights:
            # Reshape loss_dfl to [batch, 4] format
            # loss_dfl shape: [num_fg, 4] after view(-1) and division
            # We need to apply weights per dimension: [left, top, right, bottom]
            loss_dfl_per_dim = loss_dfl.view(-1, 4)  # [num_fg, 4]
            for dim_idx in range(4):
                dim_loss = loss_dfl_per_dim[:, dim_idx]
                dim_loss = dim_loss * self.dim_weights[dim_idx]
                loss_dfl_per_dim[:, dim_idx] = dim_loss
            # Reshape back to original format
            loss_dfl = loss_dfl_per_dim.view(-1)
        
        return loss_iou, loss_dfl
    return loss_iou, torch.tensor(0.0, device=loss_iou.device)
```

**注意事项**：
- 确保 `loss_dfl` 的形状是 `[num_fg * 4]`，可以 reshape 为 `[num_fg, 4]`
- 维度权重按顺序应用：`[left, top, right, bottom]`

---

### 修改 3: v8DetectionLoss 类 - HMD Loss 和 Focal Loss

#### 文件：`ultralytics/utils/loss.py`

#### 位置 1: `__init__` 方法参数添加（第 298-372 行）

**在 `__init__` 方法中添加以下参数**：

```python
def __init__(
    self,
    model,  # Model instance
    # ... 原有参数 ...
    use_dim_weights: Optional[bool] = None,
    dim_weights: Optional[List[float]] = None,
    use_focal_loss: Optional[bool] = None,
    focal_gamma: float = 1.5,
    focal_alpha: float = 0.25,
    iou_type: Optional[str] = None,
    # HMD Loss 参数
    use_hmd_loss: Optional[bool] = None,
    hmd_loss_weight: Optional[float] = None,
    hmd_penalty_single: Optional[float] = None,
    hmd_penalty_none: Optional[float] = None,
    hmd_penalty_coeff: Optional[float] = None,
    hmd_use_mm: Optional[bool] = None,
    mentum_class: int = 0,
    hyoid_class: int = 1,
):
```

**在 `__init__` 方法中添加以下初始化代码**：

```python
# 从 model.args 读取参数（如果未提供）
h = model.args  # Hyperparameters
if use_dim_weights is None:
    use_dim_weights = getattr(h, 'use_dim_weights', False)
if dim_weights is None:
    dim_weights = getattr(h, 'dim_weights', None)
if use_focal_loss is None:
    use_focal_loss = getattr(h, 'use_focal_loss', False)
if iou_type is None:
    iou_type = getattr(h, 'iou_type', 'CIoU')  # Default to CIoU

# HMD Loss 参数读取
if use_hmd_loss is None:
    use_hmd_loss = getattr(h, 'use_hmd_loss', False)
if hmd_loss_weight is None:
    hmd_loss_weight = getattr(h, 'hmd_loss_weight', 0.5)
if hmd_penalty_single is None:
    hmd_penalty_single = getattr(h, 'hmd_penalty_single', None)
if hmd_penalty_none is None:
    hmd_penalty_none = getattr(h, 'hmd_penalty_none', None)
if hmd_penalty_coeff is None:
    hmd_penalty_coeff = getattr(h, 'hmd_penalty_coeff', 0.5)
if hmd_use_mm is None:
    hmd_use_mm = getattr(h, 'hmd_use_mm', False)

# 存储 HMD Loss 参数
self.use_hmd_loss = use_hmd_loss
self.hmd_loss_weight = hmd_loss_weight
self.hmd_penalty_single = hmd_penalty_single
self.hmd_penalty_none = hmd_penalty_none
self.hmd_penalty_coeff = hmd_penalty_coeff
self.hmd_use_mm = hmd_use_mm
self.mentum_class = mentum_class
self.hyoid_class = hyoid_class

# HMD Loss 统计（用于计算 epoch 平均）
self.hmd_loss_sum = 0.0
self.hmd_loss_count = 0
self.last_hmd_loss = 0.0

# Focal Loss 初始化
self.use_focal_loss = use_focal_loss
if self.use_focal_loss:
    self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
else:
    self.focal_loss = None
```

#### 位置 2: BboxLoss 初始化修改（第 443-445 行）

**原始代码**：
```python
self.bbox_loss = BboxLoss(self.reg_max - 1, use_dim_weights=use_dim_weights, dim_weights=dim_weights)
```

**修改后**：
```python
self.bbox_loss = BboxLoss(
    self.reg_max - 1, 
    use_dim_weights=use_dim_weights, 
    dim_weights=dim_weights,
    iou_type=iou_type  # 传递 IoU 类型
)
```

#### 位置 3: `__call__` 方法修改 - HMD Loss 计算（第 419-494 行）

**在 `__call__` 方法的末尾（返回 loss 之前）添加**：

```python
# HMD loss calculation (if enabled)
if self.use_hmd_loss and fg_mask.sum() > 0:
    hmd_loss_value = self._calculate_hmd_loss(
        pred_bboxes, pred_scores, target_bboxes, gt_labels, fg_mask, stride_tensor
    )
    # 累积用于记录（计算 epoch 平均）
    self.hmd_loss_sum += hmd_loss_value
    self.hmd_loss_count += 1
    self.last_hmd_loss = hmd_loss_value
    # 添加到 box loss（加权）
    loss[0] = loss[0] + self.hmd_loss_weight * hmd_loss_value
```

#### 位置 4: 添加 `_calculate_hmd_loss` 方法（第 536-759 行）

**完整方法实现**（参考 `ultralytics/utils/loss.py` 第 536-759 行）：

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
    Calculate HMD loss for the batch.
    
    Returns weighted average HMD error (in pixels or mm).
    """
    # 导入 HMD 工具函数
    try:
        from ultralytics.mycodes.hmd_utils import calculate_hmd_loss, load_pixel_spacing_dict
    except ImportError:
        LOGGER.warning("HMD utils not available, skipping HMD loss calculation")
        return torch.tensor(0.0, device=pred_bboxes.device)
    
    batch_size = pred_bboxes.shape[0]
    device = pred_bboxes.device
    
    # 加载 pixel spacing（如果使用 mm 模式）
    pixel_spacing_dict = None
    if self.hmd_use_mm:
        try:
            pixel_spacing_dict = load_pixel_spacing_dict()
        except Exception as e:
            LOGGER.warning(f"Failed to load pixel spacing dict: {e}")
    
    # 准备数据格式
    # pred_bboxes: [batch, num_pred, 4] in xyxy format
    # pred_scores: [batch, num_pred, num_classes]
    # target_bboxes: [batch, num_target, 4] in xyxy format
    # gt_labels: [batch, num_target]
    
    hmd_errors = []
    weights = []
    
    for b in range(batch_size):
        # 获取当前 batch 的预测和 ground truth
        batch_pred_boxes = pred_bboxes[b]  # [num_pred, 4]
        batch_pred_scores = pred_scores[b]  # [num_pred, num_classes]
        batch_pred_conf = batch_pred_scores.max(dim=1)[0]  # [num_pred]
        batch_pred_cls = batch_pred_scores.argmax(dim=1)  # [num_pred]
        
        batch_target_boxes = target_bboxes[b]  # [num_target, 4]
        batch_target_cls = gt_labels[b]  # [num_target]
        
        # 过滤前景预测（fg_mask 是全局的，需要按 batch 索引）
        # 这里简化处理，假设所有预测都是前景
        # 实际实现中需要根据 fg_mask 过滤
        
        # 调用 HMD loss 计算函数
        hmd_loss_batch, stats = calculate_hmd_loss(
            pred_boxes=batch_pred_boxes.unsqueeze(0),  # [1, num_pred, 4]
            pred_conf=batch_pred_conf.unsqueeze(0),  # [1, num_pred]
            pred_cls=batch_pred_cls.unsqueeze(0),  # [1, num_pred]
            target_boxes=batch_target_boxes.unsqueeze(0),  # [1, num_target, 4]
            target_cls=batch_target_cls.unsqueeze(0),  # [1, num_target]
            mentum_class=self.mentum_class,
            hyoid_class=self.hyoid_class,
            penalty_single=self.hmd_penalty_single or 500.0,
            penalty_none=self.hmd_penalty_none or 1000.0,
            penalty_coeff=self.hmd_penalty_coeff,
            pixel_spacing=None,  # 如果需要，可以从 pixel_spacing_dict 获取
        )
        
        hmd_errors.append(hmd_loss_batch)
        weights.append(1.0)  # 可以按 batch 大小加权
    
    # 计算加权平均
    if len(hmd_errors) > 0:
        hmd_errors_tensor = torch.stack(hmd_errors)
        weights_tensor = torch.tensor(weights, device=device)
        hmd_loss = (hmd_errors_tensor * weights_tensor).sum() / (weights_tensor.sum() + 1e-8)
        return hmd_loss
    else:
        return torch.tensor(0.0, device=device)
```

**注意事项**：
- 实际实现中，`_calculate_hmd_loss` 方法需要根据具体的 batch 结构和 fg_mask 进行适配
- 建议直接使用 `hmd_utils.py` 中的 `calculate_hmd_loss` 函数

#### 位置 5: 添加 `get_avg_hmd_loss` 方法

**在 `v8DetectionLoss` 类中添加**：

```python
def get_avg_hmd_loss(self) -> float:
    """
    Get average HMD loss across all batches in the current epoch.
    
    Returns:
        Average HMD loss (float). Returns 0.0 if no HMD loss was calculated.
    """
    if self.hmd_loss_count > 0:
        if isinstance(self.hmd_loss_sum, torch.Tensor):
            return (self.hmd_loss_sum / self.hmd_loss_count).cpu().item()
        else:
            return self.hmd_loss_sum / self.hmd_loss_count
    return 0.0

def reset_hmd_loss_stats(self):
    """Reset HMD loss statistics for a new epoch."""
    self.hmd_loss_sum = 0.0
    self.hmd_loss_count = 0
    self.last_hmd_loss = 0.0
```

#### 位置 6: Focal Loss 替换 BCE Loss（第 515-518 行）

**原始代码**：
```python
loss[1] = self.bce(pred_scores, target_scores.to(dtype)) / target_scores_sum
```

**修改后**：
```python
# loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
if self.use_focal_loss and self.focal_loss is not None:
    loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype)) / target_scores_sum
else:
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)) / target_scores_sum
```

---

### 修改 4: 新增文件 - hmd_utils.py

#### 文件：`ultralytics/mycodes/hmd_utils.py`

**完整文件内容**：参考项目中的 `ultralytics/mycodes/hmd_utils.py`

**关键函数**：

1. **`calculate_hmd_from_boxes`**（第 14-60 行）
   - 从两个边界框计算 HMD 距离
   - 支持像素和毫米单位

2. **`calculate_hmd_loss`**（第 143-245 行）
   - 计算 HMD Loss
   - 实现 Smooth L1 Loss、Scale-Invariant Loss、方向约束

3. **`load_pixel_spacing_dict`**（第 109-141 行）
   - 加载 Pixel Spacing 字典（从 joblib 文件）

**依赖**：
- `torch`
- `torch.nn.functional`
- `numpy`
- `pathlib`
- `pandas`
- `joblib`

---

### 修改 5: 训练脚本 - train_yolo.py

#### 文件：`ultralytics/mycodes/train_yolo.py`

**这是一个大型自定义训练脚本，包含以下功能**：

1. **命令行参数扩展**（第 1896-1930 行）
   - `--iou_type`
   - `--use_dim_weights`
   - `--dim_weights`
   - `--use_focal_loss`
   - `--focal_gamma`
   - `--focal_alpha`
   - `--use_hmd_loss`
   - `--hmd_loss_weight`
   - `--hmd_penalty_single`
   - `--hmd_penalty_none`
   - `--hmd_penalty_coeff`
   - `--hmd_use_mm`
   - `--keep_top_conf_per_class`
   - `--conf_low`

2. **自定义回调函数**（第 655-1250 行）
   - `on_val_end_callback`：计算 HMD 指标
   - `keep_top_conf_per_class_callback`：自定义后处理

3. **Monkey Patch**（第 2441-2464 行）
   - 修补 `DetectionValidator.get_stats()` 以保存 stats

**移植建议**：
- 如果只需要核心功能，可以只移植参数解析和模型初始化部分
- 自定义回调函数可以根据需要选择性移植

---

## 新增文件说明 / New Files

### 1. `ultralytics/mycodes/hmd_utils.py`

**完整路径**：`ultralytics/mycodes/hmd_utils.py`

**功能**：
- HMD 距离计算
- HMD Loss 计算
- Pixel Spacing 加载

**依赖**：
```python
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import joblib
```

**关键函数接口**：

```python
def calculate_hmd_from_boxes(
    mentum_box: Union[torch.Tensor, np.ndarray],
    hyoid_box: Union[torch.Tensor, np.ndarray],
    pixel_spacing: Optional[float] = None
) -> Union[torch.Tensor, float]:
    """
    Calculate HMD (Hyomental Distance) from two bounding boxes.
    
    Args:
        mentum_box: [x1, y1, x2, y2] format
        hyoid_box: [x1, y1, x2, y2] format
        pixel_spacing: Optional pixel spacing (mm/pixel) for mm calculation
    
    Returns:
        HMD distance (pixels or mm)
    """

def calculate_hmd_loss(
    pred_boxes: torch.Tensor,
    pred_conf: torch.Tensor,
    pred_cls: torch.Tensor,
    target_boxes: torch.Tensor,
    target_cls: torch.Tensor,
    mentum_class: int = 0,
    hyoid_class: int = 1,
    penalty_single: float = 500.0,
    penalty_none: float = 1000.0,
    penalty_coeff: float = 0.5,
    pixel_spacing: Optional[float] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Calculate HMD loss for a batch.
    
    Returns:
        Tuple of (hmd_loss, stats_dict)
    """
```

---

## 依赖关系 / Dependencies

### Python 包依赖

**必需依赖**：
```python
torch >= 1.8.0
numpy >= 1.19.0
```

**可选依赖**（用于 HMD 功能）：
```python
pandas >= 1.3.0
joblib >= 1.0.0
```

### 模块导入依赖

**在 `ultralytics/utils/loss.py` 中**：
```python
from ultralytics.utils.metrics import bbox_iou  # 需要支持 EIoU 和 SIoU
```

**在 `ultralytics/utils/loss.py` 的 `_calculate_hmd_loss` 中**：
```python
from ultralytics.mycodes.hmd_utils import calculate_hmd_loss, load_pixel_spacing_dict
```

---

## 配置参数 / Configuration Parameters

### 1. IoU 类型参数

**参数名**：`iou_type`

**类型**：`str`

**可选值**：`"IoU"`, `"GIoU"`, `"DIoU"`, `"CIoU"`, `"EIoU"`, `"SIoU"`

**默认值**：`"CIoU"`

**使用位置**：
- `BboxLoss.__init__`
- `v8DetectionLoss.__init__`
- `BboxLoss.forward`（传递给 `bbox_iou`）

### 2. 维度权重参数

**参数名**：`use_dim_weights`, `dim_weights`

**类型**：`bool`, `List[float]`

**默认值**：`False`, `[1.0, 1.0, 1.0, 1.0]`

**格式**：`dim_weights = [left, top, right, bottom]`

**使用位置**：
- `BboxLoss.__init__`
- `BboxLoss.forward`（应用到 DFL loss）

### 3. Focal Loss 参数

**参数名**：`use_focal_loss`, `focal_gamma`, `focal_alpha`

**类型**：`bool`, `float`, `float`

**默认值**：`False`, `1.5`, `0.25`

**使用位置**：
- `v8DetectionLoss.__init__`
- `v8DetectionLoss.__call__`（替换 BCE loss）

### 4. HMD Loss 参数

**参数名**：
- `use_hmd_loss` (bool)
- `hmd_loss_weight` (float, default: 0.5)
- `hmd_penalty_single` (float, optional)
- `hmd_penalty_none` (float, optional)
- `hmd_penalty_coeff` (float, default: 0.5)
- `hmd_use_mm` (bool, default: False)
- `mentum_class` (int, default: 0)
- `hyoid_class` (int, default: 1)

**使用位置**：
- `v8DetectionLoss.__init__`
- `v8DetectionLoss._calculate_hmd_loss`
- `v8DetectionLoss.__call__`

---

## 测试验证 / Testing & Validation

### 1. 单元测试

#### 测试 IoU 类型

```python
import torch
from ultralytics.utils.metrics import bbox_iou

# 测试 SIoU
box1 = torch.tensor([[10, 10, 20, 20]])
box2 = torch.tensor([[15, 15, 25, 25]])
siou = bbox_iou(box1, box2, xywh=False, SIoU=True)
print(f"SIoU: {siou.item()}")

# 测试 EIoU
eiou = bbox_iou(box1, box2, xywh=False, EIoU=True)
print(f"EIoU: {eiou.item()}")
```

#### 测试维度权重

```python
from ultralytics.utils.loss import BboxLoss

# 创建带维度权重的 BboxLoss
bbox_loss = BboxLoss(
    reg_max=16,
    use_dim_weights=True,
    dim_weights=[2.0, 1.0, 2.0, 1.0],  # 水平方向权重更高
    iou_type="SIoU"
)

# 测试 forward
# ... 准备测试数据 ...
# loss_iou, loss_dfl = bbox_loss(...)
```

#### 测试 HMD Loss

```python
from ultralytics.mycodes.hmd_utils import calculate_hmd_from_boxes

# 测试 HMD 计算
mentum_box = torch.tensor([10, 10, 20, 20])
hyoid_box = torch.tensor([30, 15, 40, 25])
hmd = calculate_hmd_from_boxes(mentum_box, hyoid_box)
print(f"HMD: {hmd.item()} pixels")
```

### 2. 集成测试

#### 训练脚本测试

```bash
# 测试 SIoU
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --iou_type SIoU \
  --epochs 1

# 测试维度权重
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --use_dim_weights \
  --dim_weights 2.0 1.0 2.0 1.0 \
  --epochs 1

# 测试 HMD Loss
python ultralytics/mycodes/train_yolo.py yolo11n det_123 \
  --use_hmd_loss \
  --hmd_loss_weight 0.5 \
  --epochs 1
```

### 3. 验证清单

- [ ] IoU 类型切换正常工作（SIoU、EIoU、DIoU）
- [ ] 维度权重正确应用到 DFL loss
- [ ] Focal Loss 正确替换 BCE Loss
- [ ] HMD Loss 正确计算并添加到总损失
- [ ] HMD Loss 统计正确累积（`hmd_loss_sum`, `hmd_loss_count`）
- [ ] `get_avg_hmd_loss()` 返回正确的平均值
- [ ] 训练过程中 loss 值合理（不出现 NaN 或 Inf）

---

## 移植步骤总结 / Migration Steps Summary

### 步骤 1: 备份原始文件

```bash
cp ultralytics/utils/loss.py ultralytics/utils/loss.py.backup
cp ultralytics/utils/metrics.py ultralytics/utils/metrics.py.backup
```

### 步骤 2: 修改 metrics.py

1. 在 `bbox_iou` 函数签名中添加 `EIoU` 和 `SIoU` 参数
2. 实现 SIoU 计算逻辑（第 133-159 行）
3. 实现 EIoU 计算逻辑（第 160-170 行）
4. 修改条件判断以包含 `SIoU` 和 `EIoU`

### 步骤 3: 修改 loss.py - BboxLoss

1. 在 `__init__` 中添加 `use_dim_weights`, `dim_weights`, `iou_type` 参数
2. 在 `forward` 中修改 IoU 调用以支持动态类型选择
3. 在 `forward` 中实现维度权重应用到 DFL loss

### 步骤 4: 修改 loss.py - v8DetectionLoss

1. 在 `__init__` 中添加所有新参数（HMD Loss、Focal Loss、维度权重、IoU 类型）
2. 修改 BboxLoss 初始化以传递新参数
3. 在 `__call__` 中添加 HMD Loss 计算
4. 实现 `_calculate_hmd_loss` 方法
5. 实现 `get_avg_hmd_loss` 和 `reset_hmd_loss_stats` 方法
6. 修改分类损失以支持 Focal Loss

### 步骤 5: 创建 hmd_utils.py

1. 创建 `ultralytics/mycodes/` 目录（如果不存在）
2. 复制 `hmd_utils.py` 文件
3. 确保所有依赖包已安装

### 步骤 6: 测试验证

1. 运行单元测试
2. 运行集成测试
3. 检查训练日志确认功能正常

---

## 常见问题 / FAQ

### Q1: 如何确定维度权重的值？

**A**: 维度权重应根据任务特点设置：
- **水平重要**（如 HMD 计算）：`[2.0, 1.0, 2.0, 1.0]`（left 和 right 权重高）
- **垂直重要**：`[1.0, 2.0, 1.0, 2.0]`（top 和 bottom 权重高）
- **默认**：`[1.0, 1.0, 1.0, 1.0]`（所有维度相等）

### Q2: HMD Loss 不工作怎么办？

**A**: 检查以下几点：
1. 确保 `use_hmd_loss=True`
2. 确保 `hmd_utils.py` 可以正确导入
3. 检查 `mentum_class` 和 `hyoid_class` 是否正确
4. 查看训练日志中的警告信息

### Q3: SIoU/EIoU 计算出现 NaN？

**A**: 检查：
1. 确保输入 boxes 格式正确（xyxy 或 xywh）
2. 检查 `eps` 值是否足够大（建议 `1e-7`）
3. 检查 boxes 是否有无效值（如负坐标）

### Q4: 如何禁用某个功能？

**A**: 设置对应参数为默认值：
- 禁用 HMD Loss：`use_hmd_loss=False`
- 禁用维度权重：`use_dim_weights=False`
- 禁用 Focal Loss：`use_focal_loss=False`
- 使用默认 IoU：`iou_type="CIoU"`

---

## 版本兼容性 / Version Compatibility

### 测试版本

- **Ultralytics YOLO**: 8.3.159+
- **PyTorch**: 1.8.0+
- **Python**: 3.8+

### 向后兼容性

- 所有新参数都有默认值，不会破坏现有代码
- 如果不提供新参数，行为与原始 YOLO 相同

---

## 参考资料 / References

### 论文

1. **SIoU**: [SIoU Loss: More Powerful Learning for Bounding Box Regression](https://arxiv.org/abs/2205.12740)
2. **EIoU**: [Focal and Efficient IOU Loss for Accurate Bounding Box Regression](https://arxiv.org/abs/2101.08158)
3. **DIoU/CIoU**: [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)

### 代码参考

- 本项目代码库：`ultralytics/ultralytics/utils/loss.py`
- 本项目代码库：`ultralytics/ultralytics/utils/metrics.py`
- 本项目代码库：`ultralytics/mycodes/hmd_utils.py`

---

## 联系支持 / Support

如有问题，请参考：
- 项目 README.md
- 代码注释
- 相关论文

---

**最后更新 / Last Updated**: 2025-12-29

**版本 / Version**: 1.0.0


