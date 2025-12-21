# Loss 函数调整文档

本目录包含关于 YOLO 模型 Loss 函数调整的详细文档，特别是针对小目标检测的优化方法。

## 文档列表

### 1. DIMENSION_WEIGHTS_USAGE.md
维度权重使用指南。说明如何使用维度特定的权重来优化边界框损失函数，特别适用于需要强调特定方向（水平或垂直）的目标检测任务。

### 2. DIMENSION_WEIGHTS_TECHNICAL.md
维度权重技术文档。详细的技术实现说明，包括数学原理和代码实现细节。

### 3. LOSS_MODIFICATION_OPTIONS.md
Loss 修改选项文档。介绍各种 Loss 函数的修改选项，包括 Focal Loss、损失权重调整等方法的说明。

### 4. SMALL_OBJECT_LOSS_TUNING.md
小目标检测 Loss 调整指南。专门针对小目标检测场景的 Loss 函数调优方法，包括参数设置建议和最佳实践。

## 在训练脚本中使用

这些 Loss 调整功能可以在 `mycodes/train_yolo.py` 训练脚本中使用，相关参数包括：

- `--use_focal_loss` - 启用 Focal Loss
- `--focal_gamma` - Focal Loss gamma 参数
- `--focal_alpha` - Focal Loss alpha 参数
- `--use_dim_weights` - 启用维度权重
- `--dim_weights W_L W_T W_R W_B` - 设置维度权重
- `--box`, `--cls`, `--dfl` - 损失权重调整

详细使用方法请参考 `mycodes/README.md`。

D:\workplace\project_management\github_project\ultrasound-airway-detection2\ultralytics\ultralytics\utils\loss.py

