# YOLO 训练脚本使用指南

本目录包含用于训练 YOLO 模型的脚本工具。

## 文件说明

- `train_yolo.py` - 主要的训练脚本，支持完整的训练参数配置
- `best_epoch.py` - 用于查找训练过程中最佳 epoch 的工具

## 基本用法

### 基本训练命令

```bash
python mycodes/train_yolo.py <model> <database> [options]
```

### 示例命令

```bash
python mycodes/train_yolo.py yolo11n det_123 --db_version=3 --es --epochs=15 --wandb --project="ultrasound-det_123_ES-v3-small-obj" --exp_name="exp10-small-obj-optimized"
```

## 必需参数

### 模型选择 (`model`)
支持的模型：
- YOLO8: `yolo8n`, `yolo8s`, `yolo8m`, `yolo8l`, `yolo8x`
- YOLO11: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- YOLO12: `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`
- Segmentation: `yolo8n-seg`, `yolo11n-seg`, `yolo11s-seg`, 等

### 数据集选择 (`database`)
- `det_123` - 检测数据集 123
- `seg_45` - 分割数据集 45
- `det_678` - 检测数据集 678

## 常用参数

### 数据集配置
- `--db_version` (默认: 1) - 数据集版本，可选: 1, 2, 3
- `--es` - 使用 ES (Endoscopy) 数据集后缀，会使用 `*_ES.yaml` 配置文件

### 训练基本设置
- `--epochs` (默认: 50) - 训练轮数
- `--batch` (默认: 16) - 批次大小
- `--imgsz` (默认: 640) - 输入图像尺寸 (640, 1280 等)
- `--device` (默认: 'cuda:0') - 训练设备
- `--seed` (默认: 42) - 随机种子

### Wandb 日志
- `--wandb` - 启用 Wandb 日志记录
- `--project` - Wandb 项目名称（如不指定会自动生成）
- `--exp_name` - 实验名称标识符（例如: exp1, baseline, exp10-small-obj-optimized）

### 优化器设置
- `--optimizer` (默认: 'AdamW') - 优化器选择: 'SGD', 'Adam', 'AdamW'
- `--lr0` (默认: 0.01) - 初始学习率
- `--lrf` (默认: 0.01) - 最终学习率 (lr0 * lrf)
- `--weight_decay` (默认: 0.0005) - 权重衰减
- `--cos_lr` - 使用余弦学习率调度器

### Loss 权重（小目标检测重要参数）
- `--box` (默认: 7.5) - Box loss 权重，小目标建议: 8-12
- `--cls` (默认: 0.5) - Class loss 权重，类别不平衡建议: 0.7-1.0
- `--dfl` (默认: 1.5) - DFL loss 权重，建议范围: 1.5-2.0

### 小目标检测优化
- `--use_focal_loss` - 使用 Focal Loss（对小目标更有效）
- `--focal_gamma` (默认: 1.5) - Focal Loss gamma 参数，范围: 1.0-2.5
- `--focal_alpha` (默认: 0.25) - Focal Loss alpha 参数
- `--use_dim_weights` - 启用维度特定权重
- `--dim_weights W_L W_T W_R W_B` - 边界框维度权重 [left, top, right, bottom]
  - 示例: `--dim_weights 2.0 1.0 2.0 1.0` (强调水平方向)
  - 示例: `--dim_weights 1.0 2.0 1.0 2.0` (强调垂直方向)

### 数据增强（超声波图像推荐设置）
- `--hsv_h` (默认: 0) - HSV 色相增强，超声波图像设为 0（黑白图像）
- `--hsv_s` (默认: 0.7) - HSV 饱和度增强
- `--hsv_v` (默认: 0.4) - HSV 亮度增强
- `--degrees` (默认: 0.0) - 旋转角度，超声波图像建议设为 0
- `--shear` (默认: 0.0) - 剪切变换，超声波图像建议设为 0
- `--perspective` (默认: 0.0) - 透视变换，超声波图像建议设为 0
- `--scale` (默认: 0.5) - 缩放变换
- `--translate` (默认: 0.1) - 平移变换
- `--fliplr` (默认: 0.5) - 左右翻转概率
- `--mosaic` (默认: 1.0) - Mosaic 增强概率
- `--mixup` (默认: 0.0) - Mixup 增强概率
- `--copy_paste` (默认: 0.0) - Copy-paste 增强概率

### 其他训练设置
- `--patience` (默认: 100) - 早停耐心值
- `--workers` (默认: 8) - 数据加载器工作线程数
- `--resume` - 从上次检查点继续训练
- `--deterministic` (默认: True) - 确定性训练

## 常用实验配置示例

### 1. 小目标检测优化配置（基于示例命令）

```bash
python mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --epochs=15 \
  --wandb \
  --project="ultrasound-det_123_ES-v3-small-obj" \
  --exp_name="exp10-small-obj-optimized" \
  --batch=16 \
  --imgsz=640 \
  --optimizer="SGD" \
  --lr0=0.01 \
  --lrf=0.01 \
  --box=8.5 \
  --cls=0.6 \
  --dfl=2.0 \
  --use_focal_loss \
  --focal_gamma=1.5 \
  --focal_alpha=0.25 \
  --use_dim_weights \
  --dim_weights 5.0 1.0 5.0 1.0 \
  --hsv_h=0.0 \
  --hsv_s=0.8 \
  --hsv_v=0.5 \
  --scale=0.7 \
  --translate=0.15 \
  --degrees=0.0 \
  --shear=0.0 \
  --perspective=0.0 \
  --fliplr=0.5 \
  --mixup=0.0 \
  --copy_paste=0.0 \
  --close_mosaic=10 \
  --patience=100
```

### 2. 基础配置

```bash
python mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --epochs=50 \
  --wandb \
  --exp_name="baseline"
```

### 3. 高分辨率训练（更精确但更慢）

```bash
python mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --epochs=45 \
  --batch=6 \
  --imgsz=1280 \
  --wandb \
  --project="ultrasound-det_123_ES-v3" \
  --exp_name="high-res"
```

### 4. 使用维度权重优化

```bash
python mycodes/train_yolo.py yolo11n det_123 \
  --db_version=3 \
  --es \
  --epochs=15 \
  --wandb \
  --exp_name="dim-weights-vertical" \
  --use_dim_weights \
  --dim_weights 1.0 2.0 1.0 2.0
```

## 查看训练结果

### 使用 best_epoch.py 查找最佳 epoch

训练完成后，使用 `best_epoch.py` 查找最佳 epoch：

```bash
python mycodes/best_epoch.py detect 1 --run_name="yolo11n-det_123-v3-exp10-small-obj-optimized"
```

参数说明：
- `detect` 或 `segment` - 任务类型
- `1` - run number（通常为 1）
- `--run_name` - 训练运行的目录名称（与 `--exp_name` 对应）

输出会显示最佳 epoch 和对应的 fitness 值。

### Fitness 计算公式

- Detection: `fitness = mAP50 * 0.1 + mAP50-95 * 0.9`
- Segmentation: `fitness = (mAP50_box * 0.1 + mAP50-95_box * 0.9) + (mAP50_mask * 0.1 + mAP50-95_mask * 0.9)`

## 训练输出

训练结果保存在：
- `ultralytics/runs/train/{model}-{database}-v{version}-{exp_name}/`
- 最佳模型: `weights/best.pt`
- 最后检查点: `weights/last.pt`
- 训练日志: `results.csv`

如果启用了 Wandb，可以在 Wandb 网站上查看详细的训练指标和可视化。

## 注意事项

1. **数据集路径**: 确保数据集 YAML 文件存在于 `yolo_dataset/{database}/v{version}/` 目录
2. **ES 数据集**: 使用 `--es` 时，需要对应的 `{database}_ES.yaml` 文件
3. **GPU 内存**: 如果遇到 OOM 错误，尝试减小 `--batch` 或 `--imgsz`
4. **小目标检测**: 建议使用 Focal Loss 和维度权重来提高小目标检测性能
5. **超声波图像**: 建议关闭旋转相关的数据增强（`degrees=0`, `shear=0`, `perspective=0`），并设置 `hsv_h=0`

## 参数帮助

查看完整的参数列表：

```bash
python mycodes/train_yolo.py --help
```
