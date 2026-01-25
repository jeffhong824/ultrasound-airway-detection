# HMD 距离计算工具使用说明

## 功能

从 YOLO 预测结果（或 ground truth label）计算 HMD (Hyomental Distance) 真实距离（毫米）。

## 安装依赖

```bash
pip install pandas numpy pydicom opencv-python tqdm
# 或使用 PIL 替代 opencv-python
pip install pandas numpy pydicom pillow tqdm
```

## 使用方法

### 1. 单个 Patient 处理

```bash
cd ultralytics/evaluate

# 使用 ground truth label 计算
python calculate_hmd_from_yolo.py \
    --case-id det_123 \
    --patient-id 0834980 \
    --yolo-root ../../yolo_dataset \
    --dicom-root ../../dicom_dataset \
    --version v3

# 使用 YOLO 预测结果计算（需要先有预测结果）
python calculate_hmd_from_yolo.py \
    --case-id det_123 \
    --patient-id 0834980 \
    --use-pred \
    --pred-root ../../pred_video/det_123/... \
    --yolo-root ../../yolo_dataset \
    --dicom-root ../../dicom_dataset \
    --version v3
```

### 2. 批量处理

```bash
# 批量处理所有 patient
python calculate_hmd_from_yolo.py \
    --case-id det_123 \
    --batch \
    --yolo-root ../../yolo_dataset \
    --dicom-root ../../dicom_dataset \
    --version v3 \
    --output hmd_results_det_123.csv
```

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--case-id` | ✅ | 数据集 ID (如 `det_123`) |
| `--patient-id` | ⚠️ | 患者 ID (如 `0834980`)，如果不指定则必须使用 `--batch` |
| `--batch` | ⚠️ | 批量处理所有 patient（与 `--patient-id` 二选一） |
| `--yolo-root` | ❌ | yolo_dataset 根目录（默认：`../../yolo_dataset`） |
| `--dicom-root` | ❌ | dicom_dataset 根目录（默认：`../../dicom_dataset`） |
| `--version` | ❌ | 数据集版本（默认：`v3`） |
| `--use-pred` | ❌ | 使用 YOLO 预测结果（默认：使用 ground truth） |
| `--pred-root` | ⚠️ | 预测结果根目录（如果使用 `--use-pred`） |
| `--output` | ❌ | 输出 CSV 文件路径（默认：`hmd_results_{case_id}.csv`） |

## 输出格式

CSV 文件包含以下列：

- `patient_id`: 患者 ID
- `image_name`: 图片文件名
- `dicom_base`: DICOM 基础名称
- `pose`: 姿势（Neutral/Extended/Ramped/Unknown）
- `pixel_spacing`: PixelSpacing (mm/pixel)
- `hmd_mm`: HMD 距离（毫米）

## HMD 计算逻辑

根据 `distance/codes_DA/DA_metrics.py` 的实现：

```python
# 水平距离
HMD_dx = (Hyoid_xtl - Mentum_xbr) * PixelSpacing

# 垂直距离
HMD_dy = (Hyoid_y_center - Mentum_y_center) * PixelSpacing

# 欧几里得距离
HMD = sqrt(HMD_dx² + HMD_dy²)
```

其中：
- `Hyoid_xtl`: Hyoid bbox 的左边界 (x1)
- `Mentum_xbr`: Mentum bbox 的右边界 (x2)
- `Hyoid_y_center`: Hyoid bbox 的垂直中心
- `Mentum_y_center`: Mentum bbox 的垂直中心

## 注意事项

1. **文件命名规则**：
   - PNG 文件名格式：`{patient_id}_Quick ID_{timestamp}_B[_pose][{frame}].png`
   - 程序会自动提取 DICOM 基础名称和 pose 信息

2. **DICOM 文件查找**：
   - 程序会在 `dicom_dataset/{內視鏡|困難|非困難}/{patient_id}_Quick ID/` 下查找
   - 支持多种文件名格式（带或不带 pose 后缀）

3. **PixelSpacing**：
   - 从 DICOM 文件的 `PixelSpacing` 字段读取
   - 如果读取失败，该图片会被跳过

4. **Bbox 处理**：
   - 如果一张图片中有多个 Mentum 或 Hyoid bbox，目前使用第一个
   - 未来可以改进为使用置信度最高的 bbox

## 示例输出

```
📊 处理 Patient 0834980: 150 张图片
Patient 0834980: 100%|████████████| 150/150 [00:05<00:00, 28.5it/s]

📈 Patient 0834980 HMD 统计（按 pose）:
  Neutral: 中位数=45.23mm, 平均值=45.67mm, 标准差=3.21mm (n=50)
  Extended: 中位数=48.12mm, 平均值=48.45mm, 标准差=2.98mm (n=50)
  Ramped: 中位数=46.89mm, 平均值=47.12mm, 标准差=3.05mm (n=50)

📊 总体统计:
  总图片数: 150
  总 patient 数: 1

按 pose 统计:
  Extended: 中位数=48.12mm, 平均值=48.45mm (n=50)
  Neutral: 中位数=45.23mm, 平均值=45.67mm (n=50)
  Ramped: 中位数=46.89mm, 平均值=47.12mm (n=50)

✅ 结果已保存到: hmd_results_det_123.csv
```

## 扩展功能

未来可以添加：
1. 支持其他距离指标（TongueThickness, DSH, DSE 等）
2. 支持从 YOLO 预测结果直接读取（需要适配预测结果格式）
3. 可视化输出（绘制 HMD 测量线）
4. 与 ground truth 的对比评估



