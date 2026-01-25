# 實驗腳本使用說明 / Experiments Script Usage

本目錄包含兩個腳本，用於自動運行 README.md 中的所有實驗：

1. **`run_all_experiments.sh`** - Bash 腳本（適用於 Linux/macOS）
2. **`run_all_experiments.py`** - Python 腳本（適用於所有平台，包括 Windows）

## 功能特性 / Features

- ✅ 自動運行所有實驗（按順序）
- ✅ 支持 RTX 4090 和 H200 兩種配置
- ✅ 自動記錄日誌到文件
- ✅ 支持跳過失敗的實驗繼續運行
- ✅ 支持從指定實驗開始運行
- ✅ 支持運行到指定實驗停止

## 使用方法 / Usage

### Bash 腳本（Linux/macOS）

```bash
# 運行 RTX 4090 配置的所有實驗
bash run_all_experiments.sh --config 4090

# 運行 H200 配置的所有實驗
bash run_all_experiments.sh --config h200

# 如果實驗失敗，繼續運行下一個（不中斷）
bash run_all_experiments.sh --config 4090 --skip-failed

# 從指定實驗開始運行
bash run_all_experiments.sh --config 4090 --start-from "exp3 focal_loss"

# 運行到指定實驗停止
bash run_all_experiments.sh --config 4090 --stop-at "exp5-1 hmd_loss_pixel"
```

### Python 腳本（所有平台）

```bash
# 運行 RTX 4090 配置的所有實驗
python run_all_experiments.py --config 4090

# 運行 H200 配置的所有實驗
python run_all_experiments.py --config h200

# 如果實驗失敗，繼續運行下一個（不中斷）
python run_all_experiments.py --config 4090 --skip-failed

# 從指定實驗開始運行
python run_all_experiments.py --config 4090 --start-from "exp3 focal_loss"

# 運行到指定實驗停止
python run_all_experiments.py --config 4090 --stop-at "exp5-1 hmd_loss_pixel"
```

## 實驗列表 / Experiment List

### RTX 4090 配置（共 28 個實驗）

1. exp0 baseline
2. exp0 baseline+keep_top_conf_per_class
3. exp1-1 data_aug
4. exp1-1 data_aug+keep_top_conf_per_class
5. exp1-2 ultrasound_aug
6. exp1-2 ultrasound_aug+keep_top_conf_per_class
7. exp2 loss_weights
8. exp2 loss_weights+keep_top_conf_per_class
9. exp3 focal_loss
10. exp3 focal_loss+keep_top_conf_per_class
11. exp4 dim_weights
12. exp4 dim_weights+keep_top_conf_per_class
13. exp5-1 hmd_loss_pixel
14. exp5-1 hmd_loss_pixel+keep_top_conf_per_class
15. exp5-2 hmd_loss_mm
16. exp5-2 hmd_loss_mm+keep_top_conf_per_class
17. exp6-1 warmup_optimized
18. exp6-1 warmup_optimized+keep_top_conf_per_class
19. exp6-2 warmup_cosine_restart
20. exp6-2 warmup_cosine_restart+keep_top_conf_per_class
21. exp7-1 siou
22. exp7-1 siou+keep_top_conf_per_class
23. exp7-2 eiou
24. exp7-2 eiou+keep_top_conf_per_class
25. exp7-3 diou
26. exp7-3 diou+keep_top_conf_per_class

### H200 配置（共 13 個實驗）

1. exp0 baseline
2. exp1-1 data_aug
3. exp1-2 ultrasound_aug
4. exp2 loss_weights
5. exp3 focal_loss
6. exp4 dim_weights
7. exp5-1 hmd_loss_pixel
8. exp5-2 hmd_loss_mm
9. exp6-1 warmup_optimized
10. exp6-2 warmup_cosine_restart
11. exp7-1 siou
12. exp7-2 eiou
13. exp7-3 diou

## 日誌文件 / Log Files

腳本會自動創建日誌文件，文件名格式為：
- `experiments_4090_YYYYMMDD_HHMMSS.log`（RTX 4090 配置）
- `experiments_h200_YYYYMMDD_HHMMSS.log`（H200 配置）

日誌文件包含：
- 每個實驗的開始時間
- 完整的命令
- 訓練過程的所有輸出
- 實驗的成功/失敗狀態

## 注意事項 / Notes

1. **運行時間**：所有實驗按順序運行，總時間取決於每個實驗的訓練時間（`--epochs=10`）
2. **GPU 資源**：確保有足夠的 GPU 資源，避免多個實驗同時運行導致 OOM
3. **Wandb 登錄**：確保已登錄 Wandb（`wandb login`）
4. **中斷恢復**：如果腳本被中斷，可以使用 `--start-from` 參數從中斷的實驗繼續運行
5. **錯誤處理**：使用 `--skip-failed` 時，失敗的實驗會被記錄但不會中斷整個流程

## 範例場景 / Example Scenarios

### 場景 1：完整運行所有實驗（RTX 4090）

```bash
# 這將運行所有 28 個實驗，可能需要數天時間
python run_all_experiments.py --config 4090
```

### 場景 2：測試運行（只運行前幾個實驗）

```bash
# 運行到 exp2 停止
python run_all_experiments.py --config 4090 --stop-at "exp2 loss_weights"
```

### 場景 3：從中斷處恢復

```bash
# 假設上次運行到 exp5-1 時中斷，從 exp5-2 繼續
python run_all_experiments.py --config 4090 --start-from "exp5-2 hmd_loss_mm"
```

### 場景 4：容錯運行（允許部分失敗）

```bash
# 即使某些實驗失敗，也繼續運行後續實驗
python run_all_experiments.py --config 4090 --skip-failed
```

## 故障排除 / Troubleshooting

### 問題 1：權限錯誤（Bash 腳本）

```bash
chmod +x run_all_experiments.sh
```

### 問題 2：Python 找不到模組

確保在項目根目錄運行腳本：
```bash
cd /path/to/ultrasound-airway-detection2
python run_all_experiments.py --config 4090
```

### 問題 3：GPU 記憶體不足

可以修改腳本中的 `--batch` 參數，或使用 `--start-from` 和 `--stop-at` 分批運行實驗。

## 與 README.md 的對應關係

腳本中的實驗命令與 `README.md` 中的實驗命令完全一致，確保：
- 相同的參數設置
- 相同的項目名稱（project）
- 相同的實驗名稱（exp_name）
- 相同的硬體配置（batch size, device）

所有實驗結果都會記錄到對應的 Wandb 項目中，方便後續分析和比較。


