from pathlib import Path

# 設定參數
dataset_dir = Path("./seg_45/v1/")  # 含 train/val/test.txt 的資料夾
old_prefix = "/root/ultrasound/DifficultAirway/"
new_prefix = "D:/workplace/project_management/github_project/ultrasound-airway-detection/"

# 可處理的檔案列表
split_files = ["train.txt", "val.txt", "test.txt"]

for split in split_files:
    file_path = dataset_dir / split
    if not file_path.exists():
        print(f"❌ {file_path} 不存在，略過")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(old_prefix):
            new_line = line.replace(old_prefix, new_prefix)
            new_lines.append(new_line)
        else:
            new_lines.append(line)  # 保留沒改的行

    # 你可以選擇：是否要覆蓋原檔
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"✅ 已處理並更新: {file_path}")
