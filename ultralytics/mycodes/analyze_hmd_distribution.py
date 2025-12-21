"""
分析 det_123 数据集中 HMD 三种情况的分布
Analyze HMD distribution in det_123 dataset
"""
import argparse
from pathlib import Path
from collections import defaultdict

def count_classes_in_label(label_file: Path) -> tuple[bool, bool]:
    """
    读取 YOLO label 文件，检查是否包含 Mentum (class 0) 和 Hyoid (class 1)
    
    Returns:
        (has_mentum, has_hyoid): 是否包含 Mentum 和 Hyoid
    """
    if not label_file.exists():
        return False, False
    
    has_mentum = False
    has_hyoid = False
    
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id == 0:  # Mentum
                        has_mentum = True
                    elif class_id == 1:  # Hyoid
                        has_hyoid = True
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return False, False
    
    return has_mentum, has_hyoid

def parse_yaml(yaml_file: Path) -> dict:
    """
    简单解析 YAML 文件（不使用 yaml 库）
    """
    config = {}
    with open(yaml_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'path':
                    config['path'] = value
                elif key == 'train':
                    config['train'] = value
                elif key == 'val':
                    config['val'] = value
                elif key == 'test':
                    config['test'] = value
    return config

def analyze_dataset(yaml_file: Path):
    """
    分析数据集，统计 HMD 三种情况的分布
    """
    # 读取 YAML 文件
    config = parse_yaml(yaml_file)
    
    dataset_path = Path(config['path'])
    print(f"\n{'='*80}")
    print(f"分析数据集: {yaml_file.name}")
    print(f"数据集路径: {dataset_path}")
    print(f"{'='*80}\n")
    
    # 统计每个 split
    splits = ['train', 'val', 'test']
    total_stats = {
        'case1_both': 0,      # 情况1：两个都有
        'case2_single': 0,    # 情况2：只有一个
        'case3_none': 0,      # 情况3：都没有
        'total': 0
    }
    
    # 获取 split 文件名映射
    split_files_map = {
        'train': config.get('train', 'train.txt'),
        'val': config.get('val', 'val.txt'),
        'test': config.get('test', 'test.txt')
    }
    
    for split in splits:
        split_filename = split_files_map[split]
        split_file = dataset_path / split_filename
        if not split_file.exists():
            print(f"⚠️  {split_filename} 不存在，跳过")
            continue
        
        print(f"\n📊 分析 {split} 集...")
        
        # 读取图像列表
        image_paths = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    image_paths.append(Path(line))
        
        print(f"   图像数量: {len(image_paths)}")
        
        # 统计
        split_stats = {
            'case1_both': 0,
            'case2_single': 0,
            'case3_none': 0,
            'total': len(image_paths)
        }
        
        for img_path_str in image_paths:
            # 处理路径（可能是字符串或 Path 对象）
            if isinstance(img_path_str, str):
                img_path = Path(img_path_str)
            else:
                img_path = img_path_str
            
            # 找到对应的 label 文件
            # YOLO 格式：图像在 patient_data/xxx/xxx.png，label 在相同目录下
            if img_path.is_absolute():
                # 绝对路径：label 在同一目录
                label_path = img_path.parent / f"{img_path.stem}.txt"
            else:
                # 相对路径：相对于 dataset_path
                label_path = dataset_path / img_path.parent / f"{img_path.stem}.txt"
            
            # 如果找不到，尝试其他可能的路径
            if not label_path.exists():
                # 尝试从图像路径推断（patient_data/xxx/xxx.png -> patient_data/xxx/xxx.txt）
                img_str = str(img_path)
                if 'patient_data' in img_str:
                    # 替换扩展名
                    label_path = Path(img_str.rsplit('.', 1)[0] + '.txt')
                else:
                    # 尝试在 dataset_path 下查找
                    label_path = dataset_path / 'labels' / f"{img_path.stem}.txt"
            
            # 检查类别
            has_mentum, has_hyoid = count_classes_in_label(label_path)
            
            if has_mentum and has_hyoid:
                split_stats['case1_both'] += 1
                total_stats['case1_both'] += 1
            elif has_mentum or has_hyoid:
                split_stats['case2_single'] += 1
                total_stats['case2_single'] += 1
            else:
                split_stats['case3_none'] += 1
                total_stats['case3_none'] += 1
            
            total_stats['total'] += 1
        
        # 打印 split 统计
        print(f"   情况1（两个都有）: {split_stats['case1_both']:6d} ({split_stats['case1_both']/split_stats['total']*100:5.2f}%)")
        print(f"   情况2（只有一个）: {split_stats['case2_single']:6d} ({split_stats['case2_single']/split_stats['total']*100:5.2f}%)")
        print(f"   情况3（都没有）  : {split_stats['case3_none']:6d} ({split_stats['case3_none']/split_stats['total']*100:5.2f}%)")
        print(f"   总计            : {split_stats['total']:6d}")
    
    # 打印总体统计
    print(f"\n{'='*80}")
    print(f"📈 总体统计 ({yaml_file.name}):")
    print(f"{'='*80}")
    print(f"情况1（两个都有）: {total_stats['case1_both']:6d} ({total_stats['case1_both']/total_stats['total']*100:5.2f}%)")
    print(f"情况2（只有一个）: {total_stats['case2_single']:6d} ({total_stats['case2_single']/total_stats['total']*100:5.2f}%)")
    print(f"情况3（都没有）  : {total_stats['case3_none']:6d} ({total_stats['case3_none']/total_stats['total']*100:5.2f}%)")
    print(f"总计            : {total_stats['total']:6d}")
    print(f"{'='*80}\n")
    
    return total_stats

def main():
    parser = argparse.ArgumentParser(description='分析 det_123 数据集中 HMD 三种情况的分布')
    parser.add_argument('--yaml-dir', type=str, default='yolo_dataset/det_123/v3',
                       help='YAML 文件所在目录')
    args = parser.parse_args()
    
    yaml_dir = Path(args.yaml_dir)
    
    # 分析两个 YAML 文件
    yaml_files = [
        yaml_dir / 'det_123.yaml',
        yaml_dir / 'det_123_ES.yaml'
    ]
    
    all_stats = {}
    for yaml_file in yaml_files:
        if yaml_file.exists():
            stats = analyze_dataset(yaml_file)
            all_stats[yaml_file.name] = stats
        else:
            print(f"⚠️  文件不存在: {yaml_file}")
    
    # 对比总结
    if len(all_stats) == 2:
        print(f"\n{'='*80}")
        print(f"📊 对比总结")
        print(f"{'='*80}")
        print(f"{'数据集':<20} {'情况1':>12} {'情况2':>12} {'情况3':>12} {'总计':>12}")
        print(f"{'-'*80}")
        for name, stats in all_stats.items():
            print(f"{name:<20} {stats['case1_both']:>8} ({stats['case1_both']/stats['total']*100:>5.1f}%) "
                  f"{stats['case2_single']:>8} ({stats['case2_single']/stats['total']*100:>5.1f}%) "
                  f"{stats['case3_none']:>8} ({stats['case3_none']/stats['total']*100:>5.1f}%) "
                  f"{stats['total']:>12}")
        print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

