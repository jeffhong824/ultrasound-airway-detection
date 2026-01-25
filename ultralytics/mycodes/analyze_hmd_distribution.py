"""
Analyze HMD distribution in det_123 dataset (three cases)
"""
import argparse
from pathlib import Path
from collections import defaultdict

def count_classes_in_label(label_file: Path) -> tuple[bool, bool]:
    """
    Read YOLO label file and check if it contains Mentum (class 0) and Hyoid (class 1)
    
    Returns:
        (has_mentum, has_hyoid): Whether Mentum and Hyoid are present
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
    Simple YAML file parser (without using yaml library)
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
    Analyze dataset and count distribution of HMD three cases
    """
    # Read YAML file
    config = parse_yaml(yaml_file)
    
    dataset_path = Path(config['path'])
    print(f"\n{'='*80}")
    print(f"Analyzing dataset: {yaml_file.name}")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*80}\n")
    
    # Count statistics for each split
    splits = ['train', 'val', 'test']
    total_stats = {
        'case1_both': 0,      # Case 1: Both present
        'case2_single': 0,    # Case 2: Only one present
        'case3_none': 0,      # Case 3: Neither present
        'total': 0
    }
    
    # Get split filename mapping
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
        
        print(f"\n📊 Analyzing {split} set...")
        
        # Read image list
        image_paths = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    image_paths.append(Path(line))
        
        print(f"   Number of images: {len(image_paths)}")
        
        # Count statistics
        split_stats = {
            'case1_both': 0,
            'case2_single': 0,
            'case3_none': 0,
            'total': len(image_paths)
        }
        
        for img_path_str in image_paths:
            # Process path (could be string or Path object)
            if isinstance(img_path_str, str):
                img_path = Path(img_path_str)
            else:
                img_path = img_path_str
            
            # Find corresponding label file
            # YOLO format: image in patient_data/xxx/xxx.png, label in same directory
            if img_path.is_absolute():
                # Absolute path: label in same directory
                label_path = img_path.parent / f"{img_path.stem}.txt"
            else:
                # Relative path: relative to dataset_path
                label_path = dataset_path / img_path.parent / f"{img_path.stem}.txt"
            
            # If not found, try other possible paths
            if not label_path.exists():
                # Try to infer from image path (patient_data/xxx/xxx.png -> patient_data/xxx/xxx.txt)
                img_str = str(img_path)
                if 'patient_data' in img_str:
                    # Replace extension
                    label_path = Path(img_str.rsplit('.', 1)[0] + '.txt')
                else:
                    # Try to find in dataset_path
                    label_path = dataset_path / 'labels' / f"{img_path.stem}.txt"
            
            # Check classes
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
        
        # Print split statistics
        print(f"   Case 1 (both present): {split_stats['case1_both']:6d} ({split_stats['case1_both']/split_stats['total']*100:5.2f}%)")
        print(f"   Case 2 (only one):     {split_stats['case2_single']:6d} ({split_stats['case2_single']/split_stats['total']*100:5.2f}%)")
        print(f"   Case 3 (neither):      {split_stats['case3_none']:6d} ({split_stats['case3_none']/split_stats['total']*100:5.2f}%)")
        print(f"   Total:                 {split_stats['total']:6d}")
    
    # Print overall statistics
    print(f"\n{'='*80}")
    print(f"📈 Overall Statistics ({yaml_file.name}):")
    print(f"{'='*80}")
    print(f"Case 1 (both present): {total_stats['case1_both']:6d} ({total_stats['case1_both']/total_stats['total']*100:5.2f}%)")
    print(f"Case 2 (only one):     {total_stats['case2_single']:6d} ({total_stats['case2_single']/total_stats['total']*100:5.2f}%)")
    print(f"Case 3 (neither):      {total_stats['case3_none']:6d} ({total_stats['case3_none']/total_stats['total']*100:5.2f}%)")
    print(f"Total:                 {total_stats['total']:6d}")
    print(f"{'='*80}\n")
    
    return total_stats

def main():
    parser = argparse.ArgumentParser(description='Analyze HMD distribution (three cases) in det_123 dataset')
    parser.add_argument('--yaml-dir', type=str, default='yolo_dataset/det_123/v3',
                       help='Directory containing YAML files')
    args = parser.parse_args()
    
    yaml_dir = Path(args.yaml_dir)
    
    # Analyze two YAML files
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
    
    # Comparison summary
    if len(all_stats) == 2:
        print(f"\n{'='*80}")
        print(f"📊 Comparison Summary")
        print(f"{'='*80}")
        print(f"{'Dataset':<20} {'Case 1':>12} {'Case 2':>12} {'Case 3':>12} {'Total':>12}")
        print(f"{'-'*80}")
        for name, stats in all_stats.items():
            print(f"{name:<20} {stats['case1_both']:>8} ({stats['case1_both']/stats['total']*100:>5.1f}%) "
                  f"{stats['case2_single']:>8} ({stats['case2_single']/stats['total']*100:>5.1f}%) "
                  f"{stats['case3_none']:>8} ({stats['case3_none']/stats['total']*100:>5.1f}%) "
                  f"{stats['total']:>12}")
        print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

