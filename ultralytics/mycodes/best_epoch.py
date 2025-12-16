# reference: https://github.com/ultralytics/ultralytics/issues/14137
import os
import argparse
import pandas as pd


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs_type', type=str, choices=('detect','segment'))
    parser.add_argument('runs_num', type=int, default=1)
    parser.add_argument('--run_name', type=str, default=None, help='Name of the training run directory')
    args = parser.parse_args()
    if args.runs_num==1:
        args.runs_num = ''

    # Updated path for Windows
    DA_folder = r'D:\workplace\project_management\github_project\ultrasound-airway-detection2'
    assert os.path.isdir(DA_folder), f'DA_folder not exist: {DA_folder}'
    
    # Use run_name if provided, otherwise use default pattern
    if args.run_name:
        csv_path = os.path.join(DA_folder, 'ultralytics', 'runs', 'train', args.run_name, 'results.csv')
    else:
        csv_path = os.path.join(DA_folder, 'ultralytics', 'runs', args.runs_type, f'train{args.runs_num}', 'results.csv')
    
    # Load the training log
    results = pd.read_csv(csv_path)

    # Strip spaces
    results.columns = results.columns.str.strip()

    # Calculate fitness
    Box_fitness = results["metrics/mAP50(B)"] * 0.1 + results["metrics/mAP50-95(B)"] * 0.9
    if args.runs_type=='detect':
        results["fitness"] = Box_fitness
    elif args.runs_type=='segment':
        Mask_fitness = results["metrics/mAP50(M)"] * 0.1 + results["metrics/mAP50-95(M)"] * 0.9
        results["fitness"] = Box_fitness + Mask_fitness


    # Find the epoch with the highest fitness
    idx = results['fitness'].idxmax()
    best_epoch = idx + 1
    fn = results.loc[idx, 'fitness']

    print(f"Best model was saved at epoch: {best_epoch}, fitness = {fn:.6f}")