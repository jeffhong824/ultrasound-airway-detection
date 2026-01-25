"""
Box output format:
Use model.val to output xywh results (save_json=True)
If predicting entire patient video, use model.predict output

Mask output format:
Use model.predict and organize, output xy results
Convert json format to pandas.DataFrame output, columns as follows (segment models only output pts):
{
    'image_id': int,
    'category_id': int,
    'bbox': [],
    'score': float,
    'pts': [
        [x1, y1],
        [x2, y2],
        ...
    ],
}
"""
import os
import argparse
import shutil
import pydicom as dicom
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from ultralytics import YOLO

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from ultralytics directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, skip
    pass

import sys
# Add codes_DA path (relative to PROJECT_ROOT)
PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT:
    codes_da_path = Path(PROJECT_ROOT) / 'distance' / 'codes_DA'
    if codes_da_path.exists():
        sys.path.append(str(codes_da_path))
try:
    from xml_loader import load_project
except ImportError:
    print("Warning: xml_loader not found, some functions may not work")
    load_project = None

def locate_image_files(img_folders, subIDs, ext={'.dcm'}):
    # Syntax
    #       sub_dcm_folders, sub_dcm_files = locate_subject_files(image_folders, subIDs, {'.dcm','.mp4'})
    subfolders = dict.fromkeys(subIDs, '')
    img_files = dict.fromkeys(subIDs, '')
    for folder in img_folders:
        got_subID = []
        for sub in subIDs:
            subfolder = os.path.join(folder, f'{sub}_Quick ID')
            if os.path.isdir(subfolder):
                subfolders[sub] = subfolder
                img_files[sub] = [f.name for f in os.scandir(subfolder) if f.name.startswith(sub) and os.path.splitext(f.name)[1] in ext]
                got_subID.append(sub)
        subIDs = [sub for sub in subIDs if sub not in got_subID]
        if not subIDs:
            break
    if next((k for k,v in subfolders.items() if v==''), None) is not None:
        miss_subID = [k for k,v in subfolders.items() if v=='']
        raise AssertionError('Some subject not found, please check image folders: {}'.format(', '.join(miss_subID)))
    if next((k for k,v in img_files.items() if v==[]), None) is not None:
        miss_subID = [k for k,v in img_files.items() if v==[]]
        raise AssertionError('Some subject have no image file, please check: {}'.format(', '.join(miss_subID)))
    return subfolders, img_files

def locate_xml_files(xml_folders, subIDs):
    xml_paths = dict.fromkeys(subIDs, '')
    for folder in xml_folders:
        got_subID = []
        for sub in subIDs:
            xml_path = os.path.join(folder, f'{sub}.xml')
            if os.path.isfile(xml_path):
                xml_paths[sub] = xml_path
                got_subID.append(sub)
        subIDs = [sub for sub in subIDs if sub not in got_subID]
        if not subIDs:
            break
    if next((k for k,v in xml_paths.items() if v==''), None) is not None:
        miss_subID = [k for k,v in xml_paths.items() if v=='']
        raise AssertionError('Some subject xml file not found: {}'.format(', '.join(miss_subID)))
    return xml_paths


if __name__=='__main__':
    # define path and landmarks, modify if needed
    # Get PROJECT_ROOT from environment variable
    DA_folder = os.getenv('PROJECT_ROOT')
    if not DA_folder:
        raise ValueError('PROJECT_ROOT not found in environment variables. Please set it in ultralytics/.env file.')
    DA_folder = Path(DA_folder)
    image_folders = ( # list order according to possible patient number (descending)
        str(DA_folder / 'dicom_dataset' / '內視鏡'),
        str(DA_folder / 'dicom_dataset' / '非困難'),
        str(DA_folder / 'dicom_dataset' / '困難'),
        str(DA_folder / 'dicom_dataset' / '尚未判定'),
    )
    xml_folders = (
        str(DA_folder / 'CVAT_annotations' / 'DA_group1'),
        str(DA_folder / 'CVAT_annotations' / 'DA_group2'),
        str(DA_folder / 'CVAT_annotations' / 'DA_group3'),
        str(DA_folder / 'CVAT_annotations' / 'DA_group5'),
        str(DA_folder / 'CVAT_annotations' / '內視鏡2'),
        str(DA_folder / 'CVAT_annotations' / '內視鏡3'),
    )
    landmarks = {
        'det_123': (
            (0, 'Mentum', 'Mentum-neutral_xtl'),
            (0, 'Mentum', 'Mentum-extended_xtl'),
            (0, 'Mentum', 'Mentum-ramped_xtl'),
            (1, 'Hyoid', 'Hyoid-neutral_xtl'),
            (1, 'Hyoid', 'Hyoid-extended_xtl'),
            (1, 'Hyoid', 'Hyoid-ramped_xtl'),
        ),
        'seg_45': (
            (0, 'Tongue_Upper', '4Tongue_Upper'),
            (1, 'Tongue_Lower', '5Tongue_Lower'),
        ),
        'det_678': (
            (0, 'Hyoid_Bone', '6Hyoid Bone_ytl'),
            (1, 'Epiglottis', '7Epiglottis_ytl'),
            (2, 'Epiglottis2VC', '8Epiglottis-to-Vocal Cord_ytl'),
        )
    }


    parser = argparse.ArgumentParser()
    parser.add_argument('runs_type', type=str, default='detect', choices=('detect','segment'))
    parser.add_argument('runs_num', type=str, default='1', nargs='?', help='Run number (use empty string "" or omit for default)')
    parser.add_argument('database', type=str, choices=('det_123','seg_45','det_678'))
    parser.add_argument('--db_version', type=int, default=1, choices=(1,2,3))
    parser.add_argument('--model', type=str, default='best', choices=('best','last'))
    parser.add_argument('--dev', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subID', type=str, nargs='+', default=None) # inference all subject in testing set if assign 'all'
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights file (overrides runs_type/runs_num/model)')
    parser.add_argument('--output-name', type=str, default=None, help='Output folder name (e.g., "test_exp1" instead of "test2")')
    args = parser.parse_args()
    # args = parser.parse_args('segment 1 seg_45 --subID all'.split()) # TODO: Remove after debugging
    
    # Handle runs_num: convert to empty string if '1' or empty, otherwise keep as string
    if args.runs_num is None or args.runs_num == '' or args.runs_num == '1':
        args.runs_num = ''
    else:
        # Try to validate it's a number if not empty
        try:
            int(args.runs_num)
        except ValueError:
            raise ValueError(f"runs_num must be a number or empty string, got: {args.runs_num}")
    
    # load model
    assert DA_folder.exists() and DA_folder.is_dir(), f'DA_folder not exist: {DA_folder}'
    
    # Use specified weights file if provided, otherwise use default path
    if args.weights:
        mdl_file = Path(args.weights)
        # If relative path, try multiple locations
        if not mdl_file.is_absolute():
            # First try: relative to current working directory
            cwd_file = Path.cwd() / mdl_file
            if cwd_file.exists():
                mdl_file = cwd_file.resolve()
            # Second try: relative to PROJECT_ROOT
            elif (DA_folder / mdl_file).exists():
                mdl_file = DA_folder / mdl_file
            # Third try: relative to PROJECT_ROOT/ultralytics
            elif (DA_folder / 'ultralytics' / mdl_file).exists():
                mdl_file = DA_folder / 'ultralytics' / mdl_file
            # Default: relative to PROJECT_ROOT
            else:
                mdl_file = DA_folder / mdl_file
    else:
        # Default path structure (try new structure first, then old)
        # New structure: ultralytics/runs/train/...
        mdl_file_new = DA_folder / 'ultralytics' / 'runs' / args.runs_type / f'train{args.runs_num}' / 'weights' / f'{args.model}.pt'
        # Old structure: ultralytics-8.3.159/runs/...
        mdl_file_old = DA_folder / 'ultralytics-8.3.159' / 'runs' / args.runs_type / f'train{args.runs_num}' / 'weights' / f'{args.model}.pt'
        
        if mdl_file_new.exists():
            mdl_file = mdl_file_new
        elif mdl_file_old.exists():
            mdl_file = mdl_file_old
        else:
            mdl_file = mdl_file_new  # Use new structure as default
    
    assert mdl_file.exists() and mdl_file.is_file(), f'model file not found: {mdl_file}'
    print(f"Loading model from: {mdl_file}")
    model = YOLO(str(mdl_file))

    if args.subID is None:
        # perform validation
        yaml_file = DA_folder / 'yolo_dataset' / args.database / f'v{args.db_version}' / f'{args.database}.yaml'
        assert yaml_file.exists() and yaml_file.is_file(), f'DB YAML file not found: {yaml_file}'
        prj = DA_folder / 'ultralytics' / 'runs' / args.runs_type

        # Determine output folder name
        if args.output_name:
            test_folder_name = args.output_name
        else:
            test_folder_name = f'test{args.runs_num}'
        
        dst_folder = prj / test_folder_name
        if dst_folder.exists() and dst_folder.is_dir():
            shutil.rmtree(dst_folder)
        # Create output directory
        dst_folder.mkdir(parents=True, exist_ok=True)
        
        # model validation using testing set
        model.val(data=str(yaml_file), imgsz=640, batch=args.batch_size, device=args.dev, split='test', name=test_folder_name, verbose=False)

        # export testing set results
        is_det = args.database[:3]=='det'
        # load list of image paths
        txt_path = DA_folder / 'yolo_dataset' / args.database / f'v{args.db_version}' / 'test.txt'
        assert txt_path.exists() and txt_path.is_file(), f'file not found: {txt_path}'
        with open(txt_path, 'rt') as f:
            imgs = f.read().rstrip().split('\n')
        # batch processing
        batch_idx = list(range(0, len(imgs), args.batch_size))
        batch_idx.append(len(imgs))
        output = []
        for b in range(len(batch_idx)-1):
            sli = slice(batch_idx[b], batch_idx[b+1])
            results = model.predict(source=imgs[sli], conf=0.0, imgsz=640, device=args.dev, batch=args.batch_size, max_det=300, verbose=False)
            for result in results:
                data = {
                    'cls': result.boxes.cls.int().tolist(),
                    'conf': result.boxes.conf.tolist(),
                }
                gp = pd.DataFrame(data).groupby('cls')['conf'].idxmax() # Find index with highest confidence for each class
                if is_det: # detection task
                    for c in gp.index:
                        output.append({
                            'image_id': os.path.splitext(os.path.basename(result.path))[0],
                            'category_id': c,
                            'bbox': result.boxes[gp[c]].xyxy[0].tolist(), # xywh is actually cx cy w h, not convenient to use
                            'score': result.boxes[gp[c]].conf.item(),
                        })
                else: # should be segmentation task
                    for c in gp.index:
                        output.append({
                            'image_id': os.path.splitext(os.path.basename(result.path))[0],
                            'category_id': c,
                            'bbox': result.boxes[gp[c]].xyxy[0].tolist(), # xywh is actually cx cy w h, not convenient to use
                            'score': result.boxes[gp[c]].conf.item(),
                            'pts': result.masks[gp[c]].xy[0].tolist(),
                        })
        joblib_path = dst_folder / 'predictions.joblib'
        # Ensure directory exists before saving
        joblib_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pd.DataFrame.from_records(output), joblib_path, compress=3, protocol=4)
        print(f"✅ Predictions saved to: {joblib_path}")
        # if args.database[:3]=='det':
        #     model.val(data=yaml_file, imgsz=640, batch=args.batch_size, save_json=True, conf=0.0, max_det=300, device=args.dev, split='test', name=f'test{args.runs_num}')
        # elif args.database[:3]=='seg':
        #     os.mkdir(dst_folder)
        #     # load list of image paths
        #     txt_path = os.path.join(DA_folder, 'yolo_dataset', args.database, f'v{args.db_version}', 'test.txt')
        #     assert os.path.isfile(txt_path), f'file not found: {txt_path}'
        #     with open(txt_path, 'rt') as f:
        #         imgs = f.read().rstrip().split('\n')
        #     # batch processing
        #     batch_idx = list(range(0, len(imgs), args.batch_size))
        #     batch_idx.append(len(imgs))
        #     output = []
        #     for b in range(len(batch_idx)-1):
        #         sli = slice(batch_idx[b], batch_idx[b+1])
        #         results = model.predict(source=imgs[sli], conf=0.0, imgsz=640, device=args.dev, batch=args.batch_size, max_det=300)
        #         for result in results:
        #             data = {
        #                 'cls': result.boxes.cls.int().tolist(),
        #                 'conf': result.boxes.conf.tolist(),
        #             }
        #             gp = pd.DataFrame(data).groupby('cls')['conf'].idxmax() # Find index with highest confidence for each class
        #             for c in gp.index:
        #                 output.append({
        #                     'image_id': os.path.splitext(os.path.basename(result.path))[0],
        #                     'category_id': c,
        #                     'bbox': result.boxes[gp[c]].xyxy[0].tolist(), # xywh is actually cx cy w h, not convenient to use
        #                     'score': result.boxes[gp[c]].conf.item(),
        #                     'pts': result.masks[gp[c]].xy[0].tolist(),
        #                 })
        #     joblib_path = os.path.join(dst_folder, 'predictions.joblib')
        #     joblib.dump(pd.DataFrame.from_records(output), joblib_path, compress=3, protocol=4)
        # else:
        #     raise AssertionError('Unexpect database name')

    else: # inference by individual video
        if args.subID==['all']:
            txt_path = DA_folder / 'yolo_dataset' / args.database / f'v{args.db_version}' / 'subID_test.txt'
            with open(txt_path,'rt') as f:
                args.subID = f.read().rstrip().split('\n')
        img_folders, img_files = locate_image_files(image_folders, args.subID, ext={'.dcm','.mp4'})
        xml_paths = locate_xml_files(xml_folders, args.subID)
        # remove mp4 files convert from dicom
        for sub in args.subID:
            img_files[sub] = [f for f in img_files[sub] if f[-4:]=='.dcm' or (f[-4:]=='.mp4' and f[:-4]+'.dcm' not in img_files[sub])]

        # remove image files not suit for current model
        for sub in args.subID:
            img_name = [os.path.splitext(file)[0] for file in img_files[sub]]
            db_subfolder = DA_folder / 'yolo_dataset' / args.database / f'v{args.db_version}' / 'patient_data' / sub
            sub_img = dict.fromkeys([f.name[:-9] for f in os.scandir(db_subfolder) if f.name[-4:]=='.txt'])
            img_files[sub] = [img_files[sub][img_name.index(name)] for name in sub_img]

        # inference one by one
        is_det = args.database[:3]=='det'
        
        # Determine output folder name
        if args.output_name:
            test_folder_name = args.output_name
        else:
            test_folder_name = f'test{args.runs_num}'
        
        for sub in args.subID:
            dst_folder = DA_folder / 'ultralytics' / 'runs' / args.runs_type / test_folder_name / sub
            if dst_folder.exists() and dst_folder.is_dir():
                shutil.rmtree(dst_folder)
            dst_folder.mkdir(parents=True, exist_ok=True)
            for img_file in img_files[sub]:
                if img_file[-4:]=='.dcm':
                    dcm_path = os.path.join(img_folders[sub], img_file)
                    ds = dicom.dcmread(dcm_path)
                    images = ds.pixel_array # (n_frame, height, width, RGB)
                    if ds.PhotometricInterpretation!='RGB': # (0x0028,0x0004)
                        if ds.PhotometricInterpretation in {'YBR_FULL','YBR_FULL_422'}:
                            images = dicom.pixel_data_handlers.convert_color_space(images, ds.PhotometricInterpretation, 'RGB')
                        else:
                            raise AssertionError(f'Unexpect Photometric Interpretation value (0028, 0004): {ds[0x0028,0x0004].value}')
                    images = np.flip(images, axis=3) # RGB -> BGR
                elif img_file[-4:]=='.mp4':
                    raise AssertionError('Cannot process mp4 file currently')
                else:
                    raise AssertionError('Unexpect image format')

                batch_idx = list(range(0, images.shape[0], args.batch_size))
                batch_idx.append(images.shape[0])
                output = []
                for b in range(len(batch_idx)-1):
                    source = [images[i] for i in range(batch_idx[b], batch_idx[b+1])]
                    results = model.predict(source=source, conf=0.0, imgsz=640, device=args.dev, batch=args.batch_size, max_det=300, verbose=False)
                    for i,result in enumerate(results):
                        data = {
                            'cls': result.boxes.cls.int().tolist(),
                            'conf': result.boxes.conf.tolist(),
                        }
                        gp = pd.DataFrame(data).groupby('cls')['conf'].idxmax() # Find index with highest confidence for each class
                        if is_det: # detection task
                            for c in gp.index:
                                xyxy = result.boxes[gp[c]].xyxy[0].tolist()
                                output.append({
                                    'image_id': batch_idx[b] + i,
                                    'category_id': c,
                                    'bbox': result.boxes[gp[c]].xyxy[0].tolist(), # xywh is actually cx cy w h, not convenient to use
                                    'score': result.boxes[gp[c]].conf.item(),
                                })
                        else: # should be segmentation task
                            for c in gp.index:
                                xyxy = result.boxes[gp[c]].xyxy[0].tolist()
                                output.append({
                                    'image_id': batch_idx[b] + i,
                                    'category_id': c,
                                    'bbox': result.boxes[gp[c]].xyxy[0].tolist(), # xywh is actually cx cy w h, not convenient to use
                                    'score': result.boxes[gp[c]].conf.item(),
                                    'pts': result.masks[gp[c]].xy[0].tolist(),
                                })
                joblib_path = dst_folder / (os.path.splitext(img_file)[0]+'.joblib')
                joblib.dump(pd.DataFrame.from_records(output), joblib_path, compress=3, protocol=4)