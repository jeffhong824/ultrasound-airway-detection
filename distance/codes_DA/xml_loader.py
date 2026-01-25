import os
import argparse
import yaml
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import pydicom as dicom
import joblib

import sys
sys.path.append(os.path.dirname(__file__))
from utils import scan_dcm
from utils import GeofitCircle

def verify_points(pts):
    # remove duplicate points (前後兩點的座標完全相同，移除以避免發生數值錯誤)
    lg = np.r_[True, (pts[1:]!=pts[:-1]).any(axis=1)]
    pts = pts[lg]
    assert np.unique(pts, axis=0).shape[0]==pts.shape[0], 'Some points are repeated!'
    return pts

def add_points(polylines, df, sframe, col, chk_overlap):
    if chk_overlap:
        for polyline in polylines:
            if polyline.get('outside')=='0':
                frame = int(polyline.get('frame')) - sframe
                assert isinstance(df.at[frame, col],float) and pd.isna(df.at[frame, col]), f'Track overlapped! please check label: {col}, frame: {frame}'
                pts = np.fromstring(polyline.get('points').replace(';',','), dtype=float, sep=',').reshape(-1,2)
                try:
                    pts = verify_points(pts)
                except AssertionError as err:
                    err.args = ('Frame: {}\n{}'.format(frame, err.args[0]), )
                    raise err
                df.at[frame, col] = pts
    else:
        for polyline in polylines:
            if polyline.get('outside')=='0':
                frame = int(polyline.get('frame')) - sframe
                pts = np.fromstring(polyline.get('points').replace(';',','), dtype=float, sep=',').reshape(-1,2)
                try:
                    pts = verify_points(pts)
                except AssertionError as err:
                    err.args = ('Frame: {}\n{}'.format(frame, err.args[0]), )
                    raise err
                df.at[frame, col] = pts


def add_box(boxes, df, sframe, label_name, chk_overlap):
    if chk_overlap:
        for box in boxes:
            if box.get('outside')=='0':
                frame = int(box.get('frame')) - sframe
                for col in ('xtl','ytl','xbr','ybr'):
                    assert pd.isna(df.at[frame, f'{label_name}_{col}']), f'Track overlapped! please check label: {label_name}, frame: {frame}'
                    df.at[frame, f'{label_name}_{col}'] = float(box.get(col))
    else:
        for box in boxes:
            if box.get('outside')=='0':
                frame = int(box.get('frame')) - sframe
                for col in ('xtl','ytl','xbr','ybr'):
                    df.at[frame, f'{label_name}_{col}'] = float(box.get(col))


def convex_points(X):
    height, width = X.shape
    xc = round(width/2)
    xrng = round(width*0.31) # image中心線，左右31%範圍
    ymax = round(height*0.20) # 從頂端往下20%範圍
    xmin = xc - xrng
    xmax = xc + xrng
    tl_y = np.nonzero(np.any(X[:ymax, xmin:xc], axis=1))[0][0]
    tl_x = np.nonzero(X[tl_y, xmin:xc])[0][0] + xmin
    tr_y = np.nonzero(np.any(X[:ymax, xc:xmax], axis=1))[0][0]
    tr_x = np.nonzero(X[tr_y, xc:xmax])[0][0] + xc
    ymax = round((tl_y+tr_y)/2 + width*0.12) # 假設最低點在最高點往下12%的範圍內
    row, col = np.nonzero(np.diff(X[:ymax, tl_x:tr_x+1], axis=0)>0)
    ix = np.argsort(col)
    py = row[ix] + 1
    px = col[ix] + tl_x
    assert np.all(np.diff(px) <= 1), 'convex_points may not correct'
    return px, py

def load_dicom_images(img_file_path):
    if os.path.splitext(img_file_path)[1].lower()=='.dcm':
        ds = dicom.dcmread(img_file_path)
        img_array = ds.pixel_array # (n_frame, height, width, RGB)
        if ds.PhotometricInterpretation!='RGB': # (0x0028,0x0004)
            if ds.PhotometricInterpretation in {'YBR_FULL','YBR_FULL_422'}:
                img_array = dicom.pixel_data_handlers.convert_color_space(img_array, ds.PhotometricInterpretation, 'RGB')
            else:
                raise AssertionError(f'Unexpect Photometric Interpretation value (0028, 0004): {ds[0x0028,0x0004].value}')
    else:
        raise AssertionError('Unexpect image file extension')
    return img_array

def detect_start_row(images):
    # img_array: (n_frame, height, width, RGB)
    start_row = 0
    tf = images[:,start_row,:,:].astype(bool).any(axis=2).any(axis=0)
    if not tf[0]: # 超音波畫面沒有從第一列開始，從左側找第一格刻度(0)出現的位置
        lg = images[:,:,1,:].astype(bool).any(axis=2).any(axis=0)
        lg = np.concatenate(([False], lg))
        diff = np.diff(lg.astype(int))
        start_row = np.nonzero(diff==1)[0][0]
        tf = images[:,start_row,:,:].astype(bool).any(axis=2).any(axis=0)
        assert tf[0], 'Unexpect image region'

        idx = np.nonzero(np.diff(tf[10:]))[0]+10
        assert idx.size % 2 == 0, 'Unexpect range'
        s = (idx[1::2] - idx[::2]).argmax() # 找出最寬的區間
        six = idx[2*s]+1
        eix = idx[2*s+1]
        assert not images[:,:start_row,six:eix+1,:].astype(bool).reshape(-1).any(), 'start_row判斷不準確，要再確定DA_metrics的計算'
    return start_row

def load_project(xml_path, img_path, img_name, probeCR=dict()):
    prj_name = os.path.splitext(os.path.basename(xml_path))[0]
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_nodes = set((c.tag for c in root))
    assert 'track' in xml_nodes, f'No track found, the task may not be annotated yet: {xml_path}'
    assert xml_nodes=={'version','meta','track'}, f'Unexpect annotation file: {xml_path}'
    # load meta data
    prj = root.find('meta').find('project')
    tasks = dict()
    cumframe = 0
    for task in prj.find('tasks').findall('task'):
        id = int(task.find('id').text)
        name = os.path.splitext(task.find('name').text)[0]
        n_frame = int(task.find('size').text)
        im_width = int(task.find('original_size').find('width').text)
        im_height = int(task.find('original_size').find('height').text)
        if len(task.find('assignee'))==0:
            assignee = ''
        else:
            assignee = task.find('assignee').find('username').text
        tasks[id] = {'name':name, 'n_frame':n_frame, 'sframe':cumframe, 'im_width':im_width, 'im_height':im_height, 'assignee':assignee}
        cumframe += n_frame
    # load track
    cache_file = os.path.join(os.path.dirname(__file__),'probeCR.cache')
    if not probeCR and os.path.isfile(cache_file):
        probeCR = joblib.load(cache_file)
    save_cache = False
    for track in root.findall('track'):
        task_id = int(track.get('task_id'))
        # print('task_id:', task_id)
        task_name = tasks[task_id]['name']
        n_frame = tasks[task_id]['n_frame']
        sframe = tasks[task_id]['sframe']
        label_name = track.get('label').replace('. ','')
        label_type = track[0].tag
        # frames = [int(box.get('frame'))-sframe for box in track.findall('box') if box.get('outside')=='0']
        try:
            if label_name in {'1Chin&Hyoid (neutral)', '2Chin&Hyoid (extended)', '3Chin&Hyoid (ramped)', '4Tongue_Upper', '5Tongue_Lower'}:
                if 'probe_center' not in tasks[task_id]:
                    if task_name not in probeCR:
                        assert task_name in img_name, f'{task_name}.dcm not found, please check dcm_folders'
                        img_file_path = img_path[img_name.index(task_name)]
                        img_array = load_dicom_images(img_file_path)
                        X = np.any(np.any(img_array,axis=3),axis=0) # RGB不為零→任一個frame曾經有值
                        X = np.cumsum(X, axis=0).astype(bool) # 同一column由上到下曾經不為零
                        px, py = convex_points(X)
                        assert len(px) > 0, 'no convex points found, maybe a linear probe?'
                        C, R = GeofitCircle(np.stack((px, py), axis=1))
                        probeCR[task_name] = (C[0], R) # dimension of C (1x2) -> (2,)
                        save_cache = True
                    tasks[task_id]['probe_center'] = probeCR[task_name][0]
                    tasks[task_id]['probe_radius'] = probeCR[task_name][1]
            else:
                if 'start_row' not in tasks[task_id]:
                    img_file_path = img_path[img_name.index(task_name)]
                    img_array = load_dicom_images(img_file_path)
                    tasks[task_id]['start_row'] = detect_start_row(img_array)
            if label_type=='polyline':
                if 'anno' in tasks[task_id]:
                    if label_name in tasks[task_id]['anno'].columns:
                        df = tasks[task_id]['anno']
                        add_points(track.findall('polyline'), df, sframe, label_name, chk_overlap=True)
                    else:
                        df = pd.DataFrame(index=range(n_frame), columns=[label_name])
                        add_points(track.findall('polyline'), df, sframe, label_name, chk_overlap=False)
                        tasks[task_id]['anno'] = pd.concat([tasks[task_id]['anno'], df], axis=1)
                else:
                    df = pd.DataFrame(index=range(n_frame), columns=[label_name])
                    add_points(track.findall('polyline'), df, sframe, label_name, chk_overlap=False)
                    tasks[task_id]['anno'] = df
                assert tasks[task_id]['anno'].index[-1]==n_frame-1, 'Unexpect error'
            elif label_type=='box':
                if label_name in {'1Chin&Hyoid (neutral)', '2Chin&Hyoid (extended)', '3Chin&Hyoid (ramped)'}:
                    # 將Mentum與Hyoid分開
                    pose = label_name.split(' ')[1][1:-1]
                    xbr_xtl = [(float(t.get('xbr')), float(t.get('xtl'))) for t in track if t.get('outside')=='0']
                    xbr, xtl = zip(*xbr_xtl)
                    if task_name in {'0583808_Quick ID_20240805_113137_B _Neutral','6231089_Quick ID_20250313_145437_B_Neutral','7317014_Quick ID_20250515_132004_B _Neutral','7146306_Quick ID_20240613_133830_B','0583808_Quick ID_20240805_113236_B _Ramped','5095917_Quick ID_20240829_135640_B_Neutral','5849309_Quick ID_20241107_150213_B_Ramped'}: # 這個task真的偏離probe_center太遠
                        p_Mentum = sum(xbr < tasks[task_id]['probe_center'][0]-100) / len(xbr)
                        p_Hyoid = sum(xtl > tasks[task_id]['probe_center'][0]-100) / len(xtl)
                    elif task_name=='2783766_Quick ID_20250227_150912_B_Neutral': # 這個task Hyoid位置幾乎都在左邊
                        p_Mentum = sum(xbr < tasks[task_id]['probe_center'][0]-150) / len(xbr)
                        p_Hyoid = sum(xtl > tasks[task_id]['probe_center'][0]-150) / len(xtl)
                    elif task_name in {'2164249_Quick ID_20250109_160139_B','4424158_Quick ID_20250428_094202_B_Ramped'}: # 這個task Hyoid位置太中間
                        box_center = [(a+b)/2 for a,b in zip(xbr, xtl)]
                        p_Mentum = sum(box_center < tasks[task_id]['probe_center'][0]) / len(box_center)
                        p_Hyoid = sum(box_center > tasks[task_id]['probe_center'][0]) / len(box_center)
                    elif task_name=='4056657_Quick ID_20240808_143909_B_Extended':
                        p_Mentum = sum(xbr < tasks[task_id]['probe_center'][0]+50) / len(xbr)
                        p_Hyoid = sum(xtl > tasks[task_id]['probe_center'][0]+50) / len(xtl)
                    elif task_name=='2539366_Quick ID_20250213_140356_B' and len(xbr)==9: # 這個task Hyoid有一段只出現在正中央
                        p_Mentum = 0
                        p_Hyoid = 1
                    elif task_name=='6231089_Quick ID_20250313_145510_B_Extended':
                        if len(xbr)==19:
                            p_Mentum = 1
                            p_Hyoid = 0
                        else:
                            p_Mentum = 0
                            p_Hyoid = 1
                    elif task_name=='6231089_Quick ID_20250313_145555_B_Ramped':
                        if len(xbr) in {4, 11}:
                            p_Mentum = 1
                            p_Hyoid = 0
                        else:
                            p_Mentum = 0
                            p_Hyoid = 1
                    elif task_name=='2596378_Quick ID_20250407_095220_B_Romped':
                        if len(xbr)==n_frame:
                            p_Mentum = 0
                            p_Hyoid = 1
                        else:
                            p_Mentum = 1
                            p_Hyoid = 0
                    else:
                        p_Mentum = sum(xbr < tasks[task_id]['probe_center'][0]) / len(xbr)
                        p_Hyoid = sum(xtl > tasks[task_id]['probe_center'][0]) / len(xtl)
                    if p_Mentum > p_Hyoid and p_Mentum > 0.5:
                        label_name = f'Mentum-{pose}'
                    elif p_Hyoid > p_Mentum and p_Hyoid >= 0.5: # 1516553_Quick ID_20241024_154935_B_Ramped只有兩個frame
                        label_name = f'Hyoid-{pose}'
                    # if float(track[0].get('xbr')) < tasks[task_id]['probe_center'][0]:
                    #     label_name = f'Mentum-{pose}'
                    # elif float(track[0].get('xtl')) > tasks[task_id]['probe_center'][0]:
                    #     label_name = f'Hyoid-{pose}'
                    else:
                        raise AssertionError('Unexpect mentum or hyoid position')
                cols = [f'{label_name}_{col}' for col in ('xtl','ytl','xbr','ybr')]
                if 'anno' in tasks[task_id]:
                    if cols[0] in tasks[task_id]['anno'].columns:
                        df = tasks[task_id]['anno']
                        add_box(track.findall('box'), df, sframe, label_name, chk_overlap=True)
                    else:
                        df = pd.DataFrame(index=range(n_frame), columns=cols, dtype=float)
                        add_box(track.findall('box'), df, sframe, label_name, chk_overlap=False)
                        tasks[task_id]['anno'] = pd.concat([tasks[task_id]['anno'], df], axis=1)
                else:
                    df = pd.DataFrame(index=range(n_frame), columns=cols, dtype=float)
                    add_box(track.findall('box'), df, sframe, label_name, chk_overlap=False)
                    tasks[task_id]['anno'] = df
                assert tasks[task_id]['anno'].index[-1]==n_frame-1, 'Unexpect error'
            else:
                print('Warning: [{}] ({}-{}). Skipping unrecognize type: {}. Only support polyline and box currently.'.format(prj_name, tasks[task_id]['name'], label_name, label_type))
        except AssertionError as err:
            print(f'Error happened in project: {prj_name}, task: {task_name}')
            err.args = ('Project: {}, Task: {}, Label: {}\n{}'.format(prj_name, task_name, label_name, err.args[0]), )
            raise err
    if save_cache:
        joblib.dump(probeCR, cache_file, compress=3, protocol=4)
    return tasks, probeCR

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('patient', type=str, help='patient ID')
    args = parser.parse_args(['5849309'])

    DA_folder = '/root/ultrasound/DifficultAirway'
    # load apply_folders.yaml
    yaml_path = os.path.join(DA_folder, 'CVAT_annotations', 'apply_folders.yaml')
    with open(yaml_path, 'rt') as f:
        apply_folders = yaml.safe_load(f)
    
    xml_folders = [os.path.join(DA_folder,'CVAT_annotations',subfolder) for gp in ('DA','ES') for subfolder in apply_folders[gp]]
    
    dcm_folders = (
        os.path.join(DA_folder,'困難'),
        os.path.join(DA_folder,'非困難'),
        os.path.join(DA_folder,'尚未判定'),
        os.path.join(DA_folder,'待取得分類結果'),
        os.path.join(DA_folder,'內視鏡')
    )

    xml_path = next((f.path for xml_folder in xml_folders for f in os.scandir(xml_folder) if f.name==f'{args.patient}.xml'), None)
    if xml_path:
        dcm_path, dcm_name = zip(*scan_dcm(dcm_folders))
        tasks, probeCR = load_project(xml_path, dcm_path, dcm_name)
    else:
        print(f'No xml file found for patient: {args.patient}')