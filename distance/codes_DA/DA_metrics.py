# annotation前處理流程:
#   1. python codes_DA/unzip_CVAT_xml.py
#   2. python codes_DA/DA_metrics.py <xml_subfolder>
#   3. python codes_DA/DA_metrics_availability.py
#   3. python codes_DA/analyze_annotation_results.py => 產生annotation_summary.xlsx


import os
import argparse
import pandas as pd
import numpy as np
import joblib
import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from gcvspline import gcvspline
from gcvspline import splderivative
# from datetime import date
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(__file__))
from utils import scan_dcm
from xml_loader import load_project

def locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub):
    task_id = [k for k,v in tasks.items() if all((mk in v['anno'].columns for mk in landmarks))]
    if len(task_id) > 1:
        logger.warning('Subject[{}]: More than one clip found: {}'.format(sub, ','.join(landmarks)))
    # assert len(task_id)==1, 'More than one clip found, please check!'
    task_id = task_id[0]
    task_name = tasks[task_id]['name']
    assert task_name in dcm_name, f'{task_name}.dcm not found, please check dcm_folders'
    dcmfile_path = dcm_path[dcm_name.index(task_name)]
    PixelSpacing = dcm_ps[task_name]['truePixelSpacing'] # mm/pixel
    ds = dicom.dcmread(dcmfile_path)
    return task_id, PixelSpacing, ds

def xy2rtheta(pts, center):
    # pts: [npts*2]
    # center: [1*2]
    v = pts - center.reshape(1,2)
    r = np.sqrt(np.sum(v**2, axis=1))
    theta = np.arctan2(v[:,1], v[:,0]) # unit: radians
    return r, theta

def fit_polar_curve(r, theta, probe_center, PixelSpacing, mintheta, maxtheta, N=101):
    assert theta.min() <= mintheta and maxtheta <= theta.max(), 'Invalid mintheta or maxtheta value'
    NP = r.size
    assert NP > 1, f'Unexpect point number: {NP}'
    thetas = np.linspace(mintheta, maxtheta, N)
    if theta[0] > theta[-1]:
        r = np.flipud(r)
        theta = np.flipud(theta)
    # if NP < 4: # splorder=2 (cubic), minimum require 4 samples. Use linear interpolation if data points less than requirement.
    #     px = r*np.cos(theta) # 以probe_center為原點的local座標
    #     py = r*np.sin(theta)
    #     # fit line
    #     cx = px.mean()
    #     cy = py.mean()
    #     px -= cx
    #     py -= cy
    #     slope = np.sum(px*py)/np.sum(px**2)
    #     # interpolate according to theta
    #     tantheta = np.tan(thetas)
    #     dx = (cx*tantheta - cy)/(slope-tantheta)
    #     xi = probe_center[0] + cx + dx
    #     yi = probe_center[1] + cy + dx*slope
    #     Ps = np.c_[xi, yi]
    if NP <= 4:
        # if NP=2: ix = [0, N]
        # if NP=3: ix = [0, next((k for k in range(N) if thetas[k] > theta[1]), N), N]
        ix = [0]
        m = 0
        for i in range(1,NP-1):
            m = next((k for k in range(m,N) if thetas[k] > theta[i]), N)
            ix.append(m)
        ix.append(N)
        xs = np.zeros(N)
        ys = np.zeros(N)
        for s in range(NP-1):
            px = r[s:s+2]*np.cos(theta[s:s+2])
            py = r[s:s+2]*np.sin(theta[s:s+2])
            # fit line
            slope = (py[1]-py[0])/(px[1]-px[0])
            # interpolate according to theta
            idx = slice(ix[s], ix[s+1])
            tantheta = np.tan(thetas[idx])
            dx = (px[0]*tantheta - py[0])/(slope-tantheta)
            xs[idx] = probe_center[0] + px[0] + dx
            ys[idx] = probe_center[1] + py[0] + dx*slope
        Ps = np.c_[xs, ys]
    elif NP > 4: # splorder=2 (cubic), minimum require 4 samples. Use linear interpolation if data points less than requirement.
        assert np.all(theta[1:] >= theta[:-1]), 'theta is not strictly ascending'
        ese = np.ones(NP)
        VAL = 2.5/PixelSpacing # tolerance: 2.5mm
        c = gcvspline(theta, r, ese, VAL, splmode=3)[0]
        rs = np.zeros(N)
        L = 0
        for i, thetai in enumerate(thetas):
            L = next((k for k in range(L, NP-1) if theta[k+1] > thetai), NP-1)
            rs[i] = splderivative(thetai, theta, c, L=L, IDER=0)[0]
        xs = probe_center[0] + rs*np.cos(thetas)
        ys = probe_center[1] + rs*np.sin(thetas)
        Ps = np.c_[xs, ys]
    return Ps

def fit_polyline(P, PixelSpacing, N=101):
    NP = P.shape[0]
    v = P[-1] - P[0]
    if v[0] < 0: # descending
        P = np.flipud(P)
        v = -v
    if NP < 4: # splorder=2 (cubic), minimum require 4 samples. Use linear interpolation if data points less than requirement.
        xi = np.linspace(P[0,0], P[-1,0], N)
        yi = np.interp(xi, P[:,0], P[:,1])
        Ps = np.c_[xi, yi]
    else:
        Vg2l = P.mean(axis=0, keepdims=True)
        axis_x = v / np.sqrt(np.sum(v**2))
        axis_y = [-axis_x[1], axis_x[0]]
        Rg2l = np.c_[axis_x, axis_y]
        Pl = (P - Vg2l) @ Rg2l
        xl, yl = Pl[:,0], Pl[:,1]
        assert np.all(xl[1:] > xl[:-1]), 'X-coordinates not strictly ascending'
        ese = np.ones(NP)
        VAL = 2.5/PixelSpacing # tolerance: 2.5mm
        c = gcvspline(xl, yl, ese, VAL, splmode=3)[0]
        yls = np.zeros(N)
        xls = np.linspace(xl[0], xl[-1], N)
        L = 0
        for i, xi in enumerate(xls):
            L = next((k for k in range(L, NP-1) if xl[k+1] > xi), NP-1)
            yls[i] = splderivative(xi, xl, c, L=L, IDER=0)[0]
        Pls = np.c_[xls, yls]
        Ps = Vg2l + (Pls @ Rg2l.T)
    return Ps

def plot_HMD(HMD, anno, ds, fig_folder, pose):
    dis = HMD.median()
    # find frame most close to median value
    idx = np.argmin(np.abs(HMD - dis))
    X = ds.pixel_array[idx]
    Mentum_xtl = anno.at[anno.index[idx],f'Mentum-{pose}_xtl']
    Mentum_ytl = anno.at[anno.index[idx],f'Mentum-{pose}_ytl']
    Mentum_xbr = anno.at[anno.index[idx],f'Mentum-{pose}_xbr']
    Mentum_ybr = anno.at[anno.index[idx],f'Mentum-{pose}_ybr']
    Hyoid_xtl = anno.at[anno.index[idx],f'Hyoid-{pose}_xtl']
    Hyoid_ytl = anno.at[anno.index[idx],f'Hyoid-{pose}_ytl']
    Hyoid_xbr = anno.at[anno.index[idx],f'Hyoid-{pose}_xbr']
    Hyoid_ybr = anno.at[anno.index[idx],f'Hyoid-{pose}_ybr']
    Mentum_xy = (Mentum_xtl, Mentum_ytl)
    Mentum_w = Mentum_xbr - Mentum_xtl
    Mentum_h = Mentum_ybr - Mentum_ytl
    Hyoid_xy = (Hyoid_xtl, Hyoid_ytl)
    Hyoid_w = Hyoid_xbr - Hyoid_xtl
    Hyoid_h = Hyoid_ybr - Hyoid_ytl
    # arrow arguments
    Mentum_y = (Mentum_ytl + Mentum_ybr)/2
    Hyoid_y = (Hyoid_ytl + Hyoid_ybr)/2
    arrow_y = (Mentum_y + Hyoid_y)/2
    posA = (Mentum_xbr, Mentum_y)
    posB = (Hyoid_xtl, Hyoid_y)
    # text arguments
    text_x = (Mentum_xbr + Hyoid_xtl) / 2
    text_y = arrow_y - 2
    text_s = '{} mm'.format(round(dis,1))
    # start draw
    fig, ax = plt.subplots(figsize=[X.shape[1]/100, X.shape[0]/100], dpi=100)
    ax.set_position([0,0,1,1])
    ax.imshow(X)
    rec_Mentum = patches.Rectangle(Mentum_xy, Mentum_w, Mentum_h, color='#00FF00',fill=False,lw=1)
    rec_Hyoid = patches.Rectangle(Hyoid_xy, Hyoid_w, Hyoid_h, color='#00FF00',fill=False,lw=1)
    arrow = patches.FancyArrowPatch(posA, posB, arrowstyle='<|-|>', mutation_scale=20, color='#00FF00', shrinkA=0, shrinkB=0)
    ax.add_patch(rec_Mentum)
    ax.add_patch(rec_Hyoid)
    ax.add_patch(arrow)
    ax.text(text_x, text_y, text_s, horizontalalignment='center', verticalalignment='bottom', color='#00FF00', fontsize='large')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # prepare output
    fig.savefig(os.path.join(fig_folder,f'1_HMD-{pose}.png'))
    plt.close(fig)

def plot_TongueThickness(TongueThickness, task, PixelSpacing, ds, fig_folder):
    dis = np.nanmedian(TongueThickness)
    # find frame most close to median value
    idx = np.nanargmin(np.abs(TongueThickness - dis))
    X = ds.pixel_array[idx]
    anno = task['anno']
    Tongue_upper = anno.at[anno.index[idx], '4Tongue_Upper']
    Tongue_lower = anno.at[anno.index[idx], '5Tongue_Lower']
    r_upper, theta_upper = xy2rtheta(Tongue_upper, task['probe_center'])
    r_lower, theta_lower = xy2rtheta(Tongue_lower, task['probe_center'])
    mintheta = max(theta_upper.min(), theta_lower.min())
    maxtheta = min(theta_upper.max(), theta_lower.max())
    fitted_upper = fit_polar_curve(r_upper, theta_upper, task['probe_center'], PixelSpacing, mintheta, maxtheta)
    fitted_lower = fit_polar_curve(r_lower, theta_lower, task['probe_center'], PixelSpacing, mintheta, maxtheta)
    ix = np.argmax(np.sum((fitted_lower - fitted_upper)**2, axis=1))
    # arrow arguments
    posA = fitted_upper[ix]
    posB = fitted_lower[ix]
    # text arguments
    text_xy = (posA + posB)/2
    text_xy[0] += 5
    text_s = '{} mm'.format(round(dis,1))
    # start draw
    fig, ax = plt.subplots(figsize=[X.shape[1]/100, X.shape[0]/100], dpi=100)
    ax.set_position([0,0,1,1])
    ax.imshow(X)
    ax.plot(Tongue_upper[:,0], Tongue_upper[:,1], color='#F078F0', lw=1)
    ax.plot(Tongue_lower[:,0], Tongue_lower[:,1], color='#0000FF', lw=1)
    ax.plot(fitted_upper[:,0], fitted_upper[:,1], color='#F078F0', lw=2, linestyle='--')
    ax.plot(fitted_lower[:,0], fitted_lower[:,1], color='#0000FF', lw=2, linestyle='--')
    arrow = patches.FancyArrowPatch(posA, posB, arrowstyle='<|-|>', mutation_scale=20, color='#FFE599', shrinkA=0, shrinkB=0, lw=2)
    ax.add_patch(arrow)
    ax.text(text_xy[0], text_xy[1], text_s, horizontalalignment='left', verticalalignment='center', color='#FFE599', fontsize='large')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # prepare output
    fig.savefig(os.path.join(fig_folder,'2_TongueThickness.png'))
    plt.close(fig)

def plot_Skin2Box(point_to, S, anno, start_row, landmark, rec_color, ds, fig_folder, figname, q=50):
    # point_to = "top" or "middle"
    dis = np.nanpercentile(S.to_numpy(), q)
    # find frame most close to median value
    idx = np.argmin(np.abs(S - dis))
    X = ds.pixel_array[idx]
    xtl = anno.at[anno.index[idx],f'{landmark}_xtl']
    ytl = anno.at[anno.index[idx],f'{landmark}_ytl']
    xbr = anno.at[anno.index[idx],f'{landmark}_xbr']
    ybr = anno.at[anno.index[idx],f'{landmark}_ybr']
    # rectangle arguments
    xy = (xtl, ytl)
    width = xbr - xtl
    height = ybr - ytl
    # arrow arguments
    if point_to=='top':
        posA = [(xtl+xbr)/2, ytl]
    elif point_to=='middle':
        posA = [(xtl+xbr)/2, (ytl+ybr)/2]
    posB = [posA[0], start_row]
    # text arguments
    text_x = posA[0] + 5
    text_y = posA[1]/2
    text_s = '{} mm'.format(round(dis,1))
    # start draw
    fig, ax = plt.subplots(figsize=[X.shape[1]/100, X.shape[0]/100], dpi=100)
    ax.set_position([0,0,1,1])
    ax.imshow(X)
    rec = patches.Rectangle(xy, width, height, color=rec_color, fill=False, lw=1)
    arw = patches.FancyArrowPatch(posA, posB, arrowstyle='<|-|>', mutation_scale=20, color='#FFE599', shrinkA=0, shrinkB=0, lw=1)
    ax.add_patch(rec)
    ax.add_patch(arw)
    ax.text(text_x, text_y, text_s, horizontalalignment='left', verticalalignment='center', color='#FFE599', fontsize='large')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # prepare output
    fig.savefig(os.path.join(fig_folder,f'{figname}.png'))
    plt.close(fig)

def plot_BoxHeight(S, anno, landmark, rec_color, ds, fig_folder, figname, q=50):
    dis = np.nanpercentile(S.to_numpy(), q)
    # find frame most close to median value
    idx = np.argmin(np.abs(S - dis))
    X = ds.pixel_array[idx]
    xtl = anno.at[anno.index[idx],f'{landmark}_xtl']
    ytl = anno.at[anno.index[idx],f'{landmark}_ytl']
    xbr = anno.at[anno.index[idx],f'{landmark}_xbr']
    ybr = anno.at[anno.index[idx],f'{landmark}_ybr']
    # rectangle arguments
    xy = (xtl, ytl)
    width = xbr - xtl
    height = ybr - ytl
    # arrow arguments
    posA = [(xtl+xbr)/2, ybr]
    posB = [posA[0], ytl]
    # text arguments
    text_x = posA[0] + 5
    text_y = (ytl + ybr)/2
    text_s = '{} mm'.format(round(dis,1))
    # start draw
    fig, ax = plt.subplots(figsize=[X.shape[1]/100, X.shape[0]/100], dpi=100)
    ax.set_position([0,0,1,1])
    ax.imshow(X)
    rec = patches.Rectangle(xy, width, height, color=rec_color, fill=False, lw=1)
    arw = patches.FancyArrowPatch(posA, posB, arrowstyle='<|-|>', mutation_scale=20, color='#FFE599', shrinkA=0, shrinkB=0, lw=1)
    ax.add_patch(rec)
    ax.add_patch(arw)
    ax.text(text_x, text_y, text_s, horizontalalignment='left', verticalalignment='center', color='#FFE599', fontsize='large')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # prepare output
    fig.savefig(os.path.join(fig_folder,f'{figname}.png'))
    plt.close(fig)


if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Calculate distances')
    parser.add_argument('xml_subfolder', type=str, help='xml folder name')
    args = parser.parse_args()
    # ---------- Assign variable ---------
    subjects = 'all' # 指定要分析的病人，若為'all'，則xml_folder中有找到的xml都會進行分析
    # Unexpect mentum or hyoid position
    # subjects = ['2661769','3831397','5135292','5646048','6646808','6984470','7072992','7126866','7146653','7156856','7183874'] # 共11位 
    # Track overlapped!
    # subjects = ['2676627','2692431','3979187','4295498','4567341','5135897','5813259','5968962','6603941'] # 共9位 
    # ValueError or AssertionError
    # subjects = ['4444928','4424313']
    # === Start calculate distances === 時會遇到的 error
    # subjects = ['3164357','3200270','3262442','3376168','3378720','3462733','3966008','4077072','4099071','4225273', # 共20位 
    #             '4456080','4494352','5658925','5810934','6033260','6192696','6561173','6601421','6645864','7157082']
    
    DA_folder = '/root/ultrasound/DifficultAirway' # 修改為正確的路徑
    skip_exist = True # 若為True, 程式會跳過已經分析過的病人
    # ------------------------------------
    dcmPixelSpacing_joblib = os.path.join(DA_folder, '計算實際距離', 'Dicom_PixelSpacing_DA.joblib')
    # xml_folder = os.path.join(DA_folder,'CVAT_annotations','DA_group1') # 剩下修正後無法匯出的3位病人
    # xml_folder = os.path.join(DA_folder,'CVAT_annotations','DA_group2') # Done!
    # xml_folder = os.path.join(DA_folder,'CVAT_annotations','DA_group3') # Done!
    # xml_folder = os.path.join(DA_folder,'CVAT_annotations','DA_group5')
    # xml_folder = os.path.join(DA_folder,'CVAT_annotations','內視鏡1')
    # xml_folder = os.path.join(DA_folder,'CVAT_annotations','內視鏡2')
    xml_folder = os.path.join(DA_folder,'CVAT_annotations',args.xml_subfolder)
    suffix = 'DA' if args.xml_subfolder[:2]=='DA' else 'ES'
    dcm_folders = [
        os.path.join(DA_folder, '困難'),
        os.path.join(DA_folder, '非困難'),
        os.path.join(DA_folder, '尚未判定'),
        os.path.join(DA_folder, '待取得分類結果'),
        os.path.join(DA_folder, '內視鏡')
    ]
    # today = date.today().strftime('%Y%m%d')

    logger = logging.getLogger(__name__)
    output_folder = os.path.join(DA_folder, 'CVAT_annotations', 'Distances')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    log_file = os.path.join(output_folder, 'DA_metrics.log')
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')
    
    xml_files = [f.name[:-4] for f in os.scandir(xml_folder) if f.name[-4:]=='.xml']
    dcm_path, dcm_name = zip(*scan_dcm(dcm_folders))
    dcm_ps = joblib.load(dcmPixelSpacing_joblib)
    if isinstance(subjects,str) and subjects=='all':
        subjects = [s for s in xml_files if s!='CTM']
        subjects.sort()
        
    poses = ('neutral', 'extended', 'ramped')
    cols_HMD = [f'HMD-{pose}' for pose in poses]
    cols = [*cols_HMD, 'TongueThickness','DSH','DSE','E-VC','DSG','DTI','DSN']

    xls_output = os.path.join(output_folder, f'Distance{suffix}.xlsx')
    if skip_exist and os.path.isfile(xls_output):
        xls_mtime = os.path.getmtime(xls_output)
        df = pd.read_excel(xls_output, index_col=0)
        df.index = df.index.astype(str).str.rjust(7,'0')
        subjects = [sub for sub in subjects if sub not in df.index or os.path.getmtime(os.path.join(xml_folder, f'{sub}.xml')) > xls_mtime] # xml檔案有更新過也會再刷新
        df = df.loc[~df.index.isin(subjects)] # 移除已經存在的記錄(xml有更新)
        df = pd.concat((df, pd.DataFrame(index=subjects, columns=cols, dtype=float)), axis=0)
    else:
        df = pd.DataFrame(index=subjects, columns=cols, dtype=float)

    # load xml files and check overlapping
    print('=== Checking overlapping ===')
    prj_tasks = []
    cache_file = os.path.join(os.path.dirname(__file__),'probeCR.cache')
    probeCR = joblib.load(cache_file) if os.path.isfile(cache_file) else dict()
    for sub in tqdm(subjects):
        xml_path = os.path.join(xml_folder, f'{sub}.xml')
        tasks, probeCR = load_project(xml_path, dcm_path, dcm_name, probeCR)
        prj_tasks.append({k:v for k,v in tasks.items() if 'anno' in v}) # filter out no annotation tasks

    print('Done!\n')

    print('=== Start calculate distances ===')
    for sub, tasks in tqdm(zip(subjects, prj_tasks), total=len(subjects)):
        all_landmarks = {mk for id in list(tasks) for mk in tasks[id]['anno'].columns}

        fig_folder = os.path.join(output_folder, sub)
        if not os.path.isdir(fig_folder):
            os.mkdir(fig_folder)
        # ======= PART 1: Suprahyoid evaluation =======
        # Hyomental Distance (HMD)
        for pose in poses:
            landmarks = {f'Mentum-{pose}_xtl', f'Hyoid-{pose}_xtl'}
            if all((mk in all_landmarks for mk in landmarks)):
                task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
                HMD_dx = (tasks[task_id]['anno'][f'Hyoid-{pose}_xtl'] - tasks[task_id]['anno'][f'Mentum-{pose}_xbr'])*PixelSpacing
                HMD_dy = (tasks[task_id]['anno'][[f'Hyoid-{pose}_ytl',f'Hyoid-{pose}_ybr']].mean(axis=1) - tasks[task_id]['anno'][[f'Mentum-{pose}_ytl',f'Mentum-{pose}_ybr']].mean(axis=1))*PixelSpacing
                HMD = np.sqrt(HMD_dx**2 + HMD_dy**2)
                plot_HMD(HMD, tasks[task_id]['anno'], ds, fig_folder, pose)
                df.loc[sub, f'HMD-{pose}'] = HMD.median()
            else:
                logger.warning(f'Subject[{sub}]: Cannot calculate 1_HMD-{pose} - Missing landmarks: 1. Chin&Hyoid ({pose})')
        # tongue thickness
        landmarks = {'4Tongue_Upper', '5Tongue_Lower'}
        if all((mk in all_landmarks for mk in landmarks)):
            task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
            task_name = tasks[task_id]['name']
            TongueThickness = np.tile(np.nan, tasks[task_id]['anno'].shape[0])
            probe_center = tasks[task_id]['probe_center'] # unit: pixel
            probe_radius = tasks[task_id]['probe_radius']
            anno = tasks[task_id]['anno']
            lg = anno['4Tongue_Upper'].notna() & anno['5Tongue_Lower'].notna()
            assert lg.any(), f'4. Tongue_Upper與5. Tongue_Lower沒有同時出現的frame，無法計算Tongue Thickness. 請檢查task: {task_name}'
            for i in np.nonzero(lg.to_numpy())[0]:
                P_upper = anno.at[anno.index[i], '4Tongue_Upper']
                P_lower = anno.at[anno.index[i], '5Tongue_Lower']
                r_upper, theta_upper = xy2rtheta(P_upper, probe_center)
                r_lower, theta_lower = xy2rtheta(P_lower, probe_center)
                mintheta = max(theta_upper.min(), theta_lower.min())
                maxtheta = min(theta_upper.max(), theta_lower.max())
                try:
                    Ps_upper = fit_polar_curve(r_upper, theta_upper, probe_center, PixelSpacing, mintheta, maxtheta)
                except AssertionError as err:
                    print(f'Error happened! Please check project: {sub}, task: {task_name}, frame: {i}, label: 4.Tongue_Upper')
                    raise err
                try:
                    Ps_lower = fit_polar_curve(r_lower, theta_lower, probe_center, PixelSpacing, mintheta, maxtheta)
                except AssertionError as err:
                    print(f'Error happened! Please check project: {sub}, task: {task_name}, frame: {i}, label: 5.Tongue_Lower')
                    raise err
                TongueThickness[i] = np.max(np.sqrt(np.sum((Ps_lower - Ps_upper)**2, axis=1)))
            TongueThickness *= PixelSpacing
            plot_TongueThickness(TongueThickness, tasks[task_id], PixelSpacing, ds, fig_folder)
            df.loc[sub, 'TongueThickness'] = np.nanmedian(TongueThickness)
        else:
            logger.warning(f'Subject[{sub}]: Cannot calculate 2_TongueThickness - Missing landmarks: Tongue_Upper or Tongue_Lower')
        
        # ======= PART 2: Thyrohyoid, hyoid and epiglottis evaluation =======
        # Distance from skin to hyoid bone (DSH)
        landmarks = {'6Hyoid Bone_ytl'}
        if '6Hyoid Bone_ytl' in all_landmarks:
            task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
            DSH = (tasks[task_id]['anno']['6Hyoid Bone_ytl']-tasks[task_id]['start_row'])*PixelSpacing
            plot_Skin2Box('top', DSH, tasks[task_id]['anno'], tasks[task_id]['start_row'], '6Hyoid Bone', '#7F007F', ds, fig_folder, '3_DSH')
            df.loc[sub, 'DSH'] = DSH.median()
        else:
            logger.warning(f'Subject[{sub}]: Cannot calculate 3_DSH - Missing landmark: Hyoid Bone')
        # Distance from skin to epiglottis (DSE)
        landmarks = {'7Epiglottis_ytl'}
        if '7Epiglottis_ytl' in all_landmarks:
            task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
            DSE = (tasks[task_id]['anno']['7Epiglottis_ytl']-tasks[task_id]['start_row'])*PixelSpacing
            plot_Skin2Box('top', DSE, tasks[task_id]['anno'], tasks[task_id]['start_row'], '7Epiglottis', '#996633', ds, fig_folder, '4_DSE')
            df.loc[sub, 'DSE'] = DSE.median()
        else:
            logger.warning(f'Subject[{sub}]: Cannot calculate 4_DSE - Missing landmark: Epiglottis')
        
        # ======= PART 3: Thyroid, Vocal cord function =======
        # Epiglottis to vocal cord distance (E-VC distance)
        landmarks = {'8Epiglottis-to-Vocal Cord_ytl'}
        if '8Epiglottis-to-Vocal Cord_ytl' in all_landmarks:
            task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
            E_VC = (tasks[task_id]['anno']['8Epiglottis-to-Vocal Cord_ybr'] - tasks[task_id]['anno']['8Epiglottis-to-Vocal Cord_ytl'])*PixelSpacing
            plot_BoxHeight(E_VC, tasks[task_id]['anno'], '8Epiglottis-to-Vocal Cord', '#00FFFF', ds, fig_folder, '5_E-VC')
            df.loc[sub, 'E-VC'] = E_VC.median()
        else:
            logger.warning(f'Subject[{sub}]: Cannot calculate 5_E-VC - Missing landmark: Epiglottis-to-Vocal Cord')
        # Skin to anterior commissure of vocal cords (DSG)
        landmarks = {'9Anterior Commissure of VC_ytl'}
        if '9Anterior Commissure of VC_ytl' in all_landmarks:
            task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
            DSG = (tasks[task_id]['anno']['9Anterior Commissure of VC_ytl']-tasks[task_id]['start_row'])*PixelSpacing
            plot_Skin2Box('top', DSG, tasks[task_id]['anno'], tasks[task_id]['start_row'], '9Anterior Commissure of VC', '#FF00FF', ds, fig_folder, '6_DSG', 10)
            # df.loc[sub, 'DSG'] = DSG.median()
            df.loc[sub, 'DSG'] = np.nanpercentile(DSG.to_numpy(), 10)
        else:
            logger.warning(f'Subject[{sub}]: Cannot calculate 6_DSG - Missing landmark: Anterior Commissure of VC')
        # Skin to thyroid isthmus
        landmarks = {'10Thyroid Isthmus_ytl'}
        if '10Thyroid Isthmus_ytl' in all_landmarks:
            task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
            DTI = (tasks[task_id]['anno']['10Thyroid Isthmus_ytl']-tasks[task_id]['start_row'])*PixelSpacing
            plot_Skin2Box('top', DTI, tasks[task_id]['anno'], tasks[task_id]['start_row'], '10Thyroid Isthmus', '#9437FF', ds, fig_folder, '7_Skin_to_Thyroid_Isthmus')
            df.loc[sub, 'DTI'] = DTI.median()
        else:
            logger.warning(f'Subject[{sub}]: Cannot calculate 7_Skin_to_Thyroid_Isthmus - Missing landmark: Thyroid Isthmus')

        # ======= PART 5: Suprasternal =======
        # Skin to suprasternal notch
        landmarks = {'11Suprasternal Notch_ytl'}
        if '11Suprasternal Notch_ytl' in all_landmarks:
            task_id, PixelSpacing, ds = locate_task(tasks, landmarks, dcm_path, dcm_name, dcm_ps, logger, sub)
            DSN = (tasks[task_id]['anno']['11Suprasternal Notch_ytl']-tasks[task_id]['start_row'])*PixelSpacing
            plot_Skin2Box('top', DSN, tasks[task_id]['anno'], tasks[task_id]['start_row'], '11Suprasternal Notch', '#79D6F9', ds, fig_folder, '8_Skin_to_suprasternal_notch')
            df.loc[sub, 'DSN'] = DSN.median()
        else:
            logger.warning(f'Subject[{sub}]: Cannot calculate Skin_to_suprasternal_notch - Missing landmark: Suprasternal Notch')

    df.sort_index(inplace=True)
    df.to_excel(xls_output)
    print('Done!\n')