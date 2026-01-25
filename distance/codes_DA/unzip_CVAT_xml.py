import os
import zipfile

def unzip_cvat_annotations(zip_folder, overwrite=False):
    zip_files = [f.name[:-4] for f in os.scandir(zip_folder) if f.name[-4:]=='.zip']
    for name in zip_files:
        file_path = os.path.join(zip_folder, f'{name}.zip')
        src = os.path.join(zip_folder, 'annotations.xml')
        dst = os.path.join(zip_folder, f'{name}.xml')
        assert not os.path.isfile(src), 'annotations.xml appear in zip folder, please delete it.'
        zip_mtime = os.path.getmtime(file_path)
        if overwrite or not os.path.isfile(dst) or zip_mtime > os.path.getmtime(dst): # zip檔有更新的話會自動解壓覆蓋舊檔
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(zip_folder)
            os.rename(src, dst)
            print(f'unzip file: {name}.zip')


if __name__=='__main__':
    anno_folder = '/root/ultrasound/DifficultAirway/CVAT_annotations'
    for subfolder in ('DA_group1','DA_group2','DA_group3','DA_group5','DA_group6','內視鏡1','內視鏡2','內視鏡3','內視鏡4','內視鏡5'): # 讀yaml會比較慢
        zip_folder = os.path.join(anno_folder, subfolder)
        if os.path.isdir(zip_folder):
            unzip_cvat_annotations(zip_folder)