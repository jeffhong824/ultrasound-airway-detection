import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom as dicom
from scipy.linalg import lstsq
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

def scan_dcm(dirs, exclude=None, ext='.dcm'):
    # output [(dcm1_path, dcm1_name), (dcm2_path, dcm2_name), ...]
    if exclude is None:
        exclude = {'2847642_Quick ID_20230221_115320_B.dcm'} # 檔案毀損
    dcms = []
    L = len(ext)
    for dirname in dirs:
        subfolders = []
        for f in os.scandir(dirname):
            if f.is_file() and f.name[-L:]==ext and f.name not in exclude:
                dcms.append((f.path, f.name[:-L]))
            elif f.is_dir():
                subfolders.append(f.path)
        if subfolders:
            dcms.extend(scan_dcm(subfolders, exclude, ext))
    return dcms

def AlgfitCircle(x, return_resi=False):
    # X - (n*2) or (n*3)
    x = np.asarray(x)
    m = x.shape[0]
    x0 = x.sum(axis=0, keepdims=True)/m
    A = x - x0
    B = np.sum(A**2, axis=1, keepdims=True)
    X = lstsq(A, B, overwrite_a=True)[0] /2
    C = x0 + X.T
    R = np.sqrt((X**2).sum() + B.sum()/m)
    if return_resi:
        Vc2x = x - C
        L = np.sqrt((Vc2x**2).sum(axis=1, keepdims=True))
        CP = C + (R/L)*Vc2x
        resi = np.sqrt(np.sum((x-CP)**2))
        return C, R, CP, resi
    else:
        return C, R

def GeofitCircle(X, return_resi=False, return_rho=False):
    # Syntax
    #       C, R = GeofitCircle(X)
    #       C, R, CP, resi = GeofitCircle(X, return_resi=True)
    #       C, R, CP, resi, varP, rho = GeofitCircle(X, return_resi=True, return_rho=True)
    #   Input
    #       X - (n*2) or (n*3)
    lamda = 1
    # algebraic fitting
    C, R = AlgfitCircle(X)
    # geometric fitting
    m, n = X.shape
    p = m*n
    q = n+1
    n2 = n**2
    delta = np.inf
    # ix1 = np.tile(np.r_[:n],n)
    # ix2 = np.repeat(np.r_[:n],n)
    while np.sqrt(np.sum(delta**2)) > np.finfo(np.float32).eps:
        Vc2x = X - C
        Ls = np.sum(Vc2x**2, axis=1, keepdims=True)
        L = np.sqrt(Ls)
        RL = R/L
        S = 1 - RL
        edv = (S*Vc2x).reshape((p,1), order='F') # orthogonal error distance vector
        J1 = (Vc2x/L).reshape((p,1), order='F')
        J2 = np.zeros((m, n2))
        J2[:,np.arange(0,n2,n+1)] = S
        v1 = RL*Vc2x
        v2 = Vc2x/Ls
        # J2 += v1[:,ix1]*v2[:,ix2]
        J2 += np.tile(v1, n)*np.repeat(v2, n, axis=1)
        J2 = J2.reshape((p,n), order='F') # 2nd and 3rd (and 4th while fitting sphere) columns of Jacobian matrix
        J = np.c_[J1, J2]
        delta = lstsq(J, edv)[0]
        R += lamda*delta[0,0]
        C += lamda*delta[1:,:1].T
    if return_resi:
        CP = -edv.reshape((m,n),order='F') + X
        resi = np.sqrt(np.sum(edv**2))
        if return_rho:
            Cov = np.linalg.inv(J.T@J)
            varP = resi*np.sqrt(np.diag(Cov)/(p-q))
            rho = np.zeros((q,q))
            for j in range(q-1):
                for k in range(j+1,q):
                    rho[j,k] = Cov[j,k]/np.sqrt(Cov[j,j]*Cov[k,k])
            rho += rho.T
            rho[np.r_[:q], np.r_[:q]] = 1
            return C, R, CP, resi, varP, rho
        else:
            return C, R, CP, resi
    else:
        return C, R
    
def fitCircle_plot(**kwargs):
    # Syntax:
    #   < perform Demo >
    #       fitCircle_plot(func=AlgfitCircle, ndims=2)
    #       fitCircle_plot(func=GeofitCircle, ndims=3)
    #   < plot fitting result >
    #       fitCircle_plot(X=X, C=C, R=R, CP=CP, resi=resi, img=img)
    if not kwargs:
        kwargs = {'func':AlgfitCircle, 'ndims':2}
    if 'func' in kwargs: # random generate points
        if 'ndims' not in kwargs:
            kwargs['ndims'] = 2
        R = 200
        m = 10
        NoiseLevel = 10 # Will add Gaussian noise of this std.dev. to points
        if kwargs['ndims']==2: # create a circle
            C = np.array([[100, 100]])
            t = np.random.normal(scale=np.pi, size=m)
            X = C + R*np.c_[np.cos(t), np.sin(t)] + np.random.normal(size=(m,2))*NoiseLevel
        elif kwargs['ndims']==3: # create a sphere
            C = np.array([[100, 100, 100]])
            t = np.random.normal(scale=np.pi, size=m)
            phi = np.random.normal(scale=np.pi/2, size=m)
            X = C + R*np.c_[np.cos(phi)*np.cos(t), np.cos(phi)*np.sin(t), np.sin(phi)] + np.random.normal(size=(m,3))*NoiseLevel
        else:
            raise AssertionError('Unexpect dimension: {}'.format(kwargs['ndims']))
        # fit circle or sphere
        if kwargs['func'] is AlgfitCircle:
            C, R, CP, resi = AlgfitCircle(X, return_resi=True)
        elif kwargs['func'] is GeofitCircle:
            C, R, CP, resi = GeofitCircle(X, return_resi=True)
        else:
            raise AssertionError('Unexpect function handle')
    elif 'X' in kwargs and 'C' in kwargs and 'R' in kwargs and 'CP' in kwargs and 'resi' in kwargs:
        X = kwargs['X']
        C = kwargs['C']
        R = kwargs['R']
        CP = kwargs['CP']
        resi = kwargs['resi']
    else:
        raise AssertionError('Unexpect input')
    # prepare plot
    # matplotlib.use('qtagg') # choose interactive backend
    m, n = X.shape
    segs = [np.stack((p1,p2)) for p1,p2 in zip(X,CP)]
    if n==2: # plot 2D circle
        fig, ax = plt.subplots()
        if 'img' in kwargs:
            ax.imshow(kwargs['img'])
        circle = Circle(C[0], R, edgecolor='r', facecolor=None, fill=False)
        line_segments = LineCollection(segs, colors='k', lw=0.5)
        ax.add_patch(circle)
        ax.add_collection(line_segments)
        ax.scatter(C[0,0], C[0,1], c='k', marker='+', lw=0.5)
        ax.scatter(X[:,0], X[:,1], s=20, c='b')
        ax.set_title(f'Radius: {R:.2f}, Center: [{C[0,0]:.2f}, {C[0,1]:.2f}], Residual: {resi:.4f}')
        plt.autoscale(True)
        ax.set_aspect('equal')
        plt.show()
    elif n==3: # plot 3D sphere
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        pass
    if 'func' in kwargs:
        if kwargs['func'] is AlgfitCircle:
            fig.canvas.manager.set_window_title(fig.canvas.manager.get_window_title() + ' - AlgfitCircle')
        elif kwargs['func'] is GeofitCircle:
            fig.canvas.manager.set_window_title(fig.canvas.manager.get_window_title() + ' - GeofitCircle')
        else:
            raise AssertionError(f'Unexpect dimension: {n}')
