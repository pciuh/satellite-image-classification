#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:13:42 2021

@author: pciuh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sci

from matplotlib import cm
from matplotlib.patches import Patch
#from sklearn.model_selection import train_test_split

iDir = 'input/'

TYP = 'rf'
sFLG,rFLG = (True,True)

mName = {'et' : 'Extra Trees', 'rf' : 'Random Forest', 'ba' : 'Bagging'}

fnam= 'index-'+TYP+'.csv'
df = pd.read_csv(fnam,sep=',')

idx = df.Index.values

fnam = 'labels.csv'
df = pd.read_csv(iDir + fnam,sep=',')
lbl = df.label.drop_duplicates().to_list()

print(idx)

col_p = cm.get_cmap('Paired',len(lbl))
col = col_p(np.linspace(0,1,len(lbl)))

col = np.array(
       [[1.0,1.0,0.0,1.0],
        [1.0,1.0,0.0,1.0],
        [0.0,1.0,0.0,1.0],
        [0.0,0.0,1.0,1.0],
        [1.0,0.0,0.0,1.0],
        [1.0,0.0,0.0,1.0],
        [1.0,0.0,0.0,1.0],
        [0.0,0.6,0.0,1.0],
        [0.6,0.6,0.6,1.0],
        [0.8,0.4,0.0,1.0]])

mar = ['.','o','X','d',10,6,'^','p','4','$///$']

fnam = 'satellite_image.csv'
dfs = pd.read_csv(iDir + fnam,sep=',')

print(dfs.shape)

xMin = min(dfs.x)
yMin = min(dfs.y)

xr,yr = (dfs.x-xMin,dfs.y-yMin)

rr = dfs.band4/max(dfs.band4)
rg = dfs.band3/max(dfs.band3)
rb = dfs.band2/max(dfs.band2)

_,num = np.unique(yr.to_numpy(),return_counts=True)

handles = [
    Patch(facecolor=color, label=label) 
    for label, color in zip(lbl, col)
]

Nx = num[0]
Ny = int(xr.shape[0]/num[0])

print(Nx,Ny)

if sFLG:
    xg,yg = np.meshgrid(np.linspace(min(xr),max(xr),Nx),np.linspace(min(yr),max(yr),Ny))

    rrg = sci.griddata((xr,yr),rr,(xg,yg))
    rgg = sci.griddata((xr,yr),rg,(xg,yg))
    rbg = sci.griddata((xr,yr),rb,(xg,yg))

    data = np.dstack((rrg,rgg,rbg))

    print(data.shape)
    fig,ax = plt.subplots(1,figsize=(7,7))

    ax.set_title('Satellite Image with Features')
    ax.imshow(data,origin='lower',extent=(min(xr),max(xr),min(yr),max(yr)))

    for i,l in enumerate(lbl):
        xl,yl = (df.x[df.label==l]-xMin,df.y[df.label==l]-yMin)
        ax.scatter(xl,yl,color=col[i],marker=mar[i],label=lbl[i],alpha=1.0)

    #ax.grid()
    ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
    fig.savefig('sat.png',dpi=300, bbox_inches='tight')

if rFLG:
    fig,ax = plt.subplots(1,figsize=(7,7))
    ax.set_title('Model: '+mName[TYP])

    # ax.imshow(data,origin='lower',extent=(min(xr),max(xr),min(yr),max(yr)))

    for i,l in enumerate(lbl):
        ax.scatter(xr[idx==i],yr[idx==i],color=col[i],marker='.',s=1,label=lbl[i])

    #ax.grid()
    ax.axis('off')
    ax.set_aspect(1)
    ax.legend(handles = handles, loc='center left',bbox_to_anchor=(1.0, 0.5))
    fig.savefig('pred-'+TYP+'.png',dpi=300, bbox_inches='tight')



