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
import seaborn as sns

from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

def ploconfmat(clf_cfm,lbl,TIT,SUF):
    fig,ax = plt.subplots(1,figsize=(11,9))
    gc = ["{0:0.0f}".format(x) for x in clf_cfm.flatten()]

    gp = ["{0:.1%}".format(x) for x in
                         clf_cfm.flatten()/np.sum(clf_cfm)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
              zip(gc,gp)]

    labels = np.asarray(labels).reshape(len(lbl),len(lbl))

    ax = sns.heatmap(clf_cfm, annot=labels,fmt='',cmap='Blues')

    ax.set_title('Confusion Matrix ('+TIT+')\n\n');
    ax.set_xlabel('\nPredicted Labels Category')
    ax.set_ylabel('Actual Labels Category ');
    ax.xaxis.set_ticklabels(lbl)
    ax.yaxis.set_ticklabels(lbl)

    fig.savefig('conf_mat-'+SUF+'.png',dpi=300,bbox_inches='tight')

SEED = 30082024

iDir = 'input/'

fnam = 'labels.csv'
df = pd.read_csv(iDir + fnam,sep=',')
lbl = df.label.drop_duplicates().to_numpy()

_,num = np.unique(df.label,return_counts=True)

variables=['x','y','band4','band3','band2']
variables=['x','y','band1','band2','band3','band4','band5','band6']

X = df[variables].to_numpy()
Y = []
for i,l in enumerate(lbl):
    for ni in range(num[i]):
        Y=np.append(Y,i)

TIT = {'RF' : 'Random Forest', 'ET' : 'Extra Tree', 'BA' : 'Bagging'}

mvec = [ExtraTreesClassifier(),
        RandomForestClassifier(),
        BaggingClassifier()]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1, random_state = SEED)

print('Total Size:',X.shape,Y.shape)
print('Train Size:',X_train.shape, Y_train.shape)
print(' Test Size:',X_test.shape, Y_test.shape)

ofnam = 'class_report-%.8d.txt'%SEED
of = open(ofnam,'w')
of.write('            %10s%10s%10s\n'%('Samples','Category','Outcome'))
of.write('Total Size: %10.0f%10.0f%10.0f\n'%(X.shape[0],X.shape[1],Y.shape[0]))
of.write('Train Size: %10.0f%10.0f%10.0f\n'%(X_train.shape[0],X_train.shape[1],Y_train.shape[0]))
of.write(' Test Size: %10.0f%10.0f%10.0f\n'%(X_test.shape[0],X_test.shape[1],Y_test.shape[0]))

pvec = []
for v in mvec:
    v.fit(X_train, Y_train)
    pvec.append(v.predict(X_test))


p_crf,p_cet,p_cba = pvec
mNam = ['RF','ET','BA']

of.write('\n%36s\n'%'Model Performace')
print('\nModel Performance')
cfm = []
for i,v in enumerate(pvec):
    print('\n%18s:'%TIT[mNam[i]])
    print(classification_report(Y_test, v, target_names=lbl))
    print('Kappa Score:',cohen_kappa_score(Y_test, v))

    of.write('\n%14s:\n'%TIT[mNam[i]])
    of.write(classification_report(Y_test, v, target_names=lbl))
    of.write('\nKappa Score:%12.3f\n'%(cohen_kappa_score(Y_test,v))) 

    cfm.append(confusion_matrix(Y_test, v))
of.close()

for i,c in enumerate(cfm):
    ploconfmat(c,lbl,TIT[mNam[i]],mNam[i])

key = ['Index']
for i,v in enumerate(pvec):
    df = pd.DataFrame(dict(zip(key,[v]))).astype('int32')
    df.to_csv('index-'+mNam[i]+'.csv',sep=',')
