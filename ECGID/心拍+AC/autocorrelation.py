import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal    
import matplotlib as mpl
import math  
import pandas as pd
import matplotlib.colors
from sklearn.decomposition import PCA 
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import random
import scipy.io as scio

time_start = time.time()
Mulu = ['1.mat', '2.mat', '3.mat', '4.mat', '5.mat', '6.mat', '7.mat', '8.mat', '9.mat',
         '10.mat', '11.mat', '12.mat', '13.mat', '14.mat', '15.mat', '16.mat', '17.mat', '18.mat',
         '19.mat', '20.mat', '21.mat', '22.mat', '23.mat', '24.mat', '25.mat', '26.mat', '27.mat',
         '28.mat', '29.mat', '30.mat', '31.mat', '32.mat', '33.mat', '34.mat', '35.mat', '36.mat',
         '37.mat', '38.mat', '39.mat', '40.mat', '41.mat', '42.mat', '43.mat', '44.mat', '45.mat', '46.mat',
         '47.mat', '48.mat', '49.mat', '50.mat', '51.mat', '52.mat', '53.mat', '54.mat', '55.mat', '56.mat',
         '57.mat', '58.mat', '59.mat', '60.mat', '61.mat', '62.mat', '63.mat', '64.mat', '65.mat', '66.mat',
         '67.mat', '68.mat', '69.mat', '70.mat', '71.mat', '72.mat', '73.mat', '74.mat', '75.mat', '76.mat',
         '77.mat', '78.mat', '79.mat', '80.mat', '81.mat', '82.mat', '83.mat', '84.mat', '85.mat', '86.mat',
         '87.mat', '88.mat', '89.mat', '90.mat']
newMulu = ['1auto.mat', '2auto.mat', '3auto.mat', '4auto.mat', '5auto.mat', '6auto.mat', '7auto.mat',
            '8auto.mat', '9auto.mat',
            '10auto.mat', '11auto.mat', '12auto.mat', '13auto.mat', '14auto.mat', '15auto.mat', '16auto.mat',
            '17auto.mat', '18auto.mat',
            '19auto.mat', '20auto.mat', '21auto.mat', '22auto.mat', '23auto.mat', '24auto.mat', '25auto.mat',
            '26auto.mat', '27auto.mat', '28auto.mat', '29auto.mat', '30auto.mat', '31auto.mat', '32auto.mat',
             '33auto.mat', '34auto.mat','35auto.mat', '36auto.mat',
             '37auto.mat', '38auto.mat', '39auto.mat', '40auto.mat', '41auto.mat', '42auto.mat', '43auto.mat',
                    '44auto.mat', '45auto.mat', '46auto.mat',
             '47auto.mat', '48auto.mat', '49auto.mat', '50auto.mat', '51auto.mat', '52auto.mat', '53auto.mat',
                    '54auto.mat', '55auto.mat', '56auto.mat',
             '57auto.mat', '58auto.mat', '59auto.mat', '60auto.mat', '61auto.mat', '62auto.mat', '63auto.mat',
                    '64auto.mat', '65auto.mat', '66auto.mat',
             '67auto.mat', '68auto.mat', '69auto.mat', '70auto.mat', '71auto.mat', '72auto.mat', '73auto.mat',
                    '74auto.mat', '75auto.mat', '76auto.mat',
             '77auto.mat', '78auto.mat', '79auto.mat', '80auto.mat', '81auto.mat', '82auto.mat', '83auto.mat',
                    '84auto.mat', '85auto.mat', '86auto.mat',
             '87auto.mat', '88auto.mat', '89auto.mat', '90auto.mat']
#Mulu = ['23.mat', '33.mat', '34.mat', '68.mat', '71.mat', '103.mat']
#newMulu = ['23auto.mat', '33auto.mat', '34auto.mat', '68auto.mat', '71auto.mat', '103auto.mat']
L = 80  #自相关的lags
# Rxxtotal = np.zeros((1,L+1))
M = np.zeros(len(Mulu))
for r,mulu in enumerate(Mulu):
    if r>0:
        M[r] = M[r-1]+m 
    data = scio.loadmat(mulu)['juzhen']
    m, n = data.shape
    Rxx = np.zeros((m, L))
    # Rxxl = np.zeros((m, L+1))
    for i,line in enumerate(data):
        for j in range(L):
            Rxx[i][j] = np.dot(data[i][:n-j], data[i][j:])
        Rxx0 = Rxx[i][0]
        Rxx[i] = Rxx[i]/Rxx0
        # Rxxl[i] = np.concatenate([list(Rxx[i]),[r]])
    scio.savemat(newMulu[r], {'Rxx': Rxx})
    # Rxxtotal = np.vstack((Rxxtotal,Rxxl))

