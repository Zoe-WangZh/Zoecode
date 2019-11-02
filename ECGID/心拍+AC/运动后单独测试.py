# from tpot import TPOTClassifier
from __future__ import division
import random
import math
from operator import itemgetter
import random
# from deap import creator, base, tools, algorithms
import numpy as np
import scipy.io as scio
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# Mulu = ['01feature.mat', '02feature.mat', '03feature.mat', '04feature.mat', '05feature.mat', '06feature.mat', '07feature.mat',
#         '08feature.mat', '09feature.mat', '10feature.mat', '11feature.mat', '12feature.mat', '13feature.mat', '14feature.mat',
#         '15feature.mat', '16feature.mat', '17feature.mat', '18feature.mat', '19feature.mat', '20feature.mat', '21feature.mat',
#         '22feature.mat', '23feature.mat', '24feature.mat', '25feature.mat', '26feature.mat', '27feature.mat', '28feature.mat',
#         '29feature.mat', '30feature.mat', '31feature.mat', '32feature.mat', '33feature.mat', '34feature.mat', '35feature.mat',
#         '36feature.mat', '37feature.mat', '38feature.mat', '39feature.mat', '40feature.mat', '41feature.mat', '42feature.mat',
#         '43feature.mat', '44feature.mat', '45feature.mat', '46feature.mat', '47feature.mat', '48feature.mat', '49feature.mat',
#         '50feature.mat', '51feature.mat', '52feature.mat', '53feature.mat', '54feature.mat', '55feature.mat', '56feature.mat',
#         '57feature.mat', '58feature.mat', '59feature.mat', '60feature.mat', '61feature.mat', '62feature.mat', '63feature.mat',
#         '64feature.mat', '65feature.mat', '66feature.mat', '67feature.mat', '68feature.mat', '69feature.mat', '70feature.mat',
#         '71feature.mat', '72feature.mat', '73feature.mat', '74feature.mat', '75feature.mat', '76feature.mat', '77feature.mat',
#         '78feature.mat', '79feature.mat', '80feature.mat', '81feature.mat', '82feature.mat', '83feature.mat', '84feature.mat',
#         '85feature.mat', '86feature.mat', '87feature.mat', '88feature.mat', '89feature.mat', '90feature.mat', '91feature.mat',
#         '92feature.mat', '93feature.mat', '94feature.mat', '95feature.mat', '96feature.mat', '97feature.mat', '98feature.mat',
#         '99feature.mat', '100feature.mat', '101feature.mat', '102feature.mat', '103feature.mat', '104feature.mat',
#         '105feature.mat', '106feature.mat']
#
# x = np.zeros((1, 10252))
#
# for r,mulu in enumerate(Mulu):
#     feature = scio.loadmat(mulu)['feature']
#     x = np.vstack((x, feature))
# x = x[1:,:]
# junzhi = x.mean(axis=0)
# biaozhuncha = x.std(axis=0)
# for r,mulu in enumerate(Mulu):
#     feature = scio.loadmat(mulu)['feature']
#     feature = (feature-junzhi)/biaozhuncha
#     scio.savemat(mulu[:9] + '_norm.mat', {'feature': feature})
#
# for r,mulu in enumerate(Mulu):
#     data = scio.loadmat(mulu[:9] + '_norm.mat')
#     feature = data['feature']
#     junzhi = feature.mean(axis=0)
#     fangcha = feature.var(axis=0)
#     scio.savemat(mulu[:2] + 'junzhi.mat', {'junzhi': junzhi})
#     scio.savemat(mulu[:2] + 'fangcha.mat', {'fangcha': fangcha})


ceshiMulu = ['01feature.mat', '02feature.mat', '03feature.mat', '04feature.mat', '05feature.mat', '06feature.mat',
             '07feature.mat', '08feature.mat', '09feature.mat', '10feature.mat', '11feature.mat', '12feature.mat',
             '13feature.mat', '14feature.mat', '15feature.mat', '16feature.mat', '17feature.mat', '18feature.mat',
             '19feature.mat', '20feature.mat', '21feature.mat', '22feature.mat', '23feature.mat', '24feature.mat',
             '25feature.mat', '26feature.mat', '27feature.mat', '28feature.mat', '29feature.mat', '30feature.mat',
             '31feature.mat', '32feature.mat', '33feature.mat', '34feature.mat', '35feature.mat', '36feature.mat',
             '37feature.mat', '38feature.mat', '39feature.mat', '40feature.mat', '41feature.mat', '42feature.mat',
             '43feature.mat', '44feature.mat', '45feature.mat', '46feature.mat', '47feature.mat', '48feature.mat',
             '49feature.mat', '50feature.mat', '51feature.mat', '52feature.mat', '53feature.mat', '54feature.mat',
             '55feature.mat', '56feature.mat', '57feature.mat', '58feature.mat', '59feature.mat', '60feature.mat',
             '61feature.mat', '62feature.mat', '63feature.mat', '64feature.mat', '65feature.mat', '66feature.mat',
             '67feature.mat', '68feature.mat', '69feature.mat', '70feature.mat', '71feature.mat', '72feature.mat',
             '73feature.mat', '74feature.mat', '75feature.mat', '76feature.mat', '77feature.mat', '78feature.mat',
             '79feature.mat', '80feature.mat', '81feature.mat', '82feature.mat', '83feature.mat', '84feature.mat',
             '85feature.mat', '86feature.mat', '87feature.mat', '88feature.mat', '89feature.mat', '90feature.mat',
             '91feature.mat', '92feature.mat', '93feature.mat', '94feature.mat', '95feature.mat', '96feature.mat',
             '97feature.mat', '98feature.mat', '99feature.mat', '100feature.mat', '101feature.mat', 
             '102feature.mat', '103feature.mat', '104feature.mat', '105feature.mat', '106feature.mat']

x_train = np.zeros((1, 10252))
x_test = np.zeros((1, 10252))
y_train = np.zeros((1,1))
y_test = np.zeros((1,1))

for r,mulu in enumerate(ceshiMulu):
    # if r%2==0:
        # feature = scio.loadmat(mulu)['feature']
        # x_train = np.vstack((x_train, feature))
        # label = np.ones((feature.shape[0], 1))*r
        # y_train = np.vstack((y_train, label))
    # else:
    if r%2==1:
        feature = scio.loadmat(mulu)['feature']
        print(mulu)
        label = np.ones((feature.shape[0], 1)) * (r - 1)
        print(label[0])
        num = feature.shape[0]//10*7
        print(r,num)
        x_train = np.vstack((x_train, feature[:num,:]))
        x_test = np.vstack((x_test, feature[num:,:]))
        # label = np.ones((feature.shape[0], 1))*(r-1)
        y_train = np.vstack((y_train, label[:num]))
        y_test = np.vstack((y_test, label[num:]))
x_train = x_train[1:,:]
x_test = x_test[1:,:]
y_train = y_train[1:,:]
y_test = y_test[1:,:]

# print(x_train.shape)
# x_train_select = x_train[:,top]
# x_test_select = x_test[:,top]
# print(x_train_select.shape)

#scaler = MinMaxScaler().fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

############      普通SVM   ###################
pca = PCA(n_components = 0.95)
x_train_pca = pca.fit_transform(x_train)
print(x_train_pca.shape)
x_test_pca = pca.transform(x_test)
print(x_test_pca.shape)
# C_range = np.logspace(-5, 5, 11)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份
# gamma_range = np.logspace(-5, 5, 11)

# model = GridSearchCV(svm.SVC(kernel='rbf'),
#     param_grid={'C': C_range, 'gamma': gamma_range}, cv = 5)

model = svm.SVC(kernel='rbf', C = 100, gamma = 0.0001)

model.fit(x_train_pca, y_train)

# print("The best parameters are %s with a score of %0.2f"
#       % (model.best_params_, model.best_score_))#找到最佳超参数
y_train_hat = model.predict(x_train_pca)
print('train_accuracy = ', accuracy_score(y_train_hat, y_train))
y_test_hat = model.predict(x_test_pca)
print('test_accuracy = ', accuracy_score(y_test_hat, y_test))
# print('y_test:', y_test)
# print('y_test_hat:', y_test_hat)

test_precision_score = precision_score(y_test, y_test_hat, labels=None, pos_label=1,
    average='macro', sample_weight=None)
test_recall_score = recall_score(y_test, y_test_hat, labels=None, pos_label=1,
    average='macro', sample_weight=None)
print('test_precision_score =', test_precision_score)
print('test_recall_score =', test_recall_score)