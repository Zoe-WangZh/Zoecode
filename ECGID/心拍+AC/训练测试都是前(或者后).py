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


Mulu = ['1.mat', '2.mat', '3.mat', '4.mat', '5.mat', '6.mat','7.mat', '8.mat', '9.mat','10.mat', 
             '11.mat', '12.mat', '13.mat', '14.mat', '15.mat', '16.mat', '17.mat', '18.mat', '19.mat', 
             '20.mat', '21.mat', '22.mat', '23.mat', '24.mat', '25.mat', '26.mat', '27.mat', '28.mat', 
             '29.mat', '30.mat', '31.mat', '32.mat', '33.mat', '34.mat', '35.mat', '36.mat', '37.mat', 
             '38.mat', '39.mat', '40.mat', '41.mat', '42.mat', '43.mat', '44.mat', '45.mat', '46.mat', 
             '47.mat', '48.mat', '49.mat', '50.mat', '51.mat', '52.mat', '53.mat', '54.mat', '55.mat', 
             '56.mat', '57.mat', '58.mat', '59.mat', '60.mat', '61.mat', '62.mat', '63.mat', '64.mat', 
             '65.mat', '66.mat', '67.mat', '68.mat', '69.mat', '70.mat', '71.mat', '72.mat', '73.mat', 
             '74.mat', '75.mat', '76.mat', '77.mat', '78.mat', '79.mat', '80.mat', '81.mat', '82.mat', 
             '83.mat', '84.mat', '85.mat', '86.mat', '87.mat', '88.mat', '89.mat', '90.mat'] 

L = 300  #自相关的lags
#Rxxtotaltrain = np.zeros((1,L+1))
#Rxxtotaltest = np.zeros((1,L+1))

x_train = np.zeros((1,L))
x_test = np.zeros((1,L))
y_train = np.zeros((1,1))
y_test = np.zeros((1,1))


for r,mulu in enumerate(Mulu):
    data = scio.loadmat(mulu)['juzhen']
    m, n = data.shape
    Rxx = np.zeros((m, L))
    Rxxl = np.zeros((m, L+1))
    for i,line in enumerate(data):
        for j in range(L):
            Rxx[i][j] = np.dot(data[i][:n-j], data[i][j:])
        Rxx0 = Rxx[i][0]
        Rxx[i] = Rxx[i]/Rxx0
        if r%2==0:
            Rxxl[i] = np.concatenate([list(Rxx[i]),[r]])
#        else:
#            Rxxl[i] = np.concatenate([list(Rxx[i]),[r-1]])
    m = m//10*7
    if r%2==0:
        x_train = np.vstack((x_train,Rxxl[0:m,:-1]))
        y_train = np.vstack((y_train,Rxxl[0:m,-1].reshape((-1,1))))
        x_test = np.vstack((x_test,Rxxl[m:,:-1]))
        y_test = np.vstack((y_test,Rxxl[m:,-1].reshape((-1,1))))
        
#        x_test = np.vstack((x_test,Rxxl[0:m,:-1]))
#        y_test = np.vstack((y_test,Rxxl[0:m,-1].reshape((-1,1))))
#        x_train = np.vstack((x_train,Rxxl[m:,:-1]))
#        y_train = np.vstack((y_train,Rxxl[m:,-1].reshape((-1,1))))
#    else:
#        Rxxtotaltest = np.vstack((Rxxtotaltest,Rxxl))


#x_train = Rxxtotaltrain[:,:-1]
#y_train = Rxxtotaltrain[:,-1]
#x_test = Rxxtotaltest[:,:-1]
#y_test = Rxxtotaltest[:,-1]


#for r,mulu in enumerate(ceshiMulu):
#    if r%2==0:
#        feature = scio.loadmat(mulu)['feature']
#        x_train = np.vstack((x_train, feature))
#        label = np.ones((feature.shape[0], 1))*r
#        y_train = np.vstack((y_train, label))
#    else:
#        feature = scio.loadmat(mulu)['feature']
#        x_test = np.vstack((x_test, feature))
#        label = np.ones((feature.shape[0], 1))*(r-1)
#        y_test = np.vstack((y_test, label))
x_train = x_train[1:,:]
x_test = x_test[1:,:]
y_train = y_train[1:,:]
y_test = y_test[1:,:]

print(x_train.shape)
print(x_test.shape)
#x_train_select = x_train[:,:9600]#[:,top]
#x_test_select = x_test[:,:9600]#[:,top]
#print(x_train_select.shape)


scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



############      普通SVM   ###################
pca = PCA(n_components = 0.95)
x_train_pca = pca.fit_transform(x_train)
print(x_train_pca.shape)
x_test_pca = pca.transform(x_test)
# C_range = np.logspace(-5, 5, 11)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份
# gamma_range = np.logspace(-5, 5, 11)

# model = GridSearchCV(svm.SVC(kernel='rbf'),
#     param_grid={'C': C_range, 'gamma': gamma_range}, cv = 5)

model = svm.SVC(kernel='rbf', C = 100, gamma = 1) #100,0.0001

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