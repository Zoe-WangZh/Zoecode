from __future__ import print_function
import numpy as np
import scipy.io as sio
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import keras
#from __future__ import division
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

#L = 50  #自相关的lags
#Rxxtotaltrain = np.zeros((1,L+1))
#Rxxtotaltest = np.zeros((1,L+1))
#
#for r,mulu in enumerate(Mulu):
#    data = scio.loadmat(mulu)['juzhen']
#    m, n = data.shape
#    Rxx = np.zeros((m, L))
#    Rxxl = np.zeros((m, L+1))
#    for i,line in enumerate(data):
#        for j in range(L):
#            Rxx[i][j] = np.dot(data[i][:n-j], data[i][j:])
#        Rxx0 = Rxx[i][0]
#        Rxx[i] = Rxx[i]/Rxx0
#        if r%2==0:
#            Rxxl[i] = np.concatenate([list(Rxx[i]),[r]])
#        else:
#            Rxxl[i] = np.concatenate([list(Rxx[i]),[r-1]])
#    if r%2==0:
#        Rxxtotaltrain = np.vstack((Rxxtotaltrain,Rxxl))
#    else:
#        Rxxtotaltest = np.vstack((Rxxtotaltest,Rxxl))
#
#
x_train = np.zeros((1, 300))
x_test = np.zeros((1, 300))
y_train = np.zeros((1,1))
y_test = np.zeros((1,1))


for r,mulu in enumerate(Mulu):
    if r%2==0:
        feature = scio.loadmat(mulu)['juzhen']
        x_train = np.vstack((x_train, feature))
        label = np.ones((feature.shape[0], 1))*(r/2)
        y_train = np.vstack((y_train, label))
    else:
        feature = scio.loadmat(mulu)['juzhen']
        x_test = np.vstack((x_test, feature))
        label = np.ones((feature.shape[0], 1))*(r-1)/2
        y_test = np.vstack((y_test, label))
x_train = x_train[56:,:]
x_test = x_test[76:,:]
y_train = y_train[56:,:]
y_test = y_test[76:,:]

print(x_train.shape)
print(x_test.shape)
print(np.unique(y_train))
print(np.unique(y_test))
#x_train_select = x_train[:,:9600]#[:,top]
#x_test_select = x_test[:,:9600]#[:,top]
#print(x_train_select.shape)


scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


np.random.seed(2017)  #为了复现




x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#参数
#学习率
learning_rate = 0.01 
#迭代次数
epochs = 100
#每块训练样本数
batch_size = 128
#输入
n_input = 1
#步长
n_step = 300
#LSTM Cell
n_hidden = 128
#类别
n_classes = 45

#x标准化到0-1  y使用one-hot  输入 nxm的矩阵 每行m维切成n个输入
# X_train = x_train.reshape(-1, n_step, n_input)
# X_test = x_test.reshape(-1, n_step, n_input)
print(y_train.shape)
y_train = np_utils.to_categorical(y_train, num_classes=n_classes)
y_test = np_utils.to_categorical(y_test, num_classes=n_classes)
print(y_train.shape)
model = Sequential()
model.add(LSTM(n_hidden, activation='tanh', return_sequences=True, stateful=True, batch_input_shape=(n_hidden, n_step, n_input), unroll=True))
model.add(LSTM(n_hidden, return_sequences=True, stateful=True))
model.add(LSTM(n_hidden, stateful=True))
model.add(Dropout(0.2))
#通常丢弃率控制在20%~50%比较好，可以从20%开始尝试。如果比例太低则起不到效果，
#比例太高则会导致模型的欠学习。在输入层（可见层）和隐藏层都使用dropout。
#在每层都应用dropout被证明会取得好的效果。
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

adam = Adam(lr=learning_rate)
#显示模型细节
model.summary()
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, 
#                             monitor='val_acc',
#                             verbose=1,
#                             save_best_only='True',
#                             mode='auto',
#                             period=1)
#callbacks_list = [checkpoint]
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, 
    mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
# plot_model(model, to_file='model.png')
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, 
    write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, #0不显示 1显示
          validation_data=(x_test, y_test))
# scores = model.evaluate(x_test, y_test, verbose=0)
# print('LSTM test score:', scores[0]) #loss
# print('LSTM test accuracy:', scores[1])


