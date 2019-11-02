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

Mulu = ['01feature.mat', '02feature.mat', '03feature.mat', '04feature.mat', '05feature.mat', '06feature.mat', '07feature.mat', 
        '08feature.mat', '09feature.mat', '10feature.mat', '11feature.mat', '12feature.mat', '13feature.mat', '14feature.mat', 
        '15feature.mat', '16feature.mat', '17feature.mat', '18feature.mat', '19feature.mat', '20feature.mat', '21feature.mat', 
        '22feature.mat', '23feature.mat', '24feature.mat', '25feature.mat', '26feature.mat', '27feature.mat', '28feature.mat',
        '29feature.mat', '30feature.mat', '31feature.mat', '32feature.mat', '33feature.mat', '34feature.mat', '35feature.mat', 
        '36feature.mat', '37feature.mat', '38feature.mat', '39feature.mat', '40feature.mat', '41feature.mat', '42feature.mat', 
        '43feature.mat', '44feature.mat'] #'47feature.mat', '48feature.mat', '49feature.mat', 
#        '50feature.mat', '51feature.mat', '52feature.mat', '53feature.mat', '54feature.mat' #'55feature.mat', '56feature.mat',
        # '57feature.mat', '58feature.mat', '59feature.mat', '60feature.mat', '61feature.mat', '62feature.mat', '63feature.mat', 
        # '64feature.mat', '65feature.mat', '66feature.mat', '67feature.mat', '68feature.mat', '69feature.mat', '70feature.mat', 
        # '71feature.mat', '72feature.mat', '73feature.mat', '74feature.mat', '75feature.mat', '76feature.mat', '77feature.mat', 
        # '78feature.mat', '79feature.mat', '80feature.mat', '81feature.mat', '82feature.mat', '83feature.mat', '84feature.mat', 
        # '85feature.mat', '86feature.mat', '87feature.mat', '88feature.mat', '89feature.mat', '90feature.mat', '91feature.mat', 
        # '92feature.mat', '93feature.mat', '94feature.mat', '95feature.mat', '96feature.mat', '97feature.mat', '98feature.mat', 
        # '99feature.mat', '100feature.mat', '101feature.mat', '102feature.mat', '103feature.mat', '104feature.mat', 
        # '105feature.mat', '106feature.mat']

x = np.zeros((1, 10252))

for r,mulu in enumerate(Mulu):
    feature = scio.loadmat(mulu)['feature']
    x = np.vstack((x, feature))
x = x[1:,:]
junzhi = x.mean(axis=0)
biaozhuncha = x.std(axis=0)
for r,mulu in enumerate(Mulu):
    feature = scio.loadmat(mulu)['feature']
    feature = (feature-junzhi)/biaozhuncha
    scio.savemat(mulu[:9] + '_norm.mat', {'feature': feature})

for r,mulu in enumerate(Mulu):
    data = scio.loadmat(mulu[:9] + '_norm.mat')
    feature = data['feature']
    junzhi = feature.mean(axis=0)
    fangcha = feature.var(axis=0)
    scio.savemat(mulu[:2] + 'junzhi.mat', {'junzhi': junzhi})
    scio.savemat(mulu[:2] + 'fangcha.mat', {'fangcha': fangcha})

data01 = scio.loadmat('01feature_norm.mat')
feature01 = data01['feature']
data02 = scio.loadmat('02feature_norm.mat')
feature02 = data02['feature']
feature0102 = np.vstack((feature01, feature02))
junzhi0102 = feature0102.mean(axis=0)
fangcha0102 = feature0102.var(axis=0)
scio.savemat('0102junzhi.mat',{'junzhi0102':junzhi0102})
scio.savemat('0102fangcha.mat',{'fangcha0102':fangcha0102})

data03 = scio.loadmat('03feature_norm.mat')
feature03 = data03['feature']
data04 = scio.loadmat('04feature_norm.mat')
feature04 = data04['feature']
feature0304 = np.vstack((feature03, feature04))
junzhi0304 = feature0304.mean(axis=0)
fangcha0304 = feature0304.var(axis=0)
scio.savemat('0304junzhi.mat',{'junzhi0304':junzhi0304})
scio.savemat('0304fangcha.mat',{'fangcha0304':fangcha0304})

data05 = scio.loadmat('05feature_norm.mat')
feature05 = data05['feature']
data06 = scio.loadmat('06feature_norm.mat')
feature06 = data06['feature']
feature0506 = np.vstack((feature05, feature06))
junzhi0506 = feature0506.mean(axis=0)
fangcha0506 = feature0506.var(axis=0)
scio.savemat('0506junzhi.mat',{'junzhi0506':junzhi0506})
scio.savemat('0506fangcha.mat',{'fangcha0506':fangcha0506})

data07 = scio.loadmat('07feature_norm.mat')
feature07 = data07['feature']
data08 = scio.loadmat('08feature_norm.mat')
feature08 = data08['feature']
feature0708 = np.vstack((feature07, feature08))
junzhi0708 = feature0708.mean(axis=0)
fangcha0708 = feature0708.var(axis=0)
scio.savemat('0708junzhi.mat',{'junzhi0708':junzhi0708})
scio.savemat('0708fangcha.mat',{'fangcha0708':fangcha0708})

data09 = scio.loadmat('09feature_norm.mat')
feature09 = data09['feature']
data10 = scio.loadmat('10feature_norm.mat')
feature10 = data10['feature']
feature0910 = np.vstack((feature09, feature10))
junzhi0910 = feature0910.mean(axis=0)
fangcha0910 = feature0910.var(axis=0)
scio.savemat('0910junzhi.mat',{'junzhi0910':junzhi0910})
scio.savemat('0910fangcha.mat',{'fangcha0910':fangcha0910})

data11 = scio.loadmat('11feature_norm.mat')
feature11 = data11['feature']
data12 = scio.loadmat('12feature_norm.mat')
feature12 = data12['feature']
feature1112 = np.vstack((feature11, feature12))
junzhi1112 = feature1112.mean(axis=0)
fangcha1112 = feature1112.var(axis=0)
scio.savemat('1112junzhi.mat',{'junzhi1112':junzhi1112})
scio.savemat('1112fangcha.mat',{'fangcha1112':fangcha1112})

data13 = scio.loadmat('13feature_norm.mat')
feature13 = data13['feature']
data14 = scio.loadmat('14feature_norm.mat')
feature14 = data14['feature']
feature1314 = np.vstack((feature13, feature14))
junzhi1314 = feature1314.mean(axis=0)
fangcha1314 = feature1314.var(axis=0)
scio.savemat('1314junzhi.mat',{'junzhi1314':junzhi1314})
scio.savemat('1314fangcha.mat',{'fangcha1314':fangcha1314})

data15 = scio.loadmat('15feature_norm.mat')
feature15 = data15['feature']
data16 = scio.loadmat('16feature_norm.mat')
feature16 = data16['feature']
feature1516 = np.vstack((feature15, feature16))
junzhi1516 = feature1516.mean(axis=0)
fangcha1516 = feature1516.var(axis=0)
scio.savemat('1516junzhi.mat',{'junzhi1516':junzhi1516})
scio.savemat('1516fangcha.mat',{'fangcha1516':fangcha1516})

data17 = scio.loadmat('17feature_norm.mat')
feature17 = data17['feature']
data18 = scio.loadmat('18feature_norm.mat')
feature18 = data18['feature']
feature1718 = np.vstack((feature17, feature18))
junzhi1718 = feature1718.mean(axis=0)
fangcha1718 = feature1718.var(axis=0)
scio.savemat('1718junzhi.mat',{'junzhi1718':junzhi1718})
scio.savemat('1718fangcha.mat',{'fangcha1718':fangcha1718})

data19 = scio.loadmat('19feature_norm.mat')
feature19 = data19['feature']
data20 = scio.loadmat('20feature_norm.mat')
feature20 = data20['feature']
feature1920 = np.vstack((feature19, feature20))
junzhi1920 = feature1920.mean(axis=0)
fangcha1920 = feature1920.var(axis=0)
scio.savemat('1920junzhi.mat',{'junzhi1920':junzhi1920})
scio.savemat('1920fangcha.mat',{'fangcha1920':fangcha1920})

data21 = scio.loadmat('21feature_norm.mat')
feature21 = data21['feature']
data22 = scio.loadmat('22feature_norm.mat')
feature22 = data22['feature']
feature2122 = np.vstack((feature21, feature22))
junzhi2122 = feature2122.mean(axis=0)
fangcha2122 = feature2122.var(axis=0)
scio.savemat('2122junzhi.mat',{'junzhi2122':junzhi2122})
scio.savemat('2122fangcha.mat',{'fangcha2122':fangcha2122})

data23 = scio.loadmat('23feature_norm.mat')
feature23 = data23['feature']
data24 = scio.loadmat('24feature_norm.mat')
feature24 = data24['feature']
feature2324 = np.vstack((feature23, feature24))
junzhi2324 = feature2324.mean(axis=0)
fangcha2324 = feature2324.var(axis=0)
scio.savemat('2324junzhi.mat',{'junzhi2324':junzhi2324})
scio.savemat('2324fangcha.mat',{'fangcha2324':fangcha2324})

data25 = scio.loadmat('25feature_norm.mat')
feature25 = data25['feature']
data26 = scio.loadmat('26feature_norm.mat')
feature26 = data26['feature']
feature2526 = np.vstack((feature25, feature26))
junzhi2526 = feature2526.mean(axis=0)
fangcha2526 = feature2526.var(axis=0)
scio.savemat('2526junzhi.mat',{'junzhi2526':junzhi2526})
scio.savemat('2526fangcha.mat',{'fangcha2526':fangcha2526})

data27 = scio.loadmat('27feature_norm.mat')
feature27 = data27['feature']
data28 = scio.loadmat('28feature_norm.mat')
feature28 = data28['feature']
feature2728 = np.vstack((feature27, feature28))
junzhi2728 = feature2728.mean(axis=0)
fangcha2728 = feature2728.var(axis=0)
scio.savemat('2728junzhi.mat',{'junzhi2728':junzhi2728})
scio.savemat('2728fangcha.mat',{'fangcha2728':fangcha2728})

data29 = scio.loadmat('29feature_norm.mat')
feature29 = data29['feature']
data30 = scio.loadmat('30feature_norm.mat')
feature30 = data30['feature']
feature2930 = np.vstack((feature29, feature30))
junzhi2930 = feature2930.mean(axis=0)
fangcha2930 = feature2930.var(axis=0)
scio.savemat('2930junzhi.mat',{'junzhi2930':junzhi2930})
scio.savemat('2930fangcha.mat',{'fangcha2930':fangcha2930})

data31 = scio.loadmat('31feature_norm.mat')
feature31 = data31['feature']
data32 = scio.loadmat('32feature_norm.mat')
feature32 = data32['feature']
feature3132 = np.vstack((feature31, feature32))
junzhi3132 = feature3132.mean(axis=0)
fangcha3132 = feature3132.var(axis=0)
scio.savemat('3132junzhi.mat',{'junzhi3132':junzhi3132})
scio.savemat('3132fangcha.mat',{'fangcha3132':fangcha3132})

data33 = scio.loadmat('33feature_norm.mat')
feature33 = data33['feature']
data34 = scio.loadmat('34feature_norm.mat')
feature34 = data34['feature']
feature3334 = np.vstack((feature33, feature34))
junzhi3334 = feature3334.mean(axis=0)
fangcha3334 = feature3334.var(axis=0)
scio.savemat('3334junzhi.mat',{'junzhi3334':junzhi3334})
scio.savemat('3334fangcha.mat',{'fangcha3334':fangcha3334})

data35 = scio.loadmat('35feature_norm.mat')
feature35 = data35['feature']
data36 = scio.loadmat('36feature_norm.mat')
feature36 = data36['feature']
feature3536 = np.vstack((feature35, feature36))
junzhi3536 = feature3536.mean(axis=0)
fangcha3536 = feature3536.var(axis=0)
scio.savemat('3536junzhi.mat',{'junzhi3536':junzhi3536})
scio.savemat('3536fangcha.mat',{'fangcha3536':fangcha3536})

data37 = scio.loadmat('37feature_norm.mat')
feature37 = data37['feature']
data38 = scio.loadmat('38feature_norm.mat')
feature38 = data38['feature']
feature3738 = np.vstack((feature37, feature38))
junzhi3738 = feature3738.mean(axis=0)
fangcha3738 = feature3738.var(axis=0)
scio.savemat('3738junzhi.mat',{'junzhi3738':junzhi3738})
scio.savemat('3738fangcha.mat',{'fangcha3738':fangcha3738})

data39 = scio.loadmat('39feature_norm.mat')
feature39 = data39['feature']
data40 = scio.loadmat('40feature_norm.mat')
feature40 = data40['feature']
feature3940 = np.vstack((feature39, feature40))
junzhi3940 = feature3940.mean(axis=0)
fangcha3940 = feature3940.var(axis=0)
scio.savemat('3940junzhi.mat',{'junzhi3940':junzhi3940})
scio.savemat('3940fangcha.mat',{'fangcha3940':fangcha3940})

data41 = scio.loadmat('41feature_norm.mat')
feature41 = data41['feature']
data42 = scio.loadmat('42feature_norm.mat')
feature42 = data42['feature']
feature4142 = np.vstack((feature41, feature42))
junzhi4142 = feature4142.mean(axis=0)
fangcha4142 = feature4142.var(axis=0)
scio.savemat('4142junzhi.mat',{'junzhi4142':junzhi4142})
scio.savemat('4142fangcha.mat',{'fangcha4142':fangcha4142})

data43 = scio.loadmat('43feature_norm.mat')
feature43 = data43['feature']
data44 = scio.loadmat('44feature_norm.mat')
feature44 = data44['feature']
feature4344 = np.vstack((feature43, feature44))
junzhi4344 = feature4344.mean(axis=0)
fangcha4344 = feature4344.var(axis=0)
scio.savemat('4344junzhi.mat',{'junzhi4344':junzhi4344})
scio.savemat('4344fangcha.mat',{'fangcha4344':fangcha4344})

#data45 = scio.loadmat('45feature_norm.mat')
#feature45 = data45['feature']
#data46 = scio.loadmat('46feature_norm.mat')
#feature46 = data46['feature']
#feature4546 = np.vstack((feature45, feature46))
#junzhi4546 = feature4546.mean(axis=0)
#fangcha4546 = feature4546.var(axis=0)
#scio.savemat('4546junzhi.mat',{'junzhi4546':junzhi4546})
#scio.savemat('4546fangcha.mat',{'fangcha4546':fangcha4546})

#data47 = scio.loadmat('47feature_norm.mat')
#feature47 = data47['feature']
#data48 = scio.loadmat('48feature_norm.mat')
#feature48 = data48['feature']
#feature4748 = np.vstack((feature47, feature48))
#junzhi4748 = feature4748.mean(axis=0)
#fangcha4748 = feature4748.var(axis=0)
#scio.savemat('4748junzhi.mat',{'junzhi4748':junzhi4748})
#scio.savemat('4748fangcha.mat',{'fangcha4748':fangcha4748})
#
#data49 = scio.loadmat('49feature_norm.mat')
#feature49 = data49['feature']
#data50 = scio.loadmat('50feature_norm.mat')
#feature50 = data50['feature']
#feature4950 = np.vstack((feature49, feature50))
#junzhi4950 = feature4950.mean(axis=0)
#fangcha4950 = feature4950.var(axis=0)
#scio.savemat('4950junzhi.mat',{'junzhi4950':junzhi4950})
#scio.savemat('4950fangcha.mat',{'fangcha4950':fangcha4950})
#
#data51 = scio.loadmat('51feature_norm.mat')
#feature51 = data51['feature']
#data52 = scio.loadmat('52feature_norm.mat')
#feature52 = data52['feature']
#feature5152 = np.vstack((feature51, feature52))
#junzhi5152 = feature5152.mean(axis=0)
#fangcha5152 = feature5152.var(axis=0)
#scio.savemat('5152junzhi.mat',{'junzhi5152':junzhi5152})
#scio.savemat('5152fangcha.mat',{'fangcha5152':fangcha5152})
#
#data53 = scio.loadmat('53feature_norm.mat')
#feature53 = data53['feature']
#data54 = scio.loadmat('54feature_norm.mat')
#feature54 = data54['feature']
#feature5354 = np.vstack((feature53, feature54))
#junzhi5354 = feature5354.mean(axis=0)
#fangcha5354 = feature5354.var(axis=0)
#scio.savemat('5354junzhi.mat',{'junzhi5354':junzhi5354})
#scio.savemat('5354fangcha.mat',{'fangcha5354':fangcha5354})
#
#data55 = scio.loadmat('55feature_norm.mat')
#feature55 = data55['feature']
#data56 = scio.loadmat('56feature_norm.mat')
#feature56 = data56['feature']
#feature5556 = np.vstack((feature55, feature56))
#junzhi5556 = feature5556.mean(axis=0)
#fangcha5556 = feature5556.var(axis=0)
#scio.savemat('5556junzhi.mat',{'junzhi5556':junzhi5556})
#scio.savemat('5556fangcha.mat',{'fangcha5556':fangcha5556})



mulu1 = ['01junzhi.mat', '02junzhi.mat', '03junzhi.mat', '04junzhi.mat', '05junzhi.mat', '06junzhi.mat', '07junzhi.mat',
        '08junzhi.mat', '09junzhi.mat', '10junzhi.mat', '11junzhi.mat', '12junzhi.mat', '13junzhi.mat', '14junzhi.mat',
         '15junzhi.mat', '16junzhi.mat', '17junzhi.mat', '18junzhi.mat', '19junzhi.mat', '20junzhi.mat',
     '21junzhi.mat', '22junzhi.mat', '23junzhi.mat', '24junzhi.mat',
     '25junzhi.mat', '26junzhi.mat', '27junzhi.mat', '28junzhi.mat',
     '29junzhi.mat', '30junzhi.mat', '31junzhi.mat', '32junzhi.mat', '33junzhi.mat', '34junzhi.mat',
      '35junzhi.mat', '36junzhi.mat',
         '37junzhi.mat', '38junzhi.mat', '39junzhi.mat', '40junzhi.mat',
         '41junzhi.mat', '42junzhi.mat', '43junzhi.mat', '44junzhi.mat']
#         '45junzhi.mat', '46junzhi.mat', '47junzhi.mat', '48junzhi.mat',
#         '49junzhi.mat', '50junzhi.mat', '51junzhi.mat', '52junzhi.mat',
#         '53junzhi.mat', '54junzhi.mat', '55junzhi.mat', '56junzhi.mat',]
mulu2 = ['01fangcha.mat', '02fangcha.mat', '03fangcha.mat', '04fangcha.mat', '05fangcha.mat', '06fangcha.mat', '07fangcha.mat',
        '08fangcha.mat', '09fangcha.mat', '10fangcha.mat', '11fangcha.mat', '12fangcha.mat', '13fangcha.mat', '14fangcha.mat',
         '15fangcha.mat', '16fangcha.mat', '17fangcha.mat', '18fangcha.mat', '19fangcha.mat', '20fangcha.mat',
     '21fangcha.mat', '22fangcha.mat', '23fangcha.mat', '24fangcha.mat',
     '25fangcha.mat', '26fangcha.mat', '27fangcha.mat', '28fangcha.mat',
     '29fangcha.mat', '30fangcha.mat', '31fangcha.mat', '32fangcha.mat',
        '33fangcha.mat', '34fangcha.mat', '35fangcha.mat', '36fangcha.mat',
         '37fangcha.mat', '38fangcha.mat', '39fangcha.mat', '40fangcha.mat',
         '41fangcha.mat', '42fangcha.mat', '43fangcha.mat', '44fangcha.mat']
#         '45fangcha.mat', '46fangcha.mat', '47fangcha.mat', '48fangcha.mat',
#         '49fangcha.mat', '50fangcha.mat', '51fangcha.mat', '52fangcha.mat',
#         '53fangcha.mat', '54fangcha.mat', '55fangcha.mat', '56fangcha.mat',]
junzhi01 = scio.loadmat(mulu1[0])['junzhi']
fangcha01 = scio.loadmat(mulu2[0])['fangcha']
junzhi02 = scio.loadmat(mulu1[1])['junzhi']
fangcha02 = scio.loadmat(mulu2[1])['fangcha']
junzhi03 = scio.loadmat(mulu1[2])['junzhi']
fangcha03 = scio.loadmat(mulu2[2])['fangcha']
junzhi04 = scio.loadmat(mulu1[3])['junzhi']
fangcha04 = scio.loadmat(mulu2[3])['fangcha']
junzhi05 = scio.loadmat(mulu1[4])['junzhi']
fangcha05 = scio.loadmat(mulu2[4])['fangcha']
junzhi06 = scio.loadmat(mulu1[5])['junzhi']
fangcha06 = scio.loadmat(mulu2[5])['fangcha']
junzhi07 = scio.loadmat(mulu1[6])['junzhi']
fangcha07 = scio.loadmat(mulu2[6])['fangcha']
junzhi08 = scio.loadmat(mulu1[7])['junzhi']
fangcha08 = scio.loadmat(mulu2[7])['fangcha']
junzhi09 = scio.loadmat(mulu1[8])['junzhi']
fangcha09 = scio.loadmat(mulu2[8])['fangcha']
junzhi10 = scio.loadmat(mulu1[9])['junzhi']
fangcha10 = scio.loadmat(mulu2[9])['fangcha']
junzhi11 = scio.loadmat(mulu1[10])['junzhi']
fangcha11 = scio.loadmat(mulu2[10])['fangcha']
junzhi12 = scio.loadmat(mulu1[11])['junzhi']
fangcha12 = scio.loadmat(mulu2[11])['fangcha']
junzhi13 = scio.loadmat(mulu1[12])['junzhi']
fangcha13 = scio.loadmat(mulu2[12])['fangcha']
junzhi14 = scio.loadmat(mulu1[13])['junzhi']
fangcha14 = scio.loadmat(mulu2[13])['fangcha']
junzhi15 = scio.loadmat(mulu1[14])['junzhi']
fangcha15 = scio.loadmat(mulu2[14])['fangcha']
junzhi16 = scio.loadmat(mulu1[15])['junzhi']
fangcha16 = scio.loadmat(mulu2[15])['fangcha']
junzhi17 = scio.loadmat(mulu1[16])['junzhi']
fangcha17 = scio.loadmat(mulu2[16])['fangcha']
junzhi18 = scio.loadmat(mulu1[17])['junzhi']
fangcha18 = scio.loadmat(mulu2[17])['fangcha']
junzhi19 = scio.loadmat(mulu1[18])['junzhi']
fangcha19 = scio.loadmat(mulu2[18])['fangcha']
junzhi20 = scio.loadmat(mulu1[19])['junzhi']
fangcha20 = scio.loadmat(mulu2[19])['fangcha']
junzhi21 = scio.loadmat(mulu1[20])['junzhi']
fangcha21 = scio.loadmat(mulu2[20])['fangcha']
junzhi22 = scio.loadmat(mulu1[21])['junzhi']
fangcha22 = scio.loadmat(mulu2[21])['fangcha']
junzhi23 = scio.loadmat(mulu1[22])['junzhi']
fangcha23 = scio.loadmat(mulu2[22])['fangcha']
junzhi24 = scio.loadmat(mulu1[23])['junzhi']
fangcha24 = scio.loadmat(mulu2[23])['fangcha']
junzhi25 = scio.loadmat(mulu1[24])['junzhi']
fangcha25 = scio.loadmat(mulu2[24])['fangcha']
junzhi26 = scio.loadmat(mulu1[25])['junzhi']
fangcha26 = scio.loadmat(mulu2[25])['fangcha']
junzhi27 = scio.loadmat(mulu1[26])['junzhi']
fangcha27 = scio.loadmat(mulu2[26])['fangcha']
junzhi28 = scio.loadmat(mulu1[27])['junzhi']
fangcha28 = scio.loadmat(mulu2[27])['fangcha']
junzhi29 = scio.loadmat(mulu1[28])['junzhi']
fangcha29 = scio.loadmat(mulu2[28])['fangcha']
junzhi30 = scio.loadmat(mulu1[29])['junzhi']
fangcha30 = scio.loadmat(mulu2[29])['fangcha']
junzhi31 = scio.loadmat(mulu1[30])['junzhi']
fangcha31 = scio.loadmat(mulu2[30])['fangcha']
junzhi32 = scio.loadmat(mulu1[31])['junzhi']
fangcha32 = scio.loadmat(mulu2[31])['fangcha']
junzhi33 = scio.loadmat(mulu1[32])['junzhi']
fangcha33 = scio.loadmat(mulu2[32])['fangcha']
junzhi34 = scio.loadmat(mulu1[33])['junzhi']
fangcha34 = scio.loadmat(mulu2[33])['fangcha']
junzhi35 = scio.loadmat(mulu1[34])['junzhi']
fangcha35 = scio.loadmat(mulu2[34])['fangcha']
junzhi36 = scio.loadmat(mulu1[35])['junzhi']
fangcha36 = scio.loadmat(mulu2[35])['fangcha']
junzhi37 = scio.loadmat(mulu1[36])['junzhi']
fangcha37 = scio.loadmat(mulu2[36])['fangcha']
junzhi38 = scio.loadmat(mulu1[37])['junzhi']
fangcha38 = scio.loadmat(mulu2[37])['fangcha']
junzhi39 = scio.loadmat(mulu1[38])['junzhi']
fangcha39 = scio.loadmat(mulu2[38])['fangcha']
junzhi40 = scio.loadmat(mulu1[39])['junzhi']
fangcha40 = scio.loadmat(mulu2[39])['fangcha']
junzhi41 = scio.loadmat(mulu1[40])['junzhi']
fangcha41 = scio.loadmat(mulu2[40])['fangcha']
junzhi42 = scio.loadmat(mulu1[41])['junzhi']
fangcha42 = scio.loadmat(mulu2[41])['fangcha']
junzhi43 = scio.loadmat(mulu1[42])['junzhi']
fangcha43 = scio.loadmat(mulu2[42])['fangcha']
junzhi44 = scio.loadmat(mulu1[43])['junzhi']
fangcha44 = scio.loadmat(mulu2[43])['fangcha']
#junzhi45 = scio.loadmat(mulu1[44])['junzhi']
#fangcha45 = scio.loadmat(mulu2[44])['fangcha']
#junzhi46 = scio.loadmat(mulu1[45])['junzhi']
#fangcha46 = scio.loadmat(mulu2[45])['fangcha']
#junzhi47 = scio.loadmat(mulu1[46])['junzhi']
#fangcha47 = scio.loadmat(mulu2[46])['fangcha']
#junzhi48 = scio.loadmat(mulu1[47])['junzhi']
#fangcha48 = scio.loadmat(mulu2[47])['fangcha']
#junzhi49 = scio.loadmat(mulu1[48])['junzhi']
#fangcha49 = scio.loadmat(mulu2[48])['fangcha']
#junzhi50 = scio.loadmat(mulu1[49])['junzhi']
#fangcha50 = scio.loadmat(mulu2[49])['fangcha']
#junzhi51 = scio.loadmat(mulu1[50])['junzhi']
#fangcha51 = scio.loadmat(mulu2[50])['fangcha']
#junzhi52 = scio.loadmat(mulu1[51])['junzhi']
#fangcha52 = scio.loadmat(mulu2[51])['fangcha']
#junzhi53 = scio.loadmat(mulu1[52])['junzhi']
#fangcha53 = scio.loadmat(mulu2[52])['fangcha']
#junzhi54 = scio.loadmat(mulu1[53])['junzhi']
#fangcha54 = scio.loadmat(mulu2[53])['fangcha']
# junzhi55 = scio.loadmat(mulu1[54])['junzhi']
# fangcha55 = scio.loadmat(mulu2[54])['fangcha']
# junzhi56 = scio.loadmat(mulu1[55])['junzhi']
# fangcha56 = scio.loadmat(mulu2[55])['fangcha']



w1 = 1/22 * (((fangcha0102+fangcha0304+fangcha0506+fangcha0708+fangcha0910+fangcha1112+fangcha1314+fangcha1516+fangcha1718+
              fangcha1920+fangcha2122+fangcha2324+fangcha2526+fangcha2728+fangcha2930+fangcha3132+
              fangcha3536+fangcha3738+fangcha3940+fangcha4142+fangcha4344+
              junzhi0102**2+junzhi0304**2+junzhi0506**2+junzhi0708**2+junzhi0910**2+junzhi1112**2+junzhi1314**2+junzhi1516**2+
              junzhi1718**2+junzhi1920**2+junzhi2122**2+junzhi2324**2+junzhi2526**2+junzhi2728**2+junzhi2930**2+junzhi3132**2+
              junzhi3536**2+junzhi3738**2+junzhi3940**2+junzhi4142**2+junzhi4344**2)/2)
            + (1+junzhi0102**2)/2/fangcha0102+(1+junzhi0304**2)/2/fangcha0304+(1+junzhi0506**2)/2/fangcha0506+
                (1+junzhi0708**2)/2/fangcha0708+(1+junzhi0910**2)/2/fangcha0910+(1+junzhi1112**2)/2/fangcha1112+
                (1+junzhi1314**2)/2/fangcha1314+(1+junzhi1516**2)/2/fangcha1516+(1+junzhi1718**2)/2/fangcha1718+
                (1+junzhi1920**2)/2/fangcha1920+(1+junzhi2122**2)/2/fangcha2122+(1+junzhi2324**2)/2/fangcha2324+
                (1+junzhi2526**2)/2/fangcha2526+(1+junzhi2728**2)/2/fangcha2728+(1+junzhi2930**2)/2/fangcha2930+
                (1+junzhi3132**2)/2/fangcha3132+(1+junzhi3536**2)/2/fangcha3536+
                (1+junzhi3738**2)/2/fangcha3738+(1+junzhi3940**2)/2/fangcha3940+(1+junzhi4142**2)/2/fangcha4142+
                (1+junzhi4344**2)/2/fangcha4344-22)

w2 = (fangcha01+(junzhi01-junzhi0102)**2)/2/fangcha0102 + (fangcha0102+(junzhi01-junzhi0102)**2)/2/fangcha01-1+\
    (fangcha02+(junzhi02-junzhi0102)**2)/2/fangcha0102 + (fangcha0102+(junzhi02-junzhi0102)**2)/2/fangcha02-1+\
    (fangcha03+(junzhi03-junzhi0304)**2)/2/fangcha0304 + (fangcha0304+(junzhi03-junzhi0304)**2)/2/fangcha03-1+\
    (fangcha04+(junzhi04-junzhi0304)**2)/2/fangcha0304 + (fangcha0304+(junzhi04-junzhi0304)**2)/2/fangcha04-1+\
    (fangcha05+(junzhi05-junzhi0506)**2)/2/fangcha0506 + (fangcha0506+(junzhi05-junzhi0506)**2)/2/fangcha05-1+\
    (fangcha06+(junzhi06-junzhi0506)**2)/2/fangcha0506 + (fangcha0506+(junzhi06-junzhi0506)**2)/2/fangcha06-1+\
    (fangcha07+(junzhi07-junzhi0708)**2)/2/fangcha0708 + (fangcha0708+(junzhi07-junzhi0708)**2)/2/fangcha07-1+\
    (fangcha08+(junzhi08-junzhi0708)**2)/2/fangcha0708 + (fangcha0708+(junzhi08-junzhi0708)**2)/2/fangcha08-1+\
    (fangcha09+(junzhi09-junzhi0910)**2)/2/fangcha0910 + (fangcha0910+(junzhi09-junzhi0910)**2)/2/fangcha09-1+\
    (fangcha10+(junzhi10-junzhi0910)**2)/2/fangcha0910 + (fangcha0910+(junzhi10-junzhi0910)**2)/2/fangcha10-1+\
    (fangcha11+(junzhi11-junzhi1112)**2)/2/fangcha1112 + (fangcha1112+(junzhi11-junzhi1112)**2)/2/fangcha11-1+\
    (fangcha12+(junzhi12-junzhi1112)**2)/2/fangcha1112 + (fangcha1112+(junzhi12-junzhi1112)**2)/2/fangcha12-1+\
    (fangcha13+(junzhi13-junzhi1314)**2)/2/fangcha1314 + (fangcha1314+(junzhi13-junzhi1314)**2)/2/fangcha13-1+\
    (fangcha14+(junzhi14-junzhi1314)**2)/2/fangcha1314 + (fangcha1314+(junzhi14-junzhi1314)**2)/2/fangcha14-1+\
    (fangcha15+(junzhi15-junzhi1516)**2)/2/fangcha1516 + (fangcha1516+(junzhi15-junzhi1516)**2)/2/fangcha15-1+\
    (fangcha16+(junzhi16-junzhi1516)**2)/2/fangcha1516 + (fangcha1516+(junzhi16-junzhi1516)**2)/2/fangcha16-1+\
    (fangcha17+(junzhi17-junzhi1718)**2)/2/fangcha1718 + (fangcha1718+(junzhi17-junzhi1718)**2)/2/fangcha17-1+\
    (fangcha18+(junzhi18-junzhi1718)**2)/2/fangcha1718 + (fangcha1718+(junzhi18-junzhi1718)**2)/2/fangcha18-1+\
    (fangcha19+(junzhi19-junzhi1920)**2)/2/fangcha1920 + (fangcha1920+(junzhi19-junzhi1920)**2)/2/fangcha19-1+\
    (fangcha20+(junzhi20-junzhi1920)**2)/2/fangcha1920 + (fangcha1920+(junzhi20-junzhi1920)**2)/2/fangcha20-1+\
    (fangcha21+(junzhi21-junzhi2122)**2)/2/fangcha2122 + (fangcha2122+(junzhi21-junzhi2122)**2)/2/fangcha21-1+\
    (fangcha22+(junzhi22-junzhi2122)**2)/2/fangcha2122 + (fangcha2122+(junzhi22-junzhi2122)**2)/2/fangcha22-1+\
    (fangcha23+(junzhi23-junzhi2324)**2)/2/fangcha2324 + (fangcha2324+(junzhi23-junzhi2324)**2)/2/fangcha23-1+\
    (fangcha24+(junzhi24-junzhi2324)**2)/2/fangcha2324 + (fangcha2324+(junzhi24-junzhi2324)**2)/2/fangcha24-1+\
    (fangcha25+(junzhi25-junzhi2526)**2)/2/fangcha2526 + (fangcha2526+(junzhi25-junzhi2526)**2)/2/fangcha25-1+\
    (fangcha26+(junzhi26-junzhi2526)**2)/2/fangcha2526 + (fangcha2526+(junzhi26-junzhi2526)**2)/2/fangcha26-1+\
    (fangcha27+(junzhi27-junzhi2728)**2)/2/fangcha2728 + (fangcha2728+(junzhi27-junzhi2728)**2)/2/fangcha27-1+\
    (fangcha28+(junzhi28-junzhi2728)**2)/2/fangcha2728 + (fangcha2728+(junzhi28-junzhi2728)**2)/2/fangcha28-1+\
    (fangcha29+(junzhi29-junzhi2930)**2)/2/fangcha2930 + (fangcha2930+(junzhi29-junzhi2930)**2)/2/fangcha29-1+\
    (fangcha30+(junzhi30-junzhi2930)**2)/2/fangcha2930 + (fangcha2930+(junzhi30-junzhi2930)**2)/2/fangcha30-1+\
    (fangcha31+(junzhi31-junzhi3132)**2)/2/fangcha3132 + (fangcha3132+(junzhi31-junzhi3132)**2)/2/fangcha31-1+\
    (fangcha32+(junzhi32-junzhi3132)**2)/2/fangcha3132 + (fangcha3132+(junzhi32-junzhi3132)**2)/2/fangcha32-1+\
    (fangcha35+(junzhi35-junzhi3536)**2)/2/fangcha3536 + (fangcha3536+(junzhi35-junzhi3536)**2)/2/fangcha35-1+\
    (fangcha36+(junzhi36-junzhi3536)**2)/2/fangcha3536 + (fangcha3536+(junzhi36-junzhi3536)**2)/2/fangcha36-1+\
    (fangcha37+(junzhi37-junzhi3738)**2)/2/fangcha3738 + (fangcha3738+(junzhi37-junzhi3738)**2)/2/fangcha37-1+\
    (fangcha38+(junzhi38-junzhi3738)**2)/2/fangcha3738 + (fangcha3738+(junzhi38-junzhi3738)**2)/2/fangcha38-1+\
    (fangcha39+(junzhi39-junzhi3940)**2)/2/fangcha3940 + (fangcha3940+(junzhi39-junzhi3940)**2)/2/fangcha39-1+\
    (fangcha40+(junzhi40-junzhi3940)**2)/2/fangcha3940 + (fangcha3940+(junzhi40-junzhi3940)**2)/2/fangcha40-1+\
    (fangcha41+(junzhi41-junzhi4142)**2)/2/fangcha4142 + (fangcha4142+(junzhi41-junzhi4142)**2)/2/fangcha41-1+\
    (fangcha42+(junzhi42-junzhi4142)**2)/2/fangcha4142 + (fangcha4142+(junzhi42-junzhi4142)**2)/2/fangcha42-1+\
    (fangcha43+(junzhi43-junzhi4344)**2)/2/fangcha4344 + (fangcha4344+(junzhi43-junzhi4344)**2)/2/fangcha43-1+\
    (fangcha44+(junzhi44-junzhi4344)**2)/2/fangcha4344 + (fangcha4344+(junzhi44-junzhi4344)**2)/2/fangcha44-1
#    (fangcha47+(junzhi47-junzhi4748)**2)/2/fangcha4748 + (fangcha4748+(junzhi47-junzhi4748)**2)/2/fangcha47-1+\
#    (fangcha48+(junzhi48-junzhi4748)**2)/2/fangcha4748 + (fangcha4748+(junzhi48-junzhi4748)**2)/2/fangcha48-1+\
#    (fangcha49+(junzhi49-junzhi4950)**2)/2/fangcha4950 + (fangcha4950+(junzhi49-junzhi4950)**2)/2/fangcha49-1+\
#    (fangcha50+(junzhi50-junzhi4950)**2)/2/fangcha4950 + (fangcha4950+(junzhi50-junzhi4950)**2)/2/fangcha50-1+\
#    (fangcha51+(junzhi51-junzhi5152)**2)/2/fangcha5152 + (fangcha5152+(junzhi51-junzhi5152)**2)/2/fangcha51-1+\
#    (fangcha52+(junzhi52-junzhi5152)**2)/2/fangcha5152 + (fangcha5152+(junzhi52-junzhi5152)**2)/2/fangcha52-1+\
#    (fangcha53+(junzhi53-junzhi5354)**2)/2/fangcha5354 + (fangcha5354+(junzhi53-junzhi5354)**2)/2/fangcha53-1+\
#    (fangcha54+(junzhi54-junzhi5354)**2)/2/fangcha5354 + (fangcha5354+(junzhi54-junzhi5354)**2)/2/fangcha54-1
#    (fangcha23+(junzhi23-junzhi2324)**2)/2/fangcha2324 + (fangcha2324+(junzhi23-junzhi2324)**2)/2/fangcha23-1+\
#    (fangcha24+(junzhi24-junzhi2324)**2)/2/fangcha2324 + (fangcha2324+(junzhi24-junzhi2324)**2)/2/fangcha24-1+\
#    (fangcha29+(junzhi29-junzhi2930)**2)/2/fangcha2930 + (fangcha2930+(junzhi29-junzhi2930)**2)/2/fangcha29-1+\
#    (fangcha30+(junzhi30-junzhi2930)**2)/2/fangcha2930 + (fangcha2930+(junzhi30-junzhi2930)**2)/2/fangcha30-1+\
#    (fangcha33+(junzhi33-junzhi3334)**2)/2/fangcha3334 + (fangcha3334+(junzhi33-junzhi3334)**2)/2/fangcha33-1+\
#    (fangcha34+(junzhi34-junzhi3334)**2)/2/fangcha3334 + (fangcha3334+(junzhi34-junzhi3334)**2)/2/fangcha34-1+\
#    (fangcha45+(junzhi45-junzhi4546)**2)/2/fangcha4546 + (fangcha4546+(junzhi45-junzhi4546)**2)/2/fangcha45-1+\
#    (fangcha46+(junzhi46-junzhi4546)**2)/2/fangcha4546 + (fangcha4546+(junzhi46-junzhi4546)**2)/2/fangcha46-1+\
w2 = w2/22
w = 0.2*w1 - 0.8*w2
sortedfeature = np.argsort(-w, axis=1)
sortedfeature = sortedfeature[0]
np.save('sortedfeature.npy',sortedfeature)



sortedfeature = np.load('sortedfeature.npy')
top = sortedfeature[:9000]
# print(np.sum(top<5000))
#print(np.sum(top[:1000]<9600))
# # print(top.shape)
# # ceshiMulu = ['01feature.mat', '02feature.mat', '03feature.mat', '04feature.mat', '05feature.mat', '06feature.mat',
# #              '07feature.mat', '08feature.mat', '09feature.mat',
# #     '10feature.mat', '11feature.mat', '12feature.mat', '13feature.mat', '14feature.mat', '15feature.mat',
# #              '16feature.mat', '17feature.mat', '18feature.mat',
# #     '19feature.mat', '20feature.mat', '21feature.mat', '22feature.mat', '23feature.mat', '24feature.mat',
# #              '25feature.mat', '26feature.mat', '27feature.mat',
# #     '28feature.mat', '29feature.mat', '30feature.mat', '31feature.mat', '32feature.mat','33feature.mat', '34feature.mat', '35feature.mat', '36feature.mat',
# #     '37feature.mat', '38feature.mat', '39feature.mat', '40feature.mat', '41feature.mat', '42feature.mat', '43feature.mat', '44feature.mat', 
ceshiMulu = ['45feature.mat', '46feature.mat', '47feature.mat', '48feature.mat', '49feature.mat', 
             '50feature.mat', '51feature.mat', '52feature.mat', '53feature.mat', '54feature.mat',
       '55feature.mat', '56feature.mat', '57feature.mat', '58feature.mat', '59feature.mat', 
       '60feature.mat', '61feature.mat', '62feature.mat', '63feature.mat', '64feature.mat', 
       '65feature.mat', '66feature.mat', '67feature.mat', '68feature.mat',
      '69feature.mat', '70feature.mat', '71feature.mat', '72feature.mat','73feature.mat', 
      '74feature.mat', '75feature.mat', '76feature.mat', '77feature.mat', '78feature.mat', 
      '79feature.mat', '80feature.mat', '81feature.mat', '82feature.mat', '83feature.mat', 
      '84feature.mat', '85feature.mat', '86feature.mat', '87feature.mat', '88feature.mat',   
      '89feature.mat', '90feature.mat'] #'91feature.mat', '92feature.mat', '93feature.mat', '94feature.mat', '95feature.mat', '96feature.mat',
#     '97feature.mat', '98feature.mat', '99feature.mat', '100feature.mat', '101feature.mat', '102feature.mat',  '105feature.mat',
#     '106feature.mat']

x_train = np.zeros((1, 10252))
x_test = np.zeros((1, 10252))
y_train = np.zeros((1,1))
y_test = np.zeros((1,1))

for r,mulu in enumerate(ceshiMulu):
    if r%2==0:
        feature = scio.loadmat(mulu)['feature']
        x_train = np.vstack((x_train, feature))
        label = np.ones((feature.shape[0], 1))*r
        y_train = np.vstack((y_train, label))
    else:
        feature = scio.loadmat(mulu)['feature']
        x_test = np.vstack((x_test, feature))
        label = np.ones((feature.shape[0], 1))*(r-1)
        y_test = np.vstack((y_test, label))
x_train = x_train[1:,:]
x_test = x_test[1:,:]
y_train = y_train[1:,:]
y_test = y_test[1:,:]

print(x_train.shape)
x_train_select = x_train[:,top]
x_test_select = x_test[:,top]
print(x_train_select.shape)
scaler = MinMaxScaler().fit(x_train_select)
x_train = scaler.transform(x_train_select)
x_test = scaler.transform(x_test_select)
############      普通SVM   ###################
pca = PCA(n_components = 0.95)
x_train_pca = pca.fit_transform(x_train)
print(x_train_pca.shape)
x_test_pca = pca.transform(x_test)
# C_range = np.logspace(-5, 5, 11)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份
# gamma_range = np.logspace(-5, 5, 11)

# model = GridSearchCV(svm.SVC(kernel='rbf'),
#     param_grid={'C': C_range, 'gamma': gamma_range}, cv = 5)

model = svm.SVC(kernel='rbf', C = 100, gamma = 1)

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