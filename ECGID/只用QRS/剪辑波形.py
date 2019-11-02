import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import scipy.signal as scisignal
#x = scio.loadmat('71feature.mat')['feature']
#print(x.shape)
#print(x[:,-1])
#mulu = ['1.mat',
#        '2.mat']
for i in range(50,90):
    print('i=', i)
    x = np.loadtxt(str(i) + '.txt')
    print(max(x))
    print(min(x))
    print('\n')
#    y = scisignal.medfilt(x,31)
#    print(max(y))
#    print(min(y))
    
#    print('\n')
    #plt.plot(x[1100:1400])
    
    