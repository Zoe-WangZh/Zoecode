profile on;
addpath('D:\�ĵ��ź���Ŀ\MIT���ݿ�\MIT-BIT����ʧ�����ݿ�');
signal = fullfile('D:\�ĵ��ź���Ŀ\MIT���ݿ�\MIT-BIT����ʧ�����ݿ�','105.dat');
fid = fopen(signal,'r');
A = fread(fid,'uint4');
fclose(fid);
B = dec2hex(A);
m=10
B = B(1:1600*m);

C = char(length(B),16);


for i = 1:length(B)/16
    c = DES(B(16*i-15:i*16),'1999012513578642',1);
    C(i,1:16) = c;
end
profile viewer
%B��ԭʼ���� C�Ǽ��ܺ������ D�ǽ��ܺ������