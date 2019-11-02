clear;clc;
%% �������ݣ�
fprintf('Loading data...\n');
tic;
load('N_dat.mat');
load('L_dat.mat');
load('R_dat.mat');
load('V_dat.mat');
load('data2.mat');
fprintf('Finished!\n');
toc;
fprintf('=============================================================\n');
%% ����ʹ����������ÿһ��5000�������ɱ�ǩ,one-hot���룻
fprintf('Data preprocessing...\n');
tic;
%% ==============ԭʼ�ź�==============
Nb=Nb(1:5000,:);Label1=repmat([1;0;0;0],1,5000); 
Vb=Vb(1:5000,:);Label2=repmat([0;1;0;0],1,5000);
Rb=Rb(1:5000,:);Label3=repmat([0;0;1;0],1,5000);
Lb=Lb(1:5000,:);Label4=repmat([0;0;0;1],1,5000);

Data=[Nb;Vb;Rb;Lb];
Label=[Label1,Label2,Label3,Label4];

clear Nb;clear Label1;
clear Rb;clear Label2;
clear Lb;clear Label3;
clear Vb;clear Label4;
Data=Data-repmat(mean(Data,2),1,250); %ʹ�źŵľ�ֵΪ0��ȥ�����ߵ�Ӱ�죻

%% ==============�����ź�==============
Nb1=Nb1(1:5000,:);Label11=repmat([1;0;0;0],1,5000); 
Vb1=Vb1(1:5000,:);Label12=repmat([0;1;0;0],1,5000);
Rb1=Rb1(1:5000,:);Label13=repmat([0;0;1;0],1,5000);
Lb1=Lb1(1:5000,:);Label14=repmat([0;0;0;1],1,5000);

Data1=[Nb1;Vb1;Rb1;Lb1];
Label_1=[Label11,Label12,Label13,Label14];

clear Nb1;clear Label11;
clear Rb1;clear Label12;
clear Lb1;clear Label13;
clear Vb1;clear Label14;
Data1=Data1-repmat(mean(Data1,2),1,250); %ʹ�źŵľ�ֵΪ0��ȥ�����ߵ�Ӱ��
fprintf('Finished!\n');
toc;
fprintf('=============================================================\n');
%% ���ݻ�����ģ��ѵ�����ԣ�
fprintf('Model training and testing...\n');
Nums=randperm(20000);      %�����������˳�򣬴ﵽ���ѡ��ѵ������������Ŀ�ģ�
train_x=Data(Nums(1:10000),:);
test_x=Data(Nums(10001:end),:);
train_y=Label(:,Nums(1:10000));
test_y=Label(:,Nums(10001:end));
train_x=train_x';
test_x=test_x';

newtest_x=Data1';
newtest_y=Label_1;
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 4, 'kernelsize', 31,'actv','relu') %convolution layer
    struct('type', 's', 'scale', 5,'pool','mean') %sub sampling layer
    struct('type', 'c', 'outputmaps', 8, 'kernelsize', 6,'actv','relu') %convolution layer
    struct('type', 's', 'scale', 3,'pool','mean') %subsampling layer
};
cnn.output = 'softmax';  %ȷ��cnn�ṹ��
                         %ȷ����������
opts.alpha = 0.05;       %ѧϰ�ʣ�
opts.batchsize = 16;     %batch���С��
opts.numepochs = 30;     %����epoch��

cnn = cnnsetup1d(cnn, train_x, train_y);      %����1D CNN;
cnn = cnntrain1d(cnn, train_x, train_y,opts); %ѵ��1D CNN;

%% =============ԭʼ�źŲ���==============
[er,bad,out] = cnntest1d(cnn, test_x, test_y);%����1D CNN;

[~,ptest]=max(out,[],1);
[~,test_yt]=max(test_y,[],1);

Correct_Predict=zeros(1,4);                     %ͳ�Ƹ���׼ȷ�ʣ�
Class_Num=zeros(1,4);                           %���õ���������
Conf_Mat=zeros(4);
for i=1:10000
    Class_Num(test_yt(i))=Class_Num(test_yt(i))+1;
    Conf_Mat(test_yt(i),ptest(i))=Conf_Mat(test_yt(i),ptest(i))+1;
    if ptest(i)==test_yt(i)
        Correct_Predict(test_yt(i))= Correct_Predict(test_yt(i))+1;
    end
end

ACCs=Correct_Predict./Class_Num;
fprintf('Accuracy = %.2f%%\n',(1-er)*100);
fprintf('Accuracy_N = %.2f%%\n',ACCs(1)*100);
fprintf('Accuracy_V = %.2f%%\n',ACCs(2)*100);
fprintf('Accuracy_R = %.2f%%\n',ACCs(3)*100);
fprintf('Accuracy_L = %.2f%%\n',ACCs(4)*100);

%% =============ѹ���źŲ���==============
[er1,bad1,out1] = cnntest1d(cnn, newtest_x, newtest_y);%����1D CNN;

[~,ptest1]=max(out1,[],1);
[~,test_yt1]=max(newtest_y,[],1);

Correct_Predict1=zeros(1,4);                     %ͳ�Ƹ���׼ȷ�ʣ�
Class_Num1=zeros(1,4);                           %���õ���������
Conf_Mat1=zeros(4);
for i=1:20000
    Class_Num1(test_yt1(i))=Class_Num1(test_yt1(i))+1;
    Conf_Mat1(test_yt1(i),ptest1(i))=Conf_Mat1(test_yt1(i),ptest1(i))+1;
    if ptest1(i)==test_yt1(i)
        Correct_Predict1(test_yt1(i))= Correct_Predict1(test_yt1(i))+1;
    end
end

ACCs1=Correct_Predict1./Class_Num1;
fprintf('Accuracy1 = %.2f%%\n',(1-er1)*100);
fprintf('Accuracy_N1 = %.2f%%\n',ACCs1(1)*100);
fprintf('Accuracy_V1 = %.2f%%\n',ACCs1(2)*100);
fprintf('Accuracy_R1 = %.2f%%\n',ACCs1(3)*100);
fprintf('Accuracy_L1 = %.2f%%\n',ACCs1(4)*100);