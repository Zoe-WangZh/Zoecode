clear all;
clc;

f1=imread('1.PNG');
f=rgb2gray(f1);
[m,n] = size(f);

k = 100;            %�����������ֵ����
f = double(f);      %uint8����
[u,s,v] = svds (f,5);  %��������ֵ�ֽ�,����sΪ�ԽǾ��� 
% s = diag(s);        %����ԽǾ���ĶԽ���Ԫ�أ��õ�һ������
% smax = max(s);smin = min(s);    %���������ֵ����С����ֵ
% s1=s;s1(k:end) = 0;            %ֻ����ǰ20���������ֵ����������ֵ����
% s1 = diag(s1);      %��������ɶԽǾ���
g = u*s*v';        %����ѹ���Ժ��ͼ�����
g = uint8(g);
compressratio = n^2/(k*(2*n+1));

subplot(1,2,1),imshow(mat2gray(f));title('source') %ԭͼ
subplot(1,2,2),imshow(g); title(['compress ratio',num2str(compressratio)]) %ѹ�����ͼ��
figure,plot(s,'.','Color','k')  %��������ֵ��Ӧ�ĵ�
