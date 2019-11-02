clear all;
clc;

f1=imread('1.PNG');
f=rgb2gray(f1);
[m,n] = size(f);

k = 100;            %保留最大奇异值个数
f = double(f);      %uint8类型
[u,s,v] = svds (f,5);  %进行奇异值分解,这里s为对角矩阵 
% s = diag(s);        %提出对角矩阵的对角线元素，得到一个向量
% smax = max(s);smin = min(s);    %求最大奇异值和最小奇异值
% s1=s;s1(k:end) = 0;            %只保留前20个大的奇异值，其他奇异值置零
% s1 = diag(s1);      %把向量变成对角矩阵
g = u*s*v';        %计算压缩以后的图像矩阵
g = uint8(g);
compressratio = n^2/(k*(2*n+1));

subplot(1,2,1),imshow(mat2gray(f));title('source') %原图
subplot(1,2,2),imshow(g); title(['compress ratio',num2str(compressratio)]) %压缩后的图像
figure,plot(s,'.','Color','k')  %画出奇异值对应的点
