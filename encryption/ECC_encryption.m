%file:ECC.m
%��ʾ���߼����㷨��/���ܹ���
profile on;
addpath('D:\�ĵ��ź���Ŀ\MIT���ݿ�\MIT-BIT����ʧ�����ݿ�');
signal = fullfile('D:\�ĵ��ź���Ŀ\MIT���ݿ�\MIT-BIT����ʧ�����ݿ�','105.dat');
fid = fopen(signal,'r');
A = fread(fid,'uint4');
fclose(fid);

C1=zeros(1600,2);
C2=zeros(1600,2);
B=zeros(1600,2);
R=zeros(1600,1);
E=zeros(1600,1);
m=10
for i=1:1600*m
MX = A(i,1);
 if isempty(ECCCal(4,20,29,MX))
     B(i,:)=ECCCal(1,1,23,MX);
     E(i,1)=0;
     MY=B(i,1);
     a=1;
     b=1;
     p=23;
     GX=13;
     GY=16;
     k=3;%˽Կ3 ����6
[KX,KY]=NP(a,b,p,k,GX,GY);
    r=unidrnd(6);
    R(i,1)=r;
    [rKX,rKY] = NP(a,b,p,r,KX,KY);
    [C1(i,1),C1(i,2)]=Add(a,b,p,MX,MY,rKX,rKY);
    [C2(i,1),C2(i,2)]=NP(a,b,p,r,GX,GY);
    if C1(i,1)==Inf
         r=r-1;
         R(i,1)=r;
    [rKX,rKY] = NP(a,b,p,r,KX,KY);
    [C1(i,1),C1(i,2)]=Add(a,b,p,MX,MY,rKX,rKY);
    [C2(i,1),C2(i,2)]=NP(a,b,p,r,GX,GY);
    end
 else
    E(i,1)=1;
     B(i,:)=ECCCal(4,20,29,MX);
    MY=B(i,1); 
    a=4;
    b=20;
    p=29;
    GX=16;
    GY=27;
   k=8;  %˽Կ8 ����36 ��Կ24��7
  [KX,KY]=NP(a,b,p,k,GX,GY);
    r=unidrnd(36);
     R(i,1)=r;
     [rKX,rKY] = NP(a,b,p,r,KX,KY);
  [C1(i,1),C1(i,2)]=Add(a,b,p,MX,MY,rKX,rKY);
    [C2(i,1),C2(i,2)]=NP(a,b,p,r,GX,GY);
    if C1(i,1)==Inf
         r=r-1;
         R(i,1)=r;
    [rKX,rKY] = NP(a,b,p,r,KX,KY);
    [C1(i,1),C1(i,2)]=Add(a,b,p,MX,MY,rKX,rKY);
    [C2(i,1),C2(i,2)]=NP(a,b,p,r,GX,GY);
    end
 end
end
profile viewer
%C1 C2�Ǽ��ܺ������