% k=15;                            %����������kά��������
% %% ʹ��Matlab������princomp����ʵ��PCA
% [COEFF SCORE latent]=princomp(downsampledsig_2dimension)
% pcaData1=SCORE(:,1:k)            %ȡǰk������
k=15;
[r, e]=size(downsampledsig_2dimension);
meanVec=mean(downsampledsig_2dimension);%�������ľ�ֵ
Z=(downsampledsig_2dimension-repmat(meanVec,r,1));
covMatT=Z*Z'; %����Э������󣬴˴���С��������
[V, D]=eigs(covMatT,k);%����ǰk������ֵ����������
V=Z'*V;%�õ�Э�������covMatT'����������
%����������һ����λ��������
for i=1:k
    V(:,i)=V(:,i)/norm(V(:,i));
end
pcaA=Z*V;%���Ա仯��ά��kά