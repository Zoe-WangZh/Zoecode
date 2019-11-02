function[Y,MVP] = NormalizedSignal(QRS_ind,M)
%% ��ȡ��������Ƭ�δ洢��segment_martix�С�
%% ����position�Ǵ洢ÿ��R����ֵ�Ĳ����������

l=length(QRS_ind);
segment_martix=cell(l-1,1);                     %%L-1��

for k=1:l-1                                     %%1��L-1
    n=1;
    for i=QRS_ind(k):QRS_ind(k+1)               %%�ӵ�һ�����嵽���һ������
        segment_martix{k,n}=M(i);
        n=n+1;
    end
end
%% 
[m,n]=size(segment_martix);
    for i=1:m
        b=['segment',num2str(i)];
        eval([b,'=cell2mat(segment_martix(i,:))'])
        lengths(i)=length(cell2mat(segment_martix(i,:)));               %ȡÿһ�еĳ���
    end
%% �淶��
    MVP=round(mean(lengths));
    y=cell(m,MVP-1);                  %����һ��m��n-1�е�cell
    for i=1:m
        for j=1:MVP-1
            r(j)=(j-1)*(lengths(i)-1)/(MVP-1)+1;
            j1=fix(r(j));
            y{i,j}=segment_martix{i,j1}+(segment_martix{i,j1+1}-segment_martix{i,j1})*(r(j)-j1);
        end
    end
%% ���չ淶��֮����ź�   
    Y=cell2mat(y);                 
end


