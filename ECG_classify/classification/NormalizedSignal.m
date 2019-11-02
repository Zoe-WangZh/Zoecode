function[Y,MVP] = NormalizedSignal(QRS_ind,M)
%% 截取出的周期片段存储在segment_martix中。
%% 数组position是存储每个R波峰值的采样点的数组

l=length(QRS_ind);
segment_martix=cell(l-1,1);                     %%L-1列

for k=1:l-1                                     %%1到L-1
    n=1;
    for i=QRS_ind(k):QRS_ind(k+1)               %%从第一个主峰到最后一个主峰
        segment_martix{k,n}=M(i);
        n=n+1;
    end
end
%% 
[m,n]=size(segment_martix);
    for i=1:m
        b=['segment',num2str(i)];
        eval([b,'=cell2mat(segment_martix(i,:))'])
        lengths(i)=length(cell2mat(segment_martix(i,:)));               %取每一行的长度
    end
%% 规范化
    MVP=round(mean(lengths));
    y=cell(m,MVP-1);                  %定义一个m行n-1列的cell
    for i=1:m
        for j=1:MVP-1
            r(j)=(j-1)*(lengths(i)-1)/(MVP-1)+1;
            j1=fix(r(j));
            y{i,j}=segment_martix{i,j1}+(segment_martix{i,j1+1}-segment_martix{i,j1})*(r(j)-j1);
        end
    end
%% 最终规范化之后的信号   
    Y=cell2mat(y);                 
end


