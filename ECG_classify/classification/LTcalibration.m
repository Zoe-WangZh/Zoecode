function [dstTime,dstd,dst,ANNOTDD]=LTcalibration(ATRTIME,ANNOTD,TIME,QRS_ind,YM,YN)
%% 本函数把原信号的标记映射到重构信号中
% 返回值dstTime返回标记在重构信号上的时间位置
% 返回值dstd返回重构信号上的新标记（有做删改）
%% 寻找原标记点在采样点中的位置
    for i=1:length(ATRTIME)
        for j=1:length(TIME)
            if( ATRTIME(i)==TIME(j))
                ANNOTDD(i)=j;
            end
        end
    end
   %% 线性变换
% dst是ANNOTDD在QRS_ind之内的标记再经过线性映射
    m=0;
    n=0;
    k=1;
    for i=1:length(ANNOTDD)
        for j=1:length(QRS_ind)-1
            if ANNOTDD(i)>=QRS_ind(j)&&ANNOTDD(i)<QRS_ind(j+1)
    %              if ANNOTDD(i)~=QRS_ind(1)
    %                 text1(k)=ANNOTDD(i);
                    % 偏移量
                    m=fix(YN*(ANNOTDD(i)-QRS_ind(j)+1)/(QRS_ind(j+1)-QRS_ind(j)));
                    % 段偏移
                    n=YN*(j-1);
                    dst(k)=n+m;
                    k=k+1;
    %              end
            end
        end
    end
    j=1;
    for i=1:length(ANNOTDD)
        if ANNOTDD(i)>=QRS_ind(1)&&ANNOTDD(i)<QRS_ind(length(QRS_ind))
            text2(j)=ANNOTDD(i);
            dstd(j)=ANNOTD(i);
            j=j+1;
        end
    end
   
    %% 做修改，把首端以及尾端可能存在类标去除
    if dst(1)<=YN/10
        dst=dst(2:length(dst));
        dstd=dstd(2:length(dstd));
    end
    if (YM*YN-dst(length(dst)))<YN/10
        dst=dst(1:length(dst)-1);
        dstd=dstd(1:length(dstd)-1);
    end
    
    % 如果不一致，报错
    if length(dst)~=length(dstd)
        fprintf(1,'the length of dst and dstd must be same!');
    end
    
    % 目标标定的时间点
    for k=1:length(dst)
        dstTime(k)=dst(k)/360;
    end
end




