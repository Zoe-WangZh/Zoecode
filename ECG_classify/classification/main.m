function[reY1D,YM,YN,dstTime,dstd] = main(number)
%% 主函数
%% 参数number代表记录信号；SAMPLES2READ代表采样点个数

%--------------------------------------------------------------------------
% 过程：先读取MIT-BIH的心电数据，对其做一个峰-峰检测，后进行波形截取，做规范化
%       操作，之后对信号进行加密。通过SVD分解在云端重新构造加密信号，之后进行解
%       密，之后进行分类。
%--------------------------------------------------------------------------

    % 读取心电信号数据
    [ANNOTD,ATRTIME,TIME,M]= rddata(number);
    
    % 做波形的检测，得到检测出来的峰值的大小QRS_ind
    [~,QRS_ind]= DS_detect(M);
    
    % 得到规范化的信号
    [Y,~]= NormalizedSignal(QRS_ind,M);
    
    % 把Y转成一维数组
    Y1D=[];
    [YM,YN]=size(Y);
    k=1;
    for i=1:YM
        for j=1:YN
            Y1D(k)=Y(i,j);
            k=k+1;
        end
    end
%%    
    % 做十进制转十六进制转换
    %HexM = DEC2HEX(Y1D);
    
    % encryption
    % process......
    % end
    
    % SVD
    % process......
    % end
    
    % reSVD...decryption
    % process.....
    % end
    
%%  SVD分解
    [u,s,v]=svds(Y,1);
%%  SVD重构 
    reY=u*s*v';
    
    % 把重构信号展开成一维信号
    reY1D=[];
    [rm,rn]=size(reY);
    k=1;
    for i=1:rm
        for j=1:rn
            reY1D(k)=reY(i,j);
            k=k+1;
        end
    end
    
    %频率为360Hz时间数组：
    TIME1=[];
    for i=1:YM*YN
        if(i<=length(TIME))
            TIME1(i)=TIME(i); 
        else
            TIME1(i)=TIME1(i-1)+1/360;
        end
    end
    
    % 查看原始信号和重构信号的区别
%     DrawECG(TIME,M,Y,reY1D);
    
    % 重新标定
     [dstTime,dstd,dst,ANNOTDD]=LTcalibration(ATRTIME,ANNOTD,TIME,QRS_ind,YM,YN);
    
    % 画图对比
    subplot(2,1,1);plot(TIME,M);
    for k=1:length(ATRTIME)
        text(ATRTIME(k),0,num2str(ANNOTD(k)));
    end
    
    subplot(2,1,2);plot(TIME1,reY1D);
    for k=1:length(dst)
        text(dstTime(k),0,num2str(dstd(k)));
    end
    
end





