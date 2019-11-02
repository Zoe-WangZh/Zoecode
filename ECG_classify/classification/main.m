function[reY1D,YM,YN,dstTime,dstd] = main(number)
%% ������
%% ����number�����¼�źţ�SAMPLES2READ������������

%--------------------------------------------------------------------------
% ���̣��ȶ�ȡMIT-BIH���ĵ����ݣ�������һ����-���⣬����в��ν�ȡ�����淶��
%       ������֮����źŽ��м��ܡ�ͨ��SVD�ֽ����ƶ����¹�������źţ�֮����н�
%       �ܣ�֮����з��ࡣ
%--------------------------------------------------------------------------

    % ��ȡ�ĵ��ź�����
    [ANNOTD,ATRTIME,TIME,M]= rddata(number);
    
    % �����εļ�⣬�õ��������ķ�ֵ�Ĵ�СQRS_ind
    [~,QRS_ind]= DS_detect(M);
    
    % �õ��淶�����ź�
    [Y,~]= NormalizedSignal(QRS_ind,M);
    
    % ��Yת��һά����
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
    % ��ʮ����תʮ������ת��
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
    
%%  SVD�ֽ�
    [u,s,v]=svds(Y,1);
%%  SVD�ع� 
    reY=u*s*v';
    
    % ���ع��ź�չ����һά�ź�
    reY1D=[];
    [rm,rn]=size(reY);
    k=1;
    for i=1:rm
        for j=1:rn
            reY1D(k)=reY(i,j);
            k=k+1;
        end
    end
    
    %Ƶ��Ϊ360Hzʱ�����飺
    TIME1=[];
    for i=1:YM*YN
        if(i<=length(TIME))
            TIME1(i)=TIME(i); 
        else
            TIME1(i)=TIME1(i-1)+1/360;
        end
    end
    
    % �鿴ԭʼ�źź��ع��źŵ�����
%     DrawECG(TIME,M,Y,reY1D);
    
    % ���±궨
     [dstTime,dstd,dst,ANNOTDD]=LTcalibration(ATRTIME,ANNOTD,TIME,QRS_ind,YM,YN);
    
    % ��ͼ�Ա�
    subplot(2,1,1);plot(TIME,M);
    for k=1:length(ATRTIME)
        text(ATRTIME(k),0,num2str(ANNOTD(k)));
    end
    
    subplot(2,1,2);plot(TIME1,reY1D);
    for k=1:length(dst)
        text(dstTime(k),0,num2str(dstd(k)));
    end
    
end





