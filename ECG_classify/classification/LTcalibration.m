function [dstTime,dstd,dst,ANNOTDD]=LTcalibration(ATRTIME,ANNOTD,TIME,QRS_ind,YM,YN)
%% ��������ԭ�źŵı��ӳ�䵽�ع��ź���
% ����ֵdstTime���ر�����ع��ź��ϵ�ʱ��λ��
% ����ֵdstd�����ع��ź��ϵ��±�ǣ�����ɾ�ģ�
%% Ѱ��ԭ��ǵ��ڲ������е�λ��
    for i=1:length(ATRTIME)
        for j=1:length(TIME)
            if( ATRTIME(i)==TIME(j))
                ANNOTDD(i)=j;
            end
        end
    end
   %% ���Ա任
% dst��ANNOTDD��QRS_ind֮�ڵı���پ�������ӳ��
    m=0;
    n=0;
    k=1;
    for i=1:length(ANNOTDD)
        for j=1:length(QRS_ind)-1
            if ANNOTDD(i)>=QRS_ind(j)&&ANNOTDD(i)<QRS_ind(j+1)
    %              if ANNOTDD(i)~=QRS_ind(1)
    %                 text1(k)=ANNOTDD(i);
                    % ƫ����
                    m=fix(YN*(ANNOTDD(i)-QRS_ind(j)+1)/(QRS_ind(j+1)-QRS_ind(j)));
                    % ��ƫ��
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
   
    %% ���޸ģ����׶��Լ�β�˿��ܴ������ȥ��
    if dst(1)<=YN/10
        dst=dst(2:length(dst));
        dstd=dstd(2:length(dstd));
    end
    if (YM*YN-dst(length(dst)))<YN/10
        dst=dst(1:length(dst)-1);
        dstd=dstd(1:length(dstd)-1);
    end
    
    % �����һ�£�����
    if length(dst)~=length(dstd)
        fprintf(1,'the length of dst and dstd must be same!');
    end
    
    % Ŀ��궨��ʱ���
    for k=1:length(dst)
        dstTime(k)=dst(k)/360;
    end
end




