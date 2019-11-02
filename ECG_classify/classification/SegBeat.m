function []=SegBeat()
    %% 全体数据集
    clear;
    clc;
    Name_whole=[100,101,103,104,105,106,107,108,109,111,112,113,114,115,...
               116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,...
               210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234];
    Pl=100;Pr=150;
    Nb1=[];Lb1=[];Rb1=[];Vb1=[];Ab1=[];Sb1=[];
    for x=1:47
        %% 得到返回值
        [reY1D,YM,YN,dstTime,dstd] = main(Name_whole(x));       
        %% 
        [~,QRS_ind] = DS_detect(reY1D',0);% 调用QRS检测算法；
        %% 对QRS_ind做修剪
        if QRS_ind(1)<YN/10
            QRS_ind=QRS_ind(2:length(QRS_ind));
        end
        if (YM*YN-QRS_ind(length(QRS_ind)))<YN/10
            QRS_ind=QRS_ind(1:length(QRS_ind)-1);
        end
        %%
    %     dstd=dstd';
    %     dstTime=dstTime';
        %%
        Nt=size(QRS_ind,2);                % 寻找QRS_ind第二维度的长度
        R_TIME=dstTime(dstd==1 | dstd ==2 | dstd==3 | dstd==5 | dstd==8 |dstd==9 );    %峰值对应的时间点

        REF_ind=round(R_TIME'.*360);
        if size(REF_ind,2)==1
            REF_ind=REF_ind';
        end
        Nr=size(REF_ind,2);
        ann=dstd(dstd==1 | dstd ==2 | dstd==3 | dstd==5 | dstd==8 |dstd==9);

        if Nt>Nr
            typ=0;
        else
            typ=1;
        end

        if typ==0
            for n=1:Nr
                ref=REF_ind(n);
                for m=1:Nt
                    act_ind=QRS_ind(m);
                    if abs(ref-act_ind)<=54
                       if act_ind<Pl || (650000-act_ind)<Pr
                            break;
                       else
                            SEG=reY1D((act_ind-Pl+1):(act_ind+Pr));
                       end
                       switch ann(n)
                            case 1
                                Nb1=[Nb1;SEG];
                            case 2
                                Lb1=[Lb1;SEG];
                            case 3
                                Rb1=[Rb1;SEG];
                            case 5
                                Vb1=[Vb1;SEG];
                            case 8
                                Ab1=[Ab1;SEG];
                            case 9
                                Sb1=[Sb1;SEG];
                       end

                       break; 
                    end 
                end

            end
        else
            for n=1:Nt
                act_ind=QRS_ind(n);
                for m=1:Nr
                    if abs(act_ind-REF_ind(m))<=54
                        if act_ind<Pl || (650000-act_ind)<Pr
                            break;
                       else
                            SEG=reY1D((act_ind-Pl+1):(act_ind+Pr));      %
                       end
                       switch ann(m)                           
                            case 1          
                                Nb1=[Nb1;SEG];
                            case 2
                                Lb1=[Lb1;SEG];
                            case 3
                                Rb1=[Rb1;SEG];
                            case 5
                                Vb1=[Vb1;SEG];
                            case 8
                                Ab1=[Ab1;SEG];
                            case 9
                                Sb1=[Sb1;SEG];
                       end

                        break;
                    end
                end

            end
        end
    end
   SaveData(Nb1,Lb1,Rb1,Vb1,Ab1,Sb1);
end