clear;
clc;
fs = 300;

for i = 1:90

    filename = strcat(num2str(i),'.txt');
    signal = load(filename);
%     L1 = medfilt1(signal,50);
%     signal = signal - L1;
    % signal = signal(22000:end);
    [b,a]=butter(4,[0.5/(fs/2) 40/(fs/2)]);
    signal = filter(b,a,signal);
    [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(signal,fs,0); 
    RR_interval = qrs_i_raw(2:end)-qrs_i_raw(1:end-1);
    m = size(RR_interval, 2);
    juzhen = [];
    for s = 1:m
        if s>=5
            RR_interval(s) = (RR_interval(s) + RR_interval(s-1) + RR_interval(s-2) + RR_interval(s-3) + RR_interval(s-4))/5;
        end;
    end;
    HR = RR_interval/300*60;
    for k = 2:m-1
        if(HR(k)<65)
            dt = -10;
        elseif (HR(k)>=65 && HR(k)<80)
            dt = 0;
        elseif (HR(k)>=80 && HR(k)<95)
            dt = 10;
        elseif (HR(k)>=95 && HR(k)<110)
            dt = 20;
        elseif (HR(k)>=110 && HR(k)<125)
            dt = 30;
        elseif (HR(k)>=125 && HR(k)<140)
            dt = 40;
        elseif (HR(k)>=140 && HR(k)<155)
            dt = 50;
        else
            dt = 60;
        end;
        
        PQ = signal(qrs_i_raw(k)-230*0.3+dt*0.3 : qrs_i_raw(k)-90*0.3);
        %PQ矫正
        y = PQ;
        length = 450*0.3;
        m = size(y, 2);
        x = zeros(1,length);
        for j = 1:length
            rj = (j-1)*(m-1)/(length-1)+1;
            j_star = floor(rj);
            if j<length
                x(j) = y(j_star) + (y(j_star+1)-y(j_star))*(rj-j_star);
            else
                x(j) = y(m);
            end;
        end;
        PQ = x;
        
        
        %QRS不用矫正
        QRS = signal(qrs_i_raw(k)-90*0.3 : qrs_i_raw(k)+100*0.3);
        
        %ST段矫正
        ST = signal(qrs_i_raw(k)+100*0.3 : qrs_i_raw(k)+100*0.3+round(0.08*RR_interval(k)));
        y = ST;
        length = 110*0.3;
        m = size(y, 2);
        x = zeros(1,length);
        for j = 1:length
            rj = (j-1)*(m-1)/(length-1)+1;
            j_star = floor(rj);
            if j<length
                x(j) = y(j_star) + (y(j_star+1)-y(j_star))*(rj-j_star);
            else
                x(j) = y(m);
            end;
        end;
        ST = x;
        
        
        %T段矫正
        T = signal(qrs_i_raw(k)+100*0.3+round(0.08*RR_interval(k)) : qrs_i_raw(k)+round(0.42*RR_interval(k)));
        y = T;
        length = 50*0.3;
        m = size(y, 2);
        x = zeros(1,length);
        for j = 1:length
            rj = (j-1)*(m-1)/(length-1)+1;
            j_star = floor(rj);
            if j<length
                x(j) = y(j_star) + (y(j_star+1)-y(j_star))*(rj-j_star);
            else
                x(j) = y(m);
            end;
        end;
        T = x;
%     zhongjiandian = floor((qrs_i_raw(2:end)+qrs_i_raw(1:end-1))/2);
%     zhongjiandiangeshu = size(zhongjiandian, 2);
        xinhao = [PQ, QRS, ST, T];
        juzhen = [juzhen; xinhao];
    end;
    
    
    
%     for k = 1:zhongjiandiangeshu-1
%         y = signal(zhongjiandian(k):zhongjiandian(k+1));
%         m = size(y, 2);
%         x = zeros(1,300);
%         for j = 1:300
%             rj = (j-1)*(m-1)/299+1;
%             j_star = floor(rj);
%             if j<300
%                 x(j) = y(j_star) + (y(j_star+1)-y(j_star))*(rj-j_star);
%             else
%                 x(j) = y(m);
%             end;
%         end;
%         juzhen = [juzhen;x];
%     end;
%     plot(signal(1:600));
%     hold on;
%     plot(qrs_i_raw(1:3),signal(qrs_i_raw(1:3)),'o','MarkerSize',6,'MarkerEdgeColor','r');
%     while qrs_i_raw(1)-149<1
%         qrs_i_raw = qrs_i_raw(2:end);
%     end
%     while qrs_i_raw(end)+150 > length(signal)
%         qrs_i_raw = qrs_i_raw(1:end-1);
%     end
%     len = length(qrs_i_raw);
%     juzhen = zeros(len, 300);
%     for j = 1:len
%         a = qrs_i_raw(j)-149;
%         b = qrs_i_raw(j)+150;
%         juzhen(j,:) = signal(a:b);
%     end
    filename = strcat(num2str(i),'.mat');
    save(filename, 'juzhen');
end;
       