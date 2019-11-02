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
    zhongjiandian = floor((qrs_i_raw(2:end)+qrs_i_raw(1:end-1))/2);
    zhongjiandiangeshu = size(zhongjiandian, 2);
    juzhen = [];
    for k = 1:zhongjiandiangeshu-1
        y = signal(zhongjiandian(k):zhongjiandian(k+1));
        m = size(y, 2);
        x = zeros(1,300);
        for j = 1:300
            rj = (j-1)*(m-1)/299+1;
            j_star = floor(rj);
            if j<300
                x(j) = y(j_star) + (y(j_star+1)-y(j_star))*(rj-j_star);
            else
                x(j) = y(m);
            end;
        end;
        juzhen = [juzhen;x];
    end;
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
       