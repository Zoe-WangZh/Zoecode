clear;
clc;
fs = 300;

for i = 1:90
    filename = strcat(num2str(i),'.mat');
    load(filename);
%     juzhen = juzhen(:,1:end-1);
    m = size(juzhen,1);
    coeff = zeros(m,7712);
    for j=1:m
        swd=cwt(juzhen(j,:), 1:32, 'db5');
        coeff(j,:) = reshape(swd',1,size(swd,1)*size(swd,2));
    end
    filename = strcat(num2str(i),'wavelet');
    filename = strcat(filename,'.mat');
    save(filename, 'coeff');
end;
        