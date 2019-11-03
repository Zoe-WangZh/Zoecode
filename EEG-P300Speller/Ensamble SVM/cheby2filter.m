function [y0]=cheby2filter(N,Rs,fp,signal)
Wn=2*fp/240;
[C2b,C2a]=cheby2(N,Rs,Wn,'low'); % ����MATLAB cheby2����������Ƶ�ͨ�˲���  
[C2H,C2W]=freqz(C2b,C2a); % ����Ƶ����Ӧ����  
y0=filter(C2b,C2a,signal); % ���е�ͨ�˲�  
%C2y=fft(C2f,len);  % ���˲����ź���len��FFT�任  
end
% N=0; % ����  
% Fp=50; % ͨ����ֹƵ��50Hz  
% Fc=100; % �����ֹƵ��100Hz  
% Rp=1; % ͨ���������˥��Ϊ1dB  
% Rs=60; % ���˥��Ϊ60dB