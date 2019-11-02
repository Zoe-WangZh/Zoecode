%file:Add.m
%a,b ��Բ����  p ���� x1,y1 ��һ��������� x2,y2 �ڶ����������
function [P_mx,P_my ] = Add( a,b,p,x1,y1,x2,y2 )

if x1==x2 && y1==y2
    k=modfrac(3*x1^2+a,2*y1,p);
    P_mx = mod(k^2-x1-x2,p);
    P_my = mod(k*(x1-P_mx)-y1,p);
end

if x1==x2 && y1~=y2
    P_mx = inf;
    P_my = inf;
end

if x1 ~= x2
    k=modfrac(y2-y1,x2-x1,p);
    P_mx = mod(k^2-x1-x2,p);
    P_my = mod(k*(x1-P_mx)-y1,p);    
end
end
