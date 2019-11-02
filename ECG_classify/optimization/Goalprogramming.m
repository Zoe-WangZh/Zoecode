M=2000;
N=25;
eta1=0.5;
eta2=1;
DeltaC=0.1;
DeltaA=0.1;

for i=1:M
    for j=1:N
  error(i,j)=1-accuracy(i,j);
       if real(C(i,j))>DeltaC
           L1(i,j)=real(C(i,j))-DeltaC;
       else
           L1(i,j)=0;
       end
       if error(i,j)>DeltaA
           L2(i,j)=error(i,j)-DeltaA;
       else
           L2(i,j)=0;
       end
  L(i,j)=eta1*L1(i,j)+eta2*L2(i,j);
    end
end
contourf(0:0.004:0.096,0:0.0035:6.9965,L);
% xlabel('k\Omega');
% ylabel('t\Omega/2\pi');
colorbar
           
           