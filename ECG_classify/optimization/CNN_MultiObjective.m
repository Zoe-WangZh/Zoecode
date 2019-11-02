clear;
clc;
M=5;
accuracy=[91.52,95.69,96.63,97.08,96.95];
C=[239.19,134.61,89.70,67.25,53.77];

DeltaC=90;          
DeltaA=3;
eta1=DeltaA/DeltaC;
eta2=1/eta1;

for i=1:M
  error(i)=100-accuracy(i);
       if C(i)>DeltaC
           L1(i)=0;
       else
           L1(i)=DeltaC-C(i);
       end
       if error(i)>DeltaA
           L2(i)=error(i)-DeltaA;
       else
           L2(i)=0;
       end
  L(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(2,2,1);plot(L);           
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2; clear L;

DeltaC=150;
DeltaA=3;
eta1=DeltaA/DeltaC;
eta2=1/eta1;
for i=1:M
       if C(i)>DeltaC
           L1(i)=0;
       else
           L1(i)=DeltaC-C(i);
       end
       if error(i)>DeltaA
           L2(i)=error(i)-DeltaA;
       else
           L2(i)=0;
       end
  L(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(2,2,2);plot(L);      
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2; clear L;

DeltaC=90;
DeltaA=5;
eta1=DeltaA/DeltaC;
eta2=1/eta1;
for i=1:M
       if C(i)>DeltaC
           L1(i)=0;
       else
           L1(i)=DeltaC-C(i);
       end
       if error(i)>DeltaA
           L2(i)=error(i)-DeltaA;
       else
           L2(i)=0;
       end
  L(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(2,2,3);plot(L);      
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2; clear L;

DeltaC=150;
DeltaA=5;
eta1=DeltaA/DeltaC;
eta2=1/eta1;
for i=1:M
       if C(i)>DeltaC
           L1(i)=0;
       else
           L1(i)=DeltaC-C(i);
       end
       if error(i)>DeltaA
           L2(i)=error(i)-DeltaA;
       else
           L2(i)=0;
       end
  L(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(2,2,4);plot(L);      
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
% clear L1;clear L2; clear L;