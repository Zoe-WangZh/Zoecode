clear;
clc;
M=5;
%% 1
accuracy=[0.8609,0.8980,0.9266,0.9397,0.9419];
C=[239.19,134.61,89.70,67.25,53.77];

DeltaC=90;          
DeltaA=0.05;
eta1=1;
eta2=1;
DeltaC=(DeltaC-C(5))/(C(1)- C(5));   
[C,~]=mapminmax(C,0,1);

error=1-accuracy;
%DeltaA=(DeltaA-error(4))/(error(1)-error(4));
error1=[error,DeltaA];
[error1,~]=mapminmax(error1,0,1);
error=error1(1:5);DeltaA=error1(6);

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
  L_1(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(3,3,1);plot(L_1);xlabel('(a)');ylabel('O');           
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2; 
clear C; clear error; clear accuracy;clear error1;

%% 2
accuracy=[0.8609,0.8980,0.9266,0.9397,0.9419];
C=[239.19,134.61,89.70,67.25,53.77];

DeltaC=90;
DeltaA=0.07;
eta1=1;
eta2=1;
DeltaC=(DeltaC-C(5))/(C(1)- C(5));   
[C,~]=mapminmax(C,0,1);

error=1-accuracy;
%DeltaA=(DeltaA-error(4))/(error(1)-error(4));
error1=[error,DeltaA];
[error1,~]=mapminmax(error1,0,1);
error=error1(1:5);DeltaA=error1(6);

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
  L_2(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(3,3,2);plot(L_2);      xlabel('(b)');ylabel('O');
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2;
clear C; clear error; clear accuracy;clear error1;

%% 3
accuracy=[0.8609,0.8980,0.9266,0.9397,0.9419];
C=[239.19,134.61,89.70,67.25,53.77];

DeltaC=130;
DeltaA=0.05;
eta1=1;
eta2=1;
DeltaC=(DeltaC-C(5))/(C(1)- C(5));   
[C,~]=mapminmax(C,0,1);

error=1-accuracy;
%DeltaA=(DeltaA-error(4))/(error(1)-error(4));
error1=[error,DeltaA];
[error1,~]=mapminmax(error1,0,1);
error=error1(1:5);DeltaA=error1(6);

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
  L_3(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(3,3,4);plot(L_3);      xlabel('(c)');ylabel('O');
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2;
clear C; clear error; clear accuracy;clear error1;

%% 4
accuracy=[0.8609,0.8980,0.9266,0.9397,0.9419];
C=[239.19,134.61,89.70,67.25,53.77];

DeltaC=130;
DeltaA=0.07;
eta1=1;
eta2=1;
DeltaC=(DeltaC-C(5))/(C(1)- C(5));   
[C,~]=mapminmax(C,0,1);

error=1-accuracy;
%DeltaA=(DeltaA-error(4))/(error(1)-error(4));
error1=[error,DeltaA];
[error1,~]=mapminmax(error1,0,1);
error=error1(1:5);DeltaA=error1(6);

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
  L_4(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(3,3,5);plot(L_4);      xlabel('(d)');ylabel('O');
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2;
clear C; clear error; clear accuracy;clear error1;

%% 5
accuracy=[0.8609,0.8980,0.9266,0.9397,0.9419];
C=[239.19,134.61,89.70,67.25,53.77];

DeltaC=90;
DeltaA=0.05;
eta1=0.5;
eta2=1;
DeltaC=(DeltaC-C(5))/(C(1)- C(5));   
[C,~]=mapminmax(C,0,1);

error=1-accuracy;
%DeltaA=(DeltaA-error(4))/(error(1)-error(4));
error1=[error,DeltaA];
[error1,~]=mapminmax(error1,0,1);
error=error1(1:5);DeltaA=error1(6);

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
  L_5(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(3,3,7);plot(L_5);      xlabel('(e)');ylabel('O');
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2; 
clear C; clear error; clear accuracy;clear error1;

%% 6
accuracy=[0.8609,0.8980,0.9266,0.9397,0.9419];
C=[239.19,134.61,89.70,67.25,53.77];

DeltaC=90;
DeltaA=0.05;
eta1=1;
eta2=0.5;
DeltaC=(DeltaC-C(5))/(C(1)- C(5));   
[C,~]=mapminmax(C,0,1);

error=1-accuracy;
%DeltaA=(DeltaA-error(4))/(error(1)-error(4));
error1=[error,DeltaA];
[error1,~]=mapminmax(error1,0,1);
error=error1(1:5);DeltaA=error1(6);

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
  L_6(i)=eta1*L1(i)+eta2*L2(i);
end
subplot(3,3,8);plot(L_6);      xlabel('(f)');ylabel('O');
clear eta1;clear eta2;
clear DeltaC;clear DeltaA;
clear L1;clear L2; clear error1;