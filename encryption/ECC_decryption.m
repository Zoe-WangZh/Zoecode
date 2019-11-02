profile on;
D=zeros(1600,2);
F=zeros(1600,1);
m=10
for i=1:1600*m
    if E(i,1)==1
     a=4;
    b=20;
    p=29;
    k=8;
    else 
     a=1;
     b=1;
     p=23;
     k=3;
    end
C1X=C1(i,1);
C1Y=C1(i,2);
C2X=C2(i,1);
C2Y=C2(i,2);

[kC2X,kC2Y]=NP(a,b,p,k,C2X,C2Y);

kC2Y=mod(-1*kC2Y,p);

[D(i,1),D(i,2)]=Add(a,b,p,C1X,C1Y,kC2X,kC2Y);
end
count=0;
for i=1:1600
 if A(i,1)~=D(i,1)
    count=count+1;
 else
     F(i,1)=1;
 end
end
profile viewer