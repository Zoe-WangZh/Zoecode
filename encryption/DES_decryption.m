profile on;
D = char(size(C,1),16);
for i = 1:size(C,1)
    d = DES(C(i,:),'1999012513578642',2);
    D(i,1:16) = d;
end
count=0;
for i = 1:size(D,1)
    if ~strcmp(D(i,:),B(16*i-15:i*16)')
        count=count + 1;
    end
end
profile viewer