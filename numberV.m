function [c,cnum,id] = numberV(V)
[idx,C] = kmeans(V',20);
num = zeros(20,2);
for i = 1:20
    num(i,1) = i;
end
for i = 1:1682
    for j = 1:20
        if idx(i) == j
            num(j,2) = num(j,2) + 1;
        end
    end
end
[ASorted,AIdx] = sort(num(:,2));
cnum = ASorted(16:20);
c = zeros(10,5);
for i = 1:5
    c(:,i) = C(AIdx(15+i));
end

nrm = zeros(5,1682);
for i = 1:5
    for j = 1:1682
        nrm(i,j) = norm(c(:,i) - V(:,j));
    end
end
id = zeros(10,5);
for i = 1:5
    [~,Idx] = sort(nrm(i,:));
    id(:,i) = Idx(end-9:end);
end
        