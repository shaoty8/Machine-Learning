function [golden,apollo,forrest] = nearest(V)
error1 = zeros(1682);
error2 = zeros(1682);
error3 = zeros(1682);
for i = 1:1682
    error1(i) = norm(V(:,i) - V(:,2));
    error2(i) = norm(V(:,i) - V(:,28));
    error3(i) = norm(V(:,i) - V(:,69));
end
[~,AIdx] = sort(error1);
golden = AIdx(1:6);
[~,BIdx] = sort(error2);
apollo = BIdx(1:6);
[~,CIdx] = sort(error3);
forrest = CIdx(1:6);