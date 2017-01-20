function [w_infinity,a] = computew(v,w)
w_infinity = v'/sum(v);

nrm = zeros(1,2501);
for i = 1:2501
    nrm(i) = norm(w(i,:) - w_infinity,1);
end
plot(nrm,'r')

a = norm(w(2501,:) - w_infinity,1);