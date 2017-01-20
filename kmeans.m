function [c,obj,Data] = kmeans(K)
Data = zeros(500,2);
mu1 = [0;0];
sigma1 = [1,0;0,1];
mu2 = [3;0];
sigma2 = [1,0;0,1];
mu3 = [0;3];
sigma3 = [1,0;0,1];

rng default  % For reproducibility
r1 = mvnrnd(mu1,sigma1,100);
Data(1:100,:) = r1;

r2 = mvnrnd(mu2,sigma2,250);
Data(101:350,:) = r2;

r1 = mvnrnd(mu3,sigma3,150);
Data(351:500,:) = r1;

mu = rand(K,2);
c = zeros(500,1);
obj = zeros(20,1);

for m = 1:20
    L = 0;
    for i = 1:500
        error = zeros(K,1);
        for j = 1:K
            error(j) = norm(Data(i,:) - mu(j,:));
        end
        [M,I] = min(error);
        c(i) = I;
        L = L + M;
    end
    obj(m) = L;
    
    N = zeros(K,1);
    X = zeros(K,2);
    for i = 1:K
        for j = 1:500
            if c(j) == i
                N(i) = N(i) + 1;
                X(i,:) = X(i,:) + Data(j,:);
            end
        end
    end
    
    for j = 1:K
        mu(j,:) = X(j,:)/N(j);
    end
end
plot(obj,'b-o')