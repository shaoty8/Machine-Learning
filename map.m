function[U,V,error] = map(M,omega,ratings_test)
lambda = 10;
variance = 0.25;
d = 10;
U = zeros(943,d);
V = zeros(d,1682);

%initialize V
mu = zeros(10,1);
sigma = eye(10)/lambda;
for i = 1:1682
    V(:,i) = mvnrnd(mu,sigma);
end

A = eye(10) * lambda * variance;
D = lambda * variance;
error = zeros(1,100);
for itr = 1:100
    for i = 1:943
        B = zeros(10,10);
        rate = zeros(10,1);
        for j = 1:95000
            if omega(j,1) == i
                B = B + V(:,omega(j,2)) * V(:,omega(j,2))';
                rate = rate + M(i,omega(j,2)) * V(:,omega(j,2));
            end
        end
        U(i,:) = inv(A + B) * rate;
    end
    
    for k = 1:1682
        C = 0;
        rating = zeros(1,10);
        for m = 1:95000
            if omega(m,2) == k
                C = C + U(omega(m,1),:) * U(omega(m,1),:)';
                rating = rating + M(omega(m,1),k) * U(omega(m,1),:);
            end
        end
        V(:,k) = inv(D + C) * rating;
    end
    
    
    for i = 1:5000
        a = ratings_test(i,1);
        b = ratings_test(i,2);
        e = round(U(a,:) * V(:,b)) - ratings_test(i,3);
        error(itr) = error(itr) + e^2;
    end
end
error = sqrt(error/5000);