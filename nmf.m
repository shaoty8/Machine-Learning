function [W,H,error] = nmf(X)
W = rand(1024,25);
H = rand(25,1000);
error = zeros(1,200);
for i = 1:200
     A = W' * X;
     B = W' * W * H;
     
    for j = 1:25
        for k = 1:1000
           H(j,k) = H(j,k) * A(j,k) / B(j,k);
        end
    end
    
    C = X * H';
    D = W * H * H';
    for j = 1:1024
        for k = 1:25
           W(j,k) = W(j,k) * C(j,k) / D(j,k);
        end
    end
    
    err = (X - W * H).^2;
    error(i) = sum(err(:));
end

plot(error)