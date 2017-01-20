function [W,H,error] = nyt(Xcnt, Xid)
%construct X
X = zeros(3012,8447);
X = X + 1e-16;
for i = 1:8447
    [~,n] = size(Xid{i});
    for j = 1:n
        X(Xid{i}(j),i) = Xcnt{i}(j);
    end
end

%NMF
W = rand(3012,25);
H = rand(25,8447);
error = zeros(1,200);
for i = 1:200
     A = W' * X;
     B = W' * W * H;
     
    for j = 1:25
        for k = 1:8447
           H(j,k) = H(j,k) * A(j,k) / B(j,k);
        end
    end
    
    C = X * H';
    D = W * H * H';
    for j = 1:3012
        for k = 1:25
           W(j,k) = W(j,k) * C(j,k) / D(j,k);
        end
    end
    
    err = (X.*log(W*H) - W*H).^2;
    error(i) = sum(err(:));
end

%plot objective function
plot(error)

%normalization
for  i = 1:25
    W(:,i) = W(:,i)/sum(W(:,i));
end
