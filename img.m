function [I] = img(W,H)
I = zeros(10,10);
for i = 1:10
    [~,I] = max(H(i,:));
    B = reshape(X(:,I),[32,32]);
    image(B)
end
