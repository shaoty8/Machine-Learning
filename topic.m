function [I] = topic(W,nyt_vocab)
I = zeros(10,10);
for i = 1:10
    [~,A] = sort(W(:,i),'descend');
    I(i,:) = A(1:10);
end


for i = 1:10
    'this is an interval'
    for j = 1:10
       nyt_vocab{I(i,j)}
    end
    
end