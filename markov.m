function [M,wt,top,topwt,w] = markov(scores,t,legend)
M = zeros(759,759);
for i = 1:4185
    if scores(i,2) > scores(i,4)
        a = 1;
    else
        a = 0;
    end
    M(scores(i,1),scores(i,1)) = M(scores(i,1),scores(i,1)) + a + scores(i,2)/(scores(i,2)+scores(i,4));
    M(scores(i,3),scores(i,3)) = M(scores(i,3),scores(i,3)) + 1-a + scores(i,4)/(scores(i,2)+scores(i,4));
    M(scores(i,1),scores(i,3)) = M(scores(i,1),scores(i,3)) + 1-a + scores(i,4)/(scores(i,2)+scores(i,4));
    M(scores(i,3),scores(i,1)) = M(scores(i,3),scores(i,1)) + a + scores(i,2)/(scores(i,2)+scores(i,4));
end

%normalization
for i = 1:759
    M(i,:) = M(i,:) / sum(M(i,:));
end

%state distribution
w = zeros(t+1,759);
w(1,:) = 1/759;

for i = 1:t
    w(i+1,:) = w(i,:) * M;
end

%sort
wt = w(t+1,:);
[Best,I] = sort(wt,'descend');
top = legend(I(1:25));
topwt = Best(1:25);
