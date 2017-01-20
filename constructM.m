M = zeros(943,1682);
omega = [];
for i = 1:1682
     [m,n] = size(movie(i).user_id);
    for j = 1:n
        M(movie(i).user_id(j),i) = movie(i).rating(j);
        omega(end+1,:) = [movie(i).user_id(j),i];
    end
end