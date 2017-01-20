
x1 = [];
x2 = [];
x3 = [];
x4 = [];
x5 = [];
for i = 1:500
    if c(i) == 1
        x1(end+1,:) = Data(i,:);
    elseif c(i) == 2
        x2(end+1,:) = Data(i,:);
    elseif c(i) == 3
        x3(end+1,:) = Data(i,:);
    elseif c(i) ==4
        x4(end+1,:) = Data(i,:);
    elseif c(i) ==5
        x5(end+1,:) = Data(i,:);
    end
end
scatter(x1(:,1),x1(:,2),'r'); hold on;
scatter(x2(:,1),x2(:,2),'g'); hold on;
scatter(x3(:,1),x3(:,2),'b'); hold on;
scatter(x4(:,1),x4(:,2),'y'); hold on;
scatter(x5(:,1),x5(:,2),'k'); 