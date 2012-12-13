data = load("~/latestMetricFiles/deletions.txt");
figure;
plot (data (:,3),data (:,7),"+b;quality of covered;");
hold on;
#plot (data (:,3) / 200,"-g;iterations;");
hold on;
plot (data (:,3),data (:,8),".r;quality of gaed;");
grid;
print -color /home/li9i/graphs/newQuality.jpg;