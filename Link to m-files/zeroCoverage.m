data=load("~/latestMetricFiles/zeroCoverage.txt");
figure;
plot (data (:,2),data (:,1),"-b;zero coverage deletions;");
grid;
print -color /home/li9i/graphs/zeroCoverage.jpg;