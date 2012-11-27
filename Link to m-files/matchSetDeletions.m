data=load("~/latestMetricFiles/measurements.txt");
figure;
plot (data(:,33),"-b;[M] deletions;");
grid;
print -color ~/graphs/matchsetDeletions.jpg;