data2 = load("~/latestMetricFiles/measurements.txt");
figure;
plot (data2 (:,27),"-b;exploration fitness;");
hold on;
plot (data2 (:,28),"-k;exploration fitness:covered;");
hold on;
plot (data2 (:,29),"-r;exploration fitness:gaed;");
hold on;
grid;
print -color /home/li9i/graphs/explorationFitness.jpg;

figure;
plot (data2 (:,30),"-b;pure fitness;");
hold on;
plot (data2 (:,31),"-k;pure fitness:covered;");
hold on;
plot (data2 (:,32),"-r;pure fitness:gaed;");
grid;
print -color /home/li9i/graphs/pureFitness.jpg;
