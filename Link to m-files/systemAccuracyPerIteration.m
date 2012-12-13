data= load("~/latestMetricFiles/systemProgress.txt");
data2 = load("~/latestMetricFiles/measurements.txt");
figure;
plot (data (:,1),"-b; acc during train;");
hold on;
plot (data (:,2),"-k; acc during test in pcut;");
hold on;
plot (data (:,3) * 10,"-g; mean coverage;");
grid;
print -color /home/li9i/graphs/systemAccuracy.jpg;

figure;

plot (data2 (:,24),"-k;classifier accuracy;");
hold on;
plot (data2 (:,25),"-b;covered accuracy;");
hold on;
plot (data2 (:,26),"-r;gaed accuracy;");
hold on;
plot (data2 (:,19) / 100,"-g;mean ns;");
grid;
print -color /home/li9i/graphs/classifiersAccuracy.jpg;

