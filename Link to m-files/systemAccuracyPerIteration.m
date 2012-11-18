data= load("latestMetricFiles/systemProgress.txt");
data2 = load("latestMetricFiles/measurements.txt");
figure;
plot (data (:,1),"-r; acc during test;");
hold on;
plot (data (:,2),"-b; acc during train;");
hold on;
plot (data (:,3) * 10,"-g; mean coverage;");


grid;
print -color graphs/systemAccuracy.jpg;
figure;

plot (data2 (:,24),"-k;classifier accuracy;");
hold on;
plot (data2 (:,25),"-b;covered accuracy;");
hold on;
plot (data2 (:,26),"-r;gaed accuracy;");
hold on;
plot (data2 (:,19) / 100,"-g;mean ns;");
grid;
print -color graphs/classifiersAccuracy.jpg;

