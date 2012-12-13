data = load("~/latestMetricFiles/deletions.txt");
figure;
plot (data (:,3), data (:,1),"+b;quality;");
hold on;
#plot (data (:,3) / 500,"-g;iterations;");
hold on;
plot (data (:,3), data (:,4),"+g;origin,0cov,1ga;");
grid;
print -color /home/li9i/graphs/quality.jpg;


figure;
plot (data (:,2),"+r;acc;");
hold on;
plot (data (:,3) /  500,"-b;iterations;");
hold on;
plot (data (:,4),".g;origin;");
grid;
print -color /home/li9i/graphs/accuracyOfDeleted1.jpg;

figure;
plot (data (:,3), data (:,2),".r;acc;");
hold on;
#plot (data (:,3) /  500,"-g;iterations;");
hold on;
plot (data (:,3), data (:,5),"+b;covered;");
hold on;
plot (data (:,3), data (:,6),"+r;gaed;");
grid;
print -color /home/li9i/graphs/accuracyOfDeleted2.jpg;