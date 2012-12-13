data = load("~/latestMetricFiles/measurements.txt");
figure;
plot (data (:,2),"-k;number of macros;");
hold on;
plot (data (:,13), "-g;first-timers;"); 
hold on;
plot (data (:,11),"-b;covered;");
hold on;
plot (data (:,12), "-r;gaed;"); 
hold on;
plot (data (:,14), "-c;number of classifiers;"); 
hold on;
plot (data (:,18), ".m;correct set population;"); 
grid;
print -color /home/li9i/graphs/main.jpg;

figure;

plot (data (:,15), "-r;deleted by first formula;"); 
hold on;
plot (data (:,16), "-g;deleted by second formula;"); 
hold on;
plot (data (:,17), "-b;subsumptions;"); 
hold on;
plot (data (:,22), "-k;cover;"); 
hold on;
plot (data (:,23), "-c;ga;"); 
grid;
print -color /home/li9i/graphs/deletions.jpg;
