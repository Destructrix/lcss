data = load("~/latestMetricFiles/measurements.txt");
figure;
plot (data (:,2),"-k;number of macros;");
hold on;
plot (data (:,11),"-b;covered;");
hold on;
plot (data (:,12), "-r;gaed;"); 
hold on;
plot (data (:,14), "-c;number of classifiers;"); 
hold on;
plot (data (:,22) / 10, "-k;cover-ed deleted;"); 
hold on;
plot (data (:,23) / 10, "-c;ga-ed deleted;"); 
hold on;
plot (data (:,19) , "-r;mean ns;"); 
hold on;
plot (data (:,17), "-k;subsumptions;"); 
grid;
print -color /home/li9i/graphs/classifiersProgress.jpg;


