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
plot (data (:,13), "-c;inited;"); 
hold on;
plot (data (:,14), "-c;number of classifiers;"); 
hold on;
plot (data (:,18), ".m;correct set population;"); 
grid;
print -color ~/graphs/main.jpg;
figure;
plot (data (:,15), "-r;deleted by ns * <F> / f;"); 
hold on;
plot (data (:,16), "-g;deleted by ns;"); 
hold on;
plot (data (:,17), "-b;subsumptions;"); 
hold on;
plot (data (:,22), "-k;cover;"); 
hold on;
plot (data (:,23), "-c;ga;"); 
grid;
print -color ~/graphs/deletions.jpg;

data = load("~/latestMetricFiles/deletions.txt");
figure;
plot (data (:,3), data (:,1),"+b;quality;");
hold on;
#plot (data (:,3) / 500,"-g;iterations;");
hold on;
plot (data (:,3), data (:,4),"+g;origin,0cov,1ga;");
grid;
print -color ~/graphs/quality.jpg;


figure;
plot (data (:,2),"+r;acc;");
hold on;
plot (data (:,3) /  500,"-b;iterations;");
hold on;
plot (data (:,4),".g;origin;");
grid;
print -color ~/graphs/accuracyOfDeleted1.jpg;

figure;
plot (data (:,3), data (:,2),".r;acc;");
hold on;
#plot (data (:,3) /  500,"-g;iterations;");
hold on;
plot (data (:,3), data (:,5),"+b;covered;");
hold on;
plot (data (:,3), data (:,6),"+r;gaed;");
grid;
print -color ~/graphs/accuracyOfDeleted2.jpg;

data= load("~/latestMetricFiles/systemProgress.txt");
data2 = load("~/latestMetricFiles/measurements.txt");
figure;
plot (data (:,1),"-r; acc during test;");
hold on;
plot (data (:,2),"-b; acc during train;");
hold on;
plot (data (:,3) * 10,"-g; mean coverage;");


grid;
print -color ~/graphs/systemAccuracy.jpg;
figure;

plot (data2 (:,24),"-k;classifier accuracy;");
hold on;
plot (data2 (:,25),"-b;covered accuracy;");
hold on;
plot (data2 (:,26),"-r;gaed accuracy;");
hold on;
plot (data2 (:,19) / 100,"-g;mean ns;");
grid;
print -color ~/graphs/classifiersAccuracy.jpg;

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
print -color ~/graphs/classifiersProgress.jpg;


data = load("~/latestMetricFiles/deletions.txt");
figure;
plot (data (:,3),data (:,7),"+b;quality of covered;");
hold on;
#plot (data (:,3) / 200,"-g;iterations;");
hold on;
plot (data (:,3),data (:,8),".r;quality of gaed;");
grid;
print -color ~/graphs/newQuality.jpg;

data=load("~/latestMetricFiles/zeroCoverage.txt");
figure;
plot (data (:,2),data (:,1),"-b;zero coverage deletions;");
grid;
print -color ~/graphs/zeroCoverage.jpg;


data2 = load("~/latestMetricFiles/measurements.txt");
figure;
plot (data2 (:,27),"-b;exploration fitness;");
hold on;
plot (data2 (:,28),"-k;exploration fitness:covered;");
hold on;
plot (data2 (:,29),"-r;exploration fitness:gaed;");
hold on;
grid;
print -color ~/graphs/explorationFitness.jpg;
figure;
plot (data2 (:,30),"-b;pure fitness;");
hold on;
plot (data2 (:,31),"-k;pure fitness:covered;");
hold on;
plot (data2 (:,32),"-r;pure fitness:gaed;");
hold on;
grid;
print -color ~/graphs/pureFitness.jpg;
