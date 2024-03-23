rawdata = importdata('eval_hip_adduction_001.csv');
data = rawdata.data;
x = data(:, 3);
y = data(:, 4);
z = data(:, 5);
%[pks,locs] = 
findpeaks(-x,'SortStr', 'descend', 'Npeaks',100)
%hold on
%figure(1)
%plot(x)
%plot(locs, pks, 'o')
%hold off