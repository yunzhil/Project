rawdata = importdata('eval_hip_adduction_001.csv');
data = rawdata.data;
x = data(:, 3);
y = data(:, 4);
z = data(:, 5);

curr = x;   %which column of data to look at
[rawPk, rawLoc] = findpeaks(curr);
diff = [];
dist = [];
for i = 1:(length(rawPk) - 1)   %filtering
    height = abs(rawPk(i+1) - rawPk(i));    %in case values are negative
    if (height > 0.01)  %sensitivity: the smaller the value, the more peaks will be found
        diff(end+1) = height;
        dist(end+1) = rawLoc(i+1) - rawLoc(i);  %no abs because index always positive
    end
end

prominence = mean(diff);
peakDist = mean(dist);

[pks,locs] = findpeaks(curr,'MinPeakDistance', peakDist, 'MinPeakProminence', prominence);
[valleys,vLocs] = findpeaks(-curr,'MinPeakDistance', peakDist, 'MinPeakProminence', prominence);
hold on
figure(1)
plot(curr)
plot(locs, pks, 'o')
plot(vLocs, -valleys, 'x')
hold off