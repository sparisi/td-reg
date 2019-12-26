close all

folder = './';
file_regexp = '*.mat';
Files = dir(fullfile(folder, file_regexp));
J_bound = -500;
TD_bound = inf;

figure, hold all, title('J')
for f = {Files.name}
    h = load([folder f{:}]);
    try plot(max(h.J_history, J_bound), 'displayname', f{:}), catch, end
end

legend show

figure, hold all, title('TD')
for f = {Files.name}
    h = load([folder f{:}]);
    try plot(min(log(h.td_history), TD_bound), 'displayname', f{:}), catch, end
end

legend show

figure, hold all, title('TD true')
for f = {Files.name}
    h = load([folder f{:}]);
    try plot(min(log(h.td_true_history), TD_bound), 'displayname', f{:}), catch, end
end

legend show

autolayout