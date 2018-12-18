function data = getdata(data, data_iter, nmax, vars, bfs)
% Organizes collected data. It adds new data to previous data, keeping only
% up to NMAX samples. 
%
% VARS must be a cell array of variables string names. For the possible 
% names, please check COLLECT_SAMPLES.
% 
% BFS is a cell array of pairs {name, function handle} of basis functions
% depending on the state.

% Init
if isempty(data)
    for i = 1 : numel(vars)
        data.(vars{i}) = [];
    end
    for i = 1 : numel(bfs)
        data.(bfs{i}{1}) = [];
        data.([bfs{i}{1} '_nexts']) = [];
    end
end

for i = 1 : numel(vars)
    data.(vars{i}) = [[data_iter.(vars{i})], data.(vars{i})]; % Enqueue
    data.(vars{i}) = data.(vars{i})(:, 1:min(nmax,end)); % Keep up to NMAX samples
end

for i = 1 : numel(bfs)
    data.(bfs{i}{1}) = [bfs{i}{2}([data_iter.s]), data.(bfs{i}{1})];
    data.(bfs{i}{1}) = data.(bfs{i}{1})(:, 1:min(nmax,end));
    data.([bfs{i}{1} '_nexts']) = [bfs{i}{2}([data_iter.nexts]), data.([bfs{i}{1} '_nexts'])];
    data.([bfs{i}{1} '_nexts']) = data.([bfs{i}{1} '_nexts'])(:, 1:min(nmax,end));
end
