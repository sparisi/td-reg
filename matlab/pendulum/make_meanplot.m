% Compares results of different methods averaged over many trials. 
% You need to set the folder where the data is stored and the filename
% format (eg, alg1_1.mat, alg1_2.mat, ..., alg2_1.mat, ...)
%
% Plot only mean.

close all
clear all
figure()
h = {};
plot_every = 25; % plots data with the following indices [1, 5, 10, ..., end]

%% Change entries according to your needs
folder = './data_single/';
folder = './data_double/';
separator = '_';

filenames = {'a', 'Da', 'Ra', 'RDa', ...
    't', 'Dt', 'Rt', 'RDt', ...
    'v', 'Dv', 'Rv', 'RDv', };
legendnames = {'GAE-REG', 'GAE-REG + DOUBLE', 'GAE-REG + RETR', 'GAE-REG + DOUBLE + RETR', ...
    'TD-REG', 'TD-REG + DOUBLE', 'TD-REG + RETR', 'TD-REG + DOUBLE + RETR', ...
    'NO-REG', 'NO-REG + DOUBLE', 'NO-REG + RETR', 'NO-REG + DOUBLE + RETR'};
colors = {'y' 'y', 'y', 'y', ...
    'b' 'b', 'b', 'b', ...
    'r' 'r', 'r', 'r'};
markers = {'+', '^', 'o', 'x', ...
    '+', '^', 'o', 'x', ...
    '+', '^', 'o', 'x'};

variable = 'J_history';
% variable = 'min(td_history,inf)';
title(variable, 'Interpreter', 'none')

%% Plot
name_idx = 1;
name_valid = [];
for name = filenames
    counter = 1;
    dataMatrix = [];
    for trial = 1 : 50
        try
            load([folder name{:} separator num2str(trial) '.mat'], variable)
            dataMatrix(counter,:) = eval(variable);
            counter = counter + 1;
        catch
        end
    end
    
    if ~isempty(dataMatrix)
        hold all
        lineprops = { 'LineWidth', 2, 'DisplayName', name{:} };
        if ~isempty(colors)
            lineprops = {lineprops{:}, 'Color', colors{name_idx} };
        end
        if ~isempty(markers)
            lineprops = {lineprops{:}, 'Marker', markers{name_idx} };
        end
        m = mean(dataMatrix,1);
        x = 0 : plot_every : length(m);
        x(1) = 1;
        h{end+1} = plot(x, m(x), lineprops{:});
        name_valid(end+1) = name_idx;
    end
    name_idx = name_idx + 1;
    
end

legend([h{:}], legendnames{name_valid}, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
