close all
clear all

%% Format: degree_episodes_obs-noise
folder_load = './3_1_1/';
separator = '_';

filenames = {'reinf', 'spg_td', 'spg_tdreg'};
legendnames = {'REINFORCE', 'SPG TD', 'SPG TD-REG'};
colors = {};
markers = {};

variable = 'max(J_history,-1000)';
% variable = 'min(td_history,1e7)';
% variable = 'min(td_true_history,1e7)';

%% Plot
h = {};
for name = filenames
    
    counter = 1;
    dataMatrix = [];
    for trial = 1 : 999
        try
            load([folder_load name{:} separator num2str(trial) '.mat'])
            dataMatrix(counter,:) = eval(variable);
            counter = counter + 1;
        catch
        end
    end
    
    if ~isempty(dataMatrix)
        hold all
        lineprops = { 'LineWidth', 3, 'DisplayName', name{:} };
        if ~isempty(colors)
            lineprops = {lineprops{:}, 'Color', colors{numel(h)+1} };
        end
        if ~isempty(markers)
            lineprops = {lineprops{:}, 'Marker', markers{numel(h)+1} };
        end
        tmp = shadedErrorBar( ...
            1:size(dataMatrix,2), ...
            mean(dataMatrix,1), ...
            1.96*std(dataMatrix)/sqrt(size(dataMatrix,1)), ...
            lineprops, ...
            0.1, 0 );
        h{end+1} = tmp.mainLine;
    end
    
end

legend([h{:}], legendnames, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];