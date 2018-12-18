close all
clear all

%% Format: degree_tau-omega_trans-noise
folder_data = './2_0.01_noise/';
folder_data = './2_1_noise/';
% folder_data = './3_0.01_noise/';
folder_data = './3_1_noise/';
separator = '_';

filenames = {'dpg', 'dpg_notar', 'tdreg', 'td3_nodelay', 'td3reg_nodelay'};
colors = {[0.00000,0.44700,0.74100], [0.46600,0.67400,0.18800], [0.85000,0.32500,0.09800], [0.92900,0.69400,0.12500]};
colors = {};
markers = {};
legendnames = {'DPG', 'DPG NO-TAR', 'DPG TD-REG', 'TD3 NO-DELAY', 'TD3 TD-REG NO-DELAY'};

variable = 'max(J_history,-1000)';
% variable = 'min(td_history,300000)';
% variable = 'min(td_true_history,300000)';

%% Plot
h = {};
for name = filenames
    
    counter = 1;
    dataMatrix = [];
    for trial = 1 : 999
        try
            load([folder_data name{:} separator num2str(trial) '.mat'])
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
    sum(dataMatrix(:,end) == -1000)
end

legend([h{:}], legendnames, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
