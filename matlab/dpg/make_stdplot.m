% close all
figure()
clear all

%% Format: degree_tau-omega_trans-lrate_pi-noise
folder_data = './3_1_0.0005_uniform/';
% folder_data = './3_1_0.005_non-uniform/';
% folder_data = './';
separator = '_';

filenames = {'dpg', 'dpg_notar', 'dpg_reg', 'td3', 'td3_reg'};
colors = {[0.00000,0.44700,0.74100], [0.46600,0.67400,0.18800], [0.85000,0.32500,0.09800], [0.92900,0.69400,0.12500]};
colors = {};
markers = {};
legendnames = {};

filenames = {};

if isempty(filenames) % automatically identify algorithms name
    allfiles = dir(fullfile(folder_data,'*.mat'));
    for i = 1 : length(allfiles)
        tmpname = allfiles(i).name(1:end-4); % remove .mat from string
        trial_idx = strfind(tmpname, separator); % find separator
        tmpname = tmpname(1:trial_idx(end)-1); % remove trial idx from string
        if (isempty(filenames) || ~strcmp(filenames{end}, tmpname) ) && ~any(strcmp(filenames, tmpname)) % if new name, add it
            filenames{end+1} = tmpname;
        end
    end
end

variable = 'max(J_history,-1000)';
% variable = 'mean(min(td_history,300000),1)';
% variable = 'mean(min(td_true_history,300000),1)';
% variable = 'l2_diff_history';

%% Plot
h = {};
for name = filenames
    
    counter = 1;
    dataMatrix = [];
    for trial = 1 : 99
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
%         tmp = plot(mean(dataMatrix,1), lineprops{:});
%         h{end+1} = tmp;
        tmp = shadedErrorBar( ...
            1:size(dataMatrix,2), ...
            mean(dataMatrix,1), ...
            1.96*std(dataMatrix)/sqrt(size(dataMatrix,1)), ...
            lineprops, ...
            0.1, 0 );
        h{end+1} = tmp.mainLine;

        % show number of diverged and non-converged runs
        disp([name{:} ': ' num2str(sum(dataMatrix(:,end) == -1000)) ', ' num2str(sum(dataMatrix(:,end) < -110))])
    end
end

legend([h{:}], legendnames, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
