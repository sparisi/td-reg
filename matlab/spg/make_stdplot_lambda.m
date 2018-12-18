close all
clear all

%% 
folder_load = 'lambda_3_1_0/';
base_name = 'spg_tdreg';
separator = '_';
hp1_list = {0, 0.1, 0.5, 0.9, 0.99, 0.999, 1, 1.001}; % hyperparameter lists
hp1_list = {0.1, 0.5, 0.9, 0.99}; % hyperparameter lists
filenames = {};
legendnames = {'\kappa = 0', '\kappa = 0.1', '\kappa = 0.5', '\kappa = 0.9', '\kappa = 0.99', '\kappa = 0.999', '\kappa = 1', '\kappa = 1.001'};
legendnames = {'\kappa = 0.1', '\kappa = 0.5', '\kappa = 0.9', '\kappa = 0.99, 0.999, 1, 1.001'};

variable = 'max(J_history,-1000)';
% variable = 'min(td_history,300000)';
% variable = 'min(td_true_history,300000)';

%% Plot
h = {};
for hp1 = hp1_list

name = [base_name, separator, num2str(hp1{:})];
counter = 1;
dataMatrix = [];
for trial = 1 : 999
    try
        load([folder_load name separator num2str(trial) '.mat'])
        dataMatrix(counter,:) = eval(variable);
        counter = counter + 1;
    catch
    end
end

if ~isempty(dataMatrix)
    hold all
    tmp = shadedErrorBar( ...
        1:size(dataMatrix,2), ...
        mean(dataMatrix,1), ...
        1.96*std(dataMatrix)/sqrt(size(dataMatrix,1)), ...
        { 'LineWidth', 3, 'DisplayName', name }, ...
        0.1, 0 );
        h{end+1} = tmp.mainLine;
    filenames = [filenames, name];
end

end

legend([h{:}], legendnames, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
