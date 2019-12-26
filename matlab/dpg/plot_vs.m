close all

folder = './3_1_0.0005_uniform/';
file_regexp = 'dpg*.mat';
Files = dir(fullfile(folder, file_regexp));
J_bound = -500;
TD_bound = 1e6;
eval_every = 1;

figure, hold all, title('J')
for f = {Files.name}
    h = load([folder f{:}]);
    try plot([0:length(h.J_history)-1]*eval_every, max(h.J_history, J_bound), 'displayname', f{:}), catch, end
    disp([f{:} ', ' num2str(min(h.J_history)) ', ' num2str(h.J_history(end))])
end
xlabel Steps
ylabel 'Expected return'
xlim([0, (length(h.J_history)-1)*eval_every])
legend show
legend('Interpreter', 'none')

figure, hold all, title('TD')
for f = {Files.name}
    h = load([folder f{:}]);
    try plot([0:length(h.td_history)-1]*eval_every, min(h.td_history, TD_bound), 'displayname', f{:}), catch, end
end
xlabel Steps
ylabel 'Mean Squared TD Error (Estimate)'
xlim([0, (length(h.td_history)-1)*eval_every])
legend show
legend('Interpreter', 'none')

figure, hold all, title('TD true')
for f = {Files.name}
    h = load([folder f{:}]);
    try plot([0:length(h.td_true_history)-1]*eval_every, min(h.td_true_history, TD_bound), 'displayname', f{:}), catch, end
end
xlabel Steps
ylabel 'Mean Squared TD Error (True)'
xlim([0, (length(h.td_true_history)-1)*eval_every])
legend show
legend('Interpreter', 'none')

autolayout
