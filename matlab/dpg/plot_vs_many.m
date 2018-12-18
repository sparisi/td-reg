% dpg fails in trials 1, 4, 6

close all,
figure, hold all

%%
folder = './paperplot/';
name = 'dpg';
ntrials = 10;

for idx_trial = 1 : 10

copyfile([folder(3:end) name '_' num2str(idx_trial) '.mat'], ['./' name '.mat'])

c = 'b';
if idx_trial == 1 || idx_trial == 4 || idx_trial == 6 || idx_trial == 10
    c = 'r';
end

load(name)

plot(log(td_true_history),'color',c)

end