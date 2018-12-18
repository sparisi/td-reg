% dpg trials used for the plot 1, 4, 6
clear all
close all

warning('off','all')

dim = 2;
gamma = 0.99;
mdp = LQR(dim);
mdp.gamma = gamma;
theta_opt = mdp.opt();
f = @(x,y)mdp.avg_return(diag([x y]), 0);

lims = linspace(-0.8,0,50);
for i = 1:length(lims)
    for j = 1:length(lims)
        x = lims(i);
        y = lims(j);
        fun(i,j) = f(x,y);
    end
end

colors = [1 0.2 0.2; 1 0.5 0.2; 0.2 0.5 1];
fontsize1 = 10;

%%
close all,
fig(1)
subplot(131);
clim = [-9000, -1000, -150, -130, -120, -115, -110, -108];
[C,h] = contourf(lims, lims, fun, clim);
v = [-130, -120, -115, -110, -108];
clabel(C,h,v, 'labelspacing',700, 'fontsize', fontsize1);
caxis([-300, -100]);
axis([-0.8, 0, -0.7, 0]);
hold on;
colormap(gray);
xlabel('Actor parameter 1');
ylabel('Actor parameter 2');
title('DPG');

folder = './paperplot/';

name = 'dpg';
i = 1;
for idx_trial = [1,4,6]
    load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
    stepx = 1;
    fig(1)
    subplot(131);
    hl_dpg(i) = plot(theta_history(1,1:stepx:end),theta_history(4,1:stepx:end),'linewidth', 3, 'color', colors(i,:), 'linestyle', '-.');
    hold on;
    subplot(133)
    hl1(1) = semilogy(td_history, 'linewidth', 3, 'color', colors(i,:), 'linestyle', '-.');
    hold on;
    i = i + 1;
end

fig(1)
subplot(132);
%plot(M(1),M(2),'p','MarkerSize',10,'MarkerFaceColor','r') % Goal
plot(theta_history(1,1),theta_history(4,1),'o','MarkerSize',8,'MarkerFaceColor','w') % Init
set(gca, 'xtick', [-1:0.2:0], 'ytick', [-1:0.2:0]);

fig(1)
subplot(132);
clim = [-9000, -1000, -150, -130, -120, -115, -110, -108];
[C,h] = contourf(lims, lims, fun, clim);
v = [-130, -120, -115, -110, -108];
clabel(C,h,v, 'labelspacing',700, 'fontsize', 13);
caxis([-300, -100]);
axis([-0.8, 0, -0.7, 0]);
hold on;
colormap(gray);
xlabel('Actor parameter 1');
ylabel('Actor parameter 2');
title('TD-Regularized DPG');

%%
name = 'tdreg';
i = 1;
for idx_trial = [1,4,6]
    load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
    stepx = 1;
    fig(1)
    subplot(132);
    hl_tdreg(i) = plot(theta_history(1,1:stepx:end),theta_history(4,1:stepx:end), 'linewidth', 3, 'color', colors(i,:));
    subplot(133)
    hl1(2) = semilogy(td_history, 'linewidth', 3, 'color', colors(i,:));
    hold on;
    i = i + 1;
end
%plot(M(1),M(2),'p','MarkerSize',10,'MarkerFaceColor','r') % Goal

%%
fig(1);
subplot(131);
plot(theta_history(1,1),theta_history(4,1),'o','MarkerSize',12,'MarkerFaceColor','w') % Init
set(gca, 'xtick', [-0.8:0.4:0], 'ytick', [-1:0.2:0], 'fontsize', fontsize1);
hl_dpg1 = legend(hl_dpg, 'Run 1', 'Run 2', 'Run 3', 'location', 'northwest');
set(hl_dpg1, 'fontsize', fontsize1);

subplot(132);
plot(theta_history(1,1),theta_history(4,1),'o','MarkerSize',12,'MarkerFaceColor','w') % Init
set(gca, 'xtick', [-0.8:0.4:0], 'ytick', [-1:0.2:0], 'fontsize', fontsize1);
hl_tdreg1 = legend(hl_tdreg, 'Run 1', 'Run 2', 'Run 3', 'location', 'northwest');
set(hl_tdreg1, 'fontsize', fontsize1);

subplot(133);
ylim([10^-2 10^15])
xlim([1 100]);
grid on;
title('TD Error Estimates')
xlabel('Steps [x10^2]');
ylabel('Mean Squared TD Error');
hl11 = legend(hl1, 'DPG', 'TD-Reg DPG');
set(hl11, 'fontsize', fontsize1);
set(gca, 'xtick', [20:40:100], 'fontsize', fontsize1, 'ytick', [1e-1 1e3 1e7 1e11 1e15 1e19]);

% print('-dpdf', 'figure1');
