clear all
close all
warning('off','all')

folder = './3_1_noise/'; idx_trial = 7;
folder = './2_1_noise/'; idx_trial = 2;

dim = 2;
gamma = 0.99;
mdp = LQR(dim);
mdp.gamma = gamma;
theta_opt = mdp.opt();
f = @(x,y)mdp.avg_return(diag([x y]), 0);

lims = linspace(-1.3,0,50);
for i = 1:length(lims)
    for j = 1:length(lims)
        x = lims(i);
        y = lims(j);
        fun(i,j) = f(x,y);
    end
end

colors = [1 0.2 0.2; 1 0.5 0.2; 0.2 0.5 1; 0.49400,0.18400,0.55600; 0.46600,0.67400,0.18800];
fontsize1 = 10;

%%
close all,
fig(1)
clim = [-9000, -1000, -150, -130, -120, -115, -110, -108];
[C,h] = contourf(lims, lims, fun, clim);
v = [-130, -120, -115, -110, -108];
clabel(C,h,v, 'labelspacing',700, 'fontsize', fontsize1);
caxis([-300, -100]);
axis([-1.3, 0, -1, 0]);
hold on;
colormap(gray);
xlabel('Actor parameter K_{11}');
ylabel('Actor parameter K_{22}');


%% 
name = 'tdreg';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
theta_history_tdreg = theta_history;
grad1_history_tdreg = bsxfun(@times, df_theta_history, 1./matrixnorms(df_theta_history,2));
grad2_history_tdreg = bsxfun(@times, dg_theta_history, 1./matrixnorms(dg_theta_history,2));
name = 'dpg';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
theta_history_dpg = theta_history;
grad1_history_dpg = bsxfun(@times, df_theta_history, 1./matrixnorms(df_theta_history,2));
grad2_history_dpg = -bsxfun(@times, dg_theta_history, 1./matrixnorms(dg_theta_history,2));
name = 'dpg_notar';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
theta_history_dpg_notar = theta_history;
grad1_history_dpg_notar = bsxfun(@times, df_theta_history, 1./matrixnorms(df_theta_history,2));
grad2_history_dpg_notar = -bsxfun(@times, dg_theta_history, 1./matrixnorms(dg_theta_history,2));
name = 'td3';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
theta_history_td3 = theta_history;
grad1_history_td3 = bsxfun(@times, df_theta_history, 1./matrixnorms(df_theta_history,2));
grad2_history_td3 = -bsxfun(@times, dg_theta_history, 1./matrixnorms(dg_theta_history,2));
name = 'td3reg';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
theta_history_td3reg = theta_history;
grad1_history_td3reg = bsxfun(@times, df_theta_history, 1./matrixnorms(df_theta_history,2));
grad2_history_td3reg = -bsxfun(@times, dg_theta_history, 1./matrixnorms(dg_theta_history,2));

for i = 1 : 1 : size(theta_history_tdreg,2)-1
    p1 = [theta_history_tdreg(1,i) theta_history_tdreg(4,i)];
    p2 = [theta_history_tdreg(1,i+1) theta_history_tdreg(4,i+1)];
    dp = p2-p1;
    quiver(p1(1),p1(2),dp(1),dp(2),0,'b','LineWidth',2,'MarkerSize',10,'MaxHeadSize',1)

    p1 = [theta_history_dpg(1,i) theta_history_dpg(4,i)];
    p2 = [theta_history_dpg(1,i+1) theta_history_dpg(4,i+1)];
    dp = p2-p1;
    quiver(p1(1),p1(2),dp(1),dp(2),0,'r','LineWidth',2,'MarkerSize',10,'MaxHeadSize',1)

    p1 = [theta_history_dpg_notar(1,i) theta_history_dpg_notar(4,i)];
    p2 = [theta_history_dpg_notar(1,i+1) theta_history_dpg_notar(4,i+1)];
    dp = p2-p1;
    quiver(p1(1),p1(2),dp(1),dp(2),0,'Color','y','LineWidth',2,'MarkerSize',10,'MaxHeadSize',1)

    p1 = [theta_history_td3(1,i) theta_history_td3(4,i)];
    p2 = [theta_history_td3(1,i+1) theta_history_td3(4,i+1)];
    dp = p2-p1;
    quiver(p1(1),p1(2),dp(1),dp(2),0,'Color',colors(4,:),'LineWidth',2,'MarkerSize',10,'MaxHeadSize',1)

    p1 = [theta_history_td3reg(1,i) theta_history_td3reg(4,i)];
    p2 = [theta_history_td3reg(1,i+1) theta_history_td3reg(4,i+1)];
    dp = p2-p1;
    quiver(p1(1),p1(2),dp(1),dp(2),0,'Color',colors(5,:),'LineWidth',2,'MarkerSize',10,'MaxHeadSize',1)
end

name = 'tdreg';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
hl(1) = plot(theta_history(1,1),theta_history(4,1),'linewidth', 3, 'color', 'b');
hold on;

name = 'dpg';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
hl(2) = plot(theta_history(1,1),theta_history(4,1),'linewidth', 3, 'color', 'r');
hold on;

name = 'dpg_notar';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
hl(3) = plot(theta_history(1,1),theta_history(4,1),'linewidth',3, 'color', 'y');
hold on;

name = 'td3';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
hl(4) = plot(theta_history(1,1),theta_history(4,1),'linewidth',3, 'color', colors(4,:));
hold on;

name = 'td3reg';
load([folder(3:end) name '_' num2str(idx_trial) '.mat'])
hl(5) = plot(theta_history(1,1),theta_history(4,1),'linewidth',3, 'color', colors(5,:));
hold on;


plot(theta_history(1,1),theta_history(4,1),'o','MarkerSize',12,'MarkerFaceColor','w') % Init
set(gca, 'xtick', [-0.8:0.4:0], 'ytick', [-1:0.2:0], 'fontsize', fontsize1);
hl_dpg1 = legend(hl, 'DPG TD-REG', 'DPG', 'DPG NO-TAR', 'TD3', 'TD3 TD-REG', 'location', 'northwest');
set(hl_dpg1, 'fontsize', fontsize1);
