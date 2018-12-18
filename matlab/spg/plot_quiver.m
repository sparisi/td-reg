clear all
close all
warning('off','all')

dim = 2;
mdp = LQR(dim);
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
subplot(121)

clim = [-9000, -1000, -150, -130, -120, -115, -110, -108];
clim = -130 : 1 : -108;
[C,h] = contourf(lims, lims, fun, clim);
v = [-130, -120, -115, -110, -108];
clabel(C,h,v, 'labelspacing',700, 'fontsize', fontsize1);
caxis([-300, -100]);
axis([-0.8, 0, -0.7, 0]);
hold on;
colormap(gray);
xlabel('Actor parameter K_{11}');
ylabel('Actor parameter K_{22}');

% 2 3 5 6
load 3_1_0/spg_tdreg_6.mat 

theta_history_tdreg = theta_history;
norm1 = matrixnorms(df_dtheta_history([1,4],:),2);
norm2 = matrixnorms(dg_dtheta_history([1,4],:),2);
logratio = 1./abs(log(norm1./norm2));
lrate = 0.01;
grad1_history_tdreg = bsxfun(@times, df_dtheta_history, 1./norm1)*lrate;
grad2_history_tdreg = bsxfun(@times, dg_dtheta_history, 1./norm2)*lrate;

% theta_history_tdreg = theta_history;
% norm_full = matrixnorms(df_dtheta_history + dg_dtheta_history, 2);
% logratio = ones(1,size(theta_history,2));
% lrate = 0.01;
% grad1_history_tdreg = bsxfun(@times, df_dtheta_history, 1./max(norm_full, 1))*lrate;
% grad2_history_tdreg = bsxfun(@times, dg_dtheta_history, 1./max(norm_full, 1))*lrate;

stepx = 1;

xmin = min(min([theta_history_tdreg(1,:)]));
xmax = max(max([theta_history_tdreg(1,:)]));
ymin = min(min([theta_history_tdreg(4,:)]));
ymax = max(max([theta_history_tdreg(4,:)]));


%% 
for i = 1 : stepx : size(theta_history_tdreg,2)-stepx
    p1 = [theta_history_tdreg(1,i) theta_history_tdreg(4,i)];
    p2 = [theta_history_tdreg(1,i) theta_history_tdreg(4,i)] + [grad1_history_tdreg(1,i+1) grad1_history_tdreg(4,i+1)];
    dp = p2-p1;
    quiver(p1(1),p1(2),dp(1),dp(2),0,'r','LineWidth',1,'MarkerSize',10,'MaxHeadSize',1)

    p1 = [theta_history_tdreg(1,i) theta_history_tdreg(4,i)];
    p2 = [theta_history_tdreg(1,i) theta_history_tdreg(4,i)] + [grad2_history_tdreg(1,i+1) grad2_history_tdreg(4,i+1)];
    dp = (p2-p1)*max(min(logratio(i),1.33),0.1);
    quiver(p1(1),p1(2),dp(1),dp(2),0,'b','LineWidth',1,'MarkerSize',10,'MaxHeadSize',1)
end

% plot(M(1),M(2),'p','MarkerSize',10,'MarkerFaceColor','r') % Goal
plot(theta_history(1,1),theta_history(4,1),'o','MarkerSize',8,'MarkerFaceColor','w') % Init


hl(1) = plot(theta_history(1,1),theta_history(4,1),'linewidth', 3, 'color', 'r');
hl(2) = plot(theta_history(1,1),theta_history(4,1),'linewidth', 3, 'color', 'b');

set(gca, 'xtick', [-0.8:0.4:0], 'ytick', [-1:0.2:0], 'fontsize', fontsize1);
hl_dpg1 = legend(hl, 'Maximize Q', 'Minimize \eta\delta^2', 'location', 'northwest');
set(hl_dpg1, 'fontsize', fontsize1);
title('Gradients Direction')

f = subplot(122); hold on
hl1(1) = semilogy(matrixnorms(df_dtheta_history,2),'r','linewidth', 3);
hl1(2) = semilogy(matrixnorms(dg_dtheta_history,2),'b','linewidth', 3);
set(f,'YScale','log')

grid on;
title('Gradients Magnitude')
xlabel('Iteration');
hl11 = legend(hl1, 'Maximize Q', 'Minimize \eta\delta^2');
set(hl11, 'fontsize', fontsize1);
set(gca, 'fontsize', fontsize1, 'ytick', [1e-1 1e3 1e7 1e11 1e15 1e19]);

