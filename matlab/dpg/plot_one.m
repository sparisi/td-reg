% TD error (true) is computed as Q - Q_true, where Q_true is the true
% Q-function of pi in closed form.
% TD error (learned) is computed as Q - (r + gamma*Q')

% close all

bound_J = -200; % bounds for clarity
bound_TD = 1000;
bound_DTheta = 1000;

figure
plot(max(J_history,bound_J)), title('Avg return')
figure
plot(min(td_history,bound_TD)), title('TD error (learned)')
figure
plot(min(td_true_history,bound_TD)), title('TD error (true)')

figure
plot(theta_history'), title('Theta')
figure
plot(omega_history'), title('Omega')

figure
plot(matrixnorms(df_theta_history(:,1:eval_every:end),2)), title('Norm of DF DTheta')
figure
plot(matrixnorms(dg_theta_history(:,1:eval_every:end),2)), title('Norm of DG DTheta')
% figure
% plot(matrixnorms(dg_omega_history(:,1:eval_every:end),2)), title('Norm of DG DOmega')

figure
plot(max(min(df_theta_history(:,1:eval_every:end)',bound_DTheta),-bound_DTheta)), title('DF DTheta')
figure
plot(max(min(dg_theta_history(:,1:eval_every:end)',bound_DTheta),-bound_DTheta)), title('DG DTheta')
% figure
% plot(dg_omega_history(:,1:eval_every:end)'), title('DG DOmega')


% Normalized gradients to check direction
if ~isempty(dg_theta_history)
    direction_theta = sign(df_theta_history) == sign(dg_theta_history);
    df_theta_history_norm = bsxfun(@times, df_theta_history(:,1:eval_every:end), 1 ./ matrixnorms(df_theta_history(:,1:eval_every:end),2));
    dg_theta_history_norm = bsxfun(@times, dg_theta_history(:,1:eval_every:end), 1 ./ matrixnorms(dg_theta_history(:,1:eval_every:end),2));
    figure, title('Gradients')
    for i = 1 : dim_theta
        subplot(1,dim_theta,i), hold all
        plot(df_theta_history_norm(i,:))
        plot(dg_theta_history_norm(i,:))
        legend({'E[Q]', 'E[TD^2]'})
    end
end

autolayout
