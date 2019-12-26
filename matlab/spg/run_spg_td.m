function run_spg_td(trial, folder_save, varargin)

if nargin == 0, trial = 1; folder_save = []; end
if nargin == 1, folder_save = []; end

[policy, ...
    mdp, ...
    episodes_eval, ...
    steps_eval, ...
    episodes_learn, ...
    steps_learn, ...
    basisQ, ...
    options, ...
    maxiter, ...
    omega, ...
    lambda, ...
    lambda_decay, ...
    verbose, ...
    lrate] = common_settings(trial, varargin{:});

iter = 1;

%%
while iter < maxiter
    
    [ds] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).nexta = policy.makeDeterministic.drawAction(ds(i).nexts);
    end
    S = policy.entropy([ds.s]);
    
    phi = basisQ([[ds.s];[ds.a]]);
    phiN = basisQ([[ds.nexts]; [ds.nexta]]);
    QN = omega' * phiN;
    T = [ds.r] + mdp.gamma*QN.*(~[ds.terminal]);
    omega = fminunc(@(omega)mse_linear(omega,phi,T), omega, options);
    Q = omega' * phi;
    QN = omega' * phiN;
    T = [ds.r] + mdp.gamma*QN.*(~[ds.terminal]);
    TD = Q-T;
    
    td_history(:,iter) = mean(TD.^2);
    try
    td_true_history(:,iter) = mean((Q - mdp.q_function(policy.A,0,[ds.s],[ds.a])).^2);
    catch
    td_true_history(:,iter) = mean(TD.^2);
    end
    
    dlogpi = policy.dlogPidtheta([ds.s],[ds.a]);

    grad1 = mean(bsxfun(@times,dlogpi,Q),2);
    grad = grad1;
    grad = grad / max(norm(grad),1);

    policy_old = policy;
    policy = policy.update(policy.theta + grad*lrate);
    
    if isa(mdp,'LQREnv')
        J = mdp.avg_return(policy.A,0);
    else
        J = evaluate_policies(mdp, episodes_eval, steps_eval, policy);
    end
        
    norm_g1 = norm(grad1);
    
    J_history(iter) = J;
    theta_history(:,iter) = policy.theta;
    omega_history(:,iter) = omega;
    df_dtheta_history(:,iter) = grad1;
    H_history(iter) = policy.entropy([ds.s]);
    KL_history(iter) = kl_mvn2(policy_old, policy, policy.basis([ds.s]));
    
    if verbose
        fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %.2f \tTD: %.2f\n', ...
            iter, S, norm_g1, J, td_true_history(:,iter))
    end
    
    iter = iter + 1;

end

%%
save(['./' folder_save '/spg_td_' num2str(trial) '.mat'], 'H_history', 'KL_history', 'J_history', 'td_history', 'td_true_history', 'omega_history', 'theta_history', 'df_dtheta_history')
