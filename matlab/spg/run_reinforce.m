function run_reinforce(trial, folder_save, varargin)

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
    S = policy.entropy([ds.s]);
    
    Q = mc_ret(ds, mdp.gamma);

    dlogpi1 = policy.dlogPidtheta([ds.s],[ds.a]);
    grad1 = mean(bsxfun(@times,dlogpi1,Q),2);

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
    df_dtheta_history(:,iter) = grad1;
    H_history(iter) = policy.entropy([ds.s]);
    KL_history(iter) = kl_mvn2(policy_old, policy, policy.basis([ds.s]));
    
    if verbose
        fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %.2f \n', ...
            iter, S, norm_g1, J)
    end
    
    iter = iter + 1;

end

%%
save(['./' folder_save '/reinf_' num2str(trial) '.mat'], 'H_history', 'KL_history', 'J_history', 'df_dtheta_history', 'theta_history')
