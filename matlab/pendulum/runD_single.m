function runD_single(trial, do_retrace, reg_type)

rng(trial)

mdp = Pendulum;

% tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 16*eye(mdp.daction), size(x,2));
% ds = collect_samples(mdp, 100, 100, tmp_policy);
% B = avg_pairwise_dist([ds.s]);
load B_single
bfs = @(varargin) basis_fourier(100, mdp.dstate, B, 0, varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 16*eye(mdp.daction);
policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);

episodes_eval = 1000;
episodes_learn = 10;
steps_eval = 150;
steps_learn = 50;
maxiter = 1000;


folder = ['data_single/'];
mkdir(folder)
if do_retrace
    RETR = 'R';
else
    RETR = [];
end
if reg_type == 1
    ALG = 'a';
elseif reg_type == 0
    ALG = 't';
elseif reg_type == -1
    ALG = 'v';
end


% To learn V
options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'Hessian', 'on', ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

mdp.gamma = 0.99;
kl_bound = 0.01;
lambda_trace = 0.95;

bfsV = bfs;
omega1 = (rand(bfsV(),1)-0.5)*2;
omega2 = (rand(bfsV(),1)-0.5)*2;

data = [];
varnames = {'r','s','nexts','a','t','terminal','logprob'};
bfsnames = { {'phiV', bfsV} };
iter = 1;

max_reuse = 5; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter <= maxiter
    
    % Collect data
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).logprob = policy.logpdf(ds(i).a, ds(i).s);
    end
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    prob_ratio = exp(policy.logpdf(data.a, data.s) - data.logprob);
    if do_retrace
        prob_ratio = min(1,prob_ratio);
    end
    
    % Train V
    V1 = omega1'*data.phiV;
    V2 = omega2'*data.phiV;
    A1 = gae(data,V1,mdp.gamma,lambda_trace,prob_ratio);
    A2 = gae(data,V2,mdp.gamma,lambda_trace,prob_ratio);
    if rand() < 0.5
        omega1 = fminunc(@(omega)mse_linear(omega,data.phiV,V2+A2), omega1, options);
        V = omega1'*data.phiV;
    else
        omega2 = fminunc(@(omega)mse_linear(omega,data.phiV,V1+A1), omega2, options);
        V = omega2'*data.phiV;
    end
    
    % Estimate A and TD
    A = gae(data,V,mdp.gamma,lambda_trace,prob_ratio);
    TD = gae(data,V,mdp.gamma,0,prob_ratio);
    td_history(iter) = mean(TD.^2);
    
    % Estimate natural gradient
    dlogpi = policy.dlogPidtheta(data.s,data.a);
    A = (A-mean(A))/std(A);
    TD = (TD-mean(TD))/std(TD);
    if reg_type == 1
        REG = A.^2;
        REG = (REG-mean(REG))/std(REG);
        l_base = 1;
    elseif reg_type == 0
        REG = TD.^2;
        REG = (REG-mean(REG))/std(REG);
        l_base = 1;
    elseif reg_type == -1
        REG = 0;
        l_base = 0;
    end
    X = A - l_base*0.999^iter*REG;
    grad = mean(bsxfun(@times,dlogpi,X),2);
    F = dlogpi * dlogpi' / length(A);
    [grad_nat,~,~,~,~] = pcg(F,grad,1e-10,50); % Use conjugate gradient (~ are to avoid messages)
    
    % Line search
    stepsize = sqrt(kl_bound / (0.5*grad'*grad_nat));
    max_adv = @(theta) mean(policy.update(theta).logpdf([data.a],[data.s]).*X);
    kl = @(theta) kl_mvn2(policy.update(theta), policy, policy.basis(data.s));
    [success, theta, n_back] = linesearch(max_adv, policy.theta, stepsize*grad_nat, grad'*grad_nat*stepsize, kl, kl_bound);
    if ~success, warning('Could not satisfy the KL constraint.'), end % in this case, theta = policy.theta
    
    % Print info
    norm_g1 = norm(grad);
    norm_g2 = norm(grad);
    norm_ng = norm(grad_nat);
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
%     fprintf('%d) Entropy: %.2f,   Norm (G1): %e,   Norm (G2): %e,   Norm (NG): %e,   J: %e \n', ...
%         iter, entropy, norm_g1, norm_g2, norm_ng, J);
    J_history(iter) = J;
    e_history(iter) = entropy;
    
    % Update pi
    policy = policy.update(theta);
    
    iter = iter + 1;
    
end

t = policy.theta;
save([folder RETR 'D' ALG '_' num2str(trial) '.mat'], 't', 'J_history', 'e_history', 'td_history');
