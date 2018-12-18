function run_td3reg(trial, folder_save, varargin)
% https://arxiv.org/pdf/1802.09477.pdf + TD-REG

if nargin == 0, trial = 1; folder_save = []; end
if nargin == 1, folder_save = []; end

%%
[gamma, ...
    tau_omega, ...
    tau_theta, ...
    dim, ...
    mdp, ...
    basis_pi, ...
    dim_theta, ...
    basis_q, ...
    basis_q_da, ...
    dim_omega, ...
    theta, ...
    theta_t, ...
    omega, ...
    omega_t, ...
    noise_pi, ...
    noise_decay, ...
    maxsteps, ...
    minsteps, ...
    maxdata, ...
    stepslearn, ...
    eval_every, ...
    bsize, ...
    optimQ, ...
    optimPi, ...
    lambda, ...
    lambda_decay] = common_settings(trial, varargin{:});

policy_update_every = 2; % policy update delay

% Second critic
omega2 = (rand(size(omega))-0.5)*2;
omega2_t = omega2;

totsteps = 0;
dataidx = 1;
J_history = [];
td_history = [];
td_true_history = [];
omega_history = [];
theta_history = [];
df_theta_history = [];
dg_omega_history = [];
dg_theta_history = [];

try
    J_history(end+1) = mdp.avg_return(theta,0);
catch % If the system is unstable, the avg return is -inf
    policy.drawAction = @(varargin) theta * basis_pi(varargin{:});
    J_history(end+1) = evaluate_policies(mdp, 100, maxsteps, policy);
end
theta_history(:,end+1) = theta(:);
omega_history(:,end+1) = omega;

%%
while totsteps < minsteps + stepslearn
    step = 1;
    state = mdp.initstate(1);
    done = false;

    if totsteps > minsteps
        noise_pi = noise_pi * noise_decay;
    end
    
    while ( (step < maxsteps) && ~done )
        action = theta * basis_pi(state) + (rand(mdp.daction,1) - 0.5) * 2 * noise_pi;
        [nextstate, reward, done] = mdp.simulator(state, action);

        data.a(:,dataidx) = action;
        data.r(:,dataidx) = reward;
        data.s(:,dataidx) = state;
        data.sn(:,dataidx) = nextstate;
        data.done(:,dataidx) = done;
        data.bfs_s(:,dataidx) = basis_pi(state);
        data.bfs_sn(:,dataidx) = basis_pi(nextstate);
        data.bfs_s_a(:,dataidx) = basis_q(state,action);

        step = step + 1;
        dataidx = dataidx + 1;
        totsteps = totsteps + 1;
        if dataidx > maxdata
            dataidx = 1;
        end
        
        %% Train
        if totsteps > minsteps
            mb = randperm(length(data.r),bsize);
            
            sn = data.sn(:,mb);
            s = data.s(:,mb);
            a = data.a(:,mb);
            r = data.r(:,mb);
            d = data.done(:,mb);
            bfs_s = data.bfs_s(:,mb);
            bfs_sn = data.bfs_sn(:,mb);
            a_pi = theta * bfs_s;
            an_pi = theta * bfs_sn + min(max((rand(mdp.daction,1)-0.5)*2*2, -noise_pi/2), noise_pi/2); % No target policy and add clipped noise
            bfs_s_a = data.bfs_s_a(:,mb);
            bfs_sn_anpi = basis_q(sn, an_pi);
            
            q = omega' * bfs_s_a;
            q2 = omega2' * bfs_s_a;
            q_t = min(omega_t' * bfs_sn_anpi, omega2_t' * bfs_sn_anpi); % Use target Q-function
            
            td_err = q - (r + gamma .* q_t .* ~d);
            td_err2 = q2 - (r + gamma .* q_t .* ~d);
            
            df_theta = reshape(mean(mtimescolumn(bfs_s, permute(sum(bsxfun(@times,basis_q_da(s,a_pi),omega),1),[2 3 1])), 2),[mdp.daction,mdp.dstate])';
            dg_omega = bfs_s_a * td_err' / bsize;
            dg_omega2 = bfs_s_a * td_err2' / bsize;
            
            i_min = omega_t' * bfs_sn_anpi < omega2_t' * bfs_sn_anpi;
            bfs_q_da = basis_q_da(sn,an_pi);
            Q1_da = permute(sum(bsxfun(@times,bfs_q_da,omega_t),1),[2 3 1]);
            Q2_da = permute(sum(bsxfun(@times,bfs_q_da,omega2_t),1),[2 3 1]);
            Q_da = bsxfun(@times, Q1_da, i_min) + bsxfun(@times, Q2_da, ~i_min);
            dg_theta = reshape(mean(mtimescolumn(gamma * (td_err .* bfs_sn), ...
                Q_da),2),[mdp.daction,mdp.dstate])';
            
            df_theta_history(:,end+1) = df_theta(:);
            dg_omega_history(:,end+1) = dg_omega;
            dg_theta_history(:,end+1) = dg_theta(:);
            
            omega = optimQ.step(omega', dg_omega')';
            omega2 = optimQ.step(omega2', dg_omega2')';

            lambda = lambda*lambda_decay;
            if mod(totsteps, policy_update_every) == 0
                % By default, ADAM solves a minimization problem, that's why we change the sign
                theta = reshape(optimPi.step(theta(:)', -df_theta(:)' -lambda*dg_theta(:)'), size(theta));
%                 theta = max(min(theta,zeros(size(theta))),-ones(size(theta)));
                omega_t = tau_omega * omega + (1-tau_omega) * omega_t;
                omega2_t = tau_omega * omega2 + (1-tau_omega) * omega2_t;
                
            end
        end

        %% Evaluate policy
        if mod(totsteps, eval_every) == 0 && totsteps > minsteps
            try
            J = mdp.avg_return(theta,0);
            catch
            policy.drawAction = @(varargin) theta * basis_pi(varargin{:});
            J = evaluate_policies(mdp, 100, maxsteps, policy);
            end

            an_pi = theta * data.bfs_sn; % Use target policy
            q = omega' * basis_q(data.s, data.a);
            q_t = omega_t' * basis_q(data.sn, an_pi); % Use target Q-function
            td_err = q - (data.r + gamma .* q_t .* ~data.done);
            try
                td_err_true = q - mdp.q_function(theta,0,data.s,data.a);
            catch
                td_err_true = td_err; % If the system is unstable, the TD error is inf, so use the estimated one
            end
            
            J_history(end+1) = J;
            td_history(end+1) = mean(td_err.^2);
            td_true_history(end+1) = mean(td_err_true.^2);
            theta_history(:,end+1) = theta(:);
            omega_history(:,end+1) = omega;
        end
        
    end
    
end

if policy_update_every == 1, delay_str = 'nodelay_'; else, delay_str = []; end
save([folder_save 'td3reg_' delay_str num2str(trial)], 'J_history', 'td_history', 'td_true_history', 'theta_history', 'omega_history', 'df_theta_history', 'dg_omega_history')

