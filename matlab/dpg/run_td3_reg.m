function run_td3_reg(trial, folder_save, varargin)
% https://arxiv.org/pdf/1802.09477.pdf + TD-REG
% TD-REG uses the loss of both critics: 0.5*TD_ERR_1^2 + 0.5*TD_ERR_2^2

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
    omega1, ...
    omega1_t, ...
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
use_min = 1; % max_pi E[min(Q1,Q2)] (SAC style) VS max_pi E[Q1] (TD3 style)

% Second critic
omega2 = (rand(size(omega1))-0.5)*2;
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
l2_diff_history = [];

try
    J_history(end+1) = mdp.avg_return(theta,0);
catch % If the system is unstable, the avg return is -inf
    policy.drawAction = @(varargin) theta * basis_pi(varargin{:});
    J_history(end+1) = evaluate_policies(mdp, 100, maxsteps, policy);
end
theta_history(:,end+1) = theta(:);
omega_history(:,:,end+1) = [omega1 omega2];

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
        state = nextstate;

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
            bfs_d_s_api = basis_q_da(s,a_pi);
            bfs_d_sn_anpi = basis_q_da(sn,an_pi);
            
            q1 = omega1' * bfs_s_a;
            q2 = omega2' * bfs_s_a;
            q1_t = omega1_t' * bfs_sn_anpi;
            q2_t = omega2_t' * bfs_sn_anpi;
            q_t = min(q1_t, q2_t); % Use target Q-function
            
            td_err1 = q1 - (r + gamma .* q_t .* ~d);
            td_err2 = q2 - (r + gamma .* q_t .* ~d);
            
            dg_omega1 = bfs_s_a * td_err1' / bsize;
            dg_omega2 = bfs_s_a * td_err2' / bsize;

            df_theta_1 = mtimescolumn(bfs_s, permute(sum(bsxfun(@times,bfs_d_s_api,omega1), 1), [2 3 1]));
            df_theta_2 = mtimescolumn(bfs_s, permute(sum(bsxfun(@times,bfs_d_s_api,omega2), 1), [2 3 1]));
            i_min = q1 < q2;
            if use_min
                df_theta = reshape(mean(bsxfun(@times, df_theta_1, i_min) + bsxfun(@times, df_theta_2, ~i_min), 2), [mdp.daction,mdp.dstate])';
            else
                df_theta = reshape(mean(df_theta_1, 2), [mdp.daction,mdp.dstate])';
            end
            
            it_min = q1_t < q2_t;
            Q1_da = permute(sum(bsxfun(@times,bfs_d_sn_anpi,omega1_t),1),[2 3 1]);
            Q2_da = permute(sum(bsxfun(@times,bfs_d_sn_anpi,omega2_t),1),[2 3 1]);
            Q_da = bsxfun(@times, Q1_da, it_min) + bsxfun(@times, Q2_da, ~it_min);
            dg_theta = reshape(mean( ...
                mtimescolumn(gamma * bsxfun (@times, 0.5*td_err1 + 0.5*td_err2, bfs_sn), Q_da), ...
                2), [mdp.daction,mdp.dstate])';
            
%             df_theta_history(:,end+1) = df_theta(:);
%             dg_omega_history(:,:,end+1) = [dg_omega1 dg_omega2];
%             dg_theta_history(:,end+1) = dg_theta(:);
            
            omega1 = optimQ.step(omega1', dg_omega1')';
            omega2 = optimQ.step(omega2', dg_omega2')';

            lambda = lambda*lambda_decay;
            if mod(totsteps, policy_update_every) == 0
                % By default, ADAM solves a minimization problem, that's why we change the sign
                theta_old = theta;
                theta = reshape(optimPi.step(theta(:)', -df_theta(:)' -lambda*dg_theta(:)'), size(theta));
                l2_diff_history(end+1) = norm(theta-theta_old);

                omega1_t = tau_omega * omega1 + (1-tau_omega) * omega1_t;
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

            an_pi = theta * data.bfs_sn; % No target policy
            q1 = omega1' * basis_q(data.s, data.a);
            q2 = omega2' * basis_q(data.s, data.a);
            q1_t = omega1_t' * basis_q(data.sn, an_pi); % Use target Q-function
            q2_t = omega2_t' * basis_q(data.sn, an_pi);
            td_err1 = q1 - (data.r + gamma .* min(q1_t, q2_t) .* ~data.done);
            td_err2 = q2 - (data.r + gamma .* min(q1_t, q2_t) .* ~data.done);
            try
                td_err_true = [q1; q2] - mdp.q_function(theta,0,data.s,data.a);
            catch
                td_err_true = [td_err1; td_err2]; % If the system is unstable, the TD error is inf, so use the estimated one
            end
            
            J_history(end+1) = J;
            td_history(:,end+1) = mean([td_err1; td_err2].^2,2);
            td_true_history(:,end+1) = mean(td_err_true.^2,2);
            theta_history(:,end+1) = theta(:);
%             omega_history(:,:,end+1) = [omega1 omega2];
        end
        
    end
    
end

if policy_update_every == 1, delay_str = 'nodelay_'; else, delay_str = []; end
save([folder_save 'td3_reg_' delay_str num2str(trial)], 'l2_diff_history', 'J_history', 'td_history', 'td_true_history', 'theta_history', 'omega_history', 'df_theta_history', 'dg_omega_history')

