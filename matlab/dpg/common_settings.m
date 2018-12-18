function [gamma, ...
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
    lambda_decay] = common_settings(trial, varargin)

rng(trial)

options = {'lambda', 'lambda_decay', 'basis_degree', 'tau_omega', 'tau_theta'};
defaults = {0.1, 0.999, 2, 1, 0.01};
[lambda, lambda_decay, basis_degree, tau_omega, tau_theta] = internal.stats.parseArgs(options, defaults, varargin{:});

dim = 2; % dimensionality of the LQR
gamma = 0.99;
mdp = LQR(dim);
mdp.gamma = gamma;
mdp.noisy_trans = true;

basis_pi = @(s)s;
dim_theta = mdp.daction*mdp.dstate;

if basis_degree == 2
    basis_q = @(s,a)squared_bfs(s,a);
    basis_q_da = @(s,a)squared_bfs_da(s,a);
    theta = -reshape(rand(dim_theta, 1), [mdp.daction, mdp.dstate]);
elseif basis_degree == 3
    basis_q = @(s,a)cubic_bfs(s,a);
    basis_q_da = @(s,a)cubic_bfs_da(s,a);
    theta = -unifrnd(0.1,0.5,[mdp.daction, mdp.dstate]);
else
    error('Unknown basis functions.')
end

theta = -theta'*theta;
theta_t = theta; % target policy params

dim_omega = length(basis_q(zeros(mdp.dstate,1),zeros(mdp.daction,1)));
omega = (rand(dim_omega, 1)-0.5)*2;
omega_t = omega; % target Q-function params

noise_pi = 5; % exploration noise
noise_decay = 0.95;

data.s = nan(mdp.dstate, 0);
data.sn = nan(mdp.dstate, 0);
data.a = nan(mdp.daction, 0);
data.r = nan(mdp.dreward, 0);
data.q = nan(mdp.dreward, 0);
data.done = nan(1, 0);
data.bfs_s = nan(mdp.dstate,0);
data.bfs_sn = nan(mdp.dstate,0);
data.bfs_s_a = nan(dim_omega,0);

maxsteps = 150;     % max steps per episode
minsteps = 1e2;     % warmup time
maxdata = 1e6;      % max data stored
stepslearn = 12000-minsteps; % steps of learning (after warmup)
eval_every = 100;   % evaluate every X learning steps
bsize = 32;         % minibatch size

optimQ = ADAM(length(omega(:)));
optimPi = ADAM(length(theta(:)));
optimQ.alpha = 1e-2;
optimPi.alpha = 5e-4;
