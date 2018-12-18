function [policy, ...
    mdp, ...
    ep_eval, ...
    steps_eval, ...
    ep_learn, ...
    steps_learn, ...
    basisQ, ...
    options, ...
    maxiter, ...
    omega, ...
    lambda, ...
    lambda_decay, ...
    verbose, ...
    lrate] = common_settings(trial, varargin)

options = {'lambda', 'lambda_decay', 'basis_degree', 'noisy', 'ep_learn'};
defaults = {0.1, 0.999, 3, false, 1};
[lambda, lambda_decay, basis_degree, noisy, ep_learn] = internal.stats.parseArgs(options, defaults, varargin{:});

verbose = 0;
lrate = 0.01;

if nargin == 0
    trial = 1;
end

rng(trial)

dim = 2;
mdp = LQR(dim);
mdp.gamma = 0.99;
mdp.noisy_trans = true;

if noisy
    basisPi = @(varargin)basis_poly_noise(1,dim,0,varargin{:});
else
    basisPi = @(varargin)basis_poly(1,dim,0,varargin{:});
end

Sigma0 = 5*eye(dim);
A0 = -diag(rand(dim,1))*0.1;
A0 = -unifrnd(0.1,0.5,[mdp.daction, mdp.dstate]);
A0 = -A0'*A0;
policy = GaussianLinearDiag(basisPi, dim, A0, Sigma0, true);
% ep_learn = 1;
ep_eval = 100;
steps_learn = 150;
steps_eval = 150;

if noisy
    basisQ = @(varargin) basis_poly_noise(basis_degree, mdp.dstate+mdp.daction, 1, varargin{:});
else
    basisQ = @(varargin) basis_poly(basis_degree, mdp.dstate+mdp.daction, 1, varargin{:});
end

omega = rand(basisQ(),1);
maxiter = 200;

options = optimoptions('fminunc', ...
    'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'Hessian', 'on', ...
    'MaxFunEvals', 50, ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 50);
