function show_pendulum(trial, do_retrace, reg_type)

clear basis_fourier
rng(trial)

folder = ['data_single/'];
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
load([folder RETR ALG '_' num2str(trial) '.mat'], 't');

mdp = Pendulum; % or DoubleLink
tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 16*eye(mdp.daction), size(x,2));
ds = collect_samples(mdp, 100, 100, tmp_policy);
state = [ds.s];
B = avg_pairwise_dist(state);
bfs = @(varargin) basis_fourier(100, mdp.dstate, B, 0, varargin{:});
policy = GaussianLinearChol(bfs, mdp.daction, zeros(mdp.daction,bfs()+1), eye(mdp.daction));

show_simulation(mdp, policy.update(t).makeDeterministic, 200, 0.05)
