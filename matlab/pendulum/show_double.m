function show_double(trial, do_retrace, reg_type)

clear basis_fourier
rng(trial)

folder = ['data_double/'];
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

mdp = DoubleLink;
tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 16*eye(mdp.daction), size(x,2));
ds = collect_samples(mdp, 100, 100, tmp_policy);
state = [ds.s];
state = [cos(state(1:2:end,:)); sin(state(1:2:end,:)); state(2:2:end,:)];
B = avg_pairwise_dist(state);
bfs_base = @(varargin) basis_fourier(300, mdp.dstate+mdp.dstate/2, B, 0, varargin{:});
bfs = @(varargin)basis_nlink(bfs_base, varargin{:});
policy = GaussianLinearChol(bfs, mdp.daction, zeros(mdp.daction,bfs()+1), eye(mdp.daction));

show_simulation(mdp, policy.update(t).makeDeterministic, 1000, 0.01)
