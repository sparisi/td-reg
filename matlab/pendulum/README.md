#### Summary
* `run_single` runs the single-pendulum task
* `runD_single` runs the single-pendulum task with double-critic
* same for `run_double` and `runD_double`
* Arguments: trial seed, flag to enable/disable Retrace, flag for the regularizers (-1 NO-REG, 0 TD-REG, 1 GAE-REG)
* `B_single` and `B_double` are the matrices of Fourier feature bandwidths, initialized only once

#### How to see the policy running on the environment
* clear all
* load data
* fix the seed (or Fourier basis will have different phase and shift)
* init the MDP and the policy as in the run script
* run `show_simulation`

```
clear all
trial = 2;
rng(trial)
load(['data_double/Ra_' num2str(trial) '.mat'],'t')
mdp = DoubleLink();
load B_double
bfs_base = @(varargin) basis_fourier(300, mdp.dstate+mdp.dstate/2, B, 0, varargin{:});
bfs = @(varargin)basis_nlink(bfs_base, varargin{:});
policy = GaussianLinearChol(bfs, mdp.daction, zeros(mdp.daction,bfs()+1), eye(mdp.daction));
show_simulation(mdp, policy.update(t).makeDeterministic, 1000, 0.01)
```