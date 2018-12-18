import gym
import numpy as np


def rollout(env, policy, max_trans_per_ep=np.inf, render=False):
    '''
    Generates transitions until the episode ends.
    '''
    obs = env.reset()
    iobs = obs
    done = False
    trans = 0
    while not done:
        if render:
            env.render()
        act = policy(obs)
        nobs, rwd, done, _ = env.step(np.minimum(np.maximum(act, env.action_space.low), env.action_space.high))
        trans += 1
        if trans >= max_trans_per_ep: # Override environment max steps if we want to run the episode for LESS steps than the default horizon
            done = True
        yield obs, nobs, act, rwd, done, iobs
        obs = nobs


def collect_samples(env, policy, min_trans, max_trans_per_ep=np.inf, render=False):
    '''
    Keeps calling rollout and saving the resulting path until at least min_trans transitions are collected.
    Returns the following data (everything is a 2D array):
    - obs      : observation of current state
    - nobs     : observation of next state
    - act      : action at the current state
    - rwd      : reward at the current (state,action)
    - done     : if the current state is terminal
    - iobs     : observation of the initial state
    - nb_paths : number of paths collected
    - nb_steps : number of transitions for each path
    '''
    keys = ["obs", "nobs", "act", "rwd", "done", "iobs"]  # must match order of the yield above
    paths = {}
    for k in keys:
        paths[k] = []
    nb_paths = 0
    paths["nb_steps"] = []
    while len(paths["rwd"]) < min_trans:
        nb_steps = 0
        for trans_vect in rollout(env, policy, max_trans_per_ep, render):
            for key, val in zip(keys, trans_vect):
                if (key == "iobs" and nb_steps == 0) or (key != "iobs"): # initial obs (iobs) are stored only once
                    paths[key].append(val)
            nb_steps += 1
        nb_paths += 1
        paths["nb_steps"].append(nb_steps)
    for key in keys:
        data_array = np.asarray(paths[key])
        if data_array.ndim == 1:  # ensure all data is 2D
            data_array = data_array[:,None]
        paths[key] = data_array
    paths["nb_paths"] = nb_paths
    return paths


def evaluate_policy(env, policy, min_paths, render=False):
    '''
    Runs the desired number of rollouts (aka, episodes or paths) to estimate the average return of a policy.
    '''
    tot_rwd = 0.
    nb_paths = 0
    while nb_paths < min_paths:
        for trans_vect in rollout(env, policy, render=render):
            tot_rwd += trans_vect[3]
        nb_paths += 1
    return tot_rwd / nb_paths


def merge_paths(paths_list):
    '''
    Merges a list of paths (as returned by collect_samples) in a bigger path.
    '''
    paths = paths_list[0].copy()
    for p in paths_list[1:]:
        for key, value in p.items():
            if key == "nb_paths":
                paths[key] += value
            else:
                paths[key] = np.append(paths[key], value, axis=0)
    return paths


def minibatch_idx(batch_size, dataset_size):
    '''
    Selects a single mini-batch from a bigger dataset.
    '''
    return np.random.choice(dataset_size, batch_size, replace=False)


def minibatch_idx_list(batch_size, dataset_size):
    '''
    Splits the whole dataset into many mini-batches.
    '''
    batch_idx_list = np.random.choice(dataset_size, dataset_size, replace=False)
    for batch_start in range(0, dataset_size, batch_size):
        yield batch_idx_list[batch_start:min(batch_start + batch_size, dataset_size)]



class RunningStats(object):
    # https://github.com/openai/baselines/blob/d34049cab46908614c46aba1a201bf772daffeb0/baselines/common/running_mean_std.py
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.count = new_count
        self.mean = new_mean
        self.var = new_var


def standardize(x, running_stats=None):
    if running_stats is None:
        m = np.mean(x,axis=0)
        s = np.std(x,axis=0)
    else:
        m = running_stats.mean
        s = np.sqrt(running_stats.var)
    s[s==0] = 1.
    x = (x - m) / s
    return x
