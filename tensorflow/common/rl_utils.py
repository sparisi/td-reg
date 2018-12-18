import numpy as np


def mc_ret(paths, gamma):
    '''
    Computes Monte-Carlo estimates of the return from on-policy data.
    R_t = sum_(h=t)^T gamma^(h-t)*r_h
    '''
    r_values = np.empty_like(paths["rwd"], dtype=np.float32) # specify dtype in case the reward is an integer
    r_next = 0.
    for k in reversed(range(len(paths["rwd"]))):
        r_values[k] = paths["rwd"][k] + gamma * r_next * (1.-paths["done"][k])
        r_next = r_values[k]
    return r_values


def gae(paths, v_values, gamma, lambda_trace, prob_ratio=[]):
    '''
    Computes generalized advantage estimates from potentially off-policy data.
    Data have to be ordered by episode: paths[rwd] must have first all samples
    from the first episode, then all samples from the second, and so on.

    Do not pass PROB_RATIO if data is on-policy.
    Truncate PROB_RATIO = min(1,PROB_RATIO) to use Retrace.

    Computes generalized advantage estimates from on-policy data.
    https://arxiv.org/abs/1506.02438
    https://arxiv.org/abs/1606.02647
    '''

    if len(prob_ratio) == 0:
        prob_ratio = np.ones(v_values.shape)
    a_values = np.empty_like(v_values)
    for k in reversed(range(len(v_values))):
        if paths["done"][k]:
            a_values[k] = prob_ratio[k] * (paths["rwd"][k] - v_values[k])
        else:
            a_values[k] = prob_ratio[k] * (paths["rwd"][k] + gamma * v_values[k + 1] - v_values[k] + gamma * lambda_trace * a_values[k + 1])
    return a_values
