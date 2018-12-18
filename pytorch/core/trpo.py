import numpy as np
import scipy.optimize
from utils import *
import torch.utils.data as data_utils

""" For TRPO """
def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).data

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).data
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def trpo_step_td(policy_net, value_net, states, actions, next_states, rewards, masks, gamma, advantages, max_kl, damping, \
                lambda_td=0, method_name="TRPO-TD", returns=0, mtd=1):

    if method_name == "TRPO-TD":
        values_pred = value_net(states)
        next_v = value_net(next_states)
        target_v = rewards + gamma * next_v * masks
        td_err2 = (values_pred - target_v).pow(2).detach()  # detach() just to be safe
        if mtd:
            td_err2 = (td_err2 - td_err2.mean()) / td_err2.std()
        else:
            td_err2 = td_err2 / td_err2.std()


    elif method_name == "TRPO-RET-MC" or method_name == "TRPO-RET-GAE":
        values_pred = value_net(states)
        target_v = returns.to(device)
        td_err2 = (values_pred - target_v).pow(2).detach()  # detach() just to be safe
        if mtd:
            td_err2 = (td_err2 - td_err2.mean()) / td_err2.std()
        else:
            td_err2 = td_err2 / td_err2.std()


    """update policy"""
    fixed_log_probs = policy_net.get_log_prob(states, actions).data

    """define the loss function for TRPO"""
    def get_loss(volatile=False):
        log_probs = policy_net.get_log_prob(states, actions)

        if method_name == "TRPO-TD" or method_name == "TRPO-RET-MC" or method_name == "TRPO-RET-GAE":
            action_loss = (-advantages + lambda_td * td_err2) * torch.exp(log_probs - fixed_log_probs)
        elif method_name == "TRPO":   # standard TRPO
            action_loss = (-advantages) * torch.exp(log_probs - fixed_log_probs)

        return action_loss.mean()

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(v):
        M, mu, info = policy_net.get_fim(states)
        mu = mu.view(-1)
        filter_input_ids = set() if policy_net.is_disc_action else set([info['std_id']])

        t = ones(mu.size()).requires_grad_()
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t, retain_graph=True)[0]
        MJv = M * Jv.data
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, policy_net.parameters(), filter_input_ids=filter_input_ids, retain_graph=True).data
        JTMJv /= states.shape[0]
        if not policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    stepdir = conjugate_gradients(Fvp_fim, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp_fim(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
    set_flat_params_to(policy_net, new_params)

    return success

""" GAE parts """
def estimate_advantages(rewards, masks, values, gamma, tau):

    rewards = rewards.to(device_cpu)
    masks = masks.to(device_cpu)
    values = values.to(device_cpu)

    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)
    mc_returns = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    prev_mc_return = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
        mc_returns[i] = rewards[i] + gamma * prev_mc_return * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
        prev_mc_return = mc_returns[i, 0]

    td_lambda_returns = advantages + values

    advantages = advantages.to(device)
    td_lambda_returns = td_lambda_returns.to(device)
    mc_returns = mc_returns.to(device)

    return advantages, td_lambda_returns, mc_returns

def gae_step(value_net, optimizer_value, states, returns, l2_reg=0):

    """ update critic using full batch gradient descent """
    vf_iter = 20
    for _ in range(0, vf_iter):
        value_net.zero_grad()
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()

        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg

        value_loss.backward()
        optimizer_value.step()


def gae_step_epoch(value_net, optimizer_value, states, returns, l2_reg=0, vf_iter=3, mini_batch_size=128):

    """update critic for vs_iter epochs with batch size 128, as done in baselines"""
    train = data_utils.TensorDataset(states, returns)
    train_loader = data_utils.DataLoader(train, batch_size=mini_batch_size, shuffle=True)

    for _ in range(0, vf_iter):
        for batch_idx, (state_batch, returns_batch) in enumerate(train_loader):
            value_net.zero_grad()
            values_pred = value_net(state_batch)
            value_loss = (values_pred - returns_batch).pow(2).mean()

            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * l2_reg
            value_loss.backward()
            optimizer_value.step()

    return
