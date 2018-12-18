from args_parser import arg_parser
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy, DiscretePolicy
from models.mlp_critic import Value
from torch.autograd import Variable

from core.trpo import *

from core.agent import Agent

def learn_model(args):

    print("RL result will be saved at %s" % args.rl_filename)
    print("RL model will be saved at %s" % args.rl_model_filename)
    if use_gpu:
        print("Using CUDA.")

    torch.manual_seed(args.rl_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.rl_seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(args.rl_seed)
    random.seed(args.rl_seed)

    env = gym.make(args.env_name)
    env.seed(args.rl_seed)   

    env_test = gym.make(args.env_name)
    env_test.seed(args.rl_seed)  
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    a_bound = np.asscalar(env.action_space.high[0])
    a_low = np.asscalar(env.action_space.low[0])
    assert a_bound == -a_low 

    ## Binary flag for manually cliping actions for step function after adding Gaussian noise. 
    clip = (args.env_name == "LunarLanderContinuous-v2" or args.env_name == "BipedalWalker-v2")

    print(env.observation_space)
    print(env.action_space)    

    """define actor and critic"""
    policy_net = Policy(state_dim, action_dim, log_std=args.log_std, a_bound=a_bound, hidden_size=args.hidden_size, activation=args.activation).to(device)
    value_net = Value(state_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)

    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate_v)
    decayed_lambda_td = args.lambda_td

    def update_params_c(batch, i_iter):
        states = torch.from_numpy(np.stack(batch.state)).float().to(device)
        actions = torch.from_numpy(np.stack(batch.action)).float().to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).float().to(device)
        masks = torch.from_numpy(np.stack(batch.mask).astype(np.float32)).to(device)
        
        """get advantage estimation from the trajectories"""
        values = value_net(states).data
        advantages, lambda_returns, mc_returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau)
        
        if args.lamret:
            returns = lambda_returns
        else:
            returns = mc_returns

        """perform critic update"""
        #gae_step(value_net, optimizer_value, states, lambda_returns, args.l2_reg)  # full batch GD
        gae_step_epoch(value_net, optimizer_value, states, returns, args.l2_reg)  # Stochastic GD

    """ Function to update the parameters of value and policy networks"""
    def update_params_p(batch, i_iter):

        nonlocal decayed_lambda_td
        
        states = torch.from_numpy(np.stack(batch.state)).float().to(device)
        actions = torch.from_numpy(np.stack(batch.action)).float().to(device)
        next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).float().to(device)
        masks = torch.from_numpy(np.stack(batch.mask).astype(np.float32)).to(device)

        """get advantage estimation from the trajectories, this is done after gae_step update"""
        values = value_net(states).data
        advantages, lambda_returns, mc_returns = estimate_advantages(rewards, masks, values, gamma=args.gamma, tau=args.tau)

        if args.method_name == "TRPO-RET-MC":
            returns = mc_returns.detach()       # detach() does not matter since we back prop policy network only.
        elif args.method_name == "TRPO-RET-GAE":
            returns = lambda_returns.detach()   # detach() does not matter actually.
        else:
            returns = 0   # returns is not used for TRPO and TRPO-TD.

        # standardize or not ?   
        if args.mgae:     
            advantages = (advantages - advantages.mean()) / advantages.std() # this will be m-std version
        else:
            advantages = advantages / advantages.std()  # this will be std version

        trpo_step_td(policy_net=policy_net, value_net=value_net, states=states, actions=actions, next_states=next_states, rewards=rewards, masks=masks, gamma=args.gamma, advantages=advantages, \
            max_kl=args.max_kl, damping=args.damping, \
            lambda_td=decayed_lambda_td, method_name=args.method_name, returns=returns, mtd=args.mtd)

        """ decay the td_reg parameter after update """
        decayed_lambda_td = decayed_lambda_td * args.decay_td   

    """create agent"""
    agent = Agent(env, policy_net, render=False)
    agent_test = Agent(env_test, policy_net, mean_action=True, render=args.render)

    """ The actual learning loop"""
    for i_iter in range(args.rl_max_iter_num):

        """ Save the learned policy model """
        if ( (i_iter) % args.rl_save_model_interval == 0 and args.rl_save_model_interval > 0 ) \
            or (i_iter == args.rl_max_iter_num + 1) or i_iter == 0:

            policy_net = policy_net.to(device_cpu)
            value_net = value_net.to(device_cpu)

            pickle.dump((policy_net, value_net), open(args.rl_model_filename + ("_I%d.p" % (i_iter)), 'wb'))

            policy_net = policy_net.to(device)
            value_net = value_net.to(device)

        """ Test the policy before update """
        if i_iter % args.log_interval == 0 or i_iter+1 == args.rl_max_iter_num :
            _, log_test = agent_test.collect_samples_test(max_num_episodes=20, render=args.render, clip=clip)

        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        batch, log = agent.collect_samples_train(args.min_batch_size, render=False, clip=clip) # this is on-policy samples
        t1 = time.time()
        
        """ update parameters """
        t0_d = time.time()
        update_params_c(batch, i_iter)  #critic update
        update_params_p(batch, i_iter)  #actor update
        t1_d = time.time()

        """ Print out result to stdout and save it to a text file for later usage"""
        if i_iter % args.log_interval == 0:
           
            result_text = t_format("Iter %6d (%2.2fs)+(%2.2fs)" % (i_iter, t1-t0, t1_d-t0_d)) 
            result_text += " | [R] " + t_format("Avg: %.2f (%.2f)" % (log['avg_reward'], log['std_reward']), 2)
            result_text += " | [R_test] " + t_format("min: %.2f" % log_test['min_reward'], 1) + t_format("max: %.2f" % log_test['max_reward'], 1) \
                            + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
            print(result_text)

            with open(args.rl_filename, 'a') as f:
                print(result_text, file=f) 

if __name__ == "__main__":
    args = arg_parser()
    learn_model(args)
