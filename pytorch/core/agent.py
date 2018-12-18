import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time


def collect_samples_train(env, policy, custom_reward, mean_action,
                    render, min_batch_size, clip=False):
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6

    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6

    num_episodes = 0
    t_max = 1000

    total_c_reward_2 = 0
    min_c_reward_2 = 1e6
    max_c_reward_2 = -1e6

    if not policy.is_disc_action:
        a_space_high = env.action_space.high[0]
        a_space_low = env.action_space.low[0]

    total_reward_list = []
    total_c_reward_list = []

    total_reward_list_2 = []
    total_c_reward_list_2 = []

    while num_steps < min_batch_size:
        reward_episode = 0
        c_reward_episode = 0


        reward_episode_2 = 0
        c_reward_episode_2 = 0

        state = env.reset()
        for t in range(t_max):

            state_var = torch.FloatTensor(state).unsqueeze(0)
            action = policy.select_action(state_var)

            action = action.numpy()

            if clip:
                next_state, reward, done, _ = env.step(np.clip(action, a_min=a_space_low, a_max=a_space_high) )
            else:
                next_state, reward, done, _ = env.step(action)
                
            reward_episode += reward

            if custom_reward is not None:
                reward = custom_reward(state, action)

                if isinstance(reward, tuple):
                    reward_1 = reward[0]
                    reward_2 = reward[1]
                
                    c_reward_episode += reward_1
                    min_c_reward = min(min_c_reward, reward_1)
                    max_c_reward = max(max_c_reward, reward_1)

                    c_reward_episode_2 += reward_2
                    min_c_reward_2 = min(min_c_reward_2, reward_2)
                    max_c_reward_2 = max(max_c_reward_2, reward_2)
                else:
                    c_reward_episode += reward
                    min_c_reward = min(min_c_reward, reward)
                    max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
                time.sleep(0.001)
            if done:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        total_reward_list += [reward_episode]

        total_c_reward += c_reward_episode 
        total_c_reward_list += [c_reward_episode]

        if isinstance(reward, tuple):
            total_c_reward_2 += c_reward_episode_2 
            total_c_reward_list_2 += [c_reward_episode_2]


        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = np.mean(np.array(total_reward_list))   
    log['std_reward'] = np.std(np.array(total_reward_list)) / np.sqrt(num_episodes)  #TODO: add std_reward to merge_log
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward

    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = np.mean(np.array(total_c_reward_list))
        log['std_c_reward'] = np.std(np.array(total_c_reward_list)) / np.sqrt(num_episodes)  #TODO: add std_reward to merge_log
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward
            
        if isinstance(reward, tuple):
            log['total_c_reward_2'] = total_c_reward_2
            log['avg_c_reward_2'] = np.mean(np.array(total_c_reward_list_2))
            log['std_c_reward_2'] = np.std(np.array(total_c_reward_list_2)) / np.sqrt(num_episodes)  #TODO: add std_reward to merge_log
            log['max_c_reward_2'] = max_c_reward_2
            log['min_c_reward_2'] = min_c_reward_2

    return memory, log

def collect_samples_test(env, policy, mean_action,
                    render, max_num_episodes, clip=False):
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    num_episodes = 0
    t_max = 1000

    if not policy.is_disc_action:
        a_space_high = env.action_space.high[0]
        a_space_low = env.action_space.low[0]

    total_reward_list = []

    while num_episodes < max_num_episodes:
        reward_episode = 0

        state = env.reset()

        for t in range(t_max):
            
            state_var = torch.FloatTensor(state).unsqueeze(0)

            if mean_action:
                action = policy.select_greedy_action(state_var).numpy()
            else:
                action = policy.select_action(state_var).numpy()
                
            if clip:
                next_state, reward, done, _ = env.step(np.clip(action, a_min=a_space_low, a_max=a_space_high) )
            else:
                next_state, reward, done, _ = env.step(action)
                
            reward_episode += reward

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
                time.sleep(0.001)
            if done:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        total_reward_list += [reward_episode]
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = np.mean(np.array(total_reward_list))   
    log['std_reward'] = np.std(np.array(total_reward_list)) / np.sqrt(num_episodes)  #TODO: add std_reward to merge_log
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    
    return memory, log

class Agent:

    def __init__(self, env, policy, custom_reward=None, mean_action=False, 
                 num_threads=1, render=0):
        self.policy = policy
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        
        self.env = env

    def collect_samples_train(self, min_batch_size, render=False, clip=False, return_memory=False):

        # Collect samples in CPU is faster than in GPU.
        self.policy = self.policy.to(device_cpu)
        memory, log = collect_samples_train(self.env, self.policy, self.custom_reward, self.mean_action,
                                      render, min_batch_size=min_batch_size, clip=clip)
        self.policy = self.policy.to(device)

        batch = memory.sample()

        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        if return_memory:
            return batch, log, memory
        else:
            return batch, log

    def collect_samples_test(self, max_num_episodes, render=False, clip=False, return_memory=False):

        self.policy = self.policy.to(device_cpu)
        memory, log = collect_samples_test(self.env, self.policy, self.mean_action, \
                                    render, max_num_episodes=max_num_episodes, clip=clip)
        self.policy = self.policy.to(device)

        batch = memory.sample()

        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        if return_memory:
            return batch, log, memory
        else:
            return batch, log