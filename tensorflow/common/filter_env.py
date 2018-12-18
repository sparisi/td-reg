import numpy as np
import gym

def make_filtered_env(env):
    '''
    Creates a new environment class with actions and states normalized to [-1,1].
    '''

    act_space = env.action_space
    obs_space = env.observation_space
    if not type(act_space)==gym.spaces.box.Box:
        raise RuntimeError('Environment with continous action space (i.e. Box) required.')
    if not type(obs_space)==gym.spaces.box.Box:
        raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

    env_type = type(env)

    class FilteredEnv(env_type):
        def __init__(self):
            self.__dict__.update(env.__dict__) # Transfer properties

            # Observation space
            if np.any(obs_space.high < 1e10):
                h = obs_space.high
                l = obs_space.low
                self.o_c = (h+l) / 2.
                self.o_sc = (h-l) / 2.
            else:
                self.o_c = np.zeros_like(obs_space.high)
                self.o_sc = np.ones_like(obs_space.high)

            # Action space
            h = act_space.high
            l = act_space.low
            self.a_c = (h+l) / 2.
            self.a_sc = (h-l) / 2.

            # Rewards
            self.r_sc = 1.
            self.r_c = 0.

            # Check and assign transformed spaces
            self.observation_space = gym.spaces.Box(self.filter_observation(obs_space.low), self.filter_observation(obs_space.high))
            self.action_space = gym.spaces.Box(-np.ones_like(act_space.high),np.ones_like(act_space.high))
            def assertEqual(a,b): assert np.all(a == b), "{} != {}".format(a,b)
            assertEqual(self.filter_action(self.action_space.low), act_space.low)
            assertEqual(self.filter_action(self.action_space.high), act_space.high)

        def filter_observation(self, obs):
            return (obs - self.o_c) / self.o_sc

        def filter_action(self, action):
            return self.a_sc * action + self.a_c

        def filter_reward(self, reward):
            # Has to be applied manually otherwise it makes reward_threshold invalid
            return self.r_sc * reward + self.r_c

        def step(self, action):
            ac_f = np.clip(self.filter_action(action), self.action_space.low, self.action_space.high)
            obs, reward, term, info = env_type.step(self,ac_f) # Super function
            obs_f = self.filter_observation(obs)
            return obs_f, reward, term, info

    filtered_env = FilteredEnv()

    print()
    print('True action space    : ' + str(act_space.low) + ', ' + str(act_space.high))
    print('Filtered action space: ' + str(filtered_env.action_space.low) + ', ' + str(filtered_env.action_space.high))
    print('True state space     : ' + str(obs_space.low) + ', ' + str(obs_space.high))
    print('Filtered state space : ' + str(filtered_env.observation_space.low) + ', ' + str(filtered_env.observation_space.high))
    print()

    return filtered_env
