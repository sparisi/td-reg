import tensorflow as tf
import numpy as np



class RandPolicy:
    '''
    Random normal policy.
    '''
    def __init__(self, act_size, std, name='pi'):
        self.act_size = act_size
        self.std = std
        self.name = 'rand_policy_' + name

    def draw_action(self, obs):
        return np.squeeze(np.random.normal(loc=0.0, scale=self.std, size=(np.asmatrix(obs).shape[0],self.act_size)))



class MVNPolicy:
    '''
    Gaussian policy with diagonal covariance. The mean and the std can be any
    kind of tensor (a MLP depending on the state, a simple tensor, or a fixed constant).
    '''
    def __init__(self, session, obs, mean, std, name='pi', act_bound=np.inf):
        self.session = session
        self.name = 'mvn_policy_' + name
        self.obs = obs
        # If the environment has bounded actions, bound the policy output as well
        if not np.any(np.isinf(act_bound)):
            self.mean = act_bound*tf.nn.tanh(mean)
        else:
            self.mean = mean
        self.act_size = mean.get_shape().as_list()[1]
        self.std = std
        self.act_bound = act_bound

        self.mvn = tf.contrib.distributions.MultivariateNormalDiag(self.mean, self.std)
        self.output = self.mvn.sample()

        self.entropy = tf.reduce_mean(self.mvn.entropy())

        self.act = tf.placeholder(dtype=obs.dtype, shape=[None, self.act_size], name=name+'_act')
        self.log_prob = tf.expand_dims(self.mvn.log_prob(self.act), axis=-1) # expand vector returned by log_prob to row vector



    def get_log_prob(self, obs, act):
        return self.session.run(self.log_prob, {self.obs: np.asmatrix(obs), self.act: np.asmatrix(act)})

    def estimate_entropy(self, obs):
        return np.squeeze(self.session.run(self.entropy, {self.obs: np.asmatrix(obs)}))

    def estimate_kl(self, obs, old_mean, old_std):
        return np.squeeze(self.session.run(self.kl, {self.obs: np.asmatrix(obs), self.old_std: np.asmatrix(old_std), self.old_mean: np.asmatrix(old_mean)}))

    def draw_action(self, obs):
        return np.squeeze(self.session.run(self.output, {self.obs: np.asmatrix(obs)}))

    def draw_action_det(self, obs):
        return np.squeeze(self.session.run(self.mean, {self.obs: np.asmatrix(obs)}) )
