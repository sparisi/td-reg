import tensorflow as tf

precision          = tf.float32
maxiter            = 10000       # number of learning iterations
min_trans_per_iter = 3000       # minimum number of transition steps per iteration (if an episode ends before min_trans_per_iter is reached, a new one starts)
paths_eval         = 100        # number of episodes used for evaluating a policy
batch_size         = 64         # random mini-batch size for computing the gradients
lrate_v            = 1e-4       # ADAM learning rate (for learning V)
lrate_pi           = 1e-4       # ADAM learning rate (for learning pi)
epochs_v           = 20         # epochs of gradient descent (each epoch covers the whole dataset, divided in mini-batches)
epochs_pi          = 20         # epochs of gradient descent
gamma              = 0.99       # discount factor
lambda_trace       = 0.95       # coefficient for generalized advantage
e_clip             = 0.05        # the 'step size'
std_noise          = 2.         # Gaussian policy std
filter_env         = False      # True to normalize actions and states to [-1,1]
pi_activations     = [tf.nn.tanh, tf.nn.tanh]
v_activations      = [tf.nn.tanh, tf.nn.tanh]
v_sizes            = [64, 64]
pi_sizes           = [64, 64]

config_env = {
    'Reacher-v2' : {
        'maxiter'        : 3000,
    },
}
