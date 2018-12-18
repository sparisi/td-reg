#### How to run
`python3 <ALG_SCRIPT> <ENV_NAME> <SEED>`  
(seed is optional, default is 1). At each iteration, data about the most important statistics (average return, value function loss, entropy, ...) is saved in  
`data-trial/<ALG_NAME>/<ENV_NAME>/<DATE_TIME>.dat`.  

To run more trials in parallel, see `run_multiproc.py`.


> Note that all scripts use [flexible memory](https://github.com/tensorflow/tensorflow/issues/1578), i.e.,
> ```
> config_tf = tf.ConfigProto()
> config_tf.gpu_options.allow_growth=True
> session = tf.Session(config=config_tf)
> ```


#### Requirements
* [`python 3+`](https://www.python.org/download/releases/3.0/)
* [`tensorflow 1.4.1+`](https://www.tensorflow.org/install/)
* [`gym`](https://github.com/openai/gym/)
* [`numpy`](https://docs.scipy.org/doc/numpy/user/install.html)

You can also use other physics simulators, such as [Roboschool](https://github.com/openai/roboschool/), [PyBullet](https://pypi.org/project/pybullet/) and [MuJoCo](https://github.com/openai/mujoco-py/).

#### Common files
* `approximators.py`   : neural network, random Fourier features, polynomial features
* `data_collection.py` : functions for sampling MDP transitions and getting mini-batches
* `filter_env.py`      : modifies a gym environment to have states and actions normalized in [-1,1]
* `logger.py`          : creates folders for saving data
* `policy.py`          : implementation of Gaussian policy
* `rl_utils.py`        : RL functions, such as [generalized advantage estimation](https://arxiv.org/abs/1506.02438) and [Retrace](https://arxiv.org/pdf/1606.02647.pdf)
