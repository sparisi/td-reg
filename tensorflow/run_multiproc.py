'''
This script uses MULTIPROCESSING to run in parallel many trials of the same algorithm
on different environments with fixed random seed (seed = trial number).

Command
    python3 run_multiproc.py <ALG_NAME> <N_TRIALS> <ENV_LIST>

Example
    python3 run_multiproc.py 5 ppo Pendulum-v0 Swimmer-v2

Data is still saved as usual in `data-trial`, but instead of the current date and time,
the seed (= trial number) is used. For example, for the above run data will be saved in

data-trial/ppo/Pendulum-v0/0.dat
data-trial/ppo/Pendulum-v0/1.dat
...

'''

import sys
from multiprocessing import Process

alg_name = sys.argv[1]
n_trials = int(sys.argv[2])
env_list = sys.argv[3:]

from importlib import import_module
alg = import_module(alg_name)

# create a list of arguments, one for each run
args = []
for trial in range(n_trials):
    for env_name in env_list:
        args.append((env_name, trial, str(trial)))

# submit procs
ps = []
for a in args:
    p = Process(target=alg.main, args=a)
    p.start()
    ps.append(p)

for p in ps:
    p.join()
