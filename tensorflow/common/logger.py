import os, os.path
import errno
from datetime import datetime

'''
These classes safely create new folders used for logging statistics (average
return, entropy, ...) and tensorflow models.
See demo.py for their usage.
'''

def mkdir_p(path):
    try: # safely create folder and file
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

class LoggerData:
    def __init__(self, alg_name, env_name, run_name=None, ext='.dat', log_dir='./data-trial/'):
        if run_name is None:
            run_name = datetime.utcnow().strftime("%y%m%d_%H%M%S") # create unique filename
        self.fullname = os.path.join(log_dir, alg_name, env_name, run_name+ext)
        self.pathname = os.path.join(log_dir, alg_name, env_name)
        mkdir_p(self.pathname)

class LoggerModel:
    def __init__(self, alg_name, env_name, run_name=None, ext='.ckpt', log_dir='./model-tf/'):
        if run_name is None:
            run_name = datetime.utcnow().strftime("%y%m%d_%H%M%S") # create unique filename
        self.fullname = os.path.join(log_dir, alg_name, env_name, run_name+ext)
        self.pathname = os.path.join(log_dir, alg_name, env_name)
        mkdir_p(self.pathname)
