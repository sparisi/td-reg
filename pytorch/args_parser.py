import pathlib
import argparse
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch TRPO')
    parser.add_argument('--env_name', help='name of the environment to run')
    parser.add_argument('--env_id', type=int, default=0, help='Id of of the environment to run')
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')

    ## Seeds
    parser.add_argument('--rl_seed', type=int, default=1, help='random seed for rl')

    parser.add_argument('--hidden_size', nargs='+', type=int, default=[128, 128], help='list of hidden layers')
    parser.add_argument('--activation', action="store", default="tanh", choices=["relu", "tanh", "sigmoid"], help='list of hidden layers')

    parser.add_argument('--l2_reg', type=float, default=1e-3, help='l2 regularization for GAE')
    parser.add_argument('--learning_rate_v', type=float, default=3e-4, help='learning rate GAE')

    ## Method name
    parser.add_argument('--method_name', action="store", default="TRPO", choices=["TRPO", "TRPO-TD", "TRPO-RET-MC", "TRPO-RET-GAE"], help='Name of method')

    ## RL and TRPO options. Most of these comes from baselines.trpo_mpi.default
    parser.add_argument('--log_std', type=float, default=0, help='log std for the policy (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.995, help='discount factor for GAE')
    parser.add_argument('--tau', type=float, default=0.97, help='eta for GAE')
    parser.add_argument('--max_kl', type=float, default=0.01, help='max KL for TRPO')
    parser.add_argument('--damping', type=float, default=0.1, help='damping scale of FIM for TRPO')
    parser.add_argument('--rl_max_iter_num', type=int, default=10000, help='maximal number of main iterations')
    parser.add_argument('--min_batch_size', type=int, default=3000, help='minimal batch size per update')

    parser.add_argument('--lamret', type=int, default=1, help='Fit value function to TD(lambda) return or not')
    parser.add_argument('--mgae', type=int, default=1, help='Standardize GAE with mean subtraction or not')
    parser.add_argument('--mtd', type=int, default=1, help='Standardize TD with mean subtraction or not')

    ## TD regularization
    parser.add_argument('--lambda_td', type=float, default=0.1, help='Initial eta value of TD-REG')
    parser.add_argument('--decay_td', type=float, default=0.9999, help='decay of TD-REG')

    ## Path and filename options
    parser.add_argument('--pre_model_path', help='path of pre-trained model for RL')
    parser.add_argument('--rl_filename', help="name of the result file saved by RL")
    parser.add_argument('--rl_model_filename', help="name of the model file saved by RL and loaded by TRAJ")

    ## Logging options
    parser.add_argument('--log_interval', type=int, default=10, help='interval between training status logs')
    parser.add_argument('--rl_save_model_interval', type=int, default=100, help="interval between saving RL model (0 to save only at the end)")

    args = parser.parse_args()

    #Use ID instead of name. ID is more convenient when doing mass experiments on clusters.
    if args.env_name is None:
        env_dict = {-3 : "CartPoleContinuous-v0",
                    -2 : "BipedalWalker-v2",
                    -1 : "LunarLanderContinuous-v2",
                    0 : "Pendulum-v0",
                    1 : "InvertedPendulum-v2",
                    2 : "HalfCheetah-v2",
                    3 : "Reacher-v2",
                    4 : "Swimmer-v2",
                    5 : "Ant-v2",
                    6 : "Hopper-v2",
                    7 : "Walker2d-v2",
                    8 : "InvertedDoublePendulum-v2",
                    9 : "Humanoid-v2",
                    10: "HumanoidStandup-v2",
        }
        args.env_name = env_dict[args.env_id]

    if args.method_name == "TRPO":
        exp_name_rl =  "%s_%s_h%d-%d_lamret%d_mgae%d_s%d" % \
            (args.env_name, args.method_name, args.hidden_size[0], args.hidden_size[1], args.lamret, args.mgae, args.rl_seed)
    else:
        exp_name_rl =  "%s_%s_h%d-%d_lamret%d_mgae%d_mtd%d_lambda%f_decay%f_s%d" % \
            (args.env_name, args.method_name, args.hidden_size[0], args.hidden_size[1], args.lamret, args.mgae, args.mtd, args.lambda_td, args.decay_td, args.rl_seed)

    if args.rl_filename is None:
        pathlib.Path("./results/RL/" + args.env_name).mkdir(parents=True, exist_ok=True)
        args.rl_filename = "./results/RL/" + args.env_name + "/" + exp_name_rl + ".txt"

    if args.rl_model_filename is None:
        pathlib.Path("./results/RL_models/" + args.env_name).mkdir(parents=True, exist_ok=True)
        args.rl_model_filename = "./results/RL_models/" + args.env_name + "/" + exp_name_rl


    return args
