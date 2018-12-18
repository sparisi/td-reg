from args_parser import arg_parser
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

#from https://tonysyu.github.io/plotting-error-bars.html#.WRwXWXmxjZs
def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, linestyle="-", linewidth = None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, linestyle=linestyle, color=color, linewidth = linewidth, label=label)
    #ax.plot(x, y, pltcmd, linewidth = linewidth, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def plot(args):

    seed_list = [1, 2, 3, 4, 5]
    
    R_avg_all = []
    R_std_all = []
    legend_all = []

    method_name_list = ["TRPO", "TRPO-TD", "TRPO-RET-MC", "TRPO-RET-GAE"]

    for method_name in method_name_list: 
        R_test_avg_seed = []
        R_test_std_seed = []
        for rl_seed in seed_list:
            R_test_avg = []
            R_test_std = []

            if method_name == "TRPO":
                exp_name_rl =  "%s_%s_h%d-%d_std_s%d" % \
                    (args.env_name, method_name, args.hidden_size[0], args.hidden_size[1], rl_seed)    
                
                #exp_name_rl =  "%s_%s_h%d-%d_s%d" % \
                #    (args.env_name, method_name, args.hidden_size[0], args.hidden_size[1], rl_seed)    
                
                #exp_name_rl =  "%s_%s_s%d" % \
                #    (args.env_name, method_name, rl_seed)    
            else:
                exp_name_rl =  "%s_%s_h%d-%d_std_lambda%f_decay%f_s%d" % \
                    (args.env_name, method_name, args.hidden_size[0], args.hidden_size[1], args.lambda_td, args.decay_td, rl_seed)
                
                #exp_name_rl =  "%s_%s_h%d-%d_lambda%f_decay%f_s%d" % \
                #    (args.env_name, method_name, args.hidden_size[0], args.hidden_size[1], args.lambda_td, args.decay_td, rl_seed)
                
                #exp_name_rl =  "%s_%s_lambda%f_decay%f_s%d" % \
                #    (args.env_name, method_name, args.lambda_td, args.decay_td, rl_seed)

            rl_filename = "./results/RL/" + args.env_name + "/" + exp_name_rl + ".txt"

            with open(rl_filename, 'r') as f:
                for i in range(0, args.rl_max_iter_num // args.log_interval):
                    line = f.readline()
                    line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
                    if i == 0:
                        iter_idx = line.index("Iter") + 1
                        R_test_avg_idx = len(line) - 2
                    iter_value = int(line[iter_idx])
                    R_test_avg += [ float(line[R_test_avg_idx]) ]
                    R_test_std += [ float(line[R_test_avg_idx + 1]) ]

            R_test_avg_seed += [ R_test_avg ]
            R_test_std_seed += [ R_test_std ]

        R_avg_all += [np.mean(np.array(R_test_avg_seed), 0)]
        R_std_all += [np.std(np.array(R_test_avg_seed), 0) / np.sqrt(len(seed_list))]

        legend_all += [method_name]


    linewidth = 2
    fontsize = 18   #14
    f = plt.figure(figsize=(8,6))
    ax = plt.gca()

    skipper = 5
    c_tmp = ["k", "g", "m", "b"]

    for i in range(0, len(legend_all)):

        y_plot = running_mean(R_avg_all[i][:-1], skipper)
        y_err = running_mean(R_std_all[i][:-1], skipper)

        errorfill(range(0,len(y_plot)), y_plot, yerr=y_err, color=c_tmp[i], linestyle="-", linewidth=linewidth, label=(legend_all[i]))

    plt.title(args.env_name, fontsize=fontsize)
    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("Test return", fontsize=fontsize)
    plt.xticks(np.arange(0, len(y_plot)+10, 200))

    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)

    def format_func(value, tick_number):
        return r"${0}$".format(value*10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    plt.legend(prop={"size":fontsize}) 

    
    #fig_name = "./figures/%s.png" % args.env_name 
    #f.savefig(fig_name)

    #fig_name = "./figures/%s.pdf" % args.env_name 
    #f.savefig(fig_name)

    plt.show()

if __name__ == "__main__":
    args = arg_parser()
    plot(args)
