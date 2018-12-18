#### Summary
* Run `launch.m` to launch trials in parallel. Data (`.mat`) will be saved in a separate folder, specified in `launch.m`.  
* Run single algorithm scripts (e.g., `run_spg_td.m`) for launching a single trial on a single algorithm.  
* Data will be saved in the root folder if no trial number is specified.  

#### How to plot results
* `plot_vs.m` plots the results for single trials of different algorithms. Data is loaded from root folder.
* `make_stdplot.m` plots mean and std error of all algorithms over all trials.
* `make_stdplot_lambda.m` does the same, but for different values of lambda_decay for SPG TD-REG.