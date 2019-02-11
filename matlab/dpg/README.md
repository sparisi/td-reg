#### Summary
* Run `launch.m` to launch trials in parallel. Data (`.mat`) will be saved in a separate folder, specified in `launch.m`.  
* Run single algorithm scripts (e.g., `run_dpg.m`) for launching a single trial on a single algorithm.  
* Data will be saved in the root folder if no trial number is specified.  

#### How to plot results
* `plot_one.m` plots statistics of a single trial of a single algorithm. First load the `.mat` file and then run the script.
* `plot_vs.m` plots the results for single trials of different algorithms. Data is loaded from root folder.
* `make_stdplot.m` plots mean and 95% confidence interval of all algorithms over all trials.
* `make_stdplot_lambda.m` does the same, but for different values of lambda for DPG TD-REG.
