#### Summary
* `run_single` runs the single-pendulum task
* `runD_single` runs the single-pendulum task with double-critic
* same for `run_double` and `runD_double`
* Arguments: trial seed, flag to enable/disable Retrace, flag for the regularizers (-1 NO-REG, 0 TD-REG, 1 GAE-REG)
* `show_pendulum` and `show_double` show an animation with the learned policy