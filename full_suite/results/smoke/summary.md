# Headline Suite Summary

## Final/Aggregated Metrics

env,task,graph,n,dim,budget,method,n_rows,n_seeds,return_mean_mean,return_mean_se,terminal_error_mean_mean,terminal_error_mean_se,success_rate_mean,success_rate_se,mpc_failure_rate_mean,mpc_failure_rate_se,exploitation_gap_mean_mean,exploitation_gap_mean_se,imagined_energy_drift_mean_mean,imagined_energy_drift_mean_se,one_step_rmse_mean,one_step_rmse_se,parameters_mean,parameters_se
duffing_chain,stabilization,chain,4,1,32,SG-MBRL,1,1,-1.5607504633963107,0.0,0.2488126903772354,0.0,0.0,0.0,0.0,0.0,-0.0128634674251079,0.0,0.1357011897555141,0.0,0.0118206842616745,0.0,263.0,0.0
duffing_chain,stabilization,chain,4,1,32,NoControl,1,1,-3.4189093708992004,0.0,0.8040280342102051,0.0,0.0,0.0,0.0,0.0,,0.0,,0.0,,0.0,0.0,0.0
duffing_chain,stabilization,chain,4,1,32,DirectGNN-MPC,1,1,-5.658178489148617,0.0,1.5699689388275146,0.0,0.0,0.0,0.0,0.0,1.615323852300644,0.0,3.1379168677910467,0.0,0.1579359721736164,0.0,322.0,0.0
duffing_chain,stabilization,chain,4,1,32,PETS-DirectGNN,1,1,-6.845406027078629,0.0,2.391001462936402,0.0,0.0,0.0,0.0,0.0,0.6456165837049486,0.0,1.0451328847782833,0.0,0.1797034964133742,0.0,644.0,0.0
duffing_chain,stabilization,chain,4,1,32,Random,1,1,-7.627615845888854,0.0,2.1680015325546265,0.0,0.0,0.0,0.0,0.0,,0.0,,0.0,,0.0,0.0,0.0


## Samples To Threshold

env,task,method,n_rows,n_seeds,samples_to_threshold_mean,samples_to_threshold_se
duffing_chain,stabilization,DirectGNN-MPC,1,1,,0.0
duffing_chain,stabilization,NoControl,1,1,,0.0
duffing_chain,stabilization,PETS-DirectGNN,1,1,,0.0
duffing_chain,stabilization,Random,1,1,,0.0
duffing_chain,stabilization,SG-MBRL,1,1,,0.0


## Diagnostics

See `figures/rollout_diagnostics_*.png` and `raw/rollout_diagnostics.csv`.
