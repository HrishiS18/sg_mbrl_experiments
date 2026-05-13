# Nonlinear Fixed-Feature Stress Protocol

Generated at: 2026-05-11 13:49:03

This run uses a nonlinear graph-Hamiltonian chain with quartic node and edge potentials. It is designed to stress the direct graph baselines under the same data and CEM-MPC planner.

## Final-Budget Matched Planner

| env | budget | model | return_mean_mean | return_mean_se | terminal_error_mean_mean | one_step_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- |
| nonlinear_chain_stabilization | 512 | SG-PolyHam+Symp | -51.05 | 8.187 | 0.4943 | 1.759e-10 |
| nonlinear_chain_stabilization | 512 | DirectGraphNext | -53.24 | 8.31 | 0.5791 | 0.06104 |
| nonlinear_chain_stabilization | 512 | GraphEuler | -53.24 | 8.31 | 0.5791 | 0.06104 |
| nonlinear_chain_target | 512 | SG-PolyHam+Symp | -12.11 | 1.109 | 0.03736 | 2.296e-10 |
| nonlinear_chain_target | 512 | DirectGraphNext | -12.17 | 1.119 | 0.03803 | 0.02111 |
| nonlinear_chain_target | 512 | GraphEuler | -12.17 | 1.119 | 0.03803 | 0.02111 |

## Long-Horizon Nonlinear Dynamics

| env | horizon | model | rollout_rmse_mean | max_energy_drift_mean | one_step_rmse_mean |
| --- | --- | --- | --- | --- | --- |
| nonlinear_chain_stabilization | 500 | SG-PolyHam+Symp | 2.282e-08 | 0.001974 | 1.759e-10 |
| nonlinear_chain_stabilization | 500 | DirectGraphNext | 1.681 | 3.569 | 0.06104 |
| nonlinear_chain_stabilization | 500 | GraphEuler | 1.681 | 3.569 | 0.06104 |
| nonlinear_chain_target | 500 | SG-PolyHam+Symp | 3.063e-08 | 0.001547 | 2.296e-10 |
| nonlinear_chain_target | 500 | DirectGraphNext | 0.9675 | 2.321 | 0.02111 |
| nonlinear_chain_target | 500 | GraphEuler | 0.9675 | 2.321 | 0.02111 |

## Files

- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/nonlinear_stress_5seed/raw/nonlinear_matched_planner.csv`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/nonlinear_stress_5seed/raw/nonlinear_dynamics.csv`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/nonlinear_stress_5seed/figures/nonlinear_matched_planner_return.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/nonlinear_stress_5seed/figures/nonlinear_dynamics_rollout.png`

## Interpretation

These stress tests are still fixed-feature experiments. They support the mechanism that a correctly specified graph-local Hamiltonian model can be far more sample efficient and physically stable than direct graph predictors when nonlinear energy structure matters. They should be followed by neural ResidualSympGNN vs DirectGNN experiments before making final paper-scale claims.
