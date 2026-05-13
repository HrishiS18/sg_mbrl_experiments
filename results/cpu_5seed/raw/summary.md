# Fixed-Feature SG-MBRL Protocol Run

Generated at: 2026-05-11 13:41:10

This run implements a CPU-scale fixed-feature version of the experiment protocol. It is not the final neural ResidualSympGNN/PETS/MBPO/model-free benchmark. It directly tests the clean structural mechanism with matched data and matched CEM-MPC:

- graph-local Hamiltonian model + symplectic rollout (`SG-Ham+Symp`),
- graph-local direct next-state model (`DirectGraphNext`),
- graph-local Euler vector-field model (`GraphEuler`),
- no-edge symplectic Hamiltonian ablation (`SympNoGraph`),
- dense linear next-state baseline on fixed-size tasks (`DenseLinearNext`).

## Configuration

```json
{
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "budgets": [
    64,
    128,
    256,
    512,
    1024
  ],
  "control_horizon": 40,
  "eval_episodes": 6,
  "train_rollout_horizon": 100,
  "cem_population": 96,
  "cem_iterations": 4,
  "cem_elite_frac": 0.12,
  "planning_horizons": [
    1,
    5,
    10,
    20,
    30
  ],
  "graph_transfer_train_sizes": [
    4,
    8,
    16
  ],
  "graph_transfer_test_sizes": [
    32,
    64,
    128,
    256
  ],
  "rollout_horizons": [
    1,
    10,
    25,
    50,
    100,
    250,
    500
  ],
  "dt": 0.05,
  "omega2": 1.0,
  "spring_k": 0.5,
  "action_limit": 2.0,
  "ridge": 1e-05,
  "ensemble_size": 5,
  "train_noise_std": 0.0
}
```

## Main Artifacts

- `matched_planner.csv`
- `graph_size_transfer.csv`
- `topology_transfer.csv`
- `dynamics_rollout.csv`
- `ablation.csv`
- `planning_horizon.csv`
- `parameter_table.csv`
- figures in `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/cpu_5seed/figures`

## Headline Matched-Planner Result at Final Budget

| env | budget | model | return_mean_mean | return_mean_se | terminal_error_mean_mean |
| --- | --- | --- | --- | --- | --- |
| coupled_oscillator_stabilization | 1024 | SG-Ham+Symp | -35.55 | 2.757 | 0.2048 |
| coupled_oscillator_stabilization | 1024 | DirectGraphNext | -35.56 | 2.765 | 0.2052 |
| spring_chain_target | 1024 | SG-Ham+Symp | -9.958 | 0.7105 | 0.0196 |
| spring_chain_target | 1024 | DirectGraphNext | -9.961 | 0.7092 | 0.01972 |

Interpretation: this is the highest-priority fairness test. Both methods use identical data budgets, graph inputs, action bounds, CEM horizon/population/iterations, and evaluation seeds.

## Graph-Size Transfer at Largest Chain Size

| topology | test_size | model | rollout_rmse_mean | rollout_rmse_se | max_energy_drift_mean |
| --- | --- | --- | --- | --- | --- |
| chain | 256 | GraphEuler | 0.0002342 | 8.606e-07 | 0.0008604 |
| chain | 256 | DirectGraphNext | 0.0002342 | 8.592e-07 | 0.0008604 |
| chain | 256 | SG-Ham+Symp | 0.003357 | 9.544e-05 | 0.001773 |
| chain | 256 | SympNoGraph | 0.6903 | 0.001337 | 0.03884 |

Interpretation: lower rollout RMSE and energy drift at `|V|=256` support the graph-local transfer claim.

## Long-Horizon Dynamics on Spring-Mass Chain

| env | horizon | model | rollout_rmse_mean | max_energy_drift_mean | symplecticity_defect_mean |
| --- | --- | --- | --- | --- | --- |
| spring_mass_chain | 500 | DenseLinearNext | 8.416e-06 | 0.0009258 | 2.487e-07 |
| spring_mass_chain | 500 | GraphEuler | 0.001083 | 0.0009279 | 3.484e-06 |
| spring_mass_chain | 500 | DirectGraphNext | 0.001083 | 0.0009289 | 3.495e-06 |
| spring_mass_chain | 500 | SG-Ham+Symp | 0.01546 | 0.001909 | 4.03e-11 |
| spring_mass_chain | 500 | SympNoGraph | 1.243 | 0.1332 | 2.925e-11 |

Interpretation: this tests whether low one-step error is enough for physically stable rollouts. Symplectic models should have much lower symplecticity defect and energy drift.

## Ablation Summary

| env | model | return_mean_mean | return_mean_se | max_energy_drift_mean | exploitation_gap_mean_mean |
| --- | --- | --- | --- | --- | --- |
| coupled_oscillator_stabilization | DenseLinearNext | -37.37 | 4.321 | 0.0008533 | 4.102e-06 |
| coupled_oscillator_stabilization | SG-Ham+Symp | -37.37 | 4.336 | 0.001735 | 0.003697 |
| coupled_oscillator_stabilization | DirectGraphNext | -37.43 | 4.325 | 0.000835 | 0.0002986 |
| coupled_oscillator_stabilization | GraphEuler | -37.43 | 4.325 | 0.0008349 | 0.0002982 |
| coupled_oscillator_stabilization | SympNoGraph | -37.79 | 4.309 | 0.1423 | 0.409 |
| spring_chain_target | SG-Ham+Symp | -11.75 | 0.9803 | 0.002228 | -0.0009004 |
| spring_chain_target | DirectGraphNext | -11.76 | 0.9784 | 0.001076 | -9.584e-05 |
| spring_chain_target | GraphEuler | -11.76 | 0.9784 | 0.001075 | -9.604e-05 |
| spring_chain_target | DenseLinearNext | -11.76 | 0.9798 | 0.001114 | 1.677e-06 |
| spring_chain_target | SympNoGraph | -11.76 | 0.9872 | 0.1429 | 0.01447 |

Interpretation: this tests graph-only, symplectic-only, graph+Euler, dense, and graph+symplectic variants under the same control protocol.

## Planning-Horizon Sensitivity

| planning_horizon | model | return_mean_mean | exploitation_gap_mean_mean | imagined_energy_drift_mean_mean |
| --- | --- | --- | --- | --- |
| 1 | DirectGraphNext | -34.43 | -1.548e-07 | 0.4003 |
| 1 | GraphEuler | -34.43 | -1.65e-07 | 0.4003 |
| 1 | SG-Ham+Symp | -34.43 | -8.551e-05 | 0.3999 |
| 5 | DirectGraphNext | -37.49 | 2.389e-05 | 1.032 |
| 5 | GraphEuler | -37.49 | 2.375e-05 | 1.032 |
| 5 | SG-Ham+Symp | -37.47 | -0.0001762 | 1.031 |
| 10 | DirectGraphNext | -32.44 | 0.0001344 | 1.541 |
| 10 | GraphEuler | -32.44 | 0.000134 | 1.541 |
| 10 | SG-Ham+Symp | -32.44 | 0.001551 | 1.541 |
| 20 | DirectGraphNext | -44.13 | 0.001359 | 2.983 |
| 20 | GraphEuler | -44.13 | 0.001358 | 2.983 |
| 20 | SG-Ham+Symp | -44.14 | 0.01563 | 2.983 |
| 30 | DirectGraphNext | -45.05 | 0.001993 | 4.085 |
| 30 | GraphEuler | -45.05 | 0.00199 | 4.085 |
| 30 | SG-Ham+Symp | -45.1 | 0.01848 | 4.085 |

Interpretation: this tests whether each model remains useful as the planner leans harder on imagined long-horizon rollouts.

## Figure Files

- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/cpu_5seed/figures/matched_planner_return_vs_budget.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/cpu_5seed/figures/graph_size_transfer_chain.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/cpu_5seed/figures/dynamics_rollout_spring_mass.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/cpu_5seed/figures/ablation_mpc_return.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/cpu_5seed/figures/planning_horizon_sensitivity.png`

## Limitations

This run intentionally stays within NumPy/SciPy CPU constraints. It does not replace:

- neural ResidualSympGNN training,
- PETS probabilistic neural ensembles,
- MBPO with SAC/TD3,
- tuned PPO/SAC/TD3 baselines,
- Lennard-Jones/N-body hard-control tasks.

Use these results as a validated fixed-feature experiment suite and a reproducible dry run for the full-compute paper benchmark.
