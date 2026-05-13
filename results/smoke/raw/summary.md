# Fixed-Feature SG-MBRL Protocol Run

Generated at: 2026-05-11 13:38:41

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
    1
  ],
  "budgets": [
    64,
    256
  ],
  "control_horizon": 25,
  "eval_episodes": 4,
  "train_rollout_horizon": 100,
  "cem_population": 48,
  "cem_iterations": 3,
  "cem_elite_frac": 0.12,
  "planning_horizons": [
    1,
    5,
    10
  ],
  "graph_transfer_train_sizes": [
    4,
    8,
    16
  ],
  "graph_transfer_test_sizes": [
    32,
    64
  ],
  "rollout_horizons": [
    1,
    10,
    50,
    100
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
- figures in `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/smoke/figures`

## Headline Matched-Planner Result at Final Budget

| env | budget | model | return_mean_mean | return_mean_se | terminal_error_mean_mean |
| --- | --- | --- | --- | --- | --- |
| coupled_oscillator_stabilization | 256 | DirectGraphNext | -41.15 | 1.969 | 1.111 |
| coupled_oscillator_stabilization | 256 | SG-Ham+Symp | -41.17 | 1.989 | 1.114 |
| spring_chain_target | 256 | SG-Ham+Symp | -13.05 | 0.8714 | 0.08451 |
| spring_chain_target | 256 | DirectGraphNext | -13.05 | 0.8646 | 0.08528 |

Interpretation: this is the highest-priority fairness test. Both methods use identical data budgets, graph inputs, action bounds, CEM horizon/population/iterations, and evaluation seeds.

## Graph-Size Transfer at Largest Chain Size

| topology | test_size | model | rollout_rmse_mean | rollout_rmse_se | max_energy_drift_mean |
| --- | --- | --- | --- | --- | --- |
| chain | 64 | GraphEuler | 0.0002251 | 7.183e-07 | 0.0008422 |
| chain | 64 | DirectGraphNext | 0.0002251 | 7.201e-07 | 0.0008422 |
| chain | 64 | SG-Ham+Symp | 0.003062 | 0.0001284 | 0.00169 |
| chain | 64 | SympNoGraph | 0.6725 | 0.002909 | 0.04925 |

Interpretation: lower rollout RMSE and energy drift at `|V|=64` support the graph-local transfer claim.

## Long-Horizon Dynamics on Spring-Mass Chain

| env | horizon | model | rollout_rmse_mean | max_energy_drift_mean | symplecticity_defect_mean |
| --- | --- | --- | --- | --- | --- |
| spring_mass_chain | 100 | GraphEuler | 0.0002124 | 0.0007785 | 4.67e-06 |
| spring_mass_chain | 100 | DirectGraphNext | 0.0002125 | 0.0007787 | 4.701e-06 |
| spring_mass_chain | 100 | DenseLinearNext | 1.578e-06 | 0.0007849 | 2.478e-07 |
| spring_mass_chain | 100 | SG-Ham+Symp | 0.002839 | 0.001679 | 4.135e-11 |
| spring_mass_chain | 100 | SympNoGraph | 0.6319 | 0.1269 | 2.845e-11 |

Interpretation: this tests whether low one-step error is enough for physically stable rollouts. Symplectic models should have much lower symplecticity defect and energy drift.

## Ablation Summary

| env | model | return_mean_mean | return_mean_se | max_energy_drift_mean | exploitation_gap_mean_mean |
| --- | --- | --- | --- | --- | --- |
| coupled_oscillator_stabilization | DirectGraphNext | -27.62 | 8.622 | 0.0008065 | 0.0002756 |
| coupled_oscillator_stabilization | GraphEuler | -27.62 | 8.622 | 0.0008063 | 0.0002753 |
| coupled_oscillator_stabilization | SG-Ham+Symp | -27.62 | 8.622 | 0.001615 | 0.003212 |
| coupled_oscillator_stabilization | DenseLinearNext | -27.62 | 8.622 | 0.0008291 | 4.146e-06 |
| coupled_oscillator_stabilization | SympNoGraph | -28.01 | 8.846 | 0.1633 | 0.5862 |
| spring_chain_target | SG-Ham+Symp | -10.75 | 1.425 | 0.002114 | -0.0007346 |
| spring_chain_target | DenseLinearNext | -10.75 | 1.423 | 0.001084 | 1.335e-06 |
| spring_chain_target | DirectGraphNext | -10.76 | 1.425 | 0.001051 | -8.023e-05 |
| spring_chain_target | GraphEuler | -10.76 | 1.425 | 0.00105 | -8.039e-05 |
| spring_chain_target | SympNoGraph | -10.78 | 1.463 | 0.1699 | 0.02758 |

Interpretation: this tests graph-only, symplectic-only, graph+Euler, dense, and graph+symplectic variants under the same control protocol.

## Planning-Horizon Sensitivity

| planning_horizon | model | return_mean_mean | exploitation_gap_mean_mean | imagined_energy_drift_mean_mean |
| --- | --- | --- | --- | --- |
| 1 | DirectGraphNext | -29.86 | 5.985e-06 | 0.3957 |
| 1 | GraphEuler | -29.86 | 5.976e-06 | 0.3957 |
| 1 | SG-Ham+Symp | -29.86 | -6.997e-05 | 0.3953 |
| 5 | DirectGraphNext | -32 | 3.03e-05 | 0.7691 |
| 5 | GraphEuler | -32 | 3.019e-05 | 0.7691 |
| 5 | SG-Ham+Symp | -31.92 | 0.001309 | 0.7711 |
| 10 | DirectGraphNext | -26.86 | 0.0001156 | 1.19 |
| 10 | GraphEuler | -26.86 | 0.0001153 | 1.19 |
| 10 | SG-Ham+Symp | -26.86 | 0.002189 | 1.19 |

Interpretation: this tests whether each model remains useful as the planner leans harder on imagined long-horizon rollouts.

## Figure Files

- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/smoke/figures/matched_planner_return_vs_budget.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/smoke/figures/graph_size_transfer_chain.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/smoke/figures/dynamics_rollout_spring_mass.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/smoke/figures/ablation_mpc_return.png`
- `/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/smoke/figures/planning_horizon_sensitivity.png`

## Limitations

This run intentionally stays within NumPy/SciPy CPU constraints. It does not replace:

- neural ResidualSympGNN training,
- PETS probabilistic neural ensembles,
- MBPO with SAC/TD3,
- tuned PPO/SAC/TD3 baselines,
- Lennard-Jones/N-body hard-control tasks.

Use these results as a validated fixed-feature experiment suite and a reproducible dry run for the full-compute paper benchmark.
