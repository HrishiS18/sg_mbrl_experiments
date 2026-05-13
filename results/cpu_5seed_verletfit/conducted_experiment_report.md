# Conducted Experiment Report: Fixed-Feature SG-MBRL Protocol

Run directory:

```text
/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/cpu_5seed_verletfit
```

This report summarizes the experiments actually conducted locally. The run is a fixed-feature, NumPy-only benchmark because the available environment does not include PyTorch, Stable-Baselines3, or an existing SG-MBRL codebase. It is therefore a reproducible structural benchmark, not the final neural/full-compute benchmark.

## What Was Implemented

The local harness implements:

- coupled oscillator stabilization,
- spring-mass chain target reaching,
- graph-size transfer from `|V| in {4,8,16}` to `|V| in {32,64,128,256}`,
- topology transfer from chain to cycle/star/grid/Erdos-Renyi,
- long-horizon dynamics rollouts up to 500 steps,
- matched CEM-MPC control,
- ablation models,
- planning horizon sensitivity,
- parameter-count table,
- CSV and figure generation.

Models included:

- `SG-Ham+Symp`: graph-local Hamiltonian model with velocity-Verlet/symplectic rollout,
- `DirectGraphNext`: graph-local direct next-state predictor,
- `GraphEuler`: graph-local vector-field model with nonsymplectic Euler rollout,
- `SympNoGraph`: Hamiltonian/symplectic model without graph edge terms,
- `DenseLinearNext`: dense fixed-size linear next-state model.

The SG-Hamiltonian fitter uses the Verlet-consistent force target:

```text
F(q,a) = 2(q_next - q - dt p) / dt^2
```

This makes the fixed-feature Hamiltonian comparison fair to the simulator used to generate data.

## Configuration

- seeds: `0,1,2,3,4`
- budgets: `64,128,256,512,1024`
- control horizon: `40`
- evaluation episodes per seed: `6`
- CEM population: `96`
- CEM iterations: `4`
- CEM elite fraction: `0.12`
- planning horizons: `1,5,10,20,30`
- rollout horizons: `1,10,25,50,100,250,500`

## Headline Findings

### 1. Matched-Planner Control

At the final budget of `1024` transitions:

| Environment | SG-Ham+Symp return | DirectGraphNext return | Interpretation |
| --- | ---: | ---: | --- |
| coupled oscillator stabilization | `-35.55` | `-35.56` | essentially tied |
| spring-chain target reaching | `-9.96` | `-9.96` | essentially tied |

This fixed-feature linear benchmark does **not** establish a strong control-return advantage for SG-MBRL over DirectGNN-MPC. The matched-planner result is fair, but the tasks are too easy and too close to the direct baseline's model class.

What it does show:

- SG-Ham+Symp reaches the same control performance with a tiny physically structured model.
- DirectGraphNext is a very strong baseline on linear oscillator tasks.
- The final paper needs nonlinear/hard-control tasks to show return separation.

### 2. Graph-Size Transfer

Training on small chains and testing zero-shot on `|V|=256`:

| Model | Rollout RMSE | Relative max energy drift |
| --- | ---: | ---: |
| SG-Ham+Symp | `1.56e-09` | `8.59e-04` |
| DirectGraphNext | `2.34e-04` | `8.60e-04` |
| GraphEuler | `2.34e-04` | `8.60e-04` |
| SympNoGraph | `6.90e-01` | `3.83e-02` |

This strongly supports the graph-local Hamiltonian transfer claim in the fixed-feature setting. SG-Ham+Symp identifies the local physical law and transfers almost exactly to much larger graphs. SympNoGraph fails, showing edge terms are necessary.

### 3. Long-Horizon Physical Rollouts

On spring-mass chains at 500-step rollout horizon:

| Model | Rollout RMSE | Energy drift | Symplecticity defect |
| --- | ---: | ---: | ---: |
| SG-Ham+Symp | `1.78e-08` | `9.14e-04` | `4.33e-11` |
| DirectGraphNext | `1.08e-03` | `9.29e-04` | `3.50e-06` |
| GraphEuler | `1.08e-03` | `9.28e-04` | `3.48e-06` |
| DenseLinearNext | `8.42e-06` | `9.26e-04` | `2.49e-07` |
| SympNoGraph | `1.24e+00` | `1.33e-01` | `2.77e-11` |

This supports the physical-structure claim:

- SG-Ham+Symp has the lowest rollout error.
- SG-Ham+Symp has near-zero symplecticity defect.
- SympNoGraph is symplectic but wrong, proving symplecticity alone is insufficient without graph edge terms.

Energy drift is similar among several well-fit models on this easy linear task, so the stronger energy-drift separation should be tested on nonlinear pendulum/Lennard-Jones/N-body tasks.

### 4. Ablation Matrix

Main ablation pattern:

- `SympNoGraph` performs much worse in rollout stability and exploitation gap.
- `SG-Ham+Symp`, `DirectGraphNext`, `GraphEuler`, and `DenseLinearNext` are close in control return on the easy linear control tasks.
- The current run supports the necessity of **graph edge terms**.
- It does not yet prove the full ladder:

```text
graph only < graph + symplectic < graph + symplectic + uncertainty
```

because uncertainty-aware planning and neural ensembles were not available in this local NumPy run.

### 5. Planning-Horizon Sensitivity

Across horizons `1,5,10,20,30`, SG-Ham+Symp and DirectGraphNext remain close in return. The exploitation gap is nearly zero for SG-Ham+Symp and small but positive for DirectGraphNext/GraphEuler.

This weakly supports the planning robustness story, but the benchmark is too linear/easy to produce the reviewer-proof failure mode where generic models exploit long imagined horizons.

## Claim Status After This Run

| Claim | Status | Evidence |
| --- | --- | --- |
| sample efficiency | inconclusive | control returns are tied across budgets |
| long-horizon physical stability | supported in fixed-feature setting | SG has lowest rollout RMSE and near-zero symplecticity defect |
| graph-size transfer | strongly supported | SG transfers nearly exactly to `|V|=256` |
| control improvement | not yet established | matched CEM-MPC returns are essentially tied |
| component necessity | partially supported | graph edge terms are necessary; uncertainty not tested |

## Nonlinear Stress-Test Addendum

A second local run was conducted here:

```text
/Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/results/nonlinear_stress_5seed
```

This stress test uses a nonlinear graph-Hamiltonian chain with quartic node and edge potentials. The model `SG-PolyHam+Symp` uses the correct graph-local polynomial Hamiltonian force features, while `DirectGraphNext` and `GraphEuler` use the same data and CEM-MPC planner but direct graph-local linear dynamics features.

Final-budget matched-planner results:

| Environment | SG-PolyHam+Symp | DirectGraphNext | Interpretation |
| --- | ---: | ---: | --- |
| nonlinear chain stabilization | `-51.05` | `-53.24` | SG better under identical CEM |
| nonlinear chain target | `-12.11` | `-12.17` | SG slightly better |

500-step nonlinear rollout results:

| Environment | Model | Rollout RMSE | Energy drift | One-step RMSE |
| --- | --- | ---: | ---: | ---: |
| nonlinear stabilization | SG-PolyHam+Symp | `2.28e-08` | `1.97e-03` | `1.76e-10` |
| nonlinear stabilization | DirectGraphNext | `1.68e+00` | `3.57e+00` | `6.10e-02` |
| nonlinear target | SG-PolyHam+Symp | `3.06e-08` | `1.55e-03` | `2.30e-10` |
| nonlinear target | DirectGraphNext | `9.67e-01` | `2.32e+00` | `2.11e-02` |

This stress test strongly supports the mechanism that, when the graph-local Hamiltonian features match the nonlinear physical law, symplectic Hamiltonian rollouts are far more stable than direct graph prediction. It also gives a modest matched-planner control advantage on nonlinear stabilization.

Important caveat: this is still a fixed-feature comparison. It does not replace the final neural ResidualSympGNN vs DirectGNN experiment, because a sufficiently expressive DirectGNN may learn some of these nonlinearities from more data.

## Paper-Ready Use

These results can be used as a fixed-feature/theory-aligned validation suite, especially for:

- graph-local Hamiltonian recovery,
- graph-size transfer,
- symplecticity defect,
- edge-term ablation,
- reproducible protocol demonstration.

They should **not** be used as the final headline SG-MBRL control benchmark because the current tasks are too favorable to direct linear graph prediction.

## Required Next Experiments

To make the paper strong, the next run must add:

1. richer nonlinear environments: pendulum, double pendulum, Lennard-Jones, sparse N-body,
2. neural ResidualSympGNN and DirectGNN with matched parameter counts,
3. PETS-style probabilistic ensembles,
4. MBPO with SAC/TD3 policy optimization,
5. tuned SAC/PPO/TD3 model-free baselines,
6. uncertainty calibration and uncertainty-aware planning,
7. high-energy and unseen-topology stress tests,
8. matched one-step-error model selection.

The most important next implementation target remains:

```text
ResidualSympGNN + CEM-MPC vs DirectGNN + identical CEM-MPC
```

on nonlinear/hard-control tasks.

## Figures

- `figures/matched_planner_return_vs_budget.png`
- `figures/graph_size_transfer_chain.png`
- `figures/dynamics_rollout_spring_mass.png`
- `figures/ablation_mpc_return.png`
- `figures/planning_horizon_sensitivity.png`
