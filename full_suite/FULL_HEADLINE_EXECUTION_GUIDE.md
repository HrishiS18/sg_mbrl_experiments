# Full Headline Sample-Efficiency Execution Guide

The full suite is built and smoke-tested. Use this guide to run the paper-scale headline claim.

## 1. Activate The Neural Environment

```bash
cd /Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/full_suite
source .venv/bin/activate
python -m sg_mbrl_full.scripts.check_dependencies
```

Expected dependency status:

```text
torch: ok
gymnasium: ok
stable_baselines3: ok
```

## 2. Smoke Tests Already Passed

Completed runs:

```text
results/smoke
results/model_free_smoke
```

Smoke coverage:

- SG-MBRL / ResidualSympGNN + CEM-MPC,
- DirectGNN + identical CEM-MPC,
- PETS-style DirectGNN ensemble + CEM-MPC,
- no-control and random baselines,
- SAC model-free wrapper through Stable-Baselines3,
- aggregation and plotting.

## 3. Run The Main MBRL Headline Suite

```bash
python -m sg_mbrl_full.scripts.run_headline --config configs/headline.json
python -m sg_mbrl_full.scripts.aggregate --run-dir results/headline
```

Main environments:

- graph oscillator stabilization,
- nonlinear spring/Duffing chain target reaching,
- Lennard-Jones formation control.

Main methods:

- `SG-MBRL`,
- `DirectGNN-MPC`,
- `PETS-DirectGNN`,
- `MBPO-SAC`,
- `HNN-MPC`,
- `MLP-MPC`,
- `NoControl`,
- `Random`.

Primary output files:

```text
results/headline/raw/headline_metrics.csv
results/headline/raw/headline_metrics_aggregated.csv
results/headline/raw/rollout_diagnostics.csv
results/headline/raw/samples_to_threshold.csv
results/headline/summary.md
results/headline/figures/return_vs_steps_*.png
results/headline/figures/rollout_diagnostics_*.png
```

## 4. Run Model-Free Baselines

```bash
python -m sg_mbrl_full.scripts.run_model_free --config configs/model_free.json
```

This runs:

- SAC,
- PPO,
- TD3.

Output:

```text
results/model_free/raw/model_free_metrics.csv
```

For the final paper, merge `model_free_metrics.csv` with `headline_metrics.csv` before plotting final return-vs-steps figures.

## 5. Evidence Needed For The Headline Claim

The headline claim is supported if, across the three main environments:

1. `SG-MBRL` reaches the task threshold at fewer real environment steps.
2. `SG-MBRL` has better or comparable final return.
3. `SG-MBRL` has smaller exploitation gap than generic model-based baselines.
4. `SG-MBRL` has better long-horizon rollout diagnostics.
5. `SG-MBRL` remains competitive against PETS/MBPO and substantially more sample-efficient than SAC/PPO/TD3.

## 6. Fairness Checks

Before using results in the paper, verify:

- DirectGNN-MPC and SG-MBRL use identical CEM settings.
- Parameter counts are reported from `raw/parameter_logs.csv`.
- PETS uses ensemble size 5.
- Model-free baselines use the same env/task/horizon and enough steps.
- Seeds are paired across methods.
- The final headline plots include confidence bands.

## 7. Practical Runtime Notes

The default `configs/headline.json` is intentionally paper-scale. On this local CPU/MPS setup it may take a long time. For a medium run, copy the config and reduce:

```json
"seeds": [0, 1, 2],
"budgets": [1000, 2500, 5000, 10000],
"eval": {"episodes": 20, "control_horizon": 50},
"cem_defaults": {"population": 256, "iterations": 4}
```

Do not present the medium run as the final benchmark; use it to debug and tune before the full run.

