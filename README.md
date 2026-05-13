# SG-MBRL Experiments

This repository contains the experimental harness and preliminary results for the SG-MBRL paper project. The main paper-scale benchmark studies whether a Residual Symplectic Graph Neural Network world model with CEM-MPC improves sample-efficient control over generic graph, Hamiltonian, PETS-style, MBPO-style, MLP, random, and no-control baselines.

## Repository Layout

```text
.
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md
│   └── FIVE_SEED_HEADLINE_RUN.md
├── full_suite/
│   ├── configs/
│   ├── scripts/
│   ├── sg_mbrl_full/
│   └── results/
├── results/
├── run_fixed_feature_protocol.py
└── run_nonlinear_stress_protocol.py
```

## Main Experiment

The active paper-scale run is the 5-seed neural headline suite:

```text
3 environments x 5 seeds x 7 budgets x 8 methods = 840 method runs
```

Environments:

- linear graph oscillator stabilization,
- nonlinear Duffing/spring chain target reaching,
- Lennard-Jones particle formation control.

Methods:

- `SG-MBRL`,
- `DirectGNN-MPC`,
- `PETS-DirectGNN`,
- `MBPO-SAC`,
- `HNN-MPC`,
- `MLP-MPC`,
- `NoControl`,
- `Random`.

The full run status and resume commands are documented in [docs/FIVE_SEED_HEADLINE_RUN.md](docs/FIVE_SEED_HEADLINE_RUN.md).

## Quick Start

```bash
cd /Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/full_suite
bash scripts/setup_neural_env.sh
source .venv/bin/activate
python -m sg_mbrl_full.scripts.check_dependencies
```

Run a smoke test:

```bash
python -m sg_mbrl_full.scripts.run_headline --config configs/smoke.json
python -m sg_mbrl_full.scripts.aggregate --run-dir results/smoke
```

Run the paper-scale headline suite:

```bash
python -m sg_mbrl_full.scripts.run_headline --config configs/headline.json
python -m sg_mbrl_full.scripts.aggregate --run-dir results/headline
```

## Result Artifacts

Tracked lightweight artifacts include:

- fixed-feature 5-seed results in `results/cpu_5seed_verletfit`,
- nonlinear stress 5-seed results in `results/nonlinear_stress_5seed`,
- neural smoke-test results in `full_suite/results/smoke`,
- current partial neural headline CSVs in `full_suite/results/headline_20260511_201112`.

Heavy generated artifacts such as virtual environments, checkpoints, caches, and logs are ignored by Git.

## Paper-Grade Evidence Target

The final NeurIPS/ICLR-grade claim should be based on at least 5 seeds for the main MBRL headline comparison, with 10 seeds preferred for final plots if compute permits. The cleanest causal comparison remains:

```text
SG-MBRL + CEM-MPC vs DirectGNN + identical CEM-MPC
```

See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) for the full experimental plan.
