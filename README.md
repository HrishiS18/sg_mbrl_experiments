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

```
