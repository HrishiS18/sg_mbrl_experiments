# SG-MBRL Full Neural Suite

This directory contains the full-compute experiment harness for the headline sample-efficiency claim:

```text
ResidualSympGNN + CEM-MPC
vs DirectGNN-MPC
vs PETS-style DirectGNN ensemble
vs MBPO/model-free baselines
```

The currently installed system Python is 3.14 and does not include PyTorch. Use the setup script below, which creates a Python 3.12 virtual environment from the bundled Codex runtime when available.

```bash
cd /Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/full_suite
bash scripts/setup_neural_env.sh
source .venv/bin/activate
python -m sg_mbrl_full.scripts.check_dependencies
```

Run a smoke headline experiment:

```bash
python -m sg_mbrl_full.scripts.run_headline --config configs/smoke.json
python -m sg_mbrl_full.scripts.aggregate --run-dir results/smoke
```

Run the intended paper-scale suite after smoke tests pass:

```bash
python -m sg_mbrl_full.scripts.run_headline --config configs/headline.json
python -m sg_mbrl_full.scripts.run_model_free --config configs/model_free.json
python -m sg_mbrl_full.scripts.aggregate --run-dir results/headline
```

## Scope

Implemented:

- graph-Hamiltonian simulators: linear oscillator, nonlinear Duffing/spring chain, Lennard-Jones particles,
- ResidualSympGNN world model with symplectic Verlet rollout,
- DirectGNN next-state model,
- MLP next-state model,
- HNN/SympNoGraph-style baseline,
- bootstrapped ensemble support,
- PETS-style ensemble MPC,
- matched CEM-MPC evaluation,
- model-free Stable-Baselines3 runner for SAC/PPO/TD3 when installed,
- JSON configs and result aggregation.

MBPO is included as a configurable baseline hook. For final paper claims, run it with installed Stable-Baselines3 and enough compute; otherwise the runner records it as unavailable rather than silently substituting a weak approximation.

