# Cloud Run Guide

This repo contains everything needed to run the 5-seed headline experiment on a separate cloud machine. Do not copy the local `.venv`; rebuild the environment on the cloud host.

## 1. Clone

```bash
git clone https://github.com/HrishiS18/sg_mbrl_experiments.git
cd sg_mbrl_experiments/full_suite
```

## 2. Install

Use Python 3.10-3.12. On a CUDA machine, install the PyTorch wheel appropriate for the host if your base image does not already include it, then install the remaining requirements.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-neural.txt
python -m sg_mbrl_full.scripts.check_dependencies
```

If the default PyPI `torch` wheel does not detect CUDA, follow the PyTorch install command for the cloud image's CUDA version, then rerun:

```bash
python -m sg_mbrl_full.scripts.check_dependencies
```

## 3. Run The 5-Seed Headline Benchmark

The main config is:

```text
configs/headline.json
```

It runs:

```text
3 environments x 5 seeds x 7 budgets x 8 methods = 840 method runs
```

Recommended long-running launch:

```bash
screen -S sg_mbrl_full
cd /path/to/sg_mbrl_experiments/full_suite
source .venv/bin/activate
SG_MBRL_DEVICE=cuda STAMP=cloud_$(date +%Y%m%d_%H%M%S) scripts/run_full_paper_benchmark.sh
```

Detach from screen with `Ctrl-a d`.

If you do not set `SG_MBRL_DEVICE`, the launcher auto-selects `cuda`, then `mps`, then `cpu`.

## 4. Resume Safety

`sg_mbrl_full.scripts.run_headline` writes after every method row and skips rows already marked `status=ok` in:

```text
results/<run_dir>/raw/headline_metrics.csv
```

If a run is interrupted, rerun the same config with the same `run_dir`; completed successful rows will be skipped.

## 5. Outputs

The headline run writes:

```text
results/headline_<stamp>/raw/headline_metrics.csv
results/headline_<stamp>/raw/parameter_logs.csv
results/headline_<stamp>/raw/rollout_diagnostics.csv
results/headline_<stamp>/raw/headline_metrics_aggregated.csv
results/headline_<stamp>/raw/samples_to_threshold.csv
results/headline_<stamp>/summary.md
results/headline_<stamp>/figures/
```

Manual aggregation:

```bash
python -m sg_mbrl_full.scripts.aggregate --run-dir results/headline_<stamp>
```

## 6. Model-Free Baselines

The full script runs model-free baselines after the neural headline suite:

```bash
python -m sg_mbrl_full.scripts.run_model_free --config configs/model_free.json
```

For paper-grade final figures, merge the model-free metrics with the headline metrics before making final return-vs-steps plots.
