# Five-Seed Neural Headline Run

This document records the current 5-seed paper-scale benchmark state.

## Run Identity

```text
Workspace: /Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/full_suite
Screen session: sg_mbrl_full_20260511_201112
Launch timestamp: 20260511_201112
Active config: configs/headline_mps_20260511_201112.json
Run directory: results/headline_20260511_201112
Main log: logs/screen_sg_mbrl_full_20260511_201112.log
Status file: results/full_paper_status_20260511_201112.txt
```

The local run was paused in-place with `SIGSTOP` at the user's request.

```text
Paused worker PID at time of writing: 12040
Paused process state: T+
Monitor automation: monitor-sg-mbrl-full-benchmark, paused
```

If the machine is rebooted, the paused process will not survive. In that case, resume by adding skip/resume support or relaunching from the latest completed rows.

## Config Summary

```text
Seeds: 0, 1, 2, 3, 4
Budgets: 1000, 2500, 5000, 10000, 25000, 50000, 100000
Environments: 3
Methods: 8
Total method runs: 840
```

Methods:

- `SG-MBRL`
- `DirectGNN-MPC`
- `PETS-DirectGNN`
- `MBPO-SAC`
- `HNN-MPC`
- `MLP-MPC`
- `NoControl`
- `Random`

## Completed Rows

As of the last check before packaging, the run had completed the first full block:

```text
linear_oscillator/stabilization, seed=0, budget=1000, all 8 methods
```

That is:

```text
8 / 840 method rows complete
```

Partial metrics are stored in:

```text
full_suite/results/headline_20260511_201112/raw/headline_metrics.csv
full_suite/results/headline_20260511_201112/raw/parameter_logs.csv
```

First-block highlights:

| Method | Return | Success | One-step RMSE |
| --- | ---: | ---: | ---: |
| SG-MBRL | -14.8943 | 0.42 | 0.000846 |
| DirectGNN-MPC | -17.1454 | 0.46 | 0.000668 |
| PETS-DirectGNN | -17.9233 | 0.40 | 0.000366 |
| MBPO-SAC | -20.8501 | 0.00 | |
| HNN-MPC | -18.9079 | 0.26 | 0.016509 |
| MLP-MPC | -14.7129 | 0.30 | 0.003318 |
| NoControl | -41.9369 | 0.00 | |
| Random | -50.2608 | 0.00 | |

## Resume The Paused Local Process

If the screen session and PID still exist:

```bash
cd /Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/full_suite
screen -ls
ps -o pid,ppid,etime,pcpu,pmem,rss,stat,command -p 12040
kill -CONT 12040
```

Then re-enable the heartbeat monitor from Codex if desired.

## Pause Again

```bash
kill -STOP 12040
```

Confirm it stopped:

```bash
ps -o pid,ppid,etime,pcpu,pmem,rss,stat,command -p 12040
```

Expected state after pause:

```text
STAT=T+
CPU=0.0%
```

## Aggregate When Finished

```bash
cd /Users/hrishismac/Desktop/CSC122/CS2/sg_mbrl_experiments/full_suite
source .venv/bin/activate
python -m sg_mbrl_full.scripts.aggregate --run-dir results/headline_20260511_201112
```

Expected final outputs include:

```text
results/headline_20260511_201112/raw/headline_metrics_aggregated.csv
results/headline_20260511_201112/raw/samples_to_threshold.csv
results/headline_20260511_201112/raw/rollout_diagnostics.csv
results/headline_20260511_201112/summary.md
results/headline_20260511_201112/figures/
```
