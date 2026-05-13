from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_se(df: pd.DataFrame, groups: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(groups, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(groups, key))
        row["n_rows"] = len(g)
        row["n_seeds"] = g["seed"].nunique() if "seed" in g.columns else len(g)
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy() if m in g else np.array([])
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{m}_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_return(agg: pd.DataFrame, fig_dir: Path) -> None:
    for (env, task), sub_env in agg.groupby(["env", "task"], dropna=False):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        for method, g in sub_env.groupby("method"):
            g = g.sort_values("budget")
            if "return_mean_mean" not in g:
                continue
            x = g["budget"].to_numpy(float)
            y = g["return_mean_mean"].to_numpy(float)
            se = g["return_mean_se"].to_numpy(float)
            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(x, y - se, y + se, alpha=0.14)
        ax.set_xscale("log")
        ax.set_xlabel("real environment transitions")
        ax.set_ylabel("evaluation return")
        ax.set_title(f"{env} / {task}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"return_vs_steps_{env}_{task}.png")
        plt.close(fig)


def plot_diagnostics(diag: pd.DataFrame, fig_dir: Path) -> None:
    if diag.empty:
        return
    agg = mean_se(diag, ["env", "task", "method", "horizon"], ["rollout_rmse", "energy_drift", "divergence_rate"])
    for (env, task), sub_env in agg.groupby(["env", "task"], dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)
        for ax, metric, label in [
            (axes[0], "rollout_rmse", "rollout RMSE"),
            (axes[1], "energy_drift", "energy drift"),
        ]:
            for method, g in sub_env.groupby("method"):
                g = g.sort_values("horizon")
                ax.plot(g["horizon"], g[f"{metric}_mean"], marker="o", label=method)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("rollout horizon")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{env} / {task}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"rollout_diagnostics_{env}_{task}.png")
        plt.close(fig)


def samples_to_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    good = df[df["status"] == "ok"].copy()
    for (env, task, method, seed), g in good.groupby(["env", "task", "method", "seed"]):
        g = g.sort_values("budget")
        hit = g[g["terminal_error_mean"] <= threshold]
        rows.append(
            {
                "env": env,
                "task": task,
                "method": method,
                "seed": seed,
                "samples_to_threshold": int(hit.iloc[0]["budget"]) if len(hit) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_summary(run_dir: Path, agg: pd.DataFrame, diag: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    final_rows = agg.sort_values(["env", "task", "budget", "return_mean_mean"], ascending=[True, True, True, False])
    md = ["# Headline Suite Summary\n"]
    md.append("## Final/Aggregated Metrics\n")
    md.append(final_rows.to_markdown(index=False) if _has_tabulate() else final_rows.to_csv(index=False))
    if not threshold_df.empty:
        md.append("\n## Samples To Threshold\n")
        th_agg = mean_se(threshold_df, ["env", "task", "method"], ["samples_to_threshold"])
        md.append(th_agg.to_markdown(index=False) if _has_tabulate() else th_agg.to_csv(index=False))
    if not diag.empty:
        md.append("\n## Diagnostics\n")
        md.append("See `figures/rollout_diagnostics_*.png` and `raw/rollout_diagnostics.csv`.\n")
    (run_dir / "summary.md").write_text("\n".join(md))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    raw = args.run_dir / "raw"
    fig_dir = args.run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw / "headline_metrics.csv")
    ok = df[df["status"] == "ok"].copy()
    agg = mean_se(
        ok,
        ["env", "task", "graph", "n", "dim", "budget", "method"],
        [
            "return_mean",
            "terminal_error_mean",
            "success_rate",
            "mpc_failure_rate",
            "exploitation_gap_mean",
            "imagined_energy_drift_mean",
            "one_step_rmse",
            "parameters",
        ],
    )
    agg.to_csv(raw / "headline_metrics_aggregated.csv", index=False)
    plot_return(agg, fig_dir)
    diag_path = raw / "rollout_diagnostics.csv"
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    plot_diagnostics(diag, fig_dir)
    threshold_df = samples_to_threshold(ok, args.threshold)
    threshold_df.to_csv(raw / "samples_to_threshold.csv", index=False)
    write_summary(args.run_dir, agg, diag, threshold_df)
    print(f"done: {args.run_dir}")


if __name__ == "__main__":
    main()
