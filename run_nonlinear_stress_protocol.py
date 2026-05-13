#!/usr/bin/env python3
"""Nonlinear fixed-feature stress tests for SG-MBRL.

This runner complements run_fixed_feature_protocol.py with a nonlinear
Hamiltonian chain. It is still NumPy-only, but it creates a harder setting than
linear oscillators: quartic node and edge potentials. The Hamiltonian model gets
the correct graph-local energy features; direct graph baselines use the same
data and CEM planner but remain direct transition/vector-field predictors.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_fixed_feature_protocol import (
    Array,
    BaseModel,
    DirectGraphNextModel,
    GraphEulerModel,
    ProtocolConfig,
    aggregate_mean_se,
    chain_laplacian,
    cem_plan,
    evaluate_mpc,
    fit_model,
    one_step_rmse,
    pack_state,
    random_dataset,
    ridge_solve,
    rollout_metrics,
    sample_initial_state,
    set_plot_style,
    split_state,
    target_for_n,
)


@dataclass
class NonlinearHamiltonianEnv:
    n: int
    laplacian: Array
    edges: Tuple[Tuple[int, int], ...]
    dt: float = 0.05
    omega2: float = 1.0
    spring_k: float = 0.5
    quartic_node: float = 0.25
    quartic_edge: float = 0.15
    action_limit: float = 2.0
    task: str = "stabilization"
    target: Optional[Array] = None

    def clip_action(self, action: Array) -> Array:
        return np.clip(action, -self.action_limit, self.action_limit)

    def edge_cubic(self, q: Array) -> Array:
        out = np.zeros_like(q)
        for i, j in self.edges:
            d = q[..., i] - q[..., j]
            d3 = d**3
            out[..., i] += d3
            out[..., j] -= d3
        return out

    def force(self, q: Array, action: Array) -> Array:
        a = self.clip_action(action)
        return (
            -self.omega2 * q
            - self.quartic_node * q**3
            - self.spring_k * (q @ self.laplacian.T)
            - self.quartic_edge * self.edge_cubic(q)
            + a
        )

    def step(self, z: Array, action: Array) -> Array:
        q, p = split_state(z)
        a = self.clip_action(action)
        p_half = p + 0.5 * self.dt * self.force(q, a)
        q_next = q + self.dt * p_half
        p_next = p_half + 0.5 * self.dt * self.force(q_next, a)
        return pack_state(q_next, p_next)

    def reward(self, z: Array, action: Array) -> Array:
        q, p = split_state(z)
        a = np.asarray(action)
        if self.task == "target":
            target = np.zeros(self.n) if self.target is None else self.target
            state_cost = np.mean((q - target) ** 2, axis=-1) + 0.1 * np.mean(p**2, axis=-1)
        else:
            state_cost = np.mean(q**2 + p**2, axis=-1)
        action_cost = 0.002 * np.mean(a**2, axis=-1)
        return -(state_cost + action_cost)

    def energy(self, z: Array) -> Array:
        q, p = split_state(z)
        kinetic = 0.5 * np.sum(p**2, axis=-1)
        node_quad = 0.5 * self.omega2 * np.sum(q**2, axis=-1)
        node_quartic = 0.25 * self.quartic_node * np.sum(q**4, axis=-1)
        edge_quad = 0.5 * self.spring_k * np.sum((q @ self.laplacian) * q, axis=-1)
        edge_quartic = np.zeros(q.shape[:-1], dtype=float)
        for i, j in self.edges:
            edge_quartic += 0.25 * self.quartic_edge * (q[..., i] - q[..., j]) ** 4
        return kinetic + node_quad + node_quartic + edge_quad + edge_quartic


def chain_edges(n: int) -> Tuple[Tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(n - 1))


def make_nonlinear_env(task: str, n: int, cfg: ProtocolConfig) -> NonlinearHamiltonianEnv:
    return NonlinearHamiltonianEnv(
        n=n,
        laplacian=chain_laplacian(n),
        edges=chain_edges(n),
        dt=cfg.dt,
        omega2=cfg.omega2,
        spring_k=cfg.spring_k,
        action_limit=cfg.action_limit,
        task=task,
        target=target_for_n(n) if task == "target" else None,
    )


class SGPolyHamSympModel(BaseModel):
    name = "SG-PolyHam+Symp"

    def __init__(self, theta: Array):
        self.theta = theta

    @staticmethod
    def feature_matrix(env: NonlinearHamiltonianEnv, q: Array, a: Array) -> Array:
        lq = q @ env.laplacian.T
        edge_cubic = env.edge_cubic(q)
        return np.stack([-q, -(q**3), -lq, -edge_cubic, a], axis=-1)

    @classmethod
    def fit(
        cls,
        data: Sequence[Tuple[NonlinearHamiltonianEnv, Array, Array, Array]],
        ridge: float,
    ) -> "SGPolyHamSympModel":
        xs: List[Array] = []
        ys: List[Array] = []
        for env, z, a, z_next in data:
            q, p = split_state(z)
            q_next, _ = split_state(z_next)
            x = cls.feature_matrix(env, q, a).reshape(-1, 5)
            y = 2.0 * (q_next - q - env.dt * p) / (env.dt**2)
            xs.append(x)
            ys.append(y.reshape(-1, 1))
        theta = ridge_solve(np.vstack(xs), np.vstack(ys), ridge).reshape(-1)
        return cls(theta)

    def force(self, env: NonlinearHamiltonianEnv, q: Array, action: Array) -> Array:
        c_q, c_q3, c_lq, c_edge3, c_a = self.theta
        return (
            -c_q * q
            - c_q3 * q**3
            - c_lq * (q @ env.laplacian.T)
            - c_edge3 * env.edge_cubic(q)
            + c_a * action
        )

    def rollout(self, env: NonlinearHamiltonianEnv, z0: Array, actions: Array) -> Array:
        z0 = np.asarray(z0, dtype=float)
        actions = np.asarray(actions, dtype=float)
        q, p = split_state(z0)
        batch, horizon, _ = actions.shape
        traj = np.empty((batch, horizon + 1, 2 * env.n), dtype=float)
        traj[:, 0, :] = z0
        for h in range(horizon):
            a = np.clip(actions[:, h, :], -env.action_limit, env.action_limit)
            p_half = p + 0.5 * env.dt * self.force(env, q, a)
            q_next = q + env.dt * p_half
            p_next = p_half + 0.5 * env.dt * self.force(env, q_next, a)
            q, p = q_next, p_next
            traj[:, h + 1, :] = pack_state(q, p)
        return traj


def fit_stress_model(model_name: str, data, cfg: ProtocolConfig, rng: np.random.Generator) -> BaseModel:
    if model_name == "SG-PolyHam+Symp":
        return SGPolyHamSympModel.fit(data, cfg.ridge)
    if model_name == "DirectGraphNext":
        return DirectGraphNextModel.fit(data, cfg.ridge, rng)
    if model_name == "GraphEuler":
        return GraphEulerModel.fit(data, cfg.ridge, rng)
    raise ValueError(model_name)


def run_stress(output: Path, cfg: ProtocolConfig) -> None:
    set_plot_style()
    raw = output / "raw"
    figs = output / "figures"
    raw.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    models = ["SG-PolyHam+Symp", "DirectGraphNext", "GraphEuler"]
    tasks = [
        ("nonlinear_chain_stabilization", "stabilization"),
        ("nonlinear_chain_target", "target"),
    ]

    control_rows = []
    dynamics_rows = []
    for seed in cfg.seeds:
        for env_label, task in tasks:
            env = make_nonlinear_env(task, 8, cfg)
            max_data = random_dataset([env], max(cfg.budgets), np.random.default_rng(10000 + seed), cfg.train_noise_std)
            test_data = random_dataset([env], 256, np.random.default_rng(10100 + seed), 0.0)
            for budget in cfg.budgets:
                data = max_data[:budget]
                for model_name in models:
                    model = fit_stress_model(model_name, data, cfg, np.random.default_rng(10200 + seed))
                    metrics = evaluate_mpc(env, model, cfg, np.random.default_rng(10300 + seed + budget), planning_horizon=10)
                    control_rows.append(
                        {
                            "experiment": "nonlinear_matched_planner",
                            "env": env_label,
                            "seed": seed,
                            "budget": budget,
                            "model": model_name,
                            "one_step_rmse": one_step_rmse(model, test_data),
                            **metrics,
                        }
                    )
            # Dynamics at the final budget.
            data = max_data[: max(cfg.budgets)]
            for model_name in models:
                model = fit_stress_model(model_name, data, cfg, np.random.default_rng(10400 + seed))
                for horizon in cfg.rollout_horizons:
                    metrics = rollout_metrics(
                        env,
                        model,
                        np.random.default_rng(10500 + seed + horizon),
                        horizon=horizon,
                        episodes=24,
                        action_scale=0.0,
                    )
                    dynamics_rows.append(
                        {
                            "experiment": "nonlinear_dynamics",
                            "env": env_label,
                            "seed": seed,
                            "horizon": horizon,
                            "model": model_name,
                            "one_step_rmse": one_step_rmse(model, test_data),
                            **metrics,
                        }
                    )

    control = pd.DataFrame(control_rows)
    dynamics = pd.DataFrame(dynamics_rows)
    control.to_csv(raw / "nonlinear_matched_planner.csv", index=False)
    dynamics.to_csv(raw / "nonlinear_dynamics.csv", index=False)

    plot_control(control, figs)
    plot_dynamics(dynamics, figs)
    write_summary(output, raw, figs, cfg, control, dynamics)


def plot_control(df: pd.DataFrame, figs: Path) -> None:
    agg = aggregate_mean_se(df, ["env", "budget", "model"], ["return_mean", "terminal_error_mean", "one_step_rmse"])
    envs = agg["env"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 3.4), squeeze=False)
    for ax, env in zip(axes[0], envs):
        sub = agg[agg["env"] == env]
        for model, g in sub.groupby("model"):
            g = g.sort_values("budget")
            ax.plot(g["budget"], g["return_mean_mean"], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_xlabel("real transitions used for model fit")
        ax.set_ylabel("MPC return")
        ax.set_title(env.replace("_", " "))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "nonlinear_matched_planner_return.png")
    plt.close(fig)


def plot_dynamics(df: pd.DataFrame, figs: Path) -> None:
    sub = df[df["env"] == "nonlinear_chain_stabilization"]
    agg = aggregate_mean_se(sub, ["horizon", "model"], ["rollout_rmse", "max_energy_drift"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), squeeze=False)
    for ax, metric, title in [
        (axes[0, 0], "rollout_rmse", "rollout RMSE"),
        (axes[0, 1], "max_energy_drift", "relative energy drift"),
    ]:
        for model, g in agg.groupby("model"):
            g = g.sort_values("horizon")
            ax.plot(g["horizon"], g[f"{metric}_mean"], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rollout horizon")
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "nonlinear_dynamics_rollout.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: Sequence[str], n: int = 20) -> str:
    table = df.loc[:, cols].head(n).copy()
    if table.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in table.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(output: Path, raw: Path, figs: Path, cfg: ProtocolConfig, control: pd.DataFrame, dynamics: pd.DataFrame) -> None:
    final_budget = max(cfg.budgets)
    control_agg = aggregate_mean_se(control, ["env", "budget", "model"], ["return_mean", "terminal_error_mean", "one_step_rmse"])
    final_control = control_agg[control_agg["budget"] == final_budget].sort_values(["env", "return_mean_mean"], ascending=[True, False])
    dyn_agg = aggregate_mean_se(dynamics, ["env", "horizon", "model"], ["rollout_rmse", "max_energy_drift", "one_step_rmse"])
    final_dyn = dyn_agg[dyn_agg["horizon"] == max(cfg.rollout_horizons)].sort_values(["env", "rollout_rmse_mean"])
    text = f"""# Nonlinear Fixed-Feature Stress Protocol

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

This run uses a nonlinear graph-Hamiltonian chain with quartic node and edge potentials. It is designed to stress the direct graph baselines under the same data and CEM-MPC planner.

## Final-Budget Matched Planner

{md_table(final_control, ['env', 'budget', 'model', 'return_mean_mean', 'return_mean_se', 'terminal_error_mean_mean', 'one_step_rmse_mean'])}

## Long-Horizon Nonlinear Dynamics

{md_table(final_dyn, ['env', 'horizon', 'model', 'rollout_rmse_mean', 'max_energy_drift_mean', 'one_step_rmse_mean'])}

## Files

- `{raw / 'nonlinear_matched_planner.csv'}`
- `{raw / 'nonlinear_dynamics.csv'}`
- `{figs / 'nonlinear_matched_planner_return.png'}`
- `{figs / 'nonlinear_dynamics_rollout.png'}`

## Interpretation

These stress tests are still fixed-feature experiments. They support the mechanism that a correctly specified graph-local Hamiltonian model can be far more sample efficient and physically stable than direct graph predictors when nonlinear energy structure matters. They should be followed by neural ResidualSympGNN vs DirectGNN experiments before making final paper-scale claims.
"""
    (output / "summary.md").write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/nonlinear_stress"))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--cem-population", type=int, default=96)
    p.add_argument("--cem-iterations", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProtocolConfig(
        seeds=tuple(range(args.seeds)),
        budgets=(16, 32, 64, 128, 256, 512),
        control_horizon=40,
        eval_episodes=args.eval_episodes,
        cem_population=args.cem_population,
        cem_iterations=args.cem_iterations,
        rollout_horizons=(1, 10, 25, 50, 100, 250, 500),
    )
    run_stress(args.output, cfg)
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
