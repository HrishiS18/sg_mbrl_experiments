#!/usr/bin/env python3
"""Fixed-feature SG-MBRL experiment protocol.

This is a CPU-friendly protocol runner for the paper's core experimental claims.
It does not implement the final neural ResidualSympGNN/PETS/MBPO/model-free suite.
Instead it runs a reproducible fixed-feature benchmark that isolates the central
mechanism: graph-local Hamiltonian + symplectic rollout vs graph-local direct
prediction under matched data and matched CEM-MPC.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class ProtocolConfig:
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    budgets: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    control_horizon: int = 40
    eval_episodes: int = 12
    train_rollout_horizon: int = 100
    cem_population: int = 128
    cem_iterations: int = 5
    cem_elite_frac: float = 0.12
    planning_horizons: Tuple[int, ...] = (1, 5, 10, 20, 30)
    graph_transfer_train_sizes: Tuple[int, ...] = (4, 8, 16)
    graph_transfer_test_sizes: Tuple[int, ...] = (32, 64, 128, 256)
    rollout_horizons: Tuple[int, ...] = (1, 10, 25, 50, 100, 250, 500)
    dt: float = 0.05
    omega2: float = 1.0
    spring_k: float = 0.5
    action_limit: float = 2.0
    ridge: float = 1e-5
    ensemble_size: int = 5
    train_noise_std: float = 0.0


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def chain_laplacian(n: int) -> Array:
    l = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        l[i, i] += 1
        l[i + 1, i + 1] += 1
        l[i, i + 1] -= 1
        l[i + 1, i] -= 1
    return l


def cycle_laplacian(n: int) -> Array:
    l = chain_laplacian(n)
    if n > 2:
        l[0, 0] += 1
        l[-1, -1] += 1
        l[0, -1] -= 1
        l[-1, 0] -= 1
    return l


def star_laplacian(n: int) -> Array:
    l = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        l[0, 0] += 1
        l[i, i] += 1
        l[0, i] -= 1
        l[i, 0] -= 1
    return l


def grid_laplacian(rows: int, cols: int) -> Array:
    n = rows * cols
    l = np.zeros((n, n), dtype=float)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if rr < rows and cc < cols:
                    j = idx(rr, cc)
                    l[i, i] += 1
                    l[j, j] += 1
                    l[i, j] -= 1
                    l[j, i] -= 1
    return l


def er_laplacian(n: int, p: float, rng: np.random.Generator) -> Array:
    adj = rng.random((n, n)) < p
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    # Keep the graph from becoming empty/disconnected in tiny samples.
    for i in range(n - 1):
        adj[i, i + 1] = True
        adj[i + 1, i] = True
    deg = np.diag(adj.sum(axis=1))
    return deg - adj.astype(float)


def graph_kind_laplacian(kind: str, n: int, rng: np.random.Generator) -> Array:
    if kind == "chain":
        return chain_laplacian(n)
    if kind == "cycle":
        return cycle_laplacian(n)
    if kind == "star":
        return star_laplacian(n)
    if kind == "grid":
        side = int(round(math.sqrt(n)))
        if side * side == n:
            return grid_laplacian(side, side)
        return grid_laplacian(max(1, n // 4), 4)
    if kind == "er":
        return er_laplacian(n, min(0.25, 3.0 / max(n, 1)), rng)
    raise ValueError(f"unknown graph kind {kind}")


@dataclass
class LinearHamiltonianEnv:
    n: int
    laplacian: Array
    dt: float = 0.05
    omega2: float = 1.0
    spring_k: float = 0.5
    action_limit: float = 2.0
    task: str = "stabilization"
    target: Optional[Array] = None

    def clip_action(self, action: Array) -> Array:
        return np.clip(action, -self.action_limit, self.action_limit)

    def force(self, q: Array, action: Array) -> Array:
        a = self.clip_action(action)
        return -self.omega2 * q - self.spring_k * (q @ self.laplacian.T) + a

    def step(self, z: Array, action: Array) -> Array:
        q, p = split_state(z)
        a = self.clip_action(action)
        p_half = p + 0.5 * self.dt * self.force(q, a)
        q_next = q + self.dt * p_half
        p_next = p_half + 0.5 * self.dt * self.force(q_next, a)
        return pack_state(q_next, p_next)

    def step_batch(self, z: Array, action: Array) -> Array:
        q, p = split_state(z)
        a = np.clip(action, -self.action_limit, self.action_limit)
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
        node = 0.5 * self.omega2 * np.sum(q**2, axis=-1)
        edge = 0.5 * self.spring_k * np.sum((q @ self.laplacian) * q, axis=-1)
        return kinetic + node + edge


def split_state(z: Array) -> Tuple[Array, Array]:
    z = np.asarray(z)
    n2 = z.shape[-1]
    n = n2 // 2
    return z[..., :n], z[..., n:]


def pack_state(q: Array, p: Array) -> Array:
    return np.concatenate([q, p], axis=-1)


def sample_initial_state(env: LinearHamiltonianEnv, rng: np.random.Generator) -> Array:
    if env.task == "target":
        q = rng.normal(0.0, 0.7, env.n)
        p = rng.normal(0.0, 0.25, env.n)
    else:
        q = rng.normal(0.0, 1.0, env.n)
        p = rng.normal(0.0, 0.7, env.n)
    return pack_state(q, p)


def target_for_n(n: int) -> Array:
    x = np.linspace(0.0, 1.0, n)
    return 0.8 * np.sin(2 * np.pi * x)


def make_env(task: str, n: int, kind: str, cfg: ProtocolConfig, seed: int = 0) -> LinearHamiltonianEnv:
    rng = np.random.default_rng(seed)
    target = target_for_n(n) if task == "target" else None
    return LinearHamiltonianEnv(
        n=n,
        laplacian=graph_kind_laplacian(kind, n, rng),
        dt=cfg.dt,
        omega2=cfg.omega2,
        spring_k=cfg.spring_k,
        action_limit=cfg.action_limit,
        task=task,
        target=target,
    )


def random_dataset(
    envs: Sequence[LinearHamiltonianEnv],
    budget: int,
    rng: np.random.Generator,
    noise_std: float = 0.0,
) -> List[Tuple[LinearHamiltonianEnv, Array, Array, Array]]:
    data: List[Tuple[LinearHamiltonianEnv, Array, Array, Array]] = []
    for _ in range(budget):
        env = envs[int(rng.integers(0, len(envs)))]
        z = sample_initial_state(env, rng)
        a = rng.uniform(-env.action_limit, env.action_limit, env.n)
        z_next = env.step(z, a)
        if noise_std > 0:
            z_next = z_next + rng.normal(0.0, noise_std, z_next.shape)
        data.append((env, z, a, z_next))
    return data


def ridge_solve(x: Array, y: Array, ridge: float) -> Array:
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0])
    return np.linalg.solve(xtx + reg, x.T @ y)


def bootstrap_indices(n: int, rng: np.random.Generator) -> Array:
    return rng.integers(0, n, n)


class BaseModel:
    name = "base"

    def rollout(self, env: LinearHamiltonianEnv, z0: Array, actions: Array) -> Array:
        raise NotImplementedError

    def step(self, env: LinearHamiltonianEnv, z: Array, action: Array) -> Array:
        return self.rollout(env, z[None, :], action[None, None, :])[0, -1]

    def energy_drift(self, env: LinearHamiltonianEnv, z0: Array, actions: Array) -> float:
        traj = self.rollout(env, z0[None, :], actions[None, :, :])[0]
        e = env.energy(traj)
        return float(np.max(np.abs(e - e[0])))


class SGHamSympModel(BaseModel):
    name = "SG-Ham+Symp"

    def __init__(self, theta: Array, ensemble: Optional[Array] = None):
        self.theta = theta
        self.ensemble = ensemble

    @staticmethod
    def features(env: LinearHamiltonianEnv, q: Array, a: Array, include_edges: bool = True) -> Array:
        lq = q @ env.laplacian.T
        if include_edges:
            return np.stack([-q, -lq, a], axis=-1)
        return np.stack([-q, a], axis=-1)

    @classmethod
    def fit(
        cls,
        data: Sequence[Tuple[LinearHamiltonianEnv, Array, Array, Array]],
        ridge: float,
        rng: np.random.Generator,
        ensemble_size: int = 1,
        include_edges: bool = True,
    ) -> "SGHamSympModel":
        x_rows: List[Array] = []
        y_rows: List[Array] = []
        for env, z, a, z_next in data:
            q, p = split_state(z)
            q_next, _ = split_state(z_next)
            x = cls.features(env, q, a, include_edges=include_edges)
            # The simulator uses velocity Verlet:
            # q_next = q + dt * p + 0.5 * dt^2 * F(q, a).
            # This gives a low-noise force target that is consistent with the
            # symplectic transition mechanism, unlike a crude p finite difference.
            y = 2.0 * (q_next - q - env.dt * p) / (env.dt**2)
            x_rows.append(x.reshape(-1, x.shape[-1]))
            y_rows.append(y.reshape(-1, 1))
        x_all = np.vstack(x_rows)
        y_all = np.vstack(y_rows)
        theta = ridge_solve(x_all, y_all, ridge).reshape(-1)
        ens = []
        for _ in range(max(0, ensemble_size - 1)):
            idx = bootstrap_indices(len(x_all), rng)
            ens.append(ridge_solve(x_all[idx], y_all[idx], ridge).reshape(-1))
        if ensemble_size > 1:
            ens.append(theta)
        ensemble = np.stack(ens) if ens else None
        return cls(theta=theta, ensemble=ensemble)

    def force(self, env: LinearHamiltonianEnv, q: Array, action: Array, theta: Optional[Array] = None) -> Array:
        theta = self.theta if theta is None else theta
        if theta.shape[0] == 3:
            c_q, c_lq, c_a = theta
            return -c_q * q - c_lq * (q @ env.laplacian.T) + c_a * action
        c_q, c_a = theta
        return -c_q * q + c_a * action

    def rollout(self, env: LinearHamiltonianEnv, z0: Array, actions: Array) -> Array:
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


class SympNoGraphModel(SGHamSympModel):
    name = "SympNoGraph"


class DirectGraphNextModel(BaseModel):
    name = "DirectGraphNext"

    def __init__(self, w: Array):
        self.w = w

    @staticmethod
    def feature_matrix(env: LinearHamiltonianEnv, z: Array, a: Array) -> Array:
        q, p = split_state(z)
        lq = q @ env.laplacian.T
        lp = p @ env.laplacian.T
        ones = np.ones_like(q)
        return np.stack([ones, q, p, lq, lp, a], axis=-1)

    @classmethod
    def fit(
        cls,
        data: Sequence[Tuple[LinearHamiltonianEnv, Array, Array, Array]],
        ridge: float,
        rng: Optional[np.random.Generator] = None,
    ) -> "DirectGraphNextModel":
        x_rows: List[Array] = []
        y_rows: List[Array] = []
        for env, z, a, z_next in data:
            x = cls.feature_matrix(env, z, a).reshape(-1, 6)
            q_next, p_next = split_state(z_next)
            y = np.stack([q_next, p_next], axis=-1).reshape(-1, 2)
            x_rows.append(x)
            y_rows.append(y)
        x_all = np.vstack(x_rows)
        y_all = np.vstack(y_rows)
        return cls(ridge_solve(x_all, y_all, ridge))

    def rollout(self, env: LinearHamiltonianEnv, z0: Array, actions: Array) -> Array:
        z0 = np.asarray(z0, dtype=float)
        actions = np.asarray(actions, dtype=float)
        q, p = split_state(z0)
        batch, horizon, _ = actions.shape
        traj = np.empty((batch, horizon + 1, 2 * env.n), dtype=float)
        traj[:, 0, :] = z0
        for h in range(horizon):
            z = pack_state(q, p)
            x = self.feature_matrix(env, z, actions[:, h, :])
            y = x @ self.w
            q, p = y[..., 0], y[..., 1]
            traj[:, h + 1, :] = pack_state(q, p)
        return traj


class GraphEulerModel(BaseModel):
    name = "GraphEuler"

    def __init__(self, w: Array):
        self.w = w

    @classmethod
    def fit(
        cls,
        data: Sequence[Tuple[LinearHamiltonianEnv, Array, Array, Array]],
        ridge: float,
        rng: Optional[np.random.Generator] = None,
    ) -> "GraphEulerModel":
        x_rows: List[Array] = []
        y_rows: List[Array] = []
        for env, z, a, z_next in data:
            x = DirectGraphNextModel.feature_matrix(env, z, a).reshape(-1, 6)
            q, p = split_state(z)
            q_next, p_next = split_state(z_next)
            y = np.stack([(q_next - q) / env.dt, (p_next - p) / env.dt], axis=-1).reshape(-1, 2)
            x_rows.append(x)
            y_rows.append(y)
        return cls(ridge_solve(np.vstack(x_rows), np.vstack(y_rows), ridge))

    def rollout(self, env: LinearHamiltonianEnv, z0: Array, actions: Array) -> Array:
        z0 = np.asarray(z0, dtype=float)
        actions = np.asarray(actions, dtype=float)
        q, p = split_state(z0)
        batch, horizon, _ = actions.shape
        traj = np.empty((batch, horizon + 1, 2 * env.n), dtype=float)
        traj[:, 0, :] = z0
        for h in range(horizon):
            z = pack_state(q, p)
            x = DirectGraphNextModel.feature_matrix(env, z, actions[:, h, :])
            y = x @ self.w
            q = q + env.dt * y[..., 0]
            p = p + env.dt * y[..., 1]
            traj[:, h + 1, :] = pack_state(q, p)
        return traj


class DenseLinearNextModel(BaseModel):
    name = "DenseLinearNext"

    def __init__(self, w: Array, n: int):
        self.w = w
        self.n = n

    @classmethod
    def fit(
        cls,
        data: Sequence[Tuple[LinearHamiltonianEnv, Array, Array, Array]],
        ridge: float,
        rng: Optional[np.random.Generator] = None,
    ) -> "DenseLinearNextModel":
        n = data[0][0].n
        filtered = [d for d in data if d[0].n == n]
        x_rows = []
        y_rows = []
        for env, z, a, z_next in filtered:
            if env.n != n:
                continue
            x_rows.append(np.concatenate([[1.0], z, a]))
            y_rows.append(z_next)
        return cls(ridge_solve(np.vstack(x_rows), np.vstack(y_rows), ridge), n=n)

    def rollout(self, env: LinearHamiltonianEnv, z0: Array, actions: Array) -> Array:
        if env.n != self.n:
            raise ValueError("DenseLinearNextModel cannot transfer across graph sizes")
        z = np.asarray(z0, dtype=float)
        actions = np.asarray(actions, dtype=float)
        batch, horizon, _ = actions.shape
        traj = np.empty((batch, horizon + 1, 2 * env.n), dtype=float)
        traj[:, 0, :] = z
        for h in range(horizon):
            x = np.concatenate([np.ones((batch, 1)), z, actions[:, h, :]], axis=-1)
            z = x @ self.w
            traj[:, h + 1, :] = z
        return traj


def fit_model(
    model_name: str,
    data: Sequence[Tuple[LinearHamiltonianEnv, Array, Array, Array]],
    cfg: ProtocolConfig,
    rng: np.random.Generator,
) -> BaseModel:
    if model_name == "SG-Ham+Symp":
        return SGHamSympModel.fit(data, cfg.ridge, rng, cfg.ensemble_size, include_edges=True)
    if model_name == "DirectGraphNext":
        return DirectGraphNextModel.fit(data, cfg.ridge, rng)
    if model_name == "GraphEuler":
        return GraphEulerModel.fit(data, cfg.ridge, rng)
    if model_name == "SympNoGraph":
        m = SGHamSympModel.fit(data, cfg.ridge, rng, 1, include_edges=False)
        return SympNoGraphModel(theta=m.theta, ensemble=m.ensemble)
    if model_name == "DenseLinearNext":
        return DenseLinearNextModel.fit(data, cfg.ridge, rng)
    raise ValueError(f"unknown model {model_name}")


def evaluate_action_sequences(
    env: LinearHamiltonianEnv,
    model: BaseModel,
    z0: Array,
    actions: Array,
    terminal_weight: float = 1.0,
) -> Array:
    traj = model.rollout(env, np.repeat(z0[None, :], len(actions), axis=0), actions)
    rewards = []
    for h in range(actions.shape[1]):
        rewards.append(env.reward(traj[:, h, :], actions[:, h, :]))
    total = np.sum(np.stack(rewards, axis=0), axis=0)
    total += terminal_weight * env.reward(traj[:, -1, :], np.zeros((len(actions), env.n)))
    return total


def cem_plan(
    env: LinearHamiltonianEnv,
    model: BaseModel,
    z0: Array,
    horizon: int,
    population: int,
    iterations: int,
    elite_frac: float,
    rng: np.random.Generator,
    initial_mean: Optional[Array] = None,
) -> Tuple[Array, float]:
    mean = np.zeros((horizon, env.n), dtype=float) if initial_mean is None else initial_mean.copy()
    std = np.ones((horizon, env.n), dtype=float) * env.action_limit
    n_elite = max(2, int(population * elite_frac))
    best_actions = mean
    best_score = -np.inf
    for _ in range(iterations):
        samples = rng.normal(mean[None, :, :], std[None, :, :], size=(population, horizon, env.n))
        samples = np.clip(samples, -env.action_limit, env.action_limit)
        scores = evaluate_action_sequences(env, model, z0, samples)
        elite_idx = np.argpartition(scores, -n_elite)[-n_elite:]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.05)
        local_best = int(np.argmax(scores))
        if scores[local_best] > best_score:
            best_score = float(scores[local_best])
            best_actions = samples[local_best]
    return best_actions, best_score


def evaluate_mpc(
    env: LinearHamiltonianEnv,
    model: BaseModel,
    cfg: ProtocolConfig,
    rng: np.random.Generator,
    planning_horizon: int,
    episodes: Optional[int] = None,
) -> Dict[str, float]:
    episodes = cfg.eval_episodes if episodes is None else episodes
    returns: List[float] = []
    terminal_errors: List[float] = []
    exploitation_gaps: List[float] = []
    imagined_energy_drifts: List[float] = []
    divergence_count = 0
    for _ in range(episodes):
        z = sample_initial_state(env, rng)
        total = 0.0
        first_plan = None
        first_pred = None
        first_true = None
        for t in range(cfg.control_horizon):
            plan, pred_score = cem_plan(
                env,
                model,
                z,
                planning_horizon,
                cfg.cem_population,
                cfg.cem_iterations,
                cfg.cem_elite_frac,
                rng,
            )
            if t == 0:
                first_plan = plan
                first_pred = pred_score
                imagined_energy_drifts.append(model.energy_drift(env, z, plan))
                z_true = z.copy()
                true_total = 0.0
                for h in range(planning_horizon):
                    true_total += float(env.reward(z_true[None, :], plan[h][None, :])[0])
                    z_true = env.step(z_true, plan[h])
                true_total += float(env.reward(z_true[None, :], np.zeros((1, env.n)))[0])
                first_true = true_total
            a = plan[0]
            total += float(env.reward(z[None, :], a[None, :])[0])
            z = env.step(z, a)
            if not np.all(np.isfinite(z)) or np.linalg.norm(z) > 50:
                divergence_count += 1
                break
        q, p = split_state(z)
        if env.task == "target":
            target = np.zeros(env.n) if env.target is None else env.target
            terminal_errors.append(float(np.mean((q - target) ** 2 + 0.1 * p**2)))
        else:
            terminal_errors.append(float(np.mean(q**2 + p**2)))
        returns.append(total)
        if first_pred is not None and first_true is not None:
            exploitation_gaps.append(float(first_pred - first_true))
        _ = first_plan
    return {
        "return_mean": float(np.mean(returns)),
        "return_se": float(np.std(returns, ddof=1) / math.sqrt(max(len(returns), 1))) if len(returns) > 1 else 0.0,
        "terminal_error_mean": float(np.mean(terminal_errors)),
        "terminal_error_se": float(np.std(terminal_errors, ddof=1) / math.sqrt(max(len(terminal_errors), 1)))
        if len(terminal_errors) > 1
        else 0.0,
        "exploitation_gap_mean": float(np.mean(exploitation_gaps)) if exploitation_gaps else float("nan"),
        "imagined_energy_drift_mean": float(np.mean(imagined_energy_drifts)) if imagined_energy_drifts else float("nan"),
        "divergence_rate": float(divergence_count / max(episodes, 1)),
    }


def one_step_rmse(
    model: BaseModel,
    data: Sequence[Tuple[LinearHamiltonianEnv, Array, Array, Array]],
) -> float:
    err = []
    for env, z, a, z_next in data:
        pred = model.step(env, z, a)
        err.append(np.mean((pred - z_next) ** 2))
    return float(np.sqrt(np.mean(err)))


def rollout_metrics(
    env: LinearHamiltonianEnv,
    model: BaseModel,
    rng: np.random.Generator,
    horizon: int,
    episodes: int = 32,
    action_scale: float = 0.0,
) -> Dict[str, float]:
    rmses = []
    max_drifts = []
    final_drifts = []
    div = 0
    for _ in range(episodes):
        z0 = sample_initial_state(env, rng)
        actions = rng.uniform(
            -action_scale * env.action_limit,
            action_scale * env.action_limit,
            size=(horizon, env.n),
        )
        z_true = z0.copy()
        true_traj = [z_true]
        for h in range(horizon):
            z_true = env.step(z_true, actions[h])
            true_traj.append(z_true)
        true_traj_arr = np.stack(true_traj)
        pred_traj = model.rollout(env, z0[None, :], actions[None, :, :])[0]
        if (not np.all(np.isfinite(pred_traj))) or np.max(np.linalg.norm(pred_traj, axis=-1)) > 100:
            div += 1
        rmses.append(float(np.sqrt(np.mean((pred_traj - true_traj_arr) ** 2))))
        e = env.energy(pred_traj)
        max_drifts.append(float(np.max(np.abs(e - e[0])) / max(abs(e[0]), 1e-8)))
        final_drifts.append(float(abs(e[-1] - e[0]) / max(abs(e[0]), 1e-8)))
    return {
        "rollout_rmse": float(np.mean(rmses)),
        "max_energy_drift": float(np.mean(max_drifts)),
        "final_energy_drift": float(np.mean(final_drifts)),
        "divergence_rate": float(div / episodes),
    }


def finite_difference_jacobian_step(
    model: BaseModel,
    env: LinearHamiltonianEnv,
    z: Array,
    action: Array,
    eps: float = 1e-5,
) -> Array:
    d = z.shape[0]
    base_actions = action[None, None, :]
    jac = np.zeros((d, d), dtype=float)
    for j in range(d):
        dz = np.zeros_like(z)
        dz[j] = eps
        plus = model.rollout(env, (z + dz)[None, :], base_actions)[0, -1]
        minus = model.rollout(env, (z - dz)[None, :], base_actions)[0, -1]
        jac[:, j] = (plus - minus) / (2 * eps)
    return jac


def symplecticity_defect(
    model: BaseModel,
    env: LinearHamiltonianEnv,
    rng: np.random.Generator,
    samples: int = 12,
) -> float:
    n = env.n
    jmat = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    defects = []
    for _ in range(samples):
        z = sample_initial_state(env, rng)
        a = rng.uniform(-env.action_limit, env.action_limit, env.n)
        jac = finite_difference_jacobian_step(model, env, z, a)
        defects.append(float(np.linalg.norm(jac.T @ jmat @ jac - jmat, ord="fro")))
    return float(np.mean(defects))


def run_matched_planner(cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    rows = []
    tasks = [
        ("coupled_oscillator_stabilization", "stabilization", "chain", 8),
        ("spring_chain_target", "target", "chain", 8),
    ]
    model_names = ["SG-Ham+Symp", "DirectGraphNext"]
    for seed in cfg.seeds:
        train_rng = np.random.default_rng(1000 + seed)
        for env_label, task, kind, n in tasks:
            env = make_env(task, n, kind, cfg, seed=seed)
            max_data = random_dataset([env], max(cfg.budgets), train_rng, cfg.train_noise_std)
            for budget in cfg.budgets:
                data = max_data[:budget]
                for model_name in model_names:
                    fit_rng = np.random.default_rng(2000 + seed)
                    model = fit_model(model_name, data, cfg, fit_rng)
                    eval_rng = np.random.default_rng(3000 + 31 * seed + budget)
                    metrics = evaluate_mpc(env, model, cfg, eval_rng, planning_horizon=10)
                    test_data = random_dataset([env], 128, np.random.default_rng(4000 + seed), 0.0)
                    rows.append(
                        {
                            "experiment": "matched_planner",
                            "env": env_label,
                            "seed": seed,
                            "budget": budget,
                            "model": model_name,
                            "one_step_rmse": one_step_rmse(model, test_data),
                            **metrics,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "matched_planner.csv", index=False)
    return df


def run_graph_size_transfer(cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    rows = []
    model_names = ["SG-Ham+Symp", "DirectGraphNext", "GraphEuler", "SympNoGraph"]
    for seed in cfg.seeds:
        rng = np.random.default_rng(5000 + seed)
        train_envs = [make_env("stabilization", n, "chain", cfg, seed=seed) for n in cfg.graph_transfer_train_sizes]
        data = random_dataset(train_envs, 1024, rng, cfg.train_noise_std)
        for model_name in model_names:
            model = fit_model(model_name, data, cfg, np.random.default_rng(5100 + seed))
            for test_n in cfg.graph_transfer_test_sizes:
                env = make_env("stabilization", test_n, "chain", cfg, seed=seed)
                metrics = rollout_metrics(env, model, np.random.default_rng(5200 + seed + test_n), horizon=100, episodes=20)
                rows.append(
                    {
                        "experiment": "graph_size_transfer",
                        "topology": "chain",
                        "seed": seed,
                        "train_sizes": "+".join(map(str, cfg.graph_transfer_train_sizes)),
                        "test_size": test_n,
                        "model": model_name,
                        **metrics,
                    }
                )
        # Grid transfer is run separately for models that can ingest graph-local features.
        grid_train_envs = [make_env("stabilization", 16, "grid", cfg, seed=seed)]
        grid_data = random_dataset(grid_train_envs, 1024, np.random.default_rng(5300 + seed), cfg.train_noise_std)
        for model_name in model_names:
            model = fit_model(model_name, grid_data, cfg, np.random.default_rng(5400 + seed))
            for side in (8, 16):
                env = LinearHamiltonianEnv(
                    n=side * side,
                    laplacian=grid_laplacian(side, side),
                    dt=cfg.dt,
                    omega2=cfg.omega2,
                    spring_k=cfg.spring_k,
                    action_limit=cfg.action_limit,
                    task="stabilization",
                )
                metrics = rollout_metrics(env, model, np.random.default_rng(5500 + seed + side), horizon=100, episodes=12)
                rows.append(
                    {
                        "experiment": "graph_size_transfer",
                        "topology": "grid",
                        "seed": seed,
                        "train_sizes": "4x4",
                        "test_size": env.n,
                        "model": model_name,
                        **metrics,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "graph_size_transfer.csv", index=False)
    return df


def run_topology_transfer(cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    rows = []
    model_names = ["SG-Ham+Symp", "DirectGraphNext", "GraphEuler", "SympNoGraph"]
    topologies = ["chain", "cycle", "star", "grid", "er"]
    for seed in cfg.seeds:
        train_env = make_env("stabilization", 16, "chain", cfg, seed=seed)
        data = random_dataset([train_env], 1024, np.random.default_rng(6000 + seed), cfg.train_noise_std)
        for model_name in model_names:
            model = fit_model(model_name, data, cfg, np.random.default_rng(6100 + seed))
            for topo in topologies:
                env = make_env("stabilization", 16, topo, cfg, seed=6200 + seed)
                metrics = rollout_metrics(env, model, np.random.default_rng(6300 + seed), horizon=100, episodes=24)
                rows.append(
                    {
                        "experiment": "topology_transfer",
                        "train_topology": "chain",
                        "test_topology": topo,
                        "seed": seed,
                        "model": model_name,
                        **metrics,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "topology_transfer.csv", index=False)
    return df


def run_dynamics_rollout(cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    rows = []
    model_names = ["SG-Ham+Symp", "DirectGraphNext", "GraphEuler", "SympNoGraph", "DenseLinearNext"]
    env_specs = [
        ("harmonic_oscillator", "stabilization", "chain", 1),
        ("spring_mass_chain", "stabilization", "chain", 8),
        ("oscillator_graph_cycle", "stabilization", "cycle", 8),
    ]
    for seed in cfg.seeds:
        for env_label, task, kind, n in env_specs:
            env = make_env(task, n, kind, cfg, seed=seed)
            data = random_dataset([env], 512, np.random.default_rng(7000 + seed), cfg.train_noise_std)
            test_data = random_dataset([env], 256, np.random.default_rng(7100 + seed), 0.0)
            for model_name in model_names:
                if model_name == "DenseLinearNext" and n > 16:
                    continue
                model = fit_model(model_name, data, cfg, np.random.default_rng(7200 + seed))
                defect = symplecticity_defect(model, env, np.random.default_rng(7300 + seed), samples=8)
                one_step = one_step_rmse(model, test_data)
                for horizon in cfg.rollout_horizons:
                    metrics = rollout_metrics(
                        env,
                        model,
                        np.random.default_rng(7400 + seed + horizon),
                        horizon=horizon,
                        episodes=24,
                        action_scale=0.0,
                    )
                    rows.append(
                        {
                            "experiment": "dynamics_rollout",
                            "env": env_label,
                            "seed": seed,
                            "horizon": horizon,
                            "model": model_name,
                            "one_step_rmse": one_step,
                            "symplecticity_defect": defect,
                            **metrics,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dynamics_rollout.csv", index=False)
    return df


def run_ablation(cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    rows = []
    model_names = ["DenseLinearNext", "DirectGraphNext", "SympNoGraph", "GraphEuler", "SG-Ham+Symp"]
    tasks = [
        ("coupled_oscillator_stabilization", "stabilization", "chain", 8),
        ("spring_chain_target", "target", "chain", 8),
    ]
    budget = 512
    for seed in cfg.seeds:
        for env_label, task, kind, n in tasks:
            env = make_env(task, n, kind, cfg, seed=seed)
            data = random_dataset([env], budget, np.random.default_rng(8000 + seed), cfg.train_noise_std)
            for model_name in model_names:
                model = fit_model(model_name, data, cfg, np.random.default_rng(8100 + seed))
                metrics = evaluate_mpc(env, model, cfg, np.random.default_rng(8200 + seed), planning_horizon=10)
                roll = rollout_metrics(env, model, np.random.default_rng(8300 + seed), horizon=100, episodes=20)
                rows.append(
                    {
                        "experiment": "ablation",
                        "env": env_label,
                        "seed": seed,
                        "budget": budget,
                        "model": model_name,
                        **metrics,
                        "rollout_rmse": roll["rollout_rmse"],
                        "max_energy_drift": roll["max_energy_drift"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation.csv", index=False)
    return df


def run_planning_horizon(cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    rows = []
    model_names = ["SG-Ham+Symp", "DirectGraphNext", "GraphEuler"]
    env = make_env("stabilization", 8, "chain", cfg, seed=0)
    budget = 512
    for seed in cfg.seeds:
        data = random_dataset([env], budget, np.random.default_rng(9000 + seed), cfg.train_noise_std)
        for model_name in model_names:
            model = fit_model(model_name, data, cfg, np.random.default_rng(9100 + seed))
            for horizon in cfg.planning_horizons:
                metrics = evaluate_mpc(env, model, cfg, np.random.default_rng(9200 + seed + horizon), planning_horizon=horizon)
                rows.append(
                    {
                        "experiment": "planning_horizon",
                        "env": "coupled_oscillator_stabilization",
                        "seed": seed,
                        "budget": budget,
                        "model": model_name,
                        "planning_horizon": horizon,
                        **metrics,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "planning_horizon.csv", index=False)
    return df


def aggregate_mean_se(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_seeds"] = g["seed"].nunique() if "seed" in g.columns else len(g)
        for m in metric_cols:
            vals = g[m].dropna().to_numpy()
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{m}_se"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_matched(df: pd.DataFrame, fig_dir: Path) -> None:
    agg = aggregate_mean_se(df, ["env", "budget", "model"], ["return_mean", "terminal_error_mean", "exploitation_gap_mean"])
    envs = agg["env"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 3.4), squeeze=False)
    for ax, env in zip(axes[0], envs):
        sub = agg[agg["env"] == env]
        for model, g in sub.groupby("model"):
            g = g.sort_values("budget")
            ax.plot(g["budget"], g["return_mean_mean"], marker="o", label=model)
            ax.fill_between(
                g["budget"].to_numpy(float),
                (g["return_mean_mean"] - g["return_mean_se"]).to_numpy(float),
                (g["return_mean_mean"] + g["return_mean_se"]).to_numpy(float),
                alpha=0.16,
            )
        ax.set_xscale("log")
        ax.set_title(env.replace("_", " "))
        ax.set_xlabel("real transitions used for model fit")
        ax.set_ylabel("MPC evaluation return")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "matched_planner_return_vs_budget.png")
    plt.close(fig)


def plot_graph_transfer(df: pd.DataFrame, fig_dir: Path) -> None:
    agg = aggregate_mean_se(df, ["topology", "test_size", "model"], ["rollout_rmse", "max_energy_drift"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), squeeze=False)
    for ax, metric, title in [
        (axes[0, 0], "rollout_rmse", "rollout RMSE"),
        (axes[0, 1], "max_energy_drift", "relative energy drift"),
    ]:
        sub = agg[agg["topology"] == "chain"]
        for model, g in sub.groupby("model"):
            g = g.sort_values("test_size")
            ax.plot(g["test_size"], g[f"{metric}_mean"], marker="o", label=model)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("|V|")
        ax.set_ylabel(title)
        ax.set_title(f"chain transfer: {title}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "graph_size_transfer_chain.png")
    plt.close(fig)


def plot_dynamics(df: pd.DataFrame, fig_dir: Path) -> None:
    env = "spring_mass_chain"
    sub = df[df["env"] == env]
    agg = aggregate_mean_se(sub, ["horizon", "model"], ["rollout_rmse", "max_energy_drift", "symplecticity_defect"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), squeeze=False)
    for ax, metric, title in [
        (axes[0, 0], "rollout_rmse", "multi-step rollout RMSE"),
        (axes[0, 1], "max_energy_drift", "relative energy drift"),
    ]:
        for model, g in agg.groupby("model"):
            g = g.sort_values("horizon")
            ax.plot(g["horizon"], g[f"{metric}_mean"], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rollout horizon")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "dynamics_rollout_spring_mass.png")
    plt.close(fig)


def plot_ablation(df: pd.DataFrame, fig_dir: Path) -> None:
    agg = aggregate_mean_se(df, ["env", "model"], ["return_mean", "max_energy_drift", "exploitation_gap_mean"])
    envs = agg["env"].unique()
    fig, axes = plt.subplots(len(envs), 1, figsize=(8.5, 3.1 * len(envs)), squeeze=False)
    for ax, env in zip(axes[:, 0], envs):
        sub = agg[agg["env"] == env].sort_values("return_mean_mean", ascending=False)
        ax.bar(sub["model"], sub["return_mean_mean"], yerr=sub["return_mean_se"])
        ax.set_title(env.replace("_", " "))
        ax.set_ylabel("MPC return")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_mpc_return.png")
    plt.close(fig)


def plot_horizon(df: pd.DataFrame, fig_dir: Path) -> None:
    agg = aggregate_mean_se(df, ["planning_horizon", "model"], ["return_mean", "exploitation_gap_mean", "imagined_energy_drift_mean"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), squeeze=False)
    for ax, metric, title in [
        (axes[0, 0], "return_mean", "return"),
        (axes[0, 1], "exploitation_gap_mean", "model exploitation gap"),
        (axes[0, 2], "imagined_energy_drift_mean", "imagined energy drift"),
    ]:
        for model, g in agg.groupby("model"):
            g = g.sort_values("planning_horizon")
            ax.plot(g["planning_horizon"], g[f"{metric}_mean"], marker="o", label=model)
        ax.set_xlabel("planning horizon")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "planning_horizon_sensitivity.png")
    plt.close(fig)


def make_parameter_table(out_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "Model": "DenseLinearNext",
            "Parameters": "2|V| x (3|V|+1)",
            "Hidden width": "n/a",
            "Graph?": "No",
            "Symplectic?": "No",
            "Uncertainty?": "No",
        },
        {
            "Model": "DirectGraphNext",
            "Parameters": 12,
            "Hidden width": "n/a",
            "Graph?": "Yes",
            "Symplectic?": "No",
            "Uncertainty?": "No",
        },
        {
            "Model": "GraphEuler",
            "Parameters": 12,
            "Hidden width": "n/a",
            "Graph?": "Yes",
            "Symplectic?": "No",
            "Uncertainty?": "No",
        },
        {
            "Model": "SympNoGraph",
            "Parameters": 2,
            "Hidden width": "n/a",
            "Graph?": "No edge terms",
            "Symplectic?": "Yes",
            "Uncertainty?": "No",
        },
        {
            "Model": "SG-Ham+Symp",
            "Parameters": 3,
            "Hidden width": "n/a",
            "Graph?": "Yes",
            "Symplectic?": "Yes",
            "Uncertainty?": "Bootstrap ensemble available",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "parameter_table.csv", index=False)
    return df


def write_summary(
    out_dir: Path,
    fig_dir: Path,
    cfg: ProtocolConfig,
    matched: pd.DataFrame,
    graph_transfer: pd.DataFrame,
    topology_transfer: pd.DataFrame,
    dynamics: pd.DataFrame,
    ablation: pd.DataFrame,
    horizon: pd.DataFrame,
) -> None:
    matched_agg = aggregate_mean_se(matched, ["env", "budget", "model"], ["return_mean", "terminal_error_mean"])
    final_budget = max(cfg.budgets)
    matched_final = matched_agg[matched_agg["budget"] == final_budget].sort_values(["env", "return_mean_mean"], ascending=[True, False])

    transfer_agg = aggregate_mean_se(graph_transfer, ["topology", "test_size", "model"], ["rollout_rmse", "max_energy_drift"])
    transfer_final = transfer_agg[(transfer_agg["topology"] == "chain") & (transfer_agg["test_size"] == max(cfg.graph_transfer_test_sizes))]
    transfer_final = transfer_final.sort_values("rollout_rmse_mean")

    dyn_agg = aggregate_mean_se(dynamics, ["env", "horizon", "model"], ["rollout_rmse", "max_energy_drift", "symplecticity_defect"])
    dyn_final = dyn_agg[(dyn_agg["env"] == "spring_mass_chain") & (dyn_agg["horizon"] == max(cfg.rollout_horizons))]
    dyn_final = dyn_final.sort_values("max_energy_drift_mean")

    ablation_agg = aggregate_mean_se(ablation, ["env", "model"], ["return_mean", "max_energy_drift", "exploitation_gap_mean"])
    horizon_agg = aggregate_mean_se(horizon, ["planning_horizon", "model"], ["return_mean", "exploitation_gap_mean", "imagined_energy_drift_mean"])

    def md_table(df: pd.DataFrame, cols: Sequence[str], n: int = 20) -> str:
        table = df.loc[:, cols].head(n).copy()
        if table.empty:
            return "_No rows._"
        formatted = []
        for _, row in table.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.4g}")
                else:
                    vals.append(str(val))
            formatted.append(vals)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(vals) + " |" for vals in formatted]
        return "\n".join([header, sep, *body])

    text = f"""# Fixed-Feature SG-MBRL Protocol Run

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

This run implements a CPU-scale fixed-feature version of the experiment protocol. It is not the final neural ResidualSympGNN/PETS/MBPO/model-free benchmark. It directly tests the clean structural mechanism with matched data and matched CEM-MPC:

- graph-local Hamiltonian model + symplectic rollout (`SG-Ham+Symp`),
- graph-local direct next-state model (`DirectGraphNext`),
- graph-local Euler vector-field model (`GraphEuler`),
- no-edge symplectic Hamiltonian ablation (`SympNoGraph`),
- dense linear next-state baseline on fixed-size tasks (`DenseLinearNext`).

## Configuration

```json
{json.dumps(asdict(cfg), indent=2)}
```

## Main Artifacts

- `matched_planner.csv`
- `graph_size_transfer.csv`
- `topology_transfer.csv`
- `dynamics_rollout.csv`
- `ablation.csv`
- `planning_horizon.csv`
- `parameter_table.csv`
- figures in `{fig_dir}`

## Headline Matched-Planner Result at Final Budget

{md_table(matched_final, ['env', 'budget', 'model', 'return_mean_mean', 'return_mean_se', 'terminal_error_mean_mean'])}

Interpretation: this is the highest-priority fairness test. Both methods use identical data budgets, graph inputs, action bounds, CEM horizon/population/iterations, and evaluation seeds.

## Graph-Size Transfer at Largest Chain Size

{md_table(transfer_final, ['topology', 'test_size', 'model', 'rollout_rmse_mean', 'rollout_rmse_se', 'max_energy_drift_mean'])}

Interpretation: lower rollout RMSE and energy drift at `|V|={max(cfg.graph_transfer_test_sizes)}` support the graph-local transfer claim.

## Long-Horizon Dynamics on Spring-Mass Chain

{md_table(dyn_final, ['env', 'horizon', 'model', 'rollout_rmse_mean', 'max_energy_drift_mean', 'symplecticity_defect_mean'])}

Interpretation: this tests whether low one-step error is enough for physically stable rollouts. Symplectic models should have much lower symplecticity defect and energy drift.

## Ablation Summary

{md_table(ablation_agg.sort_values(['env', 'return_mean_mean'], ascending=[True, False]), ['env', 'model', 'return_mean_mean', 'return_mean_se', 'max_energy_drift_mean', 'exploitation_gap_mean_mean'])}

Interpretation: this tests graph-only, symplectic-only, graph+Euler, dense, and graph+symplectic variants under the same control protocol.

## Planning-Horizon Sensitivity

{md_table(horizon_agg, ['planning_horizon', 'model', 'return_mean_mean', 'exploitation_gap_mean_mean', 'imagined_energy_drift_mean_mean'])}

Interpretation: this tests whether each model remains useful as the planner leans harder on imagined long-horizon rollouts.

## Figure Files

- `{fig_dir / 'matched_planner_return_vs_budget.png'}`
- `{fig_dir / 'graph_size_transfer_chain.png'}`
- `{fig_dir / 'dynamics_rollout_spring_mass.png'}`
- `{fig_dir / 'ablation_mpc_return.png'}`
- `{fig_dir / 'planning_horizon_sensitivity.png'}`

## Limitations

This run intentionally stays within NumPy/SciPy CPU constraints. It does not replace:

- neural ResidualSympGNN training,
- PETS probabilistic neural ensembles,
- MBPO with SAC/TD3,
- tuned PPO/SAC/TD3 baselines,
- Lennard-Jones/N-body hard-control tasks.

Use these results as a validated fixed-feature experiment suite and a reproducible dry run for the full-compute paper benchmark.
"""
    (out_dir / "summary.md").write_text(text)


def run_all(output_root: Path, cfg: ProtocolConfig) -> None:
    set_plot_style()
    out_dir = output_root / "raw"
    fig_dir = output_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("[1/7] matched planner")
    matched = run_matched_planner(cfg, out_dir)
    print("[2/7] graph-size transfer")
    graph_transfer = run_graph_size_transfer(cfg, out_dir)
    print("[3/7] topology transfer")
    topology_transfer = run_topology_transfer(cfg, out_dir)
    print("[4/7] dynamics rollout")
    dynamics = run_dynamics_rollout(cfg, out_dir)
    print("[5/7] ablation")
    ablation = run_ablation(cfg, out_dir)
    print("[6/7] planning horizon")
    horizon = run_planning_horizon(cfg, out_dir)
    print("[7/7] plotting and summary")
    make_parameter_table(out_dir)
    plot_matched(matched, fig_dir)
    plot_graph_transfer(graph_transfer, fig_dir)
    plot_dynamics(dynamics, fig_dir)
    plot_ablation(ablation, fig_dir)
    plot_horizon(horizon, fig_dir)
    write_summary(out_dir, fig_dir, cfg, matched, graph_transfer, topology_transfer, dynamics, ablation, horizon)
    print(f"done: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fixed_feature_protocol"))
    parser.add_argument("--quick", action="store_true", help="smaller smoke-test run")
    parser.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--cem-population", type=int, default=None)
    parser.add_argument("--cem-iterations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProtocolConfig()
    if args.quick:
        cfg = ProtocolConfig(
            seeds=(0, 1),
            budgets=(64, 256),
            control_horizon=25,
            eval_episodes=4,
            cem_population=48,
            cem_iterations=3,
            planning_horizons=(1, 5, 10),
            graph_transfer_test_sizes=(32, 64),
            rollout_horizons=(1, 10, 50, 100),
        )
    if args.seeds is not None:
        cfg = ProtocolConfig(**{**asdict(cfg), "seeds": tuple(range(args.seeds))})
    if args.eval_episodes is not None:
        cfg = ProtocolConfig(**{**asdict(cfg), "eval_episodes": args.eval_episodes})
    if args.cem_population is not None:
        cfg = ProtocolConfig(**{**asdict(cfg), "cem_population": args.cem_population})
    if args.cem_iterations is not None:
        cfg = ProtocolConfig(**{**asdict(cfg), "cem_iterations": args.cem_iterations})
    run_all(args.output, cfg)


if __name__ == "__main__":
    main()
