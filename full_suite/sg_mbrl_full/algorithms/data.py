from __future__ import annotations

from typing import List, Tuple

import numpy as np

from sg_mbrl_full.envs.hamiltonian import GraphHamiltonianEnv


Transition = Tuple[np.ndarray, np.ndarray, np.ndarray]


def collect_random_data(env: GraphHamiltonianEnv, transitions: int, seed: int, rollout_reset_prob: float = 0.1) -> List[Transition]:
    rng = np.random.default_rng(seed)
    old_rng = env.rng
    env.rng = rng
    data: List[Transition] = []
    z = env.sample_state()
    for _ in range(transitions):
        if rng.random() < rollout_reset_prob:
            z = env.sample_state()
        a = env.sample_action()
        z_next = env.step(z, a)
        data.append((z.copy(), a.copy(), z_next.copy()))
        z = z_next
        if (not np.all(np.isfinite(z))) or np.linalg.norm(z) > 100:
            z = env.sample_state()
    env.rng = old_rng
    return data


def evaluate_no_control(env: GraphHamiltonianEnv, episodes: int, horizon: int, seed: int, random_actions: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    returns = []
    terminal = []
    success = 0
    for _ in range(episodes):
        z = env.sample_state()
        total = 0.0
        for _ in range(horizon):
            if random_actions:
                a = rng.uniform(-env.cfg.action_limit, env.cfg.action_limit, env.action_dim).astype(np.float32)
            else:
                a = np.zeros(env.action_dim, dtype=np.float32)
            total += env.reward(z, a)
            z = env.step(z, a)
        err = env.terminal_error(z)
        returns.append(total)
        terminal.append(err)
        success += int(err <= 0.05)
    return {
        "return_mean": float(np.mean(returns)),
        "return_se": float(np.std(returns, ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0,
        "terminal_error_mean": float(np.mean(terminal)),
        "terminal_error_se": float(np.std(terminal, ddof=1) / np.sqrt(len(terminal))) if len(terminal) > 1 else 0.0,
        "success_rate": float(success / max(episodes, 1)),
        "mpc_failure_rate": 0.0,
        "exploitation_gap_mean": float("nan"),
        "imagined_energy_drift_mean": float("nan"),
    }

