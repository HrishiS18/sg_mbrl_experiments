from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from sg_mbrl_full.envs.hamiltonian import GraphHamiltonianEnv
from sg_mbrl_full.models.world_models import EnsembleWorldModel, WorldModel


@dataclass
class CEMConfig:
    horizon: int = 10
    population: int = 512
    iterations: int = 5
    elite_frac: float = 0.12
    init_std: float = 1.0
    min_std: float = 0.05
    gamma: float = 1.0
    uncertainty_coef: float = 0.0
    uncertainty_mode: str = "penalty"  # penalty, bonus, none
    terminal_weight: float = 1.0
    device: str = "cpu"


def reward_batch(env: GraphHamiltonianEnv, traj: np.ndarray, actions: np.ndarray, gamma: float, terminal_weight: float) -> np.ndarray:
    pop, horizon, _ = actions.shape
    returns = np.zeros(pop, dtype=np.float32)
    discount = 1.0
    for h in range(horizon):
        returns += discount * np.asarray([env.reward(traj[i, h], actions[i, h]) for i in range(pop)], dtype=np.float32)
        discount *= gamma
    returns += terminal_weight * discount * np.asarray(
        [env.reward(traj[i, -1], np.zeros(env.action_dim, dtype=np.float32)) for i in range(pop)], dtype=np.float32
    )
    return returns


def model_rollout(model: WorldModel | EnsembleWorldModel, env: GraphHamiltonianEnv, z0: np.ndarray, actions: np.ndarray, device: str) -> np.ndarray:
    z_batch = np.repeat(z0[None, :], len(actions), axis=0)
    return model.rollout_np(z_batch, actions, env.graph_arrays(), device=device)


def ensemble_uncertainty(model: WorldModel | EnsembleWorldModel, env: GraphHamiltonianEnv, z0: np.ndarray, actions: np.ndarray, device: str) -> np.ndarray:
    if not isinstance(model, EnsembleWorldModel):
        return np.zeros(len(actions), dtype=np.float32)
    z_batch = np.repeat(z0[None, :], len(actions), axis=0)
    samples = model.rollout_samples_np(z_batch, actions, env.graph_arrays(), device=device)
    # samples: members, batch, horizon+1, state_dim
    return np.mean(np.var(samples, axis=0), axis=(1, 2)).astype(np.float32)


def evaluate_sequences(
    model: WorldModel | EnsembleWorldModel,
    env: GraphHamiltonianEnv,
    z0: np.ndarray,
    actions: np.ndarray,
    cfg: CEMConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    traj = model_rollout(model, env, z0, actions, cfg.device)
    returns = reward_batch(env, traj, actions, cfg.gamma, cfg.terminal_weight)
    unc = ensemble_uncertainty(model, env, z0, actions, cfg.device)
    if cfg.uncertainty_mode == "penalty":
        objective = returns - cfg.uncertainty_coef * unc
    elif cfg.uncertainty_mode == "bonus":
        objective = returns + cfg.uncertainty_coef * unc
    else:
        objective = returns
    return objective, {"mean_model_return": float(np.mean(returns)), "mean_uncertainty": float(np.mean(unc))}


def cem_plan(
    model: WorldModel | EnsembleWorldModel,
    env: GraphHamiltonianEnv,
    z0: np.ndarray,
    cfg: CEMConfig,
    rng: np.random.Generator,
    initial_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    action_dim = env.action_dim
    mean = np.zeros((cfg.horizon, action_dim), dtype=np.float32) if initial_mean is None else initial_mean.astype(np.float32)
    std = np.ones_like(mean) * cfg.init_std * env.cfg.action_limit
    n_elite = max(2, int(cfg.population * cfg.elite_frac))
    best_actions = mean.copy()
    best_score = -np.inf
    diagnostics: Dict[str, float] = {}
    for _ in range(cfg.iterations):
        samples = rng.normal(mean[None, :, :], std[None, :, :], size=(cfg.population, cfg.horizon, action_dim)).astype(np.float32)
        samples = np.clip(samples, -env.cfg.action_limit, env.cfg.action_limit)
        scores, diagnostics = evaluate_sequences(model, env, z0, samples, cfg)
        elite_idx = np.argpartition(scores, -n_elite)[-n_elite:]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), cfg.min_std)
        local_best = int(np.argmax(scores))
        if float(scores[local_best]) > best_score:
            best_score = float(scores[local_best])
            best_actions = samples[local_best]
    return best_actions, best_score, diagnostics


def true_open_loop_return(env: GraphHamiltonianEnv, z0: np.ndarray, actions: np.ndarray, gamma: float = 1.0) -> float:
    z = z0.copy()
    total = 0.0
    discount = 1.0
    for a in actions:
        total += discount * env.reward(z, a)
        z = env.step(z, a)
        discount *= gamma
    total += discount * env.reward(z, np.zeros(env.action_dim, dtype=np.float32))
    return float(total)

