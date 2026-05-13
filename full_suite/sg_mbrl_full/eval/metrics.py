from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sg_mbrl_full.envs.hamiltonian import GraphHamiltonianEnv
from sg_mbrl_full.models.world_models import EnsembleWorldModel, WorldModel
from sg_mbrl_full.planning.cem import CEMConfig, cem_plan, model_rollout, true_open_loop_return


@dataclass
class EvalConfig:
    episodes: int = 20
    control_horizon: int = 50
    success_threshold: float = 0.05
    divergence_norm: float = 100.0
    high_energy: bool = False


def evaluate_mpc(
    model: WorldModel | EnsembleWorldModel,
    env: GraphHamiltonianEnv,
    cem_cfg: CEMConfig,
    eval_cfg: EvalConfig,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    terminal_errors: List[float] = []
    exploitation_gaps: List[float] = []
    energy_drifts: List[float] = []
    successes = 0
    failures = 0
    for _ in range(eval_cfg.episodes):
        z = env.sample_state(high_energy=eval_cfg.high_energy)
        ep_return = 0.0
        for t in range(eval_cfg.control_horizon):
            plan, pred_score, _ = cem_plan(model, env, z, cem_cfg, rng)
            if t == 0:
                true_score = true_open_loop_return(env, z, plan, gamma=cem_cfg.gamma)
                exploitation_gaps.append(float(pred_score - true_score))
                pred_traj = model_rollout(model, env, z, plan[None, :, :], cem_cfg.device)[0]
                e0 = env.energy(pred_traj[0])
                energy_drifts.append(float(max(abs(env.energy(s) - e0) for s in pred_traj)))
            a = plan[0]
            ep_return += env.reward(z, a)
            z = env.step(z, a)
            if (not np.all(np.isfinite(z))) or np.linalg.norm(z) > eval_cfg.divergence_norm:
                failures += 1
                break
        terr = env.terminal_error(z)
        terminal_errors.append(terr)
        returns.append(ep_return)
        successes += int(terr <= eval_cfg.success_threshold)
    return {
        "return_mean": float(np.mean(returns)),
        "return_se": _se(returns),
        "terminal_error_mean": float(np.mean(terminal_errors)),
        "terminal_error_se": _se(terminal_errors),
        "success_rate": float(successes / max(eval_cfg.episodes, 1)),
        "mpc_failure_rate": float(failures / max(eval_cfg.episodes, 1)),
        "exploitation_gap_mean": float(np.mean(exploitation_gaps)) if exploitation_gaps else float("nan"),
        "imagined_energy_drift_mean": float(np.mean(energy_drifts)) if energy_drifts else float("nan"),
    }


def rollout_diagnostics(
    model: WorldModel | EnsembleWorldModel,
    env: GraphHamiltonianEnv,
    horizons: List[int],
    episodes: int,
    seed: int,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for horizon in horizons:
        rmses: List[float] = []
        drifts: List[float] = []
        divs = 0
        for _ in range(episodes):
            z0 = env.sample_state()
            actions = rng.uniform(-env.cfg.action_limit, env.cfg.action_limit, size=(horizon, env.action_dim)).astype(np.float32)
            true_traj = env.rollout(z0, actions)
            pred_traj = model_rollout(model, env, z0, actions[None, :, :], device)[0]
            rmses.append(float(np.sqrt(np.mean((pred_traj - true_traj) ** 2))))
            e0 = env.energy(pred_traj[0])
            drifts.append(float(max(abs(env.energy(s) - e0) for s in pred_traj)))
            if (not np.all(np.isfinite(pred_traj))) or np.max(np.linalg.norm(pred_traj, axis=-1)) > 100:
                divs += 1
        rows.append(
            {
                "horizon": horizon,
                "rollout_rmse": float(np.mean(rmses)),
                "rollout_rmse_se": _se(rmses),
                "energy_drift": float(np.mean(drifts)),
                "energy_drift_se": _se(drifts),
                "divergence_rate": float(divs / max(episodes, 1)),
            }
        )
    return rows


def one_step_rmse(model: WorldModel | EnsembleWorldModel, env: GraphHamiltonianEnv, data, device: str = "cpu") -> float:
    errs = []
    for z, a, z_next in data:
        pred = model.predict_np(z, a, env.graph_arrays(), device=device)[0]
        errs.append(float(np.mean((pred - z_next) ** 2)))
    return float(np.sqrt(np.mean(errs)))


def _se(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

