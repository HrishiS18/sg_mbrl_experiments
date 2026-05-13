from __future__ import annotations

from typing import Dict

from sg_mbrl_full.envs.gym_wrapper import LearnedModelGymEnv, TrueGymEnv
from sg_mbrl_full.envs.hamiltonian import GraphHamiltonianEnv


def run_mbpo_sac(env: GraphHamiltonianEnv, model, train_steps: int, eval_episodes: int, horizon: int, seed: int, device: str) -> Dict[str, float]:
    try:
        from stable_baselines3 import SAC
    except Exception as exc:  # pragma: no cover
        return {"status": f"skipped_missing_stable_baselines3: {exc}"}

    model_env = LearnedModelGymEnv(env, model, horizon=min(50, horizon), seed=seed, device=device)
    agent = SAC("MlpPolicy", model_env, verbose=0, seed=seed, learning_starts=min(1000, max(100, train_steps // 10)))
    agent.learn(total_timesteps=train_steps)
    eval_env = TrueGymEnv(env, horizon=horizon, seed=seed + 999)
    returns = []
    terminal = []
    successes = 0
    for ep in range(eval_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        total = 0.0
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        terminal.append(info.get("terminal_error", float("nan")))
        successes += int(terminal[-1] <= 0.05)
    import numpy as np

    return {
        "return_mean": float(np.mean(returns)),
        "return_se": float(np.std(returns, ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0,
        "terminal_error_mean": float(np.mean(terminal)),
        "terminal_error_se": float(np.std(terminal, ddof=1) / np.sqrt(len(terminal))) if len(terminal) > 1 else 0.0,
        "success_rate": float(successes / max(eval_episodes, 1)),
        "mpc_failure_rate": 0.0,
        "exploitation_gap_mean": float("nan"),
        "imagined_energy_drift_mean": float("nan"),
        "status": "ok",
    }

