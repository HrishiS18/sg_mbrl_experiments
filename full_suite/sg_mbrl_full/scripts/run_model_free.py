from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sg_mbrl_full.envs import EnvConfig, GraphHamiltonianEnv
from sg_mbrl_full.envs.gym_wrapper import TrueGymEnv
from sg_mbrl_full.utils import ensure_dir, load_json, save_json


def eval_agent(agent, env: GraphHamiltonianEnv, episodes: int, horizon: int, seed: int) -> Dict[str, float]:
    gym_env = TrueGymEnv(env, horizon=horizon, seed=seed)
    returns = []
    terminal = []
    success = 0
    for ep in range(episodes):
        obs, _ = gym_env.reset(seed=seed + ep)
        total = 0.0
        done = False
        info = {}
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        terminal.append(info.get("terminal_error", np.nan))
        success += int(terminal[-1] <= 0.05)
    return {
        "return_mean": float(np.mean(returns)),
        "return_se": float(np.std(returns, ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0,
        "terminal_error_mean": float(np.mean(terminal)),
        "terminal_error_se": float(np.std(terminal, ddof=1) / np.sqrt(len(terminal))) if len(terminal) > 1 else 0.0,
        "success_rate": float(success / max(episodes, 1)),
    }


def build_agent(name: str, gym_env, seed: int, kwargs: Dict[str, Any]):
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except Exception as exc:  # pragma: no cover
        raise ImportError("stable-baselines3 is required for model-free baselines") from exc
    cls = {"sac": SAC, "ppo": PPO, "td3": TD3}[name.lower()]
    return cls("MlpPolicy", gym_env, seed=seed, verbose=0, **kwargs)


def run(config: Dict[str, Any]) -> Path:
    run_dir = ensure_dir(config["run_dir"])
    raw = ensure_dir(run_dir / "raw")
    save_json(run_dir / "config.resolved.json", config)
    rows: List[Dict[str, Any]] = []
    for env_idx, env_cfg_dict in enumerate(config["envs"]):
        for seed in config["seeds"]:
            env = GraphHamiltonianEnv(EnvConfig(**{**env_cfg_dict, "seed": seed}))
            gym_env = TrueGymEnv(env, horizon=config["horizon"], seed=seed)
            for method, kwargs in config["methods"].items():
                try:
                    agent = build_agent(method, gym_env, seed, kwargs.get("agent_kwargs", {}))
                    prev_budget = 0
                    for budget in config["budgets"]:
                        delta = int(budget) - int(prev_budget)
                        if delta < 0:
                            raise ValueError("model-free budgets must be sorted ascending")
                        if delta > 0:
                            agent.learn(total_timesteps=delta, reset_num_timesteps=False)
                        prev_budget = budget
                        metrics = eval_agent(agent, env, config["eval_episodes"], config["horizon"], seed=seed + budget)
                        status = "ok"
                        rows.append(
                            {
                                "env": env.cfg.env_id,
                                "task": env.cfg.task,
                                "graph": env.cfg.graph,
                                "n": env.n,
                                "dim": env.dim,
                                "seed": seed,
                                "budget": budget,
                                "method": method,
                                "status": status,
                                **metrics,
                            }
                        )
                        pd.DataFrame(rows).to_csv(raw / "model_free_metrics.csv", index=False)
                except Exception as exc:
                    rows.append(
                        {
                            "env": env.cfg.env_id,
                            "task": env.cfg.task,
                            "graph": env.cfg.graph,
                            "n": env.n,
                            "dim": env.dim,
                            "seed": seed,
                            "budget": 0,
                            "method": method,
                            "status": f"failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    pd.DataFrame(rows).to_csv(raw / "model_free_metrics.csv", index=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"done: {run(load_json(args.config))}")


if __name__ == "__main__":
    main()
