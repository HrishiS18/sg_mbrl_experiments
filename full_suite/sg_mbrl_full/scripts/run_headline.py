from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sg_mbrl_full.algorithms import collect_random_data, evaluate_no_control, run_mbpo_sac
from sg_mbrl_full.envs import EnvConfig, GraphHamiltonianEnv
from sg_mbrl_full.eval import EvalConfig, evaluate_mpc, one_step_rmse, rollout_diagnostics
from sg_mbrl_full.models import ModelConfig, TrainConfig, count_parameters, train_ensemble
from sg_mbrl_full.planning import CEMConfig
from sg_mbrl_full.utils import ensure_dir, load_json, save_json


def method_to_model(method: Dict[str, Any], env: GraphHamiltonianEnv, base_model_cfg: Dict[str, Any]) -> ModelConfig:
    cfg = copy.deepcopy(base_model_cfg)
    cfg.update(method.get("model", {}))
    cfg["dim"] = env.dim
    cfg["dt"] = env.cfg.dt
    cfg["action_limit"] = env.cfg.action_limit
    return ModelConfig(**cfg)


def method_to_train(method: Dict[str, Any], base_train_cfg: Dict[str, Any]) -> TrainConfig:
    cfg = copy.deepcopy(base_train_cfg)
    cfg.update(method.get("train", {}))
    return TrainConfig(**cfg)


def method_to_cem(method: Dict[str, Any], base_cem_cfg: Dict[str, Any]) -> CEMConfig:
    cfg = copy.deepcopy(base_cem_cfg)
    cfg.update(method.get("cem", {}))
    return CEMConfig(**cfg)


def run(config: Dict[str, Any]) -> Path:
    run_dir = ensure_dir(config["run_dir"])
    raw_dir = ensure_dir(run_dir / "raw")
    save_json(run_dir / "config.resolved.json", config)

    rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    param_rows: List[Dict[str, Any]] = []
    seeds = config["seeds"]
    budgets = config["budgets"]
    methods = config["methods"]
    base_model_cfg = config.get("model_defaults", {})
    base_train_cfg = config.get("train_defaults", {})
    base_cem_cfg = config.get("cem_defaults", {})
    eval_cfg = EvalConfig(**config.get("eval", {}))
    rollout_horizons = config.get("rollout_horizons", [1, 10, 50, 100])
    diagnostics_episodes = int(config.get("diagnostics_episodes", 8))

    for env_idx, env_cfg_dict in enumerate(config["envs"]):
        for seed in seeds:
            env_cfg = EnvConfig(**{**env_cfg_dict, "seed": seed})
            env = GraphHamiltonianEnv(env_cfg)
            max_budget = max(budgets)
            data = collect_random_data(env, max_budget, seed=10_000 + 997 * seed + env_idx)
            test_data = collect_random_data(env.clone_with_seed(seed + 77), min(512, max_budget), seed=20_000 + seed + env_idx)

            for budget in budgets:
                train_data = data[:budget]
                for method_name, method in methods.items():
                    status = "ok"
                    t0 = time.time()
                    if method.get("type") == "no_control":
                        metrics = evaluate_no_control(env, eval_cfg.episodes, eval_cfg.control_horizon, seed=seed, random_actions=False)
                        params = 0
                        logs = []
                    elif method.get("type") == "random":
                        metrics = evaluate_no_control(env, eval_cfg.episodes, eval_cfg.control_horizon, seed=seed, random_actions=True)
                        params = 0
                        logs = []
                    else:
                        model_cfg = method_to_model(method, env, base_model_cfg)
                        train_cfg = method_to_train(method, base_train_cfg)
                        model_type = method.get("model_type", model_cfg.model_type)
                        try:
                            model, logs = train_ensemble(
                                model_type,
                                model_cfg,
                                train_cfg,
                                train_data,
                                env.graph_arrays(),
                                env.n,
                                seed=30_000 + 131 * seed + budget,
                            )
                            params = count_parameters(model)
                            if method.get("type") == "mbpo_sac":
                                metrics = run_mbpo_sac(
                                    env,
                                    model,
                                    train_steps=int(method.get("policy_train_steps", 10_000)),
                                    eval_episodes=eval_cfg.episodes,
                                    horizon=eval_cfg.control_horizon,
                                    seed=seed,
                                    device=train_cfg.device,
                                )
                                status = metrics.pop("status", "ok")
                            else:
                                cem_cfg = method_to_cem(method, base_cem_cfg)
                                metrics = evaluate_mpc(model, env, cem_cfg, eval_cfg, seed=40_000 + seed + budget)
                                metrics["one_step_rmse"] = one_step_rmse(model, env, test_data, device=cem_cfg.device)
                                if config.get("run_diagnostics", True) and budget == max_budget:
                                    for r in rollout_diagnostics(
                                        model,
                                        env,
                                        rollout_horizons,
                                        diagnostics_episodes,
                                        seed=50_000 + seed,
                                        device=cem_cfg.device,
                                    ):
                                        diag_rows.append(
                                            {
                                                "env": env_cfg.env_id,
                                                "task": env_cfg.task,
                                                "graph": env_cfg.graph,
                                                "seed": seed,
                                                "budget": budget,
                                                "method": method_name,
                                                **r,
                                            }
                                        )
                        except Exception as exc:
                            metrics = {}
                            params = -1
                            logs = []
                            status = f"failed: {type(exc).__name__}: {exc}"
                    row = {
                        "env": env_cfg.env_id,
                        "task": env_cfg.task,
                        "graph": env_cfg.graph,
                        "n": env.n,
                        "dim": env.dim,
                        "seed": seed,
                        "budget": budget,
                        "method": method_name,
                        "status": status,
                        "seconds": time.time() - t0,
                        "parameters": params,
                        **metrics,
                    }
                    rows.append(row)
                    param_rows.append(
                        {
                            "method": method_name,
                            "env": env_cfg.env_id,
                            "seed": seed,
                            "budget": budget,
                            "parameters": params,
                            "train_logs": logs,
                            "status": status,
                        }
                    )
                    pd.DataFrame(rows).to_csv(raw_dir / "headline_metrics.csv", index=False)
                    if diag_rows:
                        pd.DataFrame(diag_rows).to_csv(raw_dir / "rollout_diagnostics.csv", index=False)
                    pd.DataFrame(param_rows).to_csv(raw_dir / "parameter_logs.csv", index=False)
                    print(f"{env_cfg.env_id}/{env_cfg.task} seed={seed} budget={budget} method={method_name} status={status}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_dir = run(load_json(args.config))
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()

