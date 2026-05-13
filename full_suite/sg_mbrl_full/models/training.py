from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .world_models import EnsembleWorldModel, ModelConfig, WorldModel, build_model, graph_to_torch, require_torch, to_torch

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


Transition = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    ensemble_size: int = 1
    bootstrap: bool = True
    validation_fraction: float = 0.1
    patience: int = 20
    device: str = "cpu"


def make_arrays(data: Sequence[Transition]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.stack([d[0] for d in data]).astype(np.float32)
    a = np.stack([d[1] for d in data]).astype(np.float32)
    z_next = np.stack([d[2] for d in data]).astype(np.float32)
    return z, a, z_next


def train_world_model(
    model: WorldModel,
    data: Sequence[Transition],
    graph_np: Dict[str, np.ndarray],
    cfg: TrainConfig,
    seed: int,
) -> Dict[str, float]:
    require_torch()
    rng = np.random.default_rng(seed)
    device = torch.device(cfg.device)
    model.to(device)
    z, a, z_next = make_arrays(data)
    n = len(z)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * cfg.validation_fraction)) if n > 10 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:] if n_val else idx
    graph = graph_to_torch(graph_np, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    best_state = None
    wait = 0
    history: List[float] = []
    for _ in range(cfg.epochs):
        rng.shuffle(train_idx)
        losses = []
        model.train()
        for start in range(0, len(train_idx), cfg.batch_size):
            batch = train_idx[start : start + cfg.batch_size]
            zt = to_torch(z[batch], device)
            at = to_torch(a[batch], device)
            yt = to_torch(z_next[batch], device)
            pred = model.step_torch(zt, at, graph)
            loss = F.mse_loss(pred, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = float(np.mean(losses))
        if n_val:
            model.eval()
            with torch.enable_grad():
                pred = model.step_torch(to_torch(z[val_idx], device), to_torch(a[val_idx], device), graph)
                val = float(F.mse_loss(pred, to_torch(z_next[val_idx], device)).detach().cpu())
        history.append(val)
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best, "epochs_ran": len(history), "final_loss": history[-1] if history else float("nan")}


def train_ensemble(
    model_type: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data: Sequence[Transition],
    graph_np: Dict[str, np.ndarray],
    n_nodes: int,
    seed: int,
) -> Tuple[EnsembleWorldModel | WorldModel, List[Dict[str, float]]]:
    require_torch()
    rng = np.random.default_rng(seed)
    members = []
    logs = []
    ensemble_size = max(1, train_cfg.ensemble_size)
    for m in range(ensemble_size):
        model = build_model(model_type, model_cfg, n=n_nodes)
        if train_cfg.bootstrap and ensemble_size > 1:
            idx = rng.integers(0, len(data), len(data))
            member_data = [data[int(i)] for i in idx]
        else:
            member_data = list(data)
        logs.append(train_world_model(model, member_data, graph_np, train_cfg, seed=seed + 101 * m))
        members.append(model)
    if ensemble_size == 1:
        return members[0], logs
    return EnsembleWorldModel(members), logs

