from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


Array = np.ndarray


def require_torch() -> None:
    if torch is None:
        raise ImportError(f"PyTorch is required for the neural full suite: {_TORCH_IMPORT_ERROR}")


@dataclass
class ModelConfig:
    model_type: str = "residual_symp_gnn"
    dim: int = 1
    hidden: int = 128
    depth: int = 3
    residual_scale: float = 0.05
    dt: float = 0.05
    action_limit: float = 2.0
    predict_delta: bool = True


def mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> "nn.Sequential":
    require_torch()
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(max(depth - 1, 1)):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def to_torch(x: Array, device: "torch.device") -> "torch.Tensor":
    require_torch()
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def split_z(z: "torch.Tensor", dim: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
    n = z.shape[-1] // (2 * dim)
    q = z[..., : n * dim].reshape(*z.shape[:-1], n, dim)
    p = z[..., n * dim :].reshape(*z.shape[:-1], n, dim)
    return q, p


def pack_z(q: "torch.Tensor", p: "torch.Tensor") -> "torch.Tensor":
    return torch.cat([q.reshape(q.shape[0], -1), p.reshape(p.shape[0], -1)], dim=-1)


def normalize_adj(adj: "torch.Tensor") -> "torch.Tensor":
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj / deg


class WorldModel(nn.Module):
    cfg: ModelConfig

    def step_torch(self, z: "torch.Tensor", a: "torch.Tensor", graph: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        raise NotImplementedError

    def rollout_torch(self, z0: "torch.Tensor", actions: "torch.Tensor", graph: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        zs = [z0]
        z = z0
        for h in range(actions.shape[1]):
            z = self.step_torch(z, actions[:, h], graph)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def predict_np(self, z: Array, a: Array, graph_np: Dict[str, Array], device: Optional[str] = None) -> Array:
        require_torch()
        dev = torch.device(device or next(self.parameters()).device)
        self.eval()
        with torch.enable_grad():
            zt = to_torch(np.asarray(z, dtype=np.float32), dev)
            at = to_torch(np.asarray(a, dtype=np.float32), dev)
            graph = graph_to_torch(graph_np, dev)
            if zt.ndim == 1:
                zt = zt[None]
            if at.ndim == 1:
                at = at[None]
            pred = self.step_torch(zt, at, graph)
        return pred.detach().cpu().numpy()

    def rollout_np(self, z0: Array, actions: Array, graph_np: Dict[str, Array], device: Optional[str] = None) -> Array:
        require_torch()
        dev = torch.device(device or next(self.parameters()).device)
        self.eval()
        with torch.enable_grad():
            zt = to_torch(np.asarray(z0, dtype=np.float32), dev)
            at = to_torch(np.asarray(actions, dtype=np.float32), dev)
            graph = graph_to_torch(graph_np, dev)
            if zt.ndim == 1:
                zt = zt[None]
            if at.ndim == 2:
                at = at[None]
            pred = self.rollout_torch(zt, at, graph)
        return pred.detach().cpu().numpy()


def graph_to_torch(graph_np: Dict[str, Array], device: "torch.device") -> Dict[str, "torch.Tensor"]:
    require_torch()
    graph = {
        "adjacency": to_torch(graph_np["adjacency"], device),
        "laplacian": to_torch(graph_np["laplacian"], device),
    }
    if "edges" in graph_np:
        graph["edges"] = torch.as_tensor(graph_np["edges"], dtype=torch.long, device=device)
    return graph


class DirectGNN(WorldModel):
    def __init__(self, cfg: ModelConfig):
        require_torch()
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.net = mlp(in_dim=6 * d + 1, out_dim=2 * d, hidden=cfg.hidden, depth=cfg.depth)

    def step_torch(self, z: "torch.Tensor", a: "torch.Tensor", graph: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        q, p = split_z(z, self.cfg.dim)
        aa = a.reshape(q.shape)
        adj = normalize_adj(graph["adjacency"])
        agg_q = torch.einsum("ij,bjd->bid", adj, q)
        agg_p = torch.einsum("ij,bjd->bid", adj, p)
        deg = graph["adjacency"].sum(dim=-1).reshape(1, -1, 1).expand(q.shape[0], -1, 1)
        x = torch.cat([q, p, aa, agg_q, agg_p, q - agg_q, deg], dim=-1)
        y = self.net(x)
        dq, dp = y[..., : self.cfg.dim], y[..., self.cfg.dim :]
        if self.cfg.predict_delta:
            qn, pn = q + dq, p + dp
        else:
            qn, pn = dq, dp
        return pack_z(qn, pn)


class MLPNext(WorldModel):
    def __init__(self, cfg: ModelConfig, n: int):
        require_torch()
        super().__init__()
        self.cfg = cfg
        self.n = n
        in_dim = 3 * n * cfg.dim
        out_dim = 2 * n * cfg.dim
        self.net = mlp(in_dim, out_dim, cfg.hidden, cfg.depth)

    def step_torch(self, z: "torch.Tensor", a: "torch.Tensor", graph: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        x = torch.cat([z, a], dim=-1)
        y = self.net(x)
        return z + y if self.cfg.predict_delta else y


class ResidualSympGNN(WorldModel):
    """Separable graph Hamiltonian model with velocity-Verlet rollout."""

    def __init__(self, cfg: ModelConfig, use_edges: bool = True):
        require_torch()
        super().__init__()
        self.cfg = cfg
        self.use_edges = use_edges
        d = cfg.dim
        self.node_net = mlp(in_dim=2 * d, out_dim=1, hidden=cfg.hidden, depth=cfg.depth)
        self.edge_net = mlp(in_dim=d + 1, out_dim=1, hidden=cfg.hidden, depth=cfg.depth)
        self.log_cq = nn.Parameter(torch.tensor(0.0))
        self.log_ce = nn.Parameter(torch.tensor(-0.7))
        self.log_c4 = nn.Parameter(torch.tensor(-1.4))
        self.log_ce4 = nn.Parameter(torch.tensor(-1.8))
        self.action_gain = nn.Parameter(torch.tensor(1.0))

    def potential(self, q: "torch.Tensor", a: "torch.Tensor", graph: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        aa = a.reshape(q.shape)
        cq = F.softplus(self.log_cq)
        ce = F.softplus(self.log_ce)
        c4 = F.softplus(self.log_c4)
        ce4 = F.softplus(self.log_ce4)
        v = 0.5 * cq * (q * q).sum(dim=(-1, -2))
        v = v + 0.25 * c4 * ((q * q).sum(dim=-1) ** 2).sum(dim=-1)
        v = v - self.action_gain * (aa * q).sum(dim=(-1, -2))
        v = v + self.cfg.residual_scale * self.node_net(torch.cat([q, aa], dim=-1)).sum(dim=(-1, -2))
        if self.use_edges and graph.get("edges") is not None and graph["edges"].numel() > 0:
            edges = graph["edges"]
            qi = q[:, edges[:, 0], :]
            qj = q[:, edges[:, 1], :]
            diff = qi - qj
            r2 = (diff * diff).sum(dim=-1, keepdim=True)
            v = v + 0.5 * ce * r2.squeeze(-1).sum(dim=-1)
            v = v + 0.25 * ce4 * (r2.squeeze(-1) ** 2).sum(dim=-1)
            edge_x = torch.cat([diff, r2], dim=-1)
            v = v + self.cfg.residual_scale * self.edge_net(edge_x).sum(dim=(-1, -2))
        return v

    def force(self, q: "torch.Tensor", a: "torch.Tensor", graph: Dict[str, "torch.Tensor"], create_graph: bool) -> "torch.Tensor":
        q_req = q.detach().clone().requires_grad_(True) if not q.requires_grad else q
        v = self.potential(q_req, a, graph).sum()
        grad = torch.autograd.grad(v, q_req, create_graph=create_graph, retain_graph=create_graph)[0]
        return -grad

    def step_torch(self, z: "torch.Tensor", a: "torch.Tensor", graph: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        q, p = split_z(z, self.cfg.dim)
        create_graph = self.training
        f0 = self.force(q, a, graph, create_graph=create_graph)
        p_half = p + 0.5 * self.cfg.dt * f0
        q_next = q + self.cfg.dt * p_half
        f1 = self.force(q_next, a, graph, create_graph=create_graph)
        p_next = p_half + 0.5 * self.cfg.dt * f1
        return pack_z(q_next, p_next)


class SympNoGraph(ResidualSympGNN):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg, use_edges=False)


class EnsembleWorldModel:
    def __init__(self, members: List[WorldModel]):
        if not members:
            raise ValueError("ensemble must contain at least one member")
        self.members = members

    @property
    def cfg(self) -> ModelConfig:
        return self.members[0].cfg

    def predict_np(self, z: Array, a: Array, graph_np: Dict[str, Array], device: Optional[str] = None) -> Array:
        preds = [m.predict_np(z, a, graph_np, device=device) for m in self.members]
        return np.mean(preds, axis=0)

    def rollout_np(self, z0: Array, actions: Array, graph_np: Dict[str, Array], device: Optional[str] = None) -> Array:
        preds = [m.rollout_np(z0, actions, graph_np, device=device) for m in self.members]
        return np.mean(preds, axis=0)

    def rollout_samples_np(self, z0: Array, actions: Array, graph_np: Dict[str, Array], device: Optional[str] = None) -> Array:
        return np.stack([m.rollout_np(z0, actions, graph_np, device=device) for m in self.members], axis=0)

    def disagreement_np(self, z0: Array, actions: Array, graph_np: Dict[str, Array], device: Optional[str] = None) -> Array:
        samples = self.rollout_samples_np(z0, actions, graph_np, device=device)
        return np.mean(np.var(samples, axis=0), axis=(-1, -2))


def build_model(model_type: str, cfg: ModelConfig, n: int) -> WorldModel:
    if model_type == "residual_symp_gnn":
        return ResidualSympGNN(cfg, use_edges=True)
    if model_type == "direct_gnn":
        return DirectGNN(cfg)
    if model_type == "mlp_next":
        return MLPNext(cfg, n=n)
    if model_type in {"hnn", "symp_no_graph"}:
        return SympNoGraph(cfg)
    raise ValueError(f"unknown model_type: {model_type}")


def count_parameters(model: object) -> int:
    if isinstance(model, EnsembleWorldModel):
        return sum(count_parameters(m) for m in model.members)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

