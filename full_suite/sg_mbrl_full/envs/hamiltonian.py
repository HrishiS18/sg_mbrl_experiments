from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .graphs import Array, GraphSpec, make_graph, target_positions


@dataclass
class EnvConfig:
    env_id: str = "duffing_chain"
    task: str = "stabilization"
    n: int = 8
    dim: int = 1
    graph: str = "chain"
    dt: float = 0.05
    action_limit: float = 2.0
    omega2: float = 1.0
    spring_k: float = 0.5
    quartic_node: float = 0.25
    quartic_edge: float = 0.15
    lj_epsilon: float = 0.05
    lj_sigma: float = 0.35
    lj_softening: float = 0.05
    force_clip: float = 50.0
    target_pattern: str = "sine"
    seed: int = 0


class GraphHamiltonianEnv:
    """Numpy simulator for graph-Hamiltonian control tasks."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.graph: GraphSpec = make_graph(cfg.graph, cfg.n, cfg.dim, seed=cfg.seed)
        self.n = self.graph.n
        self.dim = cfg.dim
        self.state_dim = 2 * self.n * self.dim
        self.action_dim = self.n * self.dim
        self.rng = np.random.default_rng(cfg.seed)
        self.target = target_positions(self.n, self.dim, cfg.target_pattern)
        if cfg.task == "stabilization":
            self.target = np.zeros((self.n, self.dim), dtype=np.float32)

    def clone_with_seed(self, seed: int) -> "GraphHamiltonianEnv":
        d = self.cfg.__dict__.copy()
        d["seed"] = seed
        return GraphHamiltonianEnv(EnvConfig(**d))

    def pack(self, q: Array, p: Array) -> Array:
        return np.concatenate([q.reshape(-1), p.reshape(-1)]).astype(np.float32)

    def unpack(self, z: Array) -> Tuple[Array, Array]:
        z = np.asarray(z, dtype=np.float32)
        q = z[: self.n * self.dim].reshape(self.n, self.dim)
        p = z[self.n * self.dim :].reshape(self.n, self.dim)
        return q, p

    def sample_state(self, high_energy: bool = False) -> Array:
        scale_q = 1.25 if high_energy else 0.75
        scale_p = 0.65 if high_energy else 0.35
        if self.cfg.env_id == "lennard_jones":
            q = self.rng.uniform(-1.0, 1.0, size=(self.n, self.dim)).astype(np.float32)
            q += 0.15 * self.rng.normal(size=q.shape).astype(np.float32)
        else:
            q = self.rng.normal(0.0, scale_q, size=(self.n, self.dim)).astype(np.float32)
        p = self.rng.normal(0.0, scale_p, size=(self.n, self.dim)).astype(np.float32)
        return self.pack(q, p)

    def sample_action(self) -> Array:
        return self.rng.uniform(-self.cfg.action_limit, self.cfg.action_limit, size=(self.n, self.dim)).astype(np.float32).reshape(-1)

    def clip_action(self, action: Array) -> Array:
        return np.clip(action.reshape(self.n, self.dim), -self.cfg.action_limit, self.cfg.action_limit)

    def edge_cubic(self, q: Array) -> Array:
        out = np.zeros_like(q)
        for i, j in self.graph.edges:
            diff = q[i] - q[j]
            cubic = np.sum(diff * diff) * diff
            out[i] += cubic
            out[j] -= cubic
        return out

    def spring_force(self, q: Array) -> Array:
        return -(self.graph.laplacian @ q)

    def lj_force(self, q: Array) -> Array:
        out = np.zeros_like(q)
        eps = self.cfg.lj_epsilon
        sig = self.cfg.lj_sigma
        soft = self.cfg.lj_softening
        for i, j in self.graph.edges:
            r = q[i] - q[j]
            r2 = float(np.dot(r, r) + soft**2)
            inv2 = 1.0 / r2
            sig2 = sig * sig
            sr2 = sig2 * inv2
            sr6 = sr2**3
            coeff = 24.0 * eps * inv2 * (2.0 * sr6 * sr6 - sr6)
            f = np.clip(coeff, -self.cfg.force_clip, self.cfg.force_clip) * r
            out[i] += f
            out[j] -= f
        return out

    def force(self, q: Array, action: Array) -> Array:
        a = self.clip_action(action)
        if self.cfg.env_id == "lennard_jones":
            f = self.lj_force(q) - 0.05 * q + a
        elif self.cfg.env_id == "linear_oscillator":
            f = -self.cfg.omega2 * q + self.cfg.spring_k * self.spring_force(q) + a
        else:
            f = (
                -self.cfg.omega2 * q
                - self.cfg.quartic_node * np.sum(q * q, axis=-1, keepdims=True) * q
                + self.cfg.spring_k * self.spring_force(q)
                - self.cfg.quartic_edge * self.edge_cubic(q)
                + a
            )
        return np.clip(f, -self.cfg.force_clip, self.cfg.force_clip).astype(np.float32)

    def step(self, z: Array, action: Array) -> Array:
        q, p = self.unpack(z)
        a = self.clip_action(action)
        p_half = p + 0.5 * self.cfg.dt * self.force(q, a)
        q_next = q + self.cfg.dt * p_half
        p_next = p_half + 0.5 * self.cfg.dt * self.force(q_next, a)
        return self.pack(q_next, p_next)

    def rollout(self, z0: Array, actions: Array) -> Array:
        z = np.asarray(z0, dtype=np.float32)
        traj = [z.copy()]
        for a in np.asarray(actions, dtype=np.float32):
            z = self.step(z, a)
            traj.append(z.copy())
        return np.stack(traj)

    def energy(self, z: Array) -> float:
        q, p = self.unpack(z)
        kinetic = 0.5 * float(np.sum(p * p))
        if self.cfg.env_id == "lennard_jones":
            pot = 0.025 * float(np.sum(q * q))
            eps = self.cfg.lj_epsilon
            sig = self.cfg.lj_sigma
            soft = self.cfg.lj_softening
            for i, j in self.graph.edges:
                r = q[i] - q[j]
                r2 = float(np.dot(r, r) + soft**2)
                sr6 = (sig * sig / r2) ** 3
                pot += 4.0 * eps * (sr6 * sr6 - sr6)
            return kinetic + pot
        node = 0.5 * self.cfg.omega2 * float(np.sum(q * q))
        node += 0.25 * self.cfg.quartic_node * float(np.sum(np.sum(q * q, axis=-1) ** 2))
        edge = 0.0
        for i, j in self.graph.edges:
            diff = q[i] - q[j]
            r2 = float(np.dot(diff, diff))
            edge += 0.5 * self.cfg.spring_k * r2 + 0.25 * self.cfg.quartic_edge * r2 * r2
        return kinetic + node + edge

    def reward(self, z: Array, action: Array) -> float:
        q, p = self.unpack(z)
        a = self.clip_action(action)
        if self.cfg.task in {"target", "steering"}:
            state_cost = float(np.mean((q - self.target) ** 2) + 0.1 * np.mean(p * p))
        elif self.cfg.task == "formation":
            d_cur = pairwise_distances(q)
            d_tgt = pairwise_distances(self.target)
            state_cost = float(np.mean((d_cur - d_tgt) ** 2) + 0.05 * np.mean(p * p))
        else:
            state_cost = float(np.mean(q * q + p * p))
        action_cost = 0.002 * float(np.mean(a * a))
        collision = 0.0
        if self.cfg.env_id == "lennard_jones":
            d = pairwise_distances(q)
            collision = 0.01 * float(np.mean((d[d > 0] < 0.12).astype(np.float32))) if np.any(d > 0) else 0.0
        return -(state_cost + action_cost + collision)

    def terminal_error(self, z: Array) -> float:
        q, p = self.unpack(z)
        if self.cfg.task in {"target", "steering"}:
            return float(np.mean((q - self.target) ** 2) + 0.1 * np.mean(p * p))
        if self.cfg.task == "formation":
            return float(np.mean((pairwise_distances(q) - pairwise_distances(self.target)) ** 2))
        return float(np.mean(q * q + p * p))

    def graph_arrays(self) -> Dict[str, Array]:
        return {
            "adjacency": self.graph.adjacency.copy(),
            "laplacian": self.graph.laplacian.copy(),
            "edges": np.asarray(self.graph.edges, dtype=np.int64),
        }


def pairwise_distances(q: Array) -> Array:
    diff = q[:, None, :] - q[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def make_env_from_dict(d: Dict) -> GraphHamiltonianEnv:
    return GraphHamiltonianEnv(EnvConfig(**d))

