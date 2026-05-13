from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


Array = np.ndarray
EdgeList = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class GraphSpec:
    kind: str
    n: int
    dim: int
    edges: EdgeList
    adjacency: Array
    laplacian: Array


def edges_to_mats(n: int, edges: Iterable[Tuple[int, int]]) -> Tuple[Array, Array]:
    adj = np.zeros((n, n), dtype=np.float32)
    for i, j in edges:
        if i == j:
            continue
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    deg = np.diag(adj.sum(axis=1))
    return adj, deg - adj


def chain_edges(n: int) -> EdgeList:
    return tuple((i, i + 1) for i in range(n - 1))


def cycle_edges(n: int) -> EdgeList:
    edges = list(chain_edges(n))
    if n > 2:
        edges.append((n - 1, 0))
    return tuple(edges)


def star_edges(n: int) -> EdgeList:
    return tuple((0, i) for i in range(1, n))


def grid_edges(rows: int, cols: int) -> EdgeList:
    def idx(r: int, c: int) -> int:
        return r * cols + c

    edges: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
    return tuple(edges)


def er_edges(n: int, p: float, rng: np.random.Generator) -> EdgeList:
    edges = set(chain_edges(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.add((i, j))
    return tuple(sorted(edges))


def geometric_edges(points: Array, radius: float, max_degree: int | None = None) -> EdgeList:
    n = len(points)
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    edges: set[Tuple[int, int]] = set()
    for i in range(n):
        candidates = [j for j in range(n) if j != i and dists[i, j] <= radius]
        candidates = sorted(candidates, key=lambda j: dists[i, j])
        if max_degree is not None:
            candidates = candidates[:max_degree]
        for j in candidates:
            edges.add((min(i, j), max(i, j)))
    if not edges:
        return chain_edges(n)
    return tuple(sorted(edges))


def make_graph(kind: str, n: int, dim: int = 1, seed: int = 0) -> GraphSpec:
    rng = np.random.default_rng(seed)
    if kind == "chain":
        edges = chain_edges(n)
    elif kind == "cycle":
        edges = cycle_edges(n)
    elif kind == "star":
        edges = star_edges(n)
    elif kind == "grid":
        side = int(round(math.sqrt(n)))
        if side * side == n:
            edges = grid_edges(side, side)
        else:
            rows = max(1, n // 4)
            cols = max(1, n // rows)
            edges = grid_edges(rows, cols)
            n = rows * cols
    elif kind == "er":
        edges = er_edges(n, min(0.25, 3.0 / max(n, 1)), rng)
    elif kind == "geometric":
        points = rng.uniform(-1.0, 1.0, size=(n, max(2, dim)))
        edges = geometric_edges(points, radius=0.75, max_degree=6)
    else:
        raise ValueError(f"unknown graph kind: {kind}")
    adj, lap = edges_to_mats(n, edges)
    return GraphSpec(kind=kind, n=n, dim=dim, edges=edges, adjacency=adj, laplacian=lap.astype(np.float32))


def target_positions(n: int, dim: int, pattern: str = "sine") -> Array:
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if dim == 1:
        if pattern == "zero":
            return np.zeros((n, 1), dtype=np.float32)
        return (0.8 * np.sin(2.0 * np.pi * x))[:, None].astype(np.float32)
    if pattern == "circle":
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
        return np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    return np.stack([x - 0.5, 0.35 * np.sin(2.0 * np.pi * x)], axis=-1).astype(np.float32)

