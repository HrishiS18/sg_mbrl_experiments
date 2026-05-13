from __future__ import annotations

from typing import Optional

import numpy as np

from .hamiltonian import GraphHamiltonianEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None


class TrueGymEnv(gym.Env if gym is not None else object):
    """Gymnasium-compatible wrapper for model-free baselines."""

    metadata = {"render_modes": []}

    def __init__(self, env: GraphHamiltonianEnv, horizon: int = 200, seed: int = 0):
        if gym is None or spaces is None:  # pragma: no cover
            raise ImportError("gymnasium is required for TrueGymEnv")
        self.env = env.clone_with_seed(seed)
        self.horizon = horizon
        self.t = 0
        self.z: Optional[np.ndarray] = None
        high = np.full(env.state_dim, np.inf, dtype=np.float32)
        ahigh = np.full(env.action_dim, env.cfg.action_limit, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-ahigh, ahigh, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        self.t = 0
        self.z = self.env.sample_state()
        return self.z.copy(), {}

    def step(self, action):
        assert self.z is not None
        a = np.asarray(action, dtype=np.float32)
        reward = self.env.reward(self.z, a)
        self.z = self.env.step(self.z, a)
        self.t += 1
        terminated = False
        truncated = self.t >= self.horizon or (not np.all(np.isfinite(self.z))) or np.linalg.norm(self.z) > 100
        info = {"terminal_error": self.env.terminal_error(self.z)}
        return self.z.copy(), float(reward), terminated, truncated, info


class LearnedModelGymEnv(TrueGymEnv):
    """Gymnasium wrapper that steps through a learned model for MBPO-style training."""

    def __init__(self, env: GraphHamiltonianEnv, model, horizon: int = 50, seed: int = 0, device: str = "cpu"):
        super().__init__(env, horizon=horizon, seed=seed)
        self.model = model
        self.device = device

    def step(self, action):
        assert self.z is not None
        a = np.asarray(action, dtype=np.float32)
        reward = self.env.reward(self.z, a)
        pred = self.model.predict_np(self.z, a, self.env.graph_arrays(), device=self.device)[0]
        self.z = pred.astype(np.float32)
        self.t += 1
        terminated = False
        truncated = self.t >= self.horizon or (not np.all(np.isfinite(self.z))) or np.linalg.norm(self.z) > 100
        info = {"terminal_error": self.env.terminal_error(self.z)}
        return self.z.copy(), float(reward), terminated, truncated, info
