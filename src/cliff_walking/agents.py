"""Tabular Q-learning and SARSA agents for Cliff Walking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import N_ACTIONS, N_STATES


@dataclass
class Hyperparams:
    alpha: float = 0.5
    gamma: float = 1.0
    epsilon: float = 0.1


class _TabularAgent:
    def __init__(self, hp: Hyperparams, rng: np.random.Generator):
        self.hp = hp
        self.rng = rng
        self.Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    def act(self, state: int) -> int:
        if self.rng.random() < self.hp.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        q_row = self.Q[state]
        max_q = q_row.max()
        best = np.flatnonzero(q_row == max_q)
        return int(self.rng.choice(best))

    def greedy(self, state: int) -> int:
        q_row = self.Q[state]
        best = np.flatnonzero(q_row == q_row.max())
        return int(self.rng.choice(best))


class QLearningAgent(_TabularAgent):
    name = "Q-learning"

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        target = r if done else r + self.hp.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.hp.alpha * (target - self.Q[s, a])


class SarsaAgent(_TabularAgent):
    name = "SARSA"

    def update(
        self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool
    ) -> None:
        target = r if done else r + self.hp.gamma * self.Q[s_next, a_next]
        self.Q[s, a] += self.hp.alpha * (target - self.Q[s, a])
