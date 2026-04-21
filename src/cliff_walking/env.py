"""4x12 Cliff Walking gridworld (Sutton & Barto, Example 6.6)."""
from __future__ import annotations

import numpy as np

ROWS, COLS = 4, 12
N_STATES = ROWS * COLS
N_ACTIONS = 4
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
ACTION_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}

START = (ROWS - 1, 0)
GOAL = (ROWS - 1, COLS - 1)


def rc_to_state(row: int, col: int) -> int:
    return row * COLS + col


def state_to_rc(state: int) -> tuple[int, int]:
    return divmod(state, COLS)


def is_cliff(row: int, col: int) -> bool:
    return row == ROWS - 1 and 1 <= col <= COLS - 2


class CliffWalkingEnv:
    """Deterministic 4x12 Cliff Walking environment."""

    rows = ROWS
    cols = COLS
    n_states = N_STATES
    n_actions = N_ACTIONS
    start_state = rc_to_state(*START)
    goal_state = rc_to_state(*GOAL)

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.state = self.start_state

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        row, col = state_to_rc(self.state)
        dr, dc = ACTION_DELTAS[action]
        new_row = max(0, min(ROWS - 1, row + dr))
        new_col = max(0, min(COLS - 1, col + dc))

        if is_cliff(new_row, new_col):
            reward = -100.0
            self.state = self.start_state
            done = False
        else:
            self.state = rc_to_state(new_row, new_col)
            reward = -1.0
            done = self.state == self.goal_state
        return self.state, reward, done
