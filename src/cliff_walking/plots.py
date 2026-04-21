"""Generate reward curve, policy grids, and animated GIF rollouts."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .env import (
    ACTION_ARROWS,
    COLS,
    GOAL,
    ROWS,
    START,
    CliffWalkingEnv,
    is_cliff,
    state_to_rc,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = np.full(window - 1, x[0])
    padded = np.concatenate([pad, x])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_reward_curve(
    q_rewards: np.ndarray,
    sarsa_rewards: np.ndarray,
    out_path: Path,
    smooth: int = 10,
    q_rewards_ref: np.ndarray | None = None,
    sarsa_rewards_ref: np.ndarray | None = None,
) -> None:
    q_mean = q_rewards.mean(axis=0)
    s_mean = sarsa_rewards.mean(axis=0)
    episodes = np.arange(1, q_mean.size + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(episodes, moving_average(s_mean, smooth), color="#1abcc6", lw=1.8, label="Sarsa")
    ax.plot(
        episodes, moving_average(q_mean, smooth), color="#d6322e", lw=1.8, label="Q-learning"
    )
    if sarsa_rewards_ref is not None:
        ax.plot(
            episodes,
            moving_average(sarsa_rewards_ref.mean(axis=0), smooth),
            color="#1abcc6",
            lw=1.2,
            linestyle=":",
            label="Sarsa, Sutton Pub.",
        )
    if q_rewards_ref is not None:
        ax.plot(
            episodes,
            moving_average(q_rewards_ref.mean(axis=0), smooth),
            color="#d6322e",
            lw=1.2,
            linestyle=":",
            label="Q-learning, Sutton Pub.",
        )
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward Sum for Episode")
    ax.set_title(
        f"Sarsa Vs. Q-Learning Cliff Walking\n"
        f"Epsilon=0.1, Alpha=0.5\n"
        f"(averaged over {q_rewards.shape[0]} runs)"
    )
    ax.set_ylim(-100, 0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _draw_grid(ax, title: str) -> None:
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    for r in range(ROWS):
        for c in range(COLS):
            if is_cliff(r, c):
                ax.add_patch(
                    patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, facecolor="#bfe3ef", edgecolor="black"
                    )
                )
            else:
                ax.add_patch(
                    patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, facecolor="white", edgecolor="black"
                    )
                )
    ax.text(START[1], START[0] + 0.05, "Start", ha="center", va="center", fontsize=10)
    ax.text(GOAL[1], GOAL[0] + 0.05, "Goal", ha="center", va="center", fontsize=10)
    cliff_mid_col = (COLS - 1) / 2
    ax.text(cliff_mid_col, ROWS - 1 + 0.05, "Cliff", ha="center", va="center", fontsize=11)


def _greedy_action_grid(Q: np.ndarray) -> np.ndarray:
    grid = np.full((ROWS, COLS), -1, dtype=int)
    for s in range(Q.shape[0]):
        r, c = state_to_rc(s)
        if is_cliff(r, c) or (r, c) == GOAL:
            continue
        grid[r, c] = int(np.argmax(Q[s]))
    return grid


def plot_policy(Q: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    _draw_grid(ax, title)
    grid = _greedy_action_grid(Q)
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] == -1:
                continue
            ax.text(
                c,
                r,
                ACTION_ARROWS[grid[r, c]],
                ha="center",
                va="center",
                fontsize=18,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _rollout_states(Q: np.ndarray, max_steps: int = 60) -> list[int]:
    env = CliffWalkingEnv(seed=0)
    s = env.reset()
    states = [s]
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, _, done = env.step(a)
        states.append(s)
        if done:
            break
    return states


def render_rollout_gif(Q: np.ndarray, title: str, out_path: Path) -> None:
    states = _rollout_states(Q)
    frames = []
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(exist_ok=True)
    for i, s in enumerate(states):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        _draw_grid(ax, f"{title}  (step {i})")
        r, c = state_to_rc(s)
        ax.add_patch(patches.Circle((c, r), 0.32, facecolor="#ff9f1c", edgecolor="black", zorder=3))
        fig.tight_layout()
        frame_path = tmp_dir / f"frame_{i:03d}.png"
        fig.savefig(frame_path, dpi=110)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))
    imageio.mimsave(out_path, frames, duration=0.35, loop=0)
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()


def main() -> None:
    q_rewards = np.load(ARTIFACTS_DIR / "q_rewards.npy")
    sarsa_rewards = np.load(ARTIFACTS_DIR / "sarsa_rewards.npy")
    q_Q = np.load(ARTIFACTS_DIR / "q_Q.npy")
    sarsa_Q = np.load(ARTIFACTS_DIR / "sarsa_Q.npy")
    q_ref_path = ARTIFACTS_DIR / "q_rewards_ref.npy"
    s_ref_path = ARTIFACTS_DIR / "sarsa_rewards_ref.npy"
    q_ref = np.load(q_ref_path) if q_ref_path.exists() else None
    s_ref = np.load(s_ref_path) if s_ref_path.exists() else None

    plot_reward_curve(
        q_rewards,
        sarsa_rewards,
        ARTIFACTS_DIR / "reward_curve.png",
        q_rewards_ref=q_ref,
        sarsa_rewards_ref=s_ref,
    )
    plot_policy(q_Q, "Q-learning policy", ARTIFACTS_DIR / "policy_qlearning.png")
    plot_policy(sarsa_Q, "SARSA policy", ARTIFACTS_DIR / "policy_sarsa.png")
    render_rollout_gif(q_Q, "Q-learning rollout", ARTIFACTS_DIR / "rollout_qlearning.gif")
    render_rollout_gif(sarsa_Q, "SARSA rollout", ARTIFACTS_DIR / "rollout_sarsa.gif")
    print(f"Figures written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
