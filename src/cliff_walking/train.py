"""Train Q-learning and SARSA on Cliff Walking, averaged across seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .agents import Hyperparams, QLearningAgent, SarsaAgent
from .env import CliffWalkingEnv

MAX_STEPS_PER_EPISODE = 500
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def run_qlearning(hp: Hyperparams, episodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    env = CliffWalkingEnv(seed=seed)
    agent = QLearningAgent(hp, rng)
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            a = agent.act(s)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, s_next, done)
            total += r
            s = s_next
            if done:
                break
        rewards[ep] = total
    return rewards, agent.Q


def run_sarsa(hp: Hyperparams, episodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    env = CliffWalkingEnv(seed=seed)
    agent = SarsaAgent(hp, rng)
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        s = env.reset()
        a = agent.act(s)
        total = 0.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            s_next, r, done = env.step(a)
            a_next = agent.act(s_next) if not done else 0
            agent.update(s, a, r, s_next, a_next, done)
            total += r
            s, a = s_next, a_next
            if done:
                break
        rewards[ep] = total
    return rewards, agent.Q


def train_all(episodes: int, seeds: int, hp: Hyperparams, base_seed: int = 1000) -> dict:
    q_rewards = np.zeros((seeds, episodes))
    sarsa_rewards = np.zeros((seeds, episodes))
    q_final_Q = None
    sarsa_final_Q = None
    for i in range(seeds):
        q_rewards[i], q_Q = run_qlearning(hp, episodes, seed=base_seed + i)
        sarsa_rewards[i], s_Q = run_sarsa(hp, episodes, seed=base_seed + i)
        if i == 0:
            q_final_Q = q_Q
            sarsa_final_Q = s_Q
    return {
        "q_rewards": q_rewards,
        "sarsa_rewards": sarsa_rewards,
        "q_Q": q_final_Q,
        "sarsa_Q": sarsa_final_Q,
        "hp": hp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    hp = Hyperparams(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    result = train_all(args.episodes, args.seeds, hp, base_seed=1000)
    ref = train_all(args.episodes, args.seeds, hp, base_seed=9000)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.save(ARTIFACTS_DIR / "q_rewards.npy", result["q_rewards"])
    np.save(ARTIFACTS_DIR / "sarsa_rewards.npy", result["sarsa_rewards"])
    np.save(ARTIFACTS_DIR / "q_Q.npy", result["q_Q"])
    np.save(ARTIFACTS_DIR / "sarsa_Q.npy", result["sarsa_Q"])
    np.save(ARTIFACTS_DIR / "q_rewards_ref.npy", ref["q_rewards"])
    np.save(ARTIFACTS_DIR / "sarsa_rewards_ref.npy", ref["sarsa_rewards"])
    print(f"Saved artifacts to {ARTIFACTS_DIR}")
    print(
        f"Final mean reward (last 50 eps): "
        f"Q-learning={result['q_rewards'][:, -50:].mean():.2f}, "
        f"SARSA={result['sarsa_rewards'][:, -50:].mean():.2f}"
    )


if __name__ == "__main__":
    main()
