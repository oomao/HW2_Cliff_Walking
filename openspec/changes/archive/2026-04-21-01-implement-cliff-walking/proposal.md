# Proposal: Implement Cliff Walking with Q-learning & SARSA

## Why
DRL HW2 requires a side-by-side comparison of Q-learning (off-policy) and SARSA (on-policy) on the classic 4×12 Cliff Walking Gridworld. The deliverable must show learning curves, final policies, and a qualitative analysis of the risk/safety tradeoff between the two methods. A GitHub-hosted live demo makes the results reproducible and viewable without cloning.

## What Changes
- Add a `cliff_walking` Python package implementing the Gridworld environment per the spec (4×12 grid, cliff on the bottom row between Start and Goal, step reward −1, cliff reward −100, reset on cliff).
- Add `agents.py` with ε-greedy Q-learning and SARSA sharing a common tabular-Q base class.
- Add `train.py` that runs both algorithms for ≥500 episodes across multiple seeds (averaged over 50 runs per the reference result figure) and saves raw reward traces + learned Q-tables as `.npy`.
- Add `plots.py` producing (a) the reward-curve comparison figure matching the sample, (b) policy arrow grids for each algorithm mirroring the reference figure, and (c) an animated GIF of each greedy policy rolled out from Start to Goal.
- Add a GitHub Pages `docs/` site (static HTML + JS) that loads the generated figures and GIFs for a live demo.
- Add `README.md` with theory recap, parameter table, embedded results, and written analysis covering convergence speed, stability, risk appetite, and when to pick each method.
- Add `scripts/startup.sh` and `scripts/ending.sh` per the lecture workflow (pull / handover / openspec; wrap-up / archive / push).

## Impact
- Affected specs: new capability `cliff-walking`
- Affected code: new files only (greenfield repo)
- Runtime cost: training completes in under a minute on CPU; no external services.
