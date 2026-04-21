# Tasks: 01-implement-cliff-walking

## 1. Environment
- [x] 1.1 Implement `CliffWalkingEnv` (4×12 grid, Sutton & Barto Example 6.6 semantics)
- [x] 1.2 Unit-smoke-test env: start state, cliff reset, goal termination, reward tallies

## 2. Agents
- [x] 2.1 Tabular `QLearningAgent` with ε-greedy behavior and off-policy max-backup
- [x] 2.2 Tabular `SarsaAgent` with ε-greedy behavior and on-policy backup
- [x] 2.3 Shared helpers: ε-greedy action selection, Q-table init, greedy-policy extraction

## 3. Training
- [x] 3.1 Training loop averaging 50 seeds × 500 episodes per algorithm
- [x] 3.2 Persist per-episode reward traces and final Q-tables to `artifacts/`

## 4. Visualisation
- [x] 4.1 Reward curve (Q-learning vs SARSA, moving average)
- [x] 4.2 Policy arrow-grid for each algorithm (matches reference figure)
- [x] 4.3 Animated GIF of greedy rollout for each algorithm

## 5. Live demo
- [x] 5.1 `docs/index.html` with figures + GIFs + short writeup
- [x] 5.2 Configure repo for GitHub Pages (branch `main`, folder `/docs`)

## 6. Report
- [x] 6.1 README.md with theory, parameters, results, and analysis

## 7. Dev scripts
- [x] 7.1 `scripts/startup.sh` (pull, read handover, openspec status)
- [x] 7.2 `scripts/ending.sh` (update tasks, archive, handover, push)

## 8. Wrap-up
- [x] 8.1 Archive change under `openspec/changes/archive/`
- [x] 8.2 Commit & push to https://github.com/oomao/HW2_Cliff_Walking.git
