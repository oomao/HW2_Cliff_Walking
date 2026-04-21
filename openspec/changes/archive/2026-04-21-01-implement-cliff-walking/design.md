# Design Notes

## Environment
- State: integer `row * 12 + col`, 48 states total.
- Actions: `0=up, 1=right, 2=down, 3=left`.
- Transitions: deterministic, clipped to grid. Stepping into any `(3, 1..10)` cell triggers cliff: reward −100, state reset to Start `(3, 0)` (non-terminal).
- Terminal: Goal `(3, 11)`.
- Rewards: −1 per step, −100 on cliff, 0 on terminal transition.

## Algorithms
Both agents share:
- `Q ∈ R^{48×4}` initialised to zeros.
- ε-greedy: with probability ε sample uniform action, else argmax (ties broken randomly to avoid bias toward action 0).

Update rules with α=0.5, γ=1.0 (Sutton-standard for Cliff Walking figure reproduction; also supports α=0.1 / γ=0.9 from the spec via CLI flags):
- **Q-learning:** `Q[s,a] += α (r + γ max_a' Q[s',a'] − Q[s,a])`
- **SARSA:** `Q[s,a] += α (r + γ Q[s',a'] − Q[s,a])` where `a'` is the ε-greedy action actually taken next step.

## Training protocol
- 500 episodes × 50 seeds → mean reward curve, consistent with the sample figure `result_sample.jpg`.
- Cap episode length at 500 steps to avoid pathological infinite loops early in training.

## Why reproduce Sutton parameters?
The reference plot uses ε=0.1, α=0.5. Reproducing the textbook shape (SARSA ≈ −20, Q-learning ≈ −40 asymptotic average) is the clearest evidence of correct implementation. We additionally expose the spec parameters (α=0.1, γ=0.9) via flags for completeness.

## Live demo
Static HTML in `docs/` consuming PNG + GIF artefacts copied from `artifacts/`. No runtime JS needed beyond layout; Pages serves from `/docs` on main.
