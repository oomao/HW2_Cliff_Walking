# cliff-walking Specification

## Purpose
TBD - created by archiving change 01-implement-cliff-walking. Update Purpose after archive.
## Requirements
### Requirement: Gridworld Environment
The system SHALL provide a 4×12 Cliff Walking gridworld environment with deterministic transitions.

#### Scenario: Start state
- **WHEN** the environment is reset
- **THEN** the agent state is the bottom-left cell (row 3, col 0)

#### Scenario: Step reward
- **WHEN** the agent takes any non-terminal, non-cliff action
- **THEN** the returned reward is −1

#### Scenario: Cliff penalty
- **WHEN** the agent enters any cell in row 3, columns 1 through 10
- **THEN** the returned reward is −100 and the state is reset to Start without terminating the episode

#### Scenario: Goal termination
- **WHEN** the agent enters the bottom-right cell (row 3, col 11)
- **THEN** the episode terminates

### Requirement: Tabular Q-learning Agent
The system SHALL implement an off-policy tabular Q-learning agent using ε-greedy behaviour.

#### Scenario: Off-policy update
- **WHEN** a transition (s, a, r, s') is observed
- **THEN** `Q[s,a]` is updated toward `r + γ · max_a' Q[s', a']`

### Requirement: Tabular SARSA Agent
The system SHALL implement an on-policy tabular SARSA agent using ε-greedy behaviour.

#### Scenario: On-policy update
- **WHEN** a transition (s, a, r, s', a') is observed where a' was sampled ε-greedily
- **THEN** `Q[s,a]` is updated toward `r + γ · Q[s', a']`

### Requirement: Training Protocol
The system SHALL train each algorithm for at least 500 episodes and average results over at least 50 seeds.

#### Scenario: Artifact output
- **WHEN** training completes
- **THEN** per-episode reward traces and final Q-tables are persisted to the `artifacts/` directory

### Requirement: Result Visualisation
The system SHALL generate a reward-curve comparison, per-algorithm policy grids, and animated GIF rollouts.

#### Scenario: Reward curve
- **WHEN** visualisation runs
- **THEN** a PNG comparing mean per-episode reward for Q-learning and SARSA is written to `artifacts/reward_curve.png`

#### Scenario: Policy grid
- **WHEN** visualisation runs
- **THEN** one arrow-grid PNG per algorithm is written showing the greedy action at each non-terminal cell

#### Scenario: Animated rollout
- **WHEN** visualisation runs
- **THEN** an animated GIF of a greedy rollout from Start to Goal is written for each algorithm

### Requirement: Live Demo Page
The system SHALL provide a static HTML page under `docs/` that embeds the generated figures and GIFs, suitable for hosting on GitHub Pages.

#### Scenario: Pages-ready assets
- **WHEN** the repository is configured to serve `/docs` via GitHub Pages
- **THEN** visiting the published URL shows the reward curve, both policy grids, and both animated GIFs without additional build steps

