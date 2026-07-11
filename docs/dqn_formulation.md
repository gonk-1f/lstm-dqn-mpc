# DQN Formulation

## Lower-layer meaning of `Q(s, a)`

In this project, `Q(s, a)` denotes the expected discounted cumulative control
value when the vessel microgrid is currently in state `s`, the lower-layer
controller applies discrete action `a`, and the learned policy is used
afterwards.

The control value is not an abstract game score. It is tied to physical control
quality:

- tracking the upper MPC fuel-cell and battery references
- satisfying the instantaneous load demand
- keeping battery SOC in a safe operating region
- avoiding excessive left/right imbalance
- avoiding aggressive power oscillations and constraint violations

## Bellman equation

The implemented one-step Bellman target is:

`y_t = r_t + gamma * max_a' Q_target(s_{t+1}, a') * (1 - done_t)`

This is used in [dqn_agent.py](/C:/Users/20883/OneDrive/Desktop/microgrid-mpc-master/src/dqn/agents/dqn_agent.py)
through the `bellman_target(...)` method and the MSE loss between:

- current estimate `Q(s_t, a_t)`
- Bellman target `y_t`

## Replay buffer

The replay buffer stores transitions:

- state `s_t`
- action `a_t`
- reward `r_t`
- next state `s_{t+1}`
- terminal flag `done_t`

Its role is to randomize training samples and reduce the temporal correlation
that naturally exists in sequential ship operating data. This mechanism is
implemented in [replay_buffer.py](/C:/Users/20883/OneDrive/Desktop/microgrid-mpc-master/src/dqn/memory/replay_buffer.py).

## Current lower-layer environments

- [ship_env_simple.py](/C:/Users/20883/OneDrive/Desktop/microgrid-mpc-master/src/envs/ship_env_simple.py):
  phase-1 single fuel-cell + single battery + single load
- [ship_env_dual_side.py](/C:/Users/20883/OneDrive/Desktop/microgrid-mpc-master/src/envs/ship_env_dual_side.py):
  dual-side left/right fuel-cell and battery allocation

## Note on online efficiency

DQN training can require thousands of iterations offline. Online deployment only
uses a forward pass of the trained Q-network, so the online computation burden
is far smaller than the offline learning burden.
