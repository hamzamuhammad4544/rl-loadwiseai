# rl-loadwiseai

Policy Iteration vs. Value Iteration on a data-center load-management MDP. Inspired by Loadwise.ai

## Files
- `datacenter_env.py` – the data-center environment
- `estimate_dynamics.py` – estimates transition/reward dynamics
- `mdp_utils.py` – shared MDP helpers
- `policy_iteration.py` / `value_iteration.py` – the two solvers
- `compare.py` – compares Policy Iteration and Value Iteration (rewards per iteration, running time) and generates the charts