"""
Shared helpers used by both Value Iteration and Policy Iteration:
  - Bellman backup on the estimated model (dynamics_model.pkl)
  - greedy policy extraction from a value function
  - policy evaluation on the *real* Gymnasium environment (not the estimated
    model), which is what the "rewards per iteration" chart needs
"""
import numpy as np


def q_value(s, a, V, model, gamma):
    """Q(s, a) using the estimated model. Terminal states (last decision hour)
    have no meaningful continuation, so their Q-value is just the immediate
    (already-terminal-adjusted) reward, with no bootstrapped term."""
    R = model["R"][(s, a)]
    if model["terminal"][s]:
        return R
    q = R
    for s_next, p in model["P"][(s, a)].items():
        q += gamma * p * V[s_next]
    return q


def greedy_policy(V, model, gamma):
    nS, nA = model["nS"], model["nA"]
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        qs = [q_value(s, a, V, model, gamma) for a in range(nA)]
        policy[s] = int(np.argmax(qs))
    return policy


def evaluate_policy_on_env(policy, env, n_episodes=30, seed_base=10_000):
    """Roll the (deterministic) policy out on the *real* environment, not the
    estimated model, for n_episodes test episodes. Returns (mean, std) of the
    per-episode total return."""
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed_base + ep)
        done = False
        R = 0.0
        while not done:
            a = int(policy[s])
            s, r, done, _, _ = env.step(a)
            R += r
        returns[ep] = R
    return float(returns.mean()), float(returns.std())
