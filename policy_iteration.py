"""
Policy Iteration on the estimated model.

Each PI iteration = one full policy evaluation (iterate the Bellman expectation
backup for the *current* policy to convergence) followed by one policy
improvement (act greedily w.r.t. the resulting V). PI stops once improvement
no longer changes the policy.

Same gamma=1.0 rationale as in value_iteration.py (episodic, acyclic, undiscounted total daily
cost). Same per-iteration protocol as VI: after every improvement step, freeze
the new policy and test it on the real environment for 30 episodes; that test
time is excluded from the reported training time.
"""
import pickle
import time

import numpy as np

from datacenter_env import DataCenterEnv
from mdp_utils import q_value, greedy_policy, evaluate_policy_on_env

GAMMA = 1.0
EVAL_THETA = 1e-6
EVAL_MAX_ITER = 1000
MAX_ITERATIONS = 100
N_TEST_EPISODES = 30


def policy_evaluation(policy, model, gamma, theta=EVAL_THETA, max_iter=EVAL_MAX_ITER):
    nS = model["nS"]
    V = np.zeros(nS)
    for _ in range(max_iter):
        new_V = np.empty(nS)
        delta = 0.0
        for s in range(nS):
            v = q_value(s, int(policy[s]), V, model, gamma)
            delta = max(delta, abs(v - V[s]))
            new_V[s] = v
        V = new_V
        if delta < theta:
            break
    return V


def policy_iteration(model, gamma=GAMMA, max_iterations=MAX_ITERATIONS,
                      env=None, n_test_episodes=N_TEST_EPISODES, verbose=True):
    nS, nA = model["nS"], model["nA"]
    policy = np.zeros(nS, dtype=int)   # start from the "always hold" policy
    history = []
    train_time = 0.0
    V = np.zeros(nS)

    for it in range(1, max_iterations + 1):
        t0 = time.perf_counter()
        V = policy_evaluation(policy, model, gamma)
        new_policy = greedy_policy(V, model, gamma)
        train_time += time.perf_counter() - t0

        stable = np.array_equal(new_policy, policy)
        changed = int(np.sum(new_policy != policy))
        policy = new_policy

        test_mean, test_std = (None, None)
        if env is not None:
            test_mean, test_std = evaluate_policy_on_env(policy, env, n_test_episodes)

        history.append({"iteration": it, "actions_changed": changed, "train_time_cum": train_time,
                         "test_mean_reward": test_mean, "test_std_reward": test_std})
        if verbose:
            msg = f"PI iter {it:3d}: actions_changed={changed:5d}, train_time={train_time:.2f}s"
            if test_mean is not None:
                msg += f", test avg reward={test_mean:.3f}"
            print(msg)

        if stable:
            break

    return V, policy, history, train_time


if __name__ == "__main__":
    with open("dynamics_model.pkl", "rb") as f:
        model = pickle.load(f)
    env = DataCenterEnv()

    V, policy, history, train_time = policy_iteration(model, env=env)

    with open("pi_results.pkl", "wb") as f:
        pickle.dump({"V": V, "policy": policy, "history": history,
                     "train_time": train_time, "gamma": GAMMA}, f)

    print(f"\nConverged in {len(history)} iterations, total train time {train_time:.2f}s")
    print(f"Final test avg reward: {history[-1]['test_mean_reward']:.3f}")

    # compare the two learned policies (VI vs PI) action-by-action
    try:
        with open("vi_results.pkl", "rb") as f:
            vi = pickle.load(f)
        diffs = int(np.sum(vi["policy"] != policy))
        print(f"\nActions where PI's and VI's policies differ: {diffs} / {model['nS']} states "
              f"({100 * diffs / model['nS']:.1f}%)")
        print(f"max |V_PI - V_VI| across states: {np.max(np.abs(V - vi['V'])):.4f}")
    except FileNotFoundError:
        pass
