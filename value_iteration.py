"""
Value Iteration on the estimated model.

Standard synchronous VI: sweep every state, replace V(s) with max_a Q(s,a),
repeat until the largest change (delta) drops below theta.

gamma = 1.0: the problem is an episodic, at-most-24-step MDP with no cycles
(hour strictly increases each step and the last decision hour is absorbing),
so there's no infinite-horizon sum to worry about and discounting isn't needed
to keep values finite -- gamma=1 just means "minimize total cost over the day",
which is the actual objective we want.

After *every* iteration we freeze the current greedy
policy and test it on the real environment for 30 episodes (>10),
recording the average return. Training time (the Bellman sweeps + policy
extraction) is timed separately and does not include this testing time.
"""
import pickle
import time

import numpy as np

from datacenter_env import DataCenterEnv
from mdp_utils import q_value, greedy_policy, evaluate_policy_on_env

GAMMA = 1.0
THETA = 1e-6
MAX_ITERATIONS = 100
N_TEST_EPISODES = 30


def value_iteration(model, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITERATIONS,
                     env=None, n_test_episodes=N_TEST_EPISODES, verbose=True):
    nS, nA = model["nS"], model["nA"]
    V = np.zeros(nS)
    history = []  # one row per iteration: delta, cumulative train time, test reward
    train_time = 0.0

    for it in range(1, max_iterations + 1):
        t0 = time.perf_counter()
        new_V = np.empty(nS)
        delta = 0.0
        for s in range(nS):
            qs = [q_value(s, a, V, model, gamma) for a in range(nA)]
            best = max(qs)
            delta = max(delta, abs(best - V[s]))
            new_V[s] = best
        V = new_V
        policy = greedy_policy(V, model, gamma)   # extraction counts as "training"
        train_time += time.perf_counter() - t0

        test_mean, test_std = (None, None)
        if env is not None:
            test_mean, test_std = evaluate_policy_on_env(policy, env, n_test_episodes)

        history.append({"iteration": it, "delta": delta, "train_time_cum": train_time,
                         "test_mean_reward": test_mean, "test_std_reward": test_std})
        if verbose:
            msg = f"VI iter {it:3d}: delta={delta:.5f}, train_time={train_time:.2f}s"
            if test_mean is not None:
                msg += f", test avg reward={test_mean:.3f}"
            print(msg)

        if delta < theta:
            break

    return V, policy, history, train_time


if __name__ == "__main__":
    with open("dynamics_model.pkl", "rb") as f:
        model = pickle.load(f)
    env = DataCenterEnv()

    V, policy, history, train_time = value_iteration(model, env=env)

    with open("vi_results.pkl", "wb") as f:
        pickle.dump({"V": V, "policy": policy, "history": history,
                     "train_time": train_time, "gamma": GAMMA}, f)

    print(f"\nConverged in {len(history)} iterations, total train time {train_time:.2f}s")
    print(f"Final test avg reward: {history[-1]['test_mean_reward']:.3f}")

    # sanity: what does the learned policy do from the standard start state
    # (hour 0, half-charged battery, price/congestion drawn from hour-0 dist)?
    rng = np.random.default_rng(0)
    s, _ = env.reset(seed=0)
    print("\nGreedy-policy rollout from a sample start state:")
    action_names = ["hold", "charge", "discharge", "defer"]
    for step in range(24):
        a = int(policy[s])
        hour, soc, price, cong, solar, backlog = env.decode(s)
        print(f"  hour={hour:2d} soc={soc} price={price} cong={cong} solar={solar} backlog={backlog} -> {action_names[a]}")
        s, r, done, _, _ = env.step(a)
        if done:
            break
