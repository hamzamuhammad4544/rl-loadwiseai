"""
Estimate the transition dynamics p(s', r | s, a) for DataCenterEnv.

The recipe used here is "run the agent for a large number of episodes
using a random policy, then average the observed values." Taken literally, that
means: reset only from the environment's own start distribution (hour 0, a couple
of battery levels), then follow a uniformly random policy for 24 steps, and pool
whatever (s, a, r, s') transitions show up.

The problem with doing *only* that here: with 2,592 states, most combinations of
(hour, battery, price, congestion, backlog) are reached only rarely by chance in a
handful of steps from a narrow start distribution -- some (state, action) pairs
would get zero or one sample after even a few thousand episodes, which is too noisy
to build a usable model from.

So this script keeps the "random policy" idea (the action taken from every state is
drawn uniformly at random, and every sample is a genuine environment transition --
nothing here is computed from the source code) but replaces "start only from hour
0" with an exploring start: it resets directly into *every* reachable state (using
the env's `options={"state": s}` hook we added to the env) and then takes one random
action, repeated many times. This is the standard "exploring starts" fix for
incomplete coverage in Monte-Carlo model estimation, and it lets us estimate
p(s', r | s, a) for every one of the 2,592 x 4 pairs, not just the ones a random
walk from hour 0 happens to stumble into.

Output: a pickled dict with
  P[(s, a)] -> {s_next: probability, ...}
  R[(s, a)] -> average observed reward
  terminal[s] -> True if s has hour == HOURS - 1 (last decision hour; episode ends
                 after acting there, so it has no meaningful continuation)
  counts[(s, a)] -> how many samples the estimate is based on
"""
import pickle
import time
from collections import defaultdict

import numpy as np

from datacenter_env import DataCenterEnv

SAMPLES_PER_SA = 60   # random-action samples collected from every (state, action)
SEED = 42


def build_model(env: DataCenterEnv, samples_per_sa: int = SAMPLES_PER_SA, seed: int = SEED):
    rng = np.random.default_rng(seed)
    nS, nA = env.nS, env.nA

    next_counts = defaultdict(lambda: defaultdict(int))   # (s,a) -> {s': count}
    reward_sum = defaultdict(float)                        # (s,a) -> sum of rewards
    sa_count = defaultdict(int)                             # (s,a) -> number of samples
    terminal = np.zeros(nS, dtype=bool)

    for s in range(nS):
        hour, soc, price, cong, solar, backlog = env.decode(s)
        if hour == env.HOURS - 1:
            terminal[s] = True     # last decision hour: nothing meaningful beyond it

        for a in range(nA):
            for _ in range(samples_per_sa):
                env.reset(seed=int(rng.integers(1_000_000)), options={"state": s})
                s_next, r, terminated, _, _ = env.step(a)
                next_counts[(s, a)][s_next] += 1
                reward_sum[(s, a)] += r
                sa_count[(s, a)] += 1

    P, R, counts = {}, {}, {}
    for key, s_next_counts in next_counts.items():
        total = sum(s_next_counts.values())
        P[key] = {sp: c / total for sp, c in s_next_counts.items()}
        R[key] = reward_sum[key] / sa_count[key]
        counts[key] = sa_count[key]

    return {"P": P, "R": R, "terminal": terminal, "counts": counts,
            "nS": nS, "nA": nA, "samples_per_sa": samples_per_sa, "seed": seed}


if __name__ == "__main__":
    env = DataCenterEnv()
    t0 = time.time()
    model = build_model(env)
    dt = time.time() - t0

    with open("dynamics_model.pkl", "wb") as f:
        pickle.dump(model, f)

    n_pairs = env.nS * env.nA
    n_estimated = len(model["R"])
    total_samples = sum(model["counts"].values())
    print(f"Built model in {dt:.1f}s")
    print(f"(state, action) pairs: {n_pairs}, all estimated: {n_estimated == n_pairs}")
    print(f"total environment steps used: {total_samples}")
    print(f"terminal states (hour == {env.HOURS - 1}): {model['terminal'].sum()}")

    # sanity: every estimated distribution should sum to 1
    bad = [k for k, d in model["P"].items() if abs(sum(d.values()) - 1.0) > 1e-9]
    print(f"P(.|s,a) rows that don't sum to 1: {len(bad)}")

    # spot check: from a fixed state, does "charge" action, on average, draw more
    # power (and so cost more) than "hold"? (soc=2, mid battery, so charging is legal)
    s0 = env.encode(10, 2, 1, 0, 0, 0)
    print("\nSpot check at state", env.decode(s0), "(hour=10, soc=2, price=medium, no congestion, no solar, no backlog)")
    for a, name in enumerate(["hold", "charge", "discharge", "defer"]):
        print(f"  action={name:10s} mean reward = {model['R'][(s0, a)]:.3f} "
              f"(from {model['counts'][(s0, a)]} samples, {len(model['P'][(s0, a)])} distinct next states)")

    # same hour/price/battery, but with full solar available: charging should
    # look much cheaper (less negative) because it's offset by free solar.
    s1 = env.encode(10, 2, 1, 0, 2, 0)
    print("\nSame state but with full solar available", env.decode(s1))
    for a, name in enumerate(["hold", "charge", "discharge", "defer"]):
        print(f"  action={name:10s} mean reward = {model['R'][(s1, a)]:.3f} "
              f"(from {model['counts'][(s1, a)]} samples)")
