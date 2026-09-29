"""
Compare PI and VI on two metrics --
rewards per iteration (test-time convergence) and running time (train-time
only, testing excluded) -- and produce the charts for the report.
"""
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl

# -- chart chrome (validated categorical palette; blue=VI, orange=PI) -------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
VI_COLOR = "#2a78d6"   # categorical slot 1
PI_COLOR = "#eb6834"   # categorical slot 2

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})

with open("vi_results.pkl", "rb") as f:
    vi = pickle.load(f)
with open("pi_results.pkl", "rb") as f:
    pi = pickle.load(f)

vi_hist, pi_hist = vi["history"], pi["history"]

# ---------------------------------------------------------------- Chart 1 --
# Rewards per iteration: average test-episode return vs. iteration number.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.grid(True, axis="y", zorder=0)
ax.plot([h["iteration"] for h in vi_hist], [h["test_mean_reward"] for h in vi_hist],
        color=VI_COLOR, linewidth=2, marker="o", markersize=4, label="Value Iteration", zorder=3)
ax.plot([h["iteration"] for h in pi_hist], [h["test_mean_reward"] for h in pi_hist],
        color=PI_COLOR, linewidth=2, marker="o", markersize=4, label="Policy Iteration", zorder=3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_xlabel("Iteration")
ax.set_ylabel("Average test-episode reward (30 episodes)")
ax.set_title("Rewards per iteration: VI vs. PI", color=INK, fontsize=13, fontweight="bold", loc="left")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("chart_rewards_per_iteration.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Chart 2 --
# Reward vs. wall-clock training time -- same data, x-axis is cumulative
# training seconds instead of iteration count, so the two algorithms' very
# different per-iteration cost is visible.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.grid(True, axis="y", zorder=0)
ax.plot([h["train_time_cum"] for h in vi_hist], [h["test_mean_reward"] for h in vi_hist],
        color=VI_COLOR, linewidth=2, marker="o", markersize=4, label="Value Iteration", zorder=3)
ax.plot([h["train_time_cum"] for h in pi_hist], [h["test_mean_reward"] for h in pi_hist],
        color=PI_COLOR, linewidth=2, marker="o", markersize=4, label="Policy Iteration", zorder=3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_xlabel("Cumulative training time (s, testing excluded)")
ax.set_ylabel("Average test-episode reward (30 episodes)")
ax.set_title("Reward vs. training time: VI vs. PI", color=INK, fontsize=13, fontweight="bold", loc="left")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("chart_reward_vs_time.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Chart 3 --
# Running time comparison: total training time, bar chart.
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.grid(True, axis="y", zorder=0)
bars = ax.bar(["Value Iteration", "Policy Iteration"], [vi["train_time"], pi["train_time"]],
              color=[VI_COLOR, PI_COLOR], width=0.5, zorder=3)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.2f}s", (b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                textcoords="offset points", ha="center", color=INK, fontsize=10)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_ylabel("Total training time (s)")
ax.set_title("Running time: VI vs. PI", color=INK, fontsize=13, fontweight="bold", loc="left")
fig.tight_layout()
fig.savefig("chart_running_time.png", dpi=160)
plt.close(fig)

# --------------------------------------------------------- printed summary --
print("=== Summary ===")
print(f"VI: {len(vi_hist)} iterations, {vi['train_time']:.3f}s training, "
      f"final test reward {vi_hist[-1]['test_mean_reward']:.3f}")
print(f"PI: {len(pi_hist)} iterations, {pi['train_time']:.3f}s training, "
      f"final test reward {pi_hist[-1]['test_mean_reward']:.3f}")
print(f"VI mean time/iteration: {vi['train_time'] / len(vi_hist) * 1000:.2f} ms")
print(f"PI mean time/iteration: {pi['train_time'] / len(pi_hist) * 1000:.2f} ms")
print("Charts written: chart_rewards_per_iteration.png, chart_reward_vs_time.png, chart_running_time.png")
