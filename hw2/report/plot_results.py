"""Generate report figures for HW2 from exp/*/log.csv.

Usage (from hw2/):  uv run report/plot_results.py
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "exp")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Categorical palette, fixed order (validated reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(prefix):
    """Load log.csv of the single run whose directory starts with `prefix`_sd."""
    matches = sorted(glob.glob(os.path.join(EXP, f"{prefix}_sd*")))
    if not matches:
        raise FileNotFoundError(prefix)
    return pd.read_csv(os.path.join(matches[-1], "log.csv"))


def plot(runs, title, fname, y="Eval_AverageReturn", ylabel="Eval average return",
         hline=None, vline=None, xmax=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (label, prefix) in enumerate(runs):
        df = load(prefix)
        ax.plot(df["Train_EnvstepsSoFar"], df[y], color=C[i], lw=2, label=label)
    if hline is not None:
        ax.axhline(hline, color=MUTED, lw=1, ls="--")
    if vline is not None:
        ax.axvline(vline, color=MUTED, lw=1, ls=":")
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Environment steps (Train_EnvstepsSoFar)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    print("saved", fname)


# Experiment 1: CartPole
for tag, label in [("", "small batch (b=1000)"), ("_lb", "large batch (b=4000)")]:
    plot([
        ("vanilla", f"CartPole-v0_cartpole{tag}"),
        ("reward-to-go", f"CartPole-v0_cartpole{tag}_rtg"),
        ("adv. norm.", f"CartPole-v0_cartpole{tag}_na"),
        ("reward-to-go + adv. norm.", f"CartPole-v0_cartpole{tag}_rtg_na"),
    ], f"CartPole-v0, {label}", f"cartpole{tag or '_sb'}.png", hline=200)

# Experiment 2: HalfCheetah
plot([
    ("no baseline", "HalfCheetah-v4_cheetah"),
    ("baseline (bgs=5, blr=0.01)", "HalfCheetah-v4_cheetah_baseline"),
    ("baseline, bgs=1", "HalfCheetah-v4_cheetah_baseline_bgs1"),
    ("baseline, blr=0.001", "HalfCheetah-v4_cheetah_baseline_blr0.001"),
], "HalfCheetah-v4: eval return", "cheetah_eval_return.png", hline=300)

# Baseline loss: keep the same color per run as in the eval-return figure (slots 2-4)
fig, ax = plt.subplots(figsize=(8, 4.8))
for i, (label, prefix) in enumerate([
    ("baseline (bgs=5, blr=0.01)", "HalfCheetah-v4_cheetah_baseline"),
    ("baseline, bgs=1", "HalfCheetah-v4_cheetah_baseline_bgs1"),
    ("baseline, blr=0.001", "HalfCheetah-v4_cheetah_baseline_blr0.001"),
], start=1):
    df = load(prefix)
    ax.plot(df["Train_EnvstepsSoFar"], df["Baseline Loss"], color=C[i], lw=2, label=label)
ax.set_title("HalfCheetah-v4: baseline (critic) loss", loc="left")
ax.set_xlabel("Environment steps (Train_EnvstepsSoFar)")
ax.set_ylabel("Baseline MSE loss")
ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cheetah_baseline_loss.png"), dpi=150)
plt.close(fig)
print("saved cheetah_baseline_loss.png")

# Experiment 3: LunarLander GAE
plot([(f"λ = {lam}", f"LunarLander-v2_lunar_lander_lambda{lam}")
      for lam in ["0", "0.95", "0.98", "0.99", "1"]],
     "LunarLander-v2: GAE λ sweep", "lunar_lander_gae.png", hline=150)

# Experiment 4: InvertedPendulum
plot([
    ("default (b=5000)", "InvertedPendulum-v4_pendulum"),
    ("tuned (b=1000, rtg, na, GAE λ=0.98, γ=0.99)", "InvertedPendulum-v4_pendulum_D_b1000_gae"),
], "InvertedPendulum-v4: default vs. tuned", "pendulum.png", hline=1000, vline=100_000)
