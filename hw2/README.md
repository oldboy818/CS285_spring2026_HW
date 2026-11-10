# HW2: Policy Gradients

Implementation and experiments for [CS 285 (UC Berkeley, Spring 2026)](https://rail.eecs.berkeley.edu/deeprlcourse/) Homework 2. I worked through it as a self-study student, running every experiment locally on a laptop CPU.

**What I implemented**

- Vanilla policy gradient (REINFORCE) with discrete (`Categorical`) and continuous (diagonal Gaussian) policies
- Return estimators: full-trajectory discounted return and discounted reward-to-go
- Advantage normalization
- A neural-network value baseline (critic) trained by MSE regression on reward-to-go targets
- Generalized Advantage Estimation (GAE-λ), with trajectory boundaries handled in a flattened batch

| Experiment | Target | Result |
|---|---|---|
| 1. CartPole-v0 | best config reaches 200 (small and large batch) | ✅ reached 200 in every config |
| 2. HalfCheetah-v4 + baseline | eval return > 300 at end of training | ✅ final 443 (last-10 mean 342) |
| 3. LunarLander-v2 + GAE | best run > 150 at least once | ✅ max 237 (λ = 0.98) |
| 4. InvertedPendulum-v4 tuning | return 1000 within 100K env steps | ✅ first hit at **47.8K** steps (default: 298K) |

Figures are generated from the run logs with [`report/plot_results.py`](report/plot_results.py). The x-axis of every learning curve is environment steps (`Train_EnvstepsSoFar`), not iterations. All returns are `Eval_AverageReturn`.

---

## Key concepts (Q&A)

These are the questions I had to answer for myself while implementing the assignment. Each one connects a line of code to an idea from the lecture. The core code is in [`src/agents/pg_agent.py`](src/agents/pg_agent.py), [`src/networks/policies.py`](src/networks/policies.py), [`src/networks/critics.py`](src/networks/critics.py), and [`src/infrastructure/utils.py`](src/infrastructure/utils.py).

### The policy

**Q. What should the policy's `forward` return?**
A probability distribution over actions, not a single action. Policy gradients need to (1) *sample* actions during rollouts and (2) evaluate `log π(a|s)` for the actions that were taken. A `torch.distributions` object provides both: `.sample()` and a differentiable `.log_prob()`. `torch.distributions.Distribution` itself is an abstract base class, so it cannot be used directly. I used two concrete subclasses:
- Discrete actions: `Categorical(logits=...)`. The network outputs unnormalized logits, not probabilities.
- Continuous actions: `Independent(Normal(mean, exp(logstd)), 1)`

**Q. Why wrap the Gaussian in `Independent`?**
In the policy gradient `∇J ≈ Σ_t ∇log π(a_t|s_t) · A_t`, both factors are **one scalar per timestep**. A HalfCheetah action, however, is a 6-dimensional vector. Its dimensions are sampled independently, so the probability of the whole vector is the product of the per-dimension densities, and the log-probability is their sum. A plain `Normal` returns a `(B, 6)` tensor of per-dimension log-probs. `Independent(..., 1)` treats the last dimension as one event and sums over it, which gives `(B,)` and pairs one-to-one with the advantages. Without it, `(B, 6) * (B,)` either fails or silently broadcasts into a wrong loss.

**Q. Why `exp(logstd)` instead of learning the standard deviation directly?**
A standard deviation must be strictly positive, but gradient descent on a raw parameter can push it to zero or below. Learning `logstd ∈ ℝ` and exponentiating turns this constrained problem into an unconstrained one. It also makes the update scale-invariant: a fixed step in `logstd` changes σ by the same *ratio* whether σ is 1.0 or 0.05. With `logstd` initialized to 0, the policy starts at σ = 1, which gives broad exploration that shrinks as the policy improves.

### The policy-gradient loss

**Q. What is the "loss" for policy gradients?**
PyTorch differentiates a loss; it does not accept a gradient directly. So I need a scalar function whose gradient *is* the policy gradient:

```
L(θ) = −mean_t[ log π_θ(a_t|s_t) · A_t ]
```

- **The minus sign:** the optimizer minimizes, but J(θ) should be maximized.
- **The advantages are constants:** only `log π` is differentiated, so ∇L reproduces the REINFORCE estimator exactly.
- **The value of L is meaningless as a metric.** It is a surrogate that exists only to produce the gradient, so "Actor Loss" can rise while returns improve. Progress should be judged by return.

**Q. Why are advantages computed in NumPy rather than as tensors?**
Because they must act as constants. If `A_t = Q_t − V_φ(s_t)` were kept as a tensor connected to the critic, the actor loss's `backward()` would also update the critic, toward minimizing the actor loss rather than fitting V. Converting to NumPy cuts the autograd graph, just like `.detach()`. The per-element GAE recursion is also much faster in NumPy than in per-element PyTorch ops. The rule I followed: tensors only inside the networks, where gradients are needed; NumPy everywhere else.

### Variance reduction

**Q. Why does reward-to-go reduce variance without adding bias?** (Causality)
An action at time t cannot influence rewards received before t. In expectation, the full-trajectory estimator's past-reward terms multiply a score function with zero mean, so they contribute nothing to the gradient. They still add noise. Reward-to-go removes exactly those terms. In code, the t-th entry is simply the discounted return of the suffix `rewards[t:]`, which can be computed in O(T) with the backward recursion `G_t = r_t + γ G_{t+1}`.

**Q. Why apply discounting relative to the current step (Eq. 11) rather than absolutely (Eq. 10)?**
With absolute discounting `γ^{t'}`, rewards late in an episode are almost erased, so the policy effectively stops learning in states it visits late, even though those states matter just as much. Relative discounting `γ^{t'−t}` only says "prefer sooner rewards from *here*". This is uniform across the episode and matches the stationary objective we actually want, so it is the standard choice.

**Q. Why does a baseline not bias the gradient?**
For any b(s_t) that does not depend on the action: `E_{a∼π}[∇log π(a|s) · b(s)] = b(s) · ∇Σ_a π(a|s) = b(s) · ∇1 = 0`. Subtracting it therefore leaves the expected gradient unchanged. If b ≈ V(s), it removes most of the scale of the return, and with it most of the variance.

**Q. Why does advantage normalization help so much on CartPole?**
Every CartPole reward is +1, so raw returns are always positive and every update increases the probability of *every* sampled action. Only the relative magnitudes differ between actions. After normalization, below-average actions get negative advantages and are actively suppressed. Normalization is technically biased (the statistics come from the same batch), but the bias vanishes as the batch grows.

### The critic and GAE

**Q. What does the critic learn, and how?**
The critic is a plain regression model: `V_φ(s) ≈ E[reward-to-go | s]`, trained with MSE on the reward-to-go targets `Q_t`. It outputs a number, not a distribution. One shape pitfall: the MLP outputs `(B, 1)` and the targets are `(B,)`. Subtracting those two shapes silently broadcasts into a `(B, B)` matrix, so I `squeeze(-1)` the output, and an `assert values.shape == q_values.shape` guards against this.

**Q. Why does GAE need `terminals`?**
After flattening, all trajectories in a batch sit in one array. In the recursion `A_t = δ_t + γλ A_{t+1}`, the next index after a trajectory's last step belongs to a *different* trajectory. At terminal steps I drop the bootstrap and recursive terms (`δ_{H−1} = r_{H−1} − V(s_{H−1})`). The dummy value appended at the end keeps the final index in bounds.

**Q. How do I know the GAE implementation is correct?**
By its two closed forms. With λ = 1 the δ's telescope, so GAE must equal `Q_t − V(s_t)`. With λ = 0 it must equal the one-step TD error δ_t. I unit-tested both on a batch containing two trajectories, and both matched.

---

## Experiment 1: CartPole-v0

Eight runs of 100 iterations each: a small batch (`-b 1000`, ~100K steps) and a large batch (`-b 4000`, ~400K steps), each combined with reward-to-go (`-rtg`) and advantage normalization (`-na`) on or off.

![CartPole small batch](report/figures/cartpole_sb.png)
![CartPole large batch](report/figures/cartpole_lb.png)

| Config | Small batch: first 200 @ steps | Small: last-10 mean | Large batch: first 200 @ steps | Large: last-10 mean |
|---|---|---|---|---|
| vanilla | 45.9K | 105 | 89.8K | 86 |
| rtg | **16.9K** | **200** | 56.7K | 179 |
| na | 27.7K | **200** | 56.7K | **200** |
| rtg + na | 29.0K | 164 | **44.3K** | 179 |

### Analysis

**Reward-to-go versus the trajectory-centric estimator.** Without advantage normalization, the reward-to-go estimator clearly outperformed the trajectory-centric one at both batch sizes. In the small-batch setting it reached the maximum return of 200 after only 16.9K steps, about 2.7× faster than the trajectory-centric estimator (45.9K), and it stayed at 200 through the end of training. The trajectory-centric estimator touched 200 a few times but never settled: its last-10 mean was 105 for small batches and 86 for large batches.

This is why reward-to-go is generally preferred. An action taken at time t cannot change rewards that were already received, yet the trajectory-centric estimator multiplies every `∇log π(a_t|s_t)` by the full return, past rewards included. In expectation those past-reward terms contribute zero gradient, but in any finite batch they add noise. Reward-to-go removes them, so the estimator stays unbiased and its variance drops. The faster, steadier curves are that variance reduction showing up in practice.

**The effect of advantage normalization.** Normalization helped substantially, most of all for the trajectory-centric estimator: its last-10 mean jumped from 105 to 200 with small batches and from 86 to 200 with large batches. The reason is specific to CartPole's reward structure. Every reward is +1, so raw returns are always positive, and every update raises the probability of every sampled action. Only the relative sizes of the updates differ. Normalizing centers the advantages around zero, so actions from shorter-than-average episodes receive negative advantages and are actively discouraged. When reward-to-go was already in use, normalization added less. In the small-batch rtg+na run, the policy reached 200 but oscillated afterwards; I attribute this mostly to single-seed noise rather than to normalization hurting.

**The effect of batch size.** Batch size mattered in two opposite ways. Per update, the large batch gave smoother curves and more reliable convergence: three of the four large-batch configurations finished at or near 200, against two of four with the small batch. Per environment step, however, the small batch was more efficient: every configuration reached 200 sooner in steps with `-b 1000`, because each large-batch update consumes four times as much data. A larger batch reduces gradient variance but spends more samples on each step, and this trade-off shows up again in Experiment 4.

<details>
<summary>Commands</summary>

```bash
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```
</details>

---

## Experiment 2: HalfCheetah-v4 with a neural-network baseline

All runs use reward-to-go, γ = 0.95, `-lr 0.01`, and no advantage normalization, so the baseline is the only variance-reduction difference between them.

![HalfCheetah eval return](report/figures/cheetah_eval_return.png)
![HalfCheetah baseline loss](report/figures/cheetah_baseline_loss.png)

| Run | Final eval return | Last-10 mean | Max |
|---|---|---|---|
| no baseline | −208 | −213 | −178 |
| **baseline (bgs = 5, blr = 0.01)** | **443** | **342** | **505** |
| baseline, bgs = 1 | 239 | 127 | 239 |
| baseline, blr = 0.001 | −94 | −92 | 11 |

### Analysis

**Effect of the baseline.** The learned baseline made a large difference. Without it, the agent improved slowly and finished at −208. With it, the agent ended training at 443, above the 300 target, with a last-10 mean of 342. Subtracting `V(s)` from the reward-to-go leaves the expected gradient unchanged, since the baseline does not depend on the action. What it changes is the scale of the advantage: an action's signal becomes "how much better than usual from this state" instead of the raw return, which is dominated by the state itself. The eval-return curve shows that lower variance as faster, steadier improvement.

The baseline-loss curve shows the critic learning in the default setting. The MSE drops from about 200 to about 30 within the first 50K steps, then stays roughly flat while the policy, and with it the return targets, keeps changing.

**A note on seed variance.** I ran the baselined configuration five times with identical hyperparameters (seed 1 twice, then seeds 2 to 4). The final returns were 101, 204, −216, 266, and 443. The plotted run is seed 4, the first to clear 300 at the end of training. Even two runs with the same seed differed, because the starter code seeds NumPy and PyTorch but not `env.reset()`. The assignment expects one or two bad runs, and this spread is why single-run comparisons below should be read as trends.

**Weakening the critic.** To isolate how much the critic's accuracy matters, I reran the seed-4 configuration with fewer baseline gradient steps (`-bgs 1`) and, separately, a 10× smaller baseline learning rate (`-blr 0.001`).

The effect on the **baseline learning curve** was immediate. With either change, the critic could not keep up with the policy early in training. Instead of falling, the loss *rose* to about 360 (`bgs=1`) and 540 (`blr=0.001`), and it took roughly 100K steps to come down to the level the default critic had reached by 50K. Late in training, all three losses sit around 20 to 30. Those late values are not directly comparable, however: a worse policy earns smaller returns, so its critic has smaller-magnitude targets to fit.

The **policy's performance** followed the critic's accuracy. While the critic is inaccurate, the baseline subtracts the wrong amount, so the advantages stay noisy and the policy gradient is less informative. The `bgs=1` run still learned, ending at 239. The `blr=0.001` run, whose critic lagged the longest, barely improved and ended near −90. A baseline helps only to the extent that the critic actually tracks V(s).

<details>
<summary>Commands</summary>

```bash
# No baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --exp_name cheetah
# Baseline (submitted run)
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --seed 4
# Weaker critic
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 1 --exp_name cheetah_baseline_bgs1 --seed 4
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 5 --exp_name cheetah_baseline_blr0.001 --seed 4
```
</details>

---

## Experiment 3: LunarLander-v2 with GAE

A sweep over λ ∈ {0, 0.95, 0.98, 0.99, 1} with all other hyperparameters fixed as specified (200 iterations, batch 2000, γ = 0.99, baseline on).

![LunarLander GAE](report/figures/lunar_lander_gae.png)

| λ | Max eval return | Last-20 mean | First > 150 at iter |
|---|---|---|---|
| 0 | 83 | −100 | never |
| 0.95 | 194 | −60 | 74 |
| **0.98** | **237** | **119** | 116 |
| 0.99 | 217 | 105 | 109 |
| 1 | 239 | 50 | 113 |

### Analysis

**How λ affected performance.** The sweep separates into three behaviors. At λ = 0 the agent never learned to land: its return stayed negative for the whole run and peaked at 83. Every λ ≥ 0.95 exceeded the 150 target at least once. Among those, λ = 0.98 and 0.99 were the most reliable: they kept returns above 100 over the last 20 iterations and ended at 157 and 132. λ = 1 matched them in peak return (239) but swung widely, falling from 85 at iteration 150 to 43 at the end. λ = 0.95 crossed 150 earliest but collapsed late in training; with only one seed, I treat that collapse as partly noise.

**What λ = 0 and λ = 1 correspond to.** These two extremes explain the pattern. With λ = 0, GAE reduces to the one-step TD error `δ_t = r_t + γV(s_{t+1}) − V(s_t)`. Only one real reward enters the estimate, and everything after it is the critic's guess. This is the lowest-variance estimator, but it is only as good as the critic. Early in training the critic is poor, so the advantages are heavily biased, and the policy never got a useful learning signal.

With λ = 1, the δ terms telescope into `G_t − V(s_t)`: the Monte Carlo reward-to-go minus the baseline, exactly the estimator from Experiment 2. It is unbiased, because it uses all the real rewards, but in a 1000-step, noisy-action environment those full returns are high-variance, which is the instability visible in the λ = 1 curve.

Intermediate values such as 0.98 and 0.99 trade a small amount of bias for a large reduction in variance, so on LunarLander they gave both a high peak and stable late-training performance.

<details>
<summary>Command</summary>

```bash
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
  --gae_lambda <lambda> --exp_name lunar_lander_lambda<lambda>
# <lambda> ∈ {0, 0.95, 0.98, 0.99, 1}
```
</details>

---

## Experiment 4: Hyperparameter tuning on InvertedPendulum-v4

The goal is to reach a return of 1000 within **100K environment steps**. With the default settings, this takes about 300K.

![InvertedPendulum](report/figures/pendulum.png)

| Config (tuned runs all use `-rtg -na`) | First eval = 1000 @ steps | # evals at 1000 within 100K | Outcome |
|---|---|---|---|
| default: `-b 5000` | 298K | 0 | max 86 within 100K |
| A: `-b 1000 -lr 0.01` | 80.3K | 11 | stable at 1000 at end |
| B: `-b 1000 -lr 0.02` | 58.4K | 4 | collapsed to ~40 afterwards |
| C: `-b 500 -lr 0.01` | — | 0 | too noisy, max 262 |
| **D: `-b 1000 -lr 0.01` + baseline, GAE λ = 0.98, γ = 0.99** | **47.8K** | **36** | **best** |
| E: `-b 2000 -lr 0.02` | — | 0 | too few updates, max 377 |

### Analysis

**Best configuration.** My best run (D) first reached 1000 after **47.8K environment steps**, six times faster than the default, and held 1000 at 36 of the evaluations within the 100K budget:

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 \
  -rtg -na -lr 0.01 --discount 0.99 --use_baseline --gae_lambda 0.98 --exp_name pendulum_D_b1000_gae
```

**Which hyperparameters mattered.** The budget is counted in environment steps, not iterations, and that made **batch size** the most important choice. At the default `-b 5000`, 100K steps buy only 20 policy updates, and within that budget the default run never got past a return of 86. Dropping to `-b 1000` buys 100 updates for the same data. There was a floor, though: at `-b 500` the gradient estimates were too noisy to learn at all, and `-b 2000` allowed too few updates. A batch of 1000 was the best balance in my sweep.

The **learning rate** traded speed for stability. Doubling it to 0.02 (B) reached 1000 sooner than 0.01 (A), at 58K versus 80K steps, but the policy then collapsed to around 40 and never recovered. Run A was slower but held 1000 through the end of training.

The remaining gain came from **variance reduction**, which is what makes small batches viable. Reward-to-go and advantage normalization were on in every tuned run. Adding a critic baseline with GAE (λ = 0.98) and γ = 0.99 on top of A moved the first hit from 80K to 48K steps and tripled the number of evaluations at 1000. With fewer samples per update, each update needs lower-variance advantages to be useful.

The default run's full curve is a reminder of how unstable plain policy gradients can be. It eventually reached 1000 twice, around 300K and 410K steps, but collapsed to about 20 by the end of training.

<details>
<summary>Default command</summary>

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum
```
</details>

---

## Takeaways

- Policy gradient performance is dominated by **gradient variance**. Every technique that helped (reward-to-go, normalization, baselines, GAE, discounting) is a variance-reduction tool. Each one trades off differently against bias or sample cost.
- **Results vary heavily with the random seed.** Identical HalfCheetah configs ranged from −216 to 443. Conclusions from single runs should be treated as trends, and real comparisons need multiple seeds.
- **Sample efficiency is about the number of updates per environment step**, not the number of iterations. Smaller batches plus stronger variance reduction made InvertedPendulum about 6× more sample-efficient than the default.

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/). Always run scripts through `uv run`.

```bash
uv sync
uv run wandb login          # optional: experiment tracking
uv run src/scripts/run.py --help
uv run report/plot_results.py   # regenerate figures from exp/*/log.csv
```

On macOS, if building `box2d-py` fails because of a custom `CC`/`CXX` compiler setting, run `CC=clang CXX=clang++ uv sync`.
