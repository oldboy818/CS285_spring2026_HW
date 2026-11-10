# CS 285: Deep Reinforcement Learning (UC Berkeley, Spring 2026) — Homework

My solutions to the homework of [CS 285: Deep Reinforcement Learning, Decision Making, and Control](https://rail.eecs.berkeley.edu/deeprlcourse/), completed as a self-study student. Forked from the [official course repository](https://github.com/berkeleydeeprlcourse/homework_spring2026).

Each homework folder has its own report with learning curves, analysis, and the exact commands used.

| # | Topic | Status | Highlights | Report |
|---|---|---|---|---|
| HW1 | Imitation Learning | ✅ Done | MSE policy, flow-matching policy | [hw1/](hw1/) |
| HW2 | Policy Gradients | ✅ Done | REINFORCE, reward-to-go, NN baseline, GAE; InvertedPendulum solved in 47.8K env steps (default ~300K) | [hw2/](hw2/) |
| HW3 | — | ⬜ Not started | | |
| HW4 | — | ⬜ Not started | | |
| HW5 | — | ⬜ Not started | | |

## Running

Each homework is a standalone [`uv`](https://docs.astral.sh/uv/) project. See the `README.md` in each folder for setup. For example:

```bash
cd hw2
uv sync
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
```
