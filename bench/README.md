# Benchmarks against CleanRL

DQN and PPO compared with [CleanRL](https://docs.cleanrl.dev)'s published
results. Five seeds per configuration; the score is the mean return of the
last 100 training episodes of each seed, then mean ± sample standard deviation
over seeds. CleanRL used three seeds and does not state exactly how its table
is computed, so the comparison is approximate.

| Algorithm | Environment | owl-rllib | CleanRL |
|---|---|---:|---:|
| DQN | CartPole-v1 | 488.7 ± 17.9 | 488.69 ± 16.11 |
| DQN | Acrobot-v1 | −89.9 ± 4.4 | −91.54 ± 7.20 |
| DQN | MountainCar-v0 | −188.4 ± 24.2 | −194.95 ± 8.48 |
| PPO | CartPole-v1 | 493.4 ± 10.1 | 490.04 ± 6.12 |
| PPO | Acrobot-v1 | −82.6 ± 1.5 | −86.36 ± 1.32 |
| PPO | MountainCar-v0 | −200.0 ± 0.0 | −200.00 ± 0.00 |
| PPO | Hopper-v4 | 2563.9 ± 246.9 | 2382.86 ± 271.74 |
| PPO | InvertedPendulum-v4 | 995.7 ± 9.7 | 963.09 ± 22.20 |
| PPO | HalfCheetah-v4 | 2111.3 ± 952.5 | 1442.64 ± 46.03 |

Configurations:

- `dqn_cleanrl.ml`: the defaults of CleanRL's `dqn.py` (500k steps). The
  algorithm is the same as CleanRL's: MSE loss, Adam, hard target updates, no
  bootstrap only on termination.
- `ppo_cleanrl.ml`: the defaults of `ppo.py` (4 environments, 128 steps, 500k
  steps), without learning-rate annealing, value clipping, tanh layers or
  orthogonal initialisation.
- `ppo_continuous.ml --preset cleanrl`: the details of
  `ppo_continuous_action.py` (1M steps): observation and reward normalisation,
  tanh layers with orthogonal initialisation, a state-independent log_std,
  value clipping and an annealed learning rate. The MuJoCo environments run in
  Gymnasium through the Python bridge.

**HalfCheetah** is bimodal: some seeds learn to run (3000–4000), the others
stay near 1500. CleanRL's GAE also treats a time limit as termination, while
owl-rllib bootstraps from the value at a truncation. With
`--truncation-as-termination`, the seeds in the lower mode reach 1439 and
1445, against CleanRL's 1443 ± 46; bootstrapping raises that mode by about
250. On Hopper, removing the bootstrap costs about 500.

**Pendulum**: `--preset a7` is the configuration of the A7 example. Its
deterministic evaluation is −466 ± 223, against −226 ± 82 for SB3 with the
same settings. The gap is the parameterisation of the Gaussian: with a
state-independent log_std and orthogonal initialisation (what SB3 uses) it is
−264 ± 144. Adding tanh, value clipping or annealing alone makes it markedly
worse in this setting (rewards scaled by 0.1, 100k steps).

**A2C** (`a2c_zoo.ml`, RL Zoo budget of 500k steps on CartPole): four seeds of
five reach 500 with entropy 0, one collapses to 71.

All per-seed results are in `results.csv`. To reproduce them (105 runs, one
single-threaded process each, 30 in parallel by default):

```bash
dune build bench
python3 bench/acceptance.py --output bench/results.csv
```

The `mujoco` group needs Gymnasium with MuJoCo; `--groups` selects a subset
(`dqn,ppo,mujoco,pendulum,a2c`). `native.py` starts each OCaml process with
the environment captured before Python's numerical libraries are imported,
which can otherwise change the dynamic libraries Owl loads.
