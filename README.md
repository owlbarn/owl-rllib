# owl-rllib

Reinforcement learning in OCaml, built on [Owl](https://github.com/owlbarn/owl).

owl-rllib aims to be small, readable and reproducible rather than large: one
file per algorithm, explicit random number generators, and every environment
and loss checked against a reference implementation (Gymnasium, PyTorch).
DQN and PPO match CleanRL's published results on its benchmark environments
(see [bench/](bench/README.md)).

## Contents

- **MDPs and tabular methods**: finite MDPs with explicit transition tables;
  policy evaluation, policy iteration and value iteration; Monte Carlo, TD(0),
  SARSA and Q-learning.
- **Environments**: GridWorld, FrozenLake, CliffWalking, Taxi, CartPole,
  MountainCar, Acrobot and Pendulum, written in OCaml and checked step by step
  against Gymnasium 0.29.1. Synchronous vector environments, frame stacking,
  observation and reward normalisation.
- **Deep RL**: linear function approximation, REINFORCE (with an optional
  baseline), DQN, A2C and PPO, with categorical and Gaussian policies, on
  Owl's neural networks (MLPs and small CNNs).
- **RL for language models** (`owl-rllib.llm`): a toy autoregressive model on
  a verifiable task (two-digit addition), with REINFORCE, RLOO, token-level
  PPO, GRPO, and the GRPO variants Dr. GRPO, DAPO, GSPO and CISPO as
  configuration switches.
- **Python bridge** (`owl-rllib.bridges`, optional): any Gymnasium environment
  (e.g. LunarLander, MuJoCo) and VizDoom scenarios, run in a Python child
  process and exposed as ordinary environments.

## Installation

owl-rllib needs a version of Owl newer than the latest release on opam (1.2):
it uses `Optimise.Stepper`, which is only on Owl's main branch. Build Owl from
source first:

```bash
git clone https://github.com/owlbarn/owl.git
cd owl && make install
```

Then build owl-rllib (the API documentation is generated with `dune build @doc`):

```bash
git clone https://github.com/owlbarn/owl-rllib.git
cd owl-rllib
opam install alcotest   # only needed for the tests
dune build
dune test
```

Do not run `opam install . --deps-only`: opam would replace the Owl built from
source with Owl 1.2 from the repository.

The Python bridge needs Python 3 with `gymnasium` (and `Box2D` or `mujoco` for
the corresponding environments) or `vizdoom`. Set `OWL_RLLIB_PYTHON` to choose
the interpreter; the default is `python3`. Tests that need these packages are
skipped when they are missing.

## Quick start

The package has three libraries: `owl-rllib` (module `Owl_rllib`),
`owl-rllib.llm` (`Owl_rllib_llm`) and `owl-rllib.bridges`
(`Owl_rllib_bridges`). With dune:

```
(executable (name main) (libraries owl-rllib))
```

```ocaml
open Owl_rllib
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let rngs = Rng.create 0 in
  let policy = N.mlp ~seed:rngs.network_seed ~shape:[| 4 |] ~hidden:[ 64; 64 ] ~outputs:2 in
  let critic = N.mlp ~seed:(rngs.network_seed + 1) ~shape:[| 4 |] ~hidden:[ 64; 64 ] ~outputs:1 in
  let env = Vec_env.create (Array.init 8 (fun _ -> Cartpole.make ~rng:rngs.env ())) in
  let returns = Ppo.S.train ~updates:100 ~rngs ~policy ~critic ~distribution:Policy.S.categorical env in
  Printf.printf "last episode: %.0f\n" returns.(Array.length returns - 1)
```

```bash
OPENBLAS_NUM_THREADS=1 dune exec examples/classic/a7_actor_critic.exe
```

Set `OPENBLAS_NUM_THREADS=1`: the networks here are small, and OpenBLAS's
default thread count slows them down.

## Examples

`examples/` holds runnable programs in two tracks, each building on the last:

- `examples/classic`: from MDPs to pixels. a1 MDPs, a2 dynamic programming,
  a3 tabular methods, a4 function approximation, a5 DQN, a6 REINFORCE, a7
  actor-critic and PPO (CartPole, Pendulum, LunarLander), a8 PPO from pixels
  on VizDoom.
- `examples/llm`: RL for language models, following Cameron Wolfe's
  [*Reinforcement Learning for LLMs*](https://cameronrwolfe.substack.com/p/llm-rl):
  b0 supervised fine-tuning, b1 generation as an MDP, b2 REINFORCE, b3 RLOO,
  b4 KL penalties, b5 PPO, b6 GRPO, b7a–b7d Dr. GRPO, DAPO, GSPO and CISPO.

Each file's header gives the command to run it and what it shows.

## Reproducibility

Every random choice goes through an explicit `Random.State.t` (see `Rng`):
the same seed on the same machine gives bit-identical results. One exception:
training a convolutional network in Owl occasionally differs by one unit in
the last place between runs, so the VizDoom example (a8) is not bit-for-bit
reproducible.

## License

MIT, see [LICENSE](LICENSE).
