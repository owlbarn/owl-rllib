## 0.1 (2026-11-20)

Initial release.

* MDPs and tabular methods: finite MDPs with explicit transition tables;
  policy evaluation, policy iteration and value iteration; Monte Carlo, TD(0),
  SARSA and Q-learning.
* Environments: GridWorld, FrozenLake, CliffWalking, Taxi, CartPole,
  MountainCar, Acrobot and Pendulum, checked step by step against Gymnasium
  0.29.1; synchronous vector environments, frame stacking, observation and
  reward normalisation.
* Deep RL on Owl's neural networks: linear function approximation, REINFORCE
  (with an optional baseline), DQN, A2C and PPO, with categorical and Gaussian
  policies. DQN and PPO match CleanRL's published results on its benchmark
  environments.
* RL for language models (`owl-rllib.llm`): a toy autoregressive model on
  two-digit addition, with REINFORCE, RLOO, token-level PPO, GRPO and the GRPO
  variants Dr. GRPO, DAPO, GSPO and CISPO.
* Python bridge (`owl-rllib.bridges`, optional): Gymnasium environments and
  VizDoom scenarios run in a Python child process.
* Requires Owl 1.3 or later.
