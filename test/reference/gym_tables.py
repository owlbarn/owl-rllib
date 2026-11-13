"""Export the transition tables of Gymnasium's tabular environments.

The OCaml tests compare our environments against these files.
Run from the repository root: python3 test/reference/gym_tables.py

Format, one record per line:
  N <n_states> <n_actions>
  T <state> <action> <prob> <next> <reward> <terminal>
  I <state> <prob>
"""
import gymnasium as gym

ENVS = {
    "frozen_lake": ("FrozenLake-v1", {"is_slippery": True}),
    "frozen_lake_dry": ("FrozenLake-v1", {"is_slippery": False}),
    "cliff_walking": ("CliffWalking-v0", {}),
    "taxi": ("Taxi-v3", {}),
}

print("gymnasium", gym.__version__)
for name, (env_id, kwargs) in ENVS.items():
    env = gym.make(env_id, **kwargs).unwrapped
    n_states, n_actions = env.observation_space.n, env.action_space.n
    with open(f"test/fixtures/{name}.txt", "w") as f:
        f.write(f"N {n_states} {n_actions}\n")
        for s in range(n_states):
            for a in range(n_actions):
                for prob, nxt, reward, terminal in env.P[s][a]:
                    f.write(f"T {s} {a} {float(prob)!r} {int(nxt)} {float(reward)!r} {int(terminal)}\n")
        for s, prob in enumerate(env.initial_state_distrib):
            if prob > 0:
                f.write(f"I {s} {float(prob)!r}\n")
    print("wrote", name)
