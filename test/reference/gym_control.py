"""Export Gymnasium 0.29.1 trajectories, including clipping and terminal edges.
Run from the repository root: python3 test/reference/gym_control.py
Initial internal states are supplied explicitly; RNG algorithms need not match.
"""
import gymnasium as gym
import numpy as np

CASES = {
    "cartpole": ("CartPole-v1", [[0.01, -0.02, 0.03, -0.04], [2.39, 1, 0, 0], [0, 0, 0.209, 1]]),
    "mountain_car": ("MountainCar-v0", [[-0.5, 0], [-1.19, -0.07], [0.499, 0.02], [-0.4, 0.069]]),
    "acrobot": ("Acrobot-v1", [[0.02, -0.05, 0.01, -0.03], [3.13, 0, 0.2, 0], [0, 0, 12.5, -28.2]]),
    "pendulum": ("Pendulum-v1", [[0.2, -0.1], [3.14, 7.9], [-3.14, -7.9]]),
}
print("gymnasium", gym.__version__)
for name, (env_id, states) in CASES.items():
    with open(f"test/fixtures/{name}_steps.txt", "w") as f:
        for i, state in enumerate(states):
            limit = 1 if i == 1 else 40
            env = gym.make(env_id, max_episode_steps=limit)
            env.reset(seed=0)
            env.unwrapped.state = np.array(state, dtype=np.float64)
            f.write("C " + str(limit) + " " + " ".join(map(str, state)) + "\n")
            for t in range(limit):
                action = [0, 1, 1, 0][t % 4] if name == "cartpole" else t % 3
                if name == "pendulum":
                    action = [-3.0, 0.7, 3.0][t % 3]
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32) if name == "pendulum" else action
                )
                f.write("T " + " ".join(map(str, [action, float(reward), int(terminated), int(truncated), *map(float, obs)])) + "\n")
                if terminated or truncated:
                    break
            env.close()
