"""Reference values for the CleanRL details used to match CleanRL's PPO.
Run from the repository root: python3 test/reference/ppo_details.py

ppo_normalise.txt: Gymnasium 0.29.1's NormalizeObservation and
  NormalizeReward(gamma=0.99) on CartPole-v1 (three terminated episodes) and
  Pendulum-v1 (a truncated episode, then 10 more steps). A raw copy of each
  environment, with the same seed and actions, supplies the inputs.
  E <id>; R <raw obs> | <normalised obs>;
  T <raw reward> <terminated> <truncated> <raw obs> | <normalised obs> | <normalised reward>
ppo_value.txt: CleanRL's clipped value loss, max((v - R)^2, (clip(v) - R)^2),
  per sample, and the gradient of its sum with respect to v (PyTorch).
"""
import gymnasium as gym
import numpy as np
import torch


def hexes(xs):
    return ' '.join(float(x).hex() for x in np.asarray(xs, dtype=np.float64).ravel())


with open('test/fixtures/ppo_normalise.txt', 'w') as f:
    for env_id, episodes, extra in [('CartPole-v1', 3, 0), ('Pendulum-v1', 1, 10)]:
        raw = gym.make(env_id)
        wrapped = gym.wrappers.NormalizeReward(gym.wrappers.NormalizeObservation(gym.make(env_id)), gamma=0.99)
        f.write(f'E {env_id}\n')
        t = 0
        for episode in range(episodes + (1 if extra else 0)):
            seed = 4 if episode == 0 else None
            obs, _ = raw.reset(seed=seed)
            norm, _ = wrapped.reset(seed=seed)
            f.write(f'R {hexes(obs)} | {hexes(norm)}\n')
            steps = 0
            while True:
                if env_id == 'CartPole-v1':
                    action = (t // 4) % 2
                else:
                    action = np.array([2 * np.sin(0.3 * t)], dtype=np.float32)
                t += 1
                steps += 1
                obs, reward, terminated, truncated, _ = raw.step(action)
                norm, nreward, _, _, _ = wrapped.step(action)
                f.write(f'T {float(reward).hex()} {int(terminated)} {int(truncated)} {hexes(obs)} | {hexes(norm)} | {float(nreward).hex()}\n')
                if terminated or truncated or (episode == episodes and steps == extra):
                    break

clip = 0.2
v = torch.tensor([0.5, 1.3, -0.4, 2.0, 0.1], dtype=torch.float64, requires_grad=True)
old = torch.tensor([0.4, 1.0, 0.0, 2.1, 0.9], dtype=torch.float64)
returns = torch.tensor([1.0, 0.0, -1.0, 2.05, 0.0], dtype=torch.float64)
v_clipped = old + torch.clamp(v - old, -clip, clip)
loss = torch.max((v - returns) ** 2, (v_clipped - returns) ** 2)
loss.sum().backward()
with open('test/fixtures/ppo_value.txt', 'w') as f:
    for name, xs in [('predicted', v.detach()), ('old', old), ('returns', returns), ('error', loss.detach()), ('grad', v.grad)]:
        f.write(name + ' ' + hexes(xs.numpy()) + '\n')
