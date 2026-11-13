"""Export Gymnasium 0.29.1 and VizDoom 1.2.3 behaviour for the bridge tests.
Run from the repository root: python3 test/reference/bridges.py
Floats are written with float.hex, so the bridge must reproduce them exactly.

bridge_gym.txt:  E <id> <seed> starts an environment; R <obs> is a reset;
             T <action> <reward> <terminated> <truncated> <obs> is a step.
bridge_doom.txt: the same for basic at seed 7 with frame skip 4. Observations are
             the 60x80 sums of 2x2 pixel blocks (the bridge divides by 1020);
             the first one in full, later ones as their total.
"""
import os
import gymnasium as gym
import numpy as np
import vizdoom as vzd


def hexes(xs):
    return ' '.join(float(x).hex() for x in np.asarray(xs, dtype=np.float64).ravel())


with open('test/fixtures/bridge_gym.txt', 'w') as f:
    # CartPole: two episodes, the second from an unseeded reset; ends by termination.
    # Pendulum: continuous actions, ends by the 200-step time limit.
    for env_id, seed, episodes in [('CartPole-v1', 3, 2), ('Pendulum-v1', 5, 1)]:
        env = gym.make(env_id)
        f.write(f'E {env_id} {seed}\n')
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed if episode == 0 else None)
            f.write('R ' + hexes(obs) + '\n')
            for t in range(1000):
                if env_id == 'CartPole-v1':
                    action, text = (t // 3) % 2, str((t // 3) % 2)
                else:
                    action = np.array([2 * np.sin(0.37 * t)], dtype=np.float32)
                    text = hexes(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                f.write(f'T {text} {float(reward).hex()} {int(terminated)} {int(truncated)} {hexes(obs)}\n')
                if terminated or truncated:
                    break

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, 'basic.cfg'))
game.set_window_visible(False)
game.set_screen_format(vzd.ScreenFormat.GRAY8)
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
game.set_seed(7)
game.init()


def blocks():
    screen = game.get_state().screen_buffer.astype(np.int64)
    return screen.reshape(60, 2, 80, 2).sum(axis=(1, 3))


with open('test/fixtures/bridge_doom.txt', 'w') as f:
    # Shoot at once; strafe left one step, then shoot (at seed 7 this kills the
    # monster, so the episode is terminated); only strafe, so the 300-tic limit
    # truncates the episode. The monster's position is random per episode.
    policies = [lambda t: 2, lambda t: 0 if t < 1 else 2, lambda t: 0]
    for episode, policy in enumerate(policies):
        game.new_episode()
        frame = blocks()
        f.write('R ' + (' '.join(map(str, frame.ravel())) if episode == 0 else str(frame.sum())) + '\n')
        t = 0
        while True:
            action = policy(t)
            buttons = [0, 0, 0]
            buttons[action] = 1
            reward = game.make_action(buttons, 4)
            ended = game.is_episode_finished()
            truncated = ended and game.get_episode_time() >= game.get_episode_start_time() + game.get_episode_timeout()
            if not ended:
                frame = blocks()
            f.write(f'T {action} {float(reward).hex()} {int(ended and not truncated)} {int(truncated)} {frame.sum()}\n')
            t += 1
            if ended:
                break
