"""owl-rllib against CleanRL's published results.
Five seeds per configuration, one single-threaded process per run, in parallel.
Build: dune build bench
Run: python3 bench/acceptance.py --output bench/results.csv
  dqn:      DQN with CleanRL dqn.py's defaults (CartPole, Acrobot, MountainCar)
  ppo:      PPO with CleanRL ppo.py's defaults (same three environments)
  mujoco:   PPO with CleanRL ppo_continuous_action.py's details (Hopper, HalfCheetah,
            InvertedPendulum), and the same with time limits treated as terminal
  pendulum: the A7 example's Pendulum configuration with CleanRL's details added one at a time
  a2c:      A2C on CartPole with the RL Zoo budget
Needs Gymnasium with MuJoCo for the mujoco group. Raw curves go next to the output.
"""
import native  # Capture native library paths before anything imports Python numerics.
import argparse
import concurrent.futures
import csv
from pathlib import Path
import re

ROW = re.compile(r'last100=([-\d.]+) eval_mean=([-\w.]+) seconds=(\d+)')
BUILD = '_build/default/bench/'
PENDULUM = {
    'a7': [], 'state-independent-std': ['--state-independent-std'], 'orthogonal': ['--orthogonal'],
    'state-independent-std+orthogonal': ['--state-independent-std', '--orthogonal'], 'tanh': ['--tanh'],
    'clip-value': ['--clip-value'], 'anneal': ['--anneal'],
    'all-but-normalisation': ['--state-independent-std', '--orthogonal', '--tanh', '--clip-value', '--anneal'],
}


def jobs(groups, seeds):
    for seed in range(seeds):
        if 'dqn' in groups:
            for env in ['cartpole', 'acrobot', 'mountain_car']:
                yield 'dqn', env, 'cleanrl', seed, [BUILD + 'dqn_cleanrl.exe', '--env', env]
        if 'ppo' in groups:
            for env in ['cartpole', 'acrobot', 'mountain_car']:
                yield 'ppo', env, 'cleanrl', seed, [BUILD + 'ppo_cleanrl.exe', '--env', env]
        if 'mujoco' in groups:
            for env in ['Hopper-v4', 'HalfCheetah-v4', 'InvertedPendulum-v4']:
                yield 'ppo', env, 'cleanrl', seed, [BUILD + 'ppo_continuous.exe', '--env', env]
            for env in ['Hopper-v4', 'HalfCheetah-v4']:
                yield 'ppo', env, 'cleanrl+truncation-as-termination', seed, \
                    [BUILD + 'ppo_continuous.exe', '--env', env, '--truncation-as-termination']
        if 'pendulum' in groups:
            for name, flags in PENDULUM.items():
                yield 'ppo', 'pendulum', name, seed, [BUILD + 'ppo_continuous.exe', '--env', 'pendulum', '--preset', 'a7', *flags]
        if 'a2c' in groups:
            for entropy in ['0', '0.01']:
                yield 'a2c', 'cartpole', f'zoo-entropy-{entropy}', seed, [BUILD + 'a2c_zoo.exe', '--entropy', entropy]


def run(job, curves):
    algorithm, env, config, seed, command = job
    result = native.run(command + ['--seed', str(seed)])
    (curves / f'{algorithm}-{env}-{config}-{seed}.csv').write_text(result.stdout)
    last100, evaluation, seconds = ROW.search(result.stderr).groups()
    return [algorithm, env, config, seed, last100, evaluation, seconds]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/tmp/owl-rllib-bench.csv'))
    parser.add_argument('--groups', default='dqn,ppo,mujoco,pendulum,a2c')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=30)
    args = parser.parse_args()
    curves = args.output.with_suffix('')
    curves.mkdir(parents=True, exist_ok=True)
    todo = list(jobs(args.groups.split(','), args.seeds))
    with args.output.open('w') as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['algorithm', 'environment', 'configuration', 'seed', 'last100', 'eval_mean', 'seconds'])
        for row in pool.map(lambda job: run(job, curves), todo):
            writer.writerow(row)
            f.flush()
            print(*row, flush=True)
