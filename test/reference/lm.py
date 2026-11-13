"""Independent float64 fixtures for the language-model RL losses (no Gym needed).

Run from the repository root: python3 test/reference/lm.py
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
G = 4
rewards = np.array([1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0.])  # three groups, the middle one all zero


def groups(x):
    return x.reshape(-1, G)


rloo = (groups(rewards) - (groups(rewards).sum(1, keepdims=True) - groups(rewards)) / (G - 1)).flatten()
g = torch.tensor(groups(rewards))
grpo = ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)).flatten()  # torch std is the sample std

# six token rows from three completions of lengths 1, 2, 3
sequence = np.array([0, 1, 1, 2, 2, 2])
lengths = np.array([1, 2, 3])
logp0 = np.log([0.5, 0.9, 0.3, 0.7, 0.2, 0.6])
old = np.log([0.4, 0.8, 0.5, 0.7, 0.1, 0.9])
ref = np.log([0.45, 0.6, 0.4, 0.8, 0.3, 0.5])
adv = np.array([1.5, -0.5, -0.5, 2.0, 2.0, 2.0])
weights = {
    'sequence_sum': np.array([1 / 3] * 6),
    'sequence_mean': np.array([1 / (3 * lengths[s]) for s in sequence]),
    'token_mean': np.array([1 / 6] * 6),
}

with open('test/fixtures/lm_numerics.txt', 'w') as f:
    def emit(name, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        f.write(name + ' ' + ' '.join(f'{v:.17g}' for v in np.asarray(x).flatten()) + '\n')

    emit('rloo', rloo)
    emit('grpo', grpo)
    emit('k1', logp0 - ref)
    emit('k3', np.exp(ref - logp0) - (ref - logp0) - 1)
    for name, w in weights.items():
        emit('weights_' + name, w)
    w = torch.tensor(weights['sequence_mean'])
    a = torch.tensor(adv)
    for name, fn in [
        ('pg', lambda lp: -(w * a * lp).sum()),
        ('clipped', lambda lp: -(w * torch.minimum((lp - torch.tensor(old)).exp() * a,
                                                   (lp - torch.tensor(old)).exp().clamp(0.8, 1.2) * a)).sum()),
        ('k3', lambda lp: (w * ((torch.tensor(ref) - lp).exp() - (torch.tensor(ref) - lp) - 1)).sum()),
    ]:
        lp = torch.tensor(logp0, requires_grad=True)
        loss = fn(lp)
        emit(name + '_loss', loss)
        emit(name + '_grad', torch.autograd.grad(loss, lp)[0])

    # token-level PPO advantages: rewards -beta*k1 per token plus the outcome at the
    # last token; every completion ends in a terminal state; gamma 1, lambda 0.95
    beta, lam = 0.1, 0.95
    outcome = np.array([1., 0., 1.])
    values = np.array([0.3, 0.2, 0.1, 0.6, 0.5, 0.4])
    token_rewards = -beta * (old - ref)
    advantages = np.zeros(6)
    for s in range(3):
        idx = np.where(sequence == s)[0]
        token_rewards[idx[-1]] += outcome[s]
        tail = 0.
        for k, i in reversed(list(enumerate(idx))):
            next_v = values[idx[k + 1]] if k + 1 < len(idx) else 0.
            tail = token_rewards[i] + next_v - values[i] + lam * tail
            advantages[i] = tail
    emit('token_advantages', advantages)
    emit('token_returns', advantages + values)
print('wrote test/fixtures/lm_numerics.txt')
