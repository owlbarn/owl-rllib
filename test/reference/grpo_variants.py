"""Independent float64 fixtures for the GRPO variants (Dr. GRPO, DAPO, GSPO, CISPO).

Uses the same rows as test/reference/lm.py. Run from the repository root:
python3 test/reference/grpo_variants.py
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
G = 4
rewards = np.array([1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0.])
g = rewards.reshape(-1, G)
centred = (g - g.mean(1, keepdims=True)).flatten()  # Dr. GRPO: no std

# six token rows from three completions of lengths 1, 2, 3
sequence = np.array([0, 1, 1, 2, 2, 2])
lengths = np.array([1, 2, 3])
logp0 = np.log([0.5, 0.9, 0.3, 0.7, 0.2, 0.6])
old = torch.tensor(np.log([0.4, 0.8, 0.5, 0.7, 0.1, 0.9]))
seq_adv = torch.tensor([1.5, -0.5, 2.0])
adv = seq_adv[torch.tensor(sequence)]
sequence_mean = torch.tensor([1 / (3 * lengths[s]) for s in sequence])
token_mean = torch.full((6,), 1 / 6)
fixed_length = np.full(6, 1 / (3 * 4))  # Dr. GRPO with a generation budget of 4


def clip(r, low, high):
    return r.clamp(1 - low, 1 + high)


def clipped(lp):  # DAPO clip-higher: [1 - 0.2, 1 + 0.28]
    r = (lp - old).exp()
    return -(sequence_mean * torch.minimum(r * adv, clip(r, 0.2, 0.28) * adv)).sum()


def gspo(lp):  # sequence ratio = exp(mean of token log-ratios), range [1 - 0.1, 1 + 0.2]
    lr = lp - old
    s = torch.stack([lr[torch.tensor(sequence == i)].mean() for i in range(3)]).exp()
    return -torch.minimum(s * seq_adv, clip(s, 0.1, 0.2) * seq_adv).mean()


def cispo(lp):  # the clipped ratio is a constant weight
    w = clip((lp.detach() - old).exp(), 0.2, 0.28)
    return -(token_mean * w * adv * lp).sum()


ratio = np.exp(logp0 - old.numpy())
a = adv.numpy()


def clip_fraction(a):
    return np.mean(((a > 0) & (ratio > 1.28)) | ((a < 0) & (ratio < 0.8)))


with open('test/fixtures/grpo_variants_numerics.txt', 'w') as f:
    def emit(name, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        f.write(name + ' ' + ' '.join(f'{v:.17g}' for v in np.asarray(x).flatten()) + '\n')

    emit('centred', centred)
    emit('weights_fixed_length', fixed_length)
    emit('clip_fraction', clip_fraction(a))
    emit('clip_fraction_negated', clip_fraction(-a))
    for name, fn in [('clip_higher', clipped), ('gspo', gspo), ('cispo', cispo)]:
        lp = torch.tensor(logp0, requires_grad=True)
        loss = fn(lp)
        emit(name + '_loss', loss)
        emit(name + '_grad', torch.autograd.grad(loss, lp)[0])
print('wrote test/fixtures/grpo_variants_numerics.txt')
