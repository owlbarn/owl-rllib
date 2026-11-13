"""Independent float64 GAE and PyTorch loss/gradient fixtures (no Gym needed)."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
# time-major, two envs: terminal, truncation, simultaneous flags, and horizon cut
rewards = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
values = np.array([.2, .3, .4, .5, .6, .7, .8, .9])
next_values = np.array([.4, .5, 100., 2., .8, .9, 3., 100.])
terminated = np.array([False, False, True, False, False, False, False, True])
truncated = np.array([False, False, False, True, False, False, False, True])
advantages = np.zeros(8)
for env in range(2):
    tail = 0.
    for t in reversed(range(4)):
        i = t * 2 + env
        delta = rewards[i] + .9 * (not terminated[i]) * next_values[i] - values[i]
        tail = delta + .9 * .8 * (not (terminated[i] or truncated[i])) * tail
        advantages[i] = tail
with open('test/fixtures/actor_critic_numerics.txt', 'w') as f:
    def emit(name, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        f.write(name + ' ' + ' '.join(f'{v:.17g}' for v in np.asarray(x).flatten()) + '\n')
    for name, x in [('gae', advantages), ('returns', advantages + values)]:
        emit(name, x)
    logp = torch.tensor(np.log([.5, .9, 1.1, 1.5, .5, .9, 1.1, 1.5]), requires_grad=True)
    old = torch.zeros(8)
    adv = torch.tensor([1., 2., 3., 4., -1., -2., -3., -4.])
    ratio = (logp - old).exp()
    loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
    emit('ppo_loss', loss)
    emit('ppo_grad', torch.autograd.grad(loss, logp)[0])
    logits = torch.tensor([[.2, -.1], [-.3, .4]], requires_grad=True)
    dist = torch.distributions.Categorical(logits=logits)
    loss = -(dist.log_prob(torch.tensor([0, 1])) * torch.tensor([2., -1.])).mean() - .01 * dist.entropy().mean()
    emit('a2c_loss', loss)
    emit('a2c_grad', torch.autograd.grad(loss, logits)[0])
    output = torch.tensor([[.1, -.3], [.2, .4]], requires_grad=True)
    dist = torch.distributions.Normal(output[:, :1], output[:, 1:].clamp(-5, 2).exp())
    loss = dist.log_prob(torch.tensor([[.7], [-.2]])).sum()
    emit('gaussian_grad', torch.autograd.grad(loss, output)[0])
