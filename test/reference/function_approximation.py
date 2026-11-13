"""Independent PyTorch reference values and gradients; regenerate from repo root."""
import torch

torch.set_default_dtype(torch.float64)
x = torch.tensor([[0.2, -0.3, 0.7], [-0.1, 0.5, 0.2]], requires_grad=True)
y = torch.tensor([1, 0])
q = torch.distributions.Categorical(logits=torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.2, 0.1]]))
p = torch.distributions.Categorical(logits=x)
mu = torch.tensor([[0.2, -0.1], [0.7, 0.5]], requires_grad=True)
ls = torch.tensor([[-0.3, 0.2]], requires_grad=True)
a = torch.tensor([[0.1, 0.4], [0.3, -0.2]])
g = torch.distributions.Normal(mu, ls.exp())
h = torch.distributions.Normal(mu.detach() + 0.3, (ls.detach() + 0.1).exp())
with open("test/fixtures/function_approximation_numerics.txt", "w") as f:
    def emit(name, value):
        f.write(name + " " + " ".join(f"{v:.17g}" for v in value.detach().flatten().tolist()) + "\n")
    for name, value, parameters in [
        ("categorical_log_prob", p.log_prob(y), [x]),
        ("categorical_entropy", p.entropy(), [x]),
        ("categorical_kl", torch.distributions.kl_divergence(p, q), [x]),
        ("gaussian_log_prob", g.log_prob(a).sum(-1), [mu, ls]),
        ("gaussian_entropy", g.entropy().sum(-1), [ls]),
        ("gaussian_kl", torch.distributions.kl_divergence(g, h).sum(-1), [mu, ls]),
    ]:
        emit(name, value)
        for i, derivative in enumerate(torch.autograd.grad(value.sum(), parameters, retain_graph=True)):
            emit(f"{name}_grad{i}", derivative)
print("torch", torch.__version__)
