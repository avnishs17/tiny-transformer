import torch


class AdamW:
    """Adam with decoupled weight decay; no decay on 1-D tensors (norms and biases).

    Import as `from tinytransformer.optim import AdamW` so it can never be
    confused with `torch.optim.AdamW`.
    """

    def __init__(self, params, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01):
        self.params = params
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: torch.zeros_like(t) for k, t in params.items()}
        self.v = {k: torch.zeros_like(t) for k, t in params.items()}
        self.t = 0

    def zero_grad(self):
        for p in self.params.values():
            p.grad = None

    def step(self, lr):
        self.t += 1
        b1, b2 = self.betas
        with torch.no_grad():
            for name, p in self.params.items():
                if p.grad is None:
                    continue
                m = self.m[name].mul_(b1).add_(p.grad, alpha=1 - b1)
                v = self.v[name].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m_hat = m / (1 - b1 ** self.t)
                v_hat = v / (1 - b2 ** self.t)
                if p.dim() > 1:
                    p.mul_(1 - lr * self.weight_decay)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-lr)

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load_state_dict(self, s):
        self.m, self.v, self.t = s["m"], s["v"], s["t"]
