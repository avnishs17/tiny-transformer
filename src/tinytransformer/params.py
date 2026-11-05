import math

import torch


def init_params(cfg, device):
    C, F, V = cfg.d_model, cfg.d_ff, cfg.vocab_size

    attn = {"w_q": (C, C), "w_k": (C, C), "w_v": (C, C), "w_o": (C, C)}
    ffn = {"w1": (C, F), "b1": (F,), "w2": (F, C), "b2": (C,)}
    norm = {"gamma": (C,), "beta": (C,)}

    shapes = {"embed": (V, C)}
    for i in range(cfg.num_layers):
        base = f"enc.{i}"
        shapes.update({f"{base}.{k}": v for k, v in {**attn, **ffn}.items()})
        shapes.update({f"{base}.attn_{k}": v for k, v in norm.items()})
        shapes.update({f"{base}.ffn_{k}": v for k, v in norm.items()})

        base = f"dec.{i}"
        shapes.update({f"{base}.{k}": v for k, v in ffn.items()})
        shapes.update({f"{base}.ffn_{k}": v for k, v in norm.items()})
        for kind in ("self", "cross"):
            shapes.update({f"{base}.{kind}_{k}": v for k, v in {**attn, **norm}.items()})

    params = {}
    for name, shape in shapes.items():
        if name.endswith("gamma"):
            t = torch.ones(shape, device=device)
        elif name.endswith(("beta", "b1", "b2")):
            t = torch.zeros(shape, device=device)
        elif name == "embed":
            t = torch.randn(shape, device=device) / math.sqrt(C)
        else:
            t = torch.randn(shape, device=device) / math.sqrt(shape[0])
        params[name] = t.requires_grad_()

    return params


def flatten(params):
    return list(params.values())


def num_params(params):
    return sum(t.numel() for t in params.values())
