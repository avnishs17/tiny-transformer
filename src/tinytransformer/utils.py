import math
from dataclasses import asdict

import torch
import torch.nn.functional as F

from tinytransformer.config import Config
from tinytransformer.model import forward
from tinytransformer.optim import AdamW


def noam_lr(step, cfg):
    step = max(step, 1)
    return cfg.d_model ** -0.5 * min(step ** -0.5, step * cfg.warmup_steps ** -1.5)


def loss_fn(logits, labels, cfg):
    return F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        labels.reshape(-1),
        ignore_index=cfg.pad_id,
        label_smoothing=cfg.label_smoothing,
    )


def evaluate(data, params, cfg, device):
    total, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for batch in data.batches(shuffle=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = forward(batch, params, cfg, training=False)
            mask = batch["tgt_label"] != cfg.pad_id
            n = mask.sum().item()
            total += loss_fn(logits, batch["tgt_label"], cfg).item() * n
            correct += (logits.argmax(-1) == batch["tgt_label"])[mask].sum().item()
            count += n
    return total / count, correct / count


def save_checkpoint(path, params, opt, step, best, cfg):
    torch.save({
        "params": params,
        "opt": opt.state_dict(),
        "step": step,
        "best": best,
        "cfg": asdict(cfg),
    }, path)


def load_checkpoint(path, device):
    """Rebuild everything a resumed run needs: cfg, params, optimizer, counters."""
    ck = torch.load(path, map_location=device)
    cfg = Config(**ck["cfg"])
    params = ck["params"]
    opt = AdamW(params, weight_decay=cfg.weight_decay)
    opt.load_state_dict(ck["opt"])
    return cfg, params, opt, ck["step"], ck["best"]
