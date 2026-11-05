import os
import random
import time

import torch

from tinytransformer.config import Config
from tinytransformer.data import Data, Vocab
from tinytransformer.model import forward
from tinytransformer.optim import AdamW
from tinytransformer.params import flatten, init_params, num_params
from tinytransformer.utils import (
    evaluate,
    load_checkpoint,
    loss_fn,
    noam_lr,
    save_checkpoint,
)


def train_step(batch, params, cfg, opt, step, amp):
    with torch.autocast(device_type=torch.device(batch["src"].device).type,
                        dtype=torch.bfloat16, enabled=amp):
        logits = forward(batch, params, cfg, training=True)
        loss = loss_fn(logits, batch["tgt_label"], cfg)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(flatten(params), 1.0)
    opt.step(noam_lr(step, cfg))
    return loss.item()


def train(train_data, val_data, params, cfg, opt, device, epochs, patience, step, best, amp,
          on_epoch=None):
    bad = 0
    for epoch in range(1, epochs + 1):
        t0, total, count = time.time(), 0.0, 0

        for batch in train_data.batches():
            batch = {k: v.to(device) for k, v in batch.items()}
            step += 1
            loss = train_step(batch, params, cfg, opt, step, amp)
            n = (batch["tgt_label"] != cfg.pad_id).sum().item()
            total += loss * n
            count += n

        vloss, vacc = evaluate(val_data, params, cfg, device)
        print(f"epoch {epoch} | train {total / count:.4f} | val {vloss:.4f} "
              f"| val acc {vacc:.3f} | lr {noam_lr(step, cfg):.6f} | {time.time() - t0:.0f}s")

        if vloss < best:
            best, bad = vloss, 0
            save_checkpoint("ckpt/best.pt", params, opt, step, best, cfg)
        else:
            bad += 1
        save_checkpoint("ckpt/last.pt", params, opt, step, best, cfg)

        if on_epoch:
            on_epoch(step, best)

        if bad >= patience:
            print(f"no val improvement for {bad} epochs, stopping")
            return step, best

    return step, best


def run(epochs=30, device=None, resume=None, patience=3, on_epoch=None):
    torch.manual_seed(0)
    random.seed(0)
    os.makedirs("ckpt", exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.device(device).type == "cuda"

    if resume:
        cfg, params, opt, step, best = load_checkpoint(resume, device)
        print(f"resumed at step {step}, best val {best:.4f}")
    else:
        cfg = Config()
        params = init_params(cfg, device)
        opt = AdamW(params, weight_decay=cfg.weight_decay)
        step, best = 0, float("inf")

    vocab = Vocab.load("data/bpe.json", cfg)
    train_data = Data(cfg, vocab, "train")
    val_data = Data(cfg, vocab, "val")

    print(f"device {device} | params {num_params(params):,} "
          f"| train {len(train_data.pairs):,} | val {len(val_data.pairs):,}")

    return train(train_data, val_data, params, cfg, opt, device,
                 epochs, patience, step, best, amp, on_epoch)