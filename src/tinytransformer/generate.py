import math
from collections import Counter

import torch

from tinytransformer.data import Data, Vocab
from tinytransformer.layers import padding_mask
from tinytransformer.model import decode, encode
from tinytransformer.utils import load_checkpoint


def next_logits(tgt_ids, enc, src_mask, params, cfg):
    dec = decode(tgt_ids, enc, src_mask, params, cfg, training=False)
    return dec[:, -1] @ params["embed"].T


def greedy_decode(src, params, cfg):
    src_mask = padding_mask(src, cfg.pad_id)
    enc = encode(src, src_mask, params, cfg, training=False)

    ids = [cfg.bos_id]
    for _ in range(cfg.max_len):
        tgt = torch.tensor([ids], device=src.device)
        nxt = next_logits(tgt, enc, src_mask, params, cfg).argmax(-1).item()
        if nxt == cfg.eos_id:
            break
        ids.append(nxt)
    return ids[1:]


def beam_search(src, params, cfg, beam_size=5, length_penalty=0.6):
    src_mask = padding_mask(src, cfg.pad_id)
    enc = encode(src, src_mask, params, cfg, training=False)

    beams = [([cfg.bos_id], 0.0)]
    finished = []

    for _ in range(cfg.max_len):
        cand = []
        for ids, score in beams:
            tgt = torch.tensor([ids], device=src.device)
            logp = torch.log_softmax(next_logits(tgt, enc, src_mask, params, cfg), -1)[0]
            top_v, top_i = logp.topk(beam_size)
            for v, i in zip(top_v.tolist(), top_i.tolist()):
                cand.append((ids + [i], score + v))

        cand.sort(key=lambda c: c[1], reverse=True)
        finished += [(ids[:-1], score) for ids, score in cand if ids[-1] == cfg.eos_id]
        beams = [(ids, score) for ids, score in cand if ids[-1] != cfg.eos_id][:beam_size]
        if not beams:
            break

    if not finished:
        finished = beams

    def normalized(item):
        ids, score = item
        return score / ((5 + len(ids)) / 6) ** length_penalty

    return max(finished, key=normalized)[0]


def bleu(hyps, refs, max_n=4, unit="word"):
    """Corpus BLEU, add-1 smoothed. unit='char' removes spaces first."""
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0

    tokenize = str.split if unit == "word" else lambda text: list(text.replace(" ", ""))

    for hyp, ref in zip(hyps, refs):
        h, r = tokenize(hyp), tokenize(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            ref_counts = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            for i in range(len(h) - n + 1):
                gram = tuple(h[i:i + n])
                totals[n - 1] += 1
                if ref_counts[gram] > 0:
                    matches[n - 1] += 1
                    ref_counts[gram] -= 1

    prec = [(m + 1) / (t + 1) for m, t in zip(matches, totals)]
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(sum(math.log(p) for p in prec) / max_n)


def run(ckpt="ckpt/best.pt", beam=5, n=100, split="test", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg, params, _, step, best = load_checkpoint(ckpt, device)
    vocab = Vocab.load("data/bpe.json", cfg)
    pairs = Data(cfg, vocab, split).pairs[:n]

    mode = f"beam {beam}" if beam > 1 else "greedy"
    print(f"{ckpt}: step {step}, best val {best:.4f} | {split} {len(pairs)} sentences | {mode}")

    hyps, refs = [], []
    for i, (src_ids, tgt_ids) in enumerate(pairs, 1):
        src = torch.tensor([src_ids + [cfg.eos_id]], device=device)
        out = beam_search(src, params, cfg, beam) if beam > 1 else greedy_decode(src, params, cfg)

        hyp, ref = vocab.decode(out), vocab.decode(tgt_ids)
        hyps.append(hyp)
        refs.append(ref)

        if i <= 5:
            print(f"\nsrc: {vocab.decode(src_ids)}\nref: {ref}\nhyp: {hyp}")
        elif i % 25 == 0:
            print(f"  {i}/{len(pairs)}")

    print(f"\nBLEU word {bleu(hyps, refs):.2f} | char {bleu(hyps, refs, unit='char'):.2f} "
          f"({len(pairs)} sentences)")
