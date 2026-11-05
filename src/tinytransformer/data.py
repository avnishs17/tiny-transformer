import pickle
import random

import torch

from tinytransformer.bpe import BPE
from tinytransformer.layers import padding_mask


class Vocab:
    def __init__(self, bpe, cfg):
        self.bpe = bpe
        self.cfg = cfg
        assert bpe.vocab_size == cfg.pad_id, "BPE vocab and special ids disagree"

    @classmethod
    def load(cls, path, cfg):
        return cls(BPE.load(path), cfg)

    def encode(self, text):
        return self.bpe.encode(text)

    def decode(self, ids):
        return self.bpe.decode([i for i in ids if i < self.cfg.pad_id])

    def src_input(self, ids):
        return ids + [self.cfg.eos_id]

    def tgt_input(self, ids):
        return [self.cfg.bos_id] + ids

    def tgt_label(self, ids):
        return ids + [self.cfg.eos_id]


class Data:
    def __init__(self, cfg, vocab, split):
        self.cfg = cfg
        self.vocab = vocab

        with open(f"{cfg.data_dir}/tokenized/{split}.pkl", "rb") as f:
            pairs = pickle.load(f)

        self.pairs = [
            (s, t) for s, t in pairs
            if len(s) + 1 <= cfg.max_len and len(t) + 1 <= cfg.max_len
        ]

    def batches(self, shuffle=True):

        order = list(range(len(self.pairs)))

        if shuffle:
            random.shuffle(order)

        order.sort(key=lambda i: len(self.pairs[i][0]) + len(self.pairs[i][1]))

        batches = [
            order[i:i + self.cfg.batch_size]
            for i in range(0, len(order), self.cfg.batch_size)
        ]

        if shuffle:
            random.shuffle(batches)

        for idx in batches:
            yield self._collate([self.pairs[i] for i in idx])

    def _collate(self, pairs):

        src = self._pad([self.vocab.src_input(s) for s, _ in pairs])
        tgt_in = self._pad([self.vocab.tgt_input(t) for _, t in pairs])
        tgt_label = self._pad([self.vocab.tgt_label(t) for _, t in pairs])

        return {
            "src": src,
            "src_mask": padding_mask(src, self.cfg.pad_id),
            "tgt_in": tgt_in,
            "tgt_label": tgt_label,
            "tgt_mask": padding_mask(tgt_in, self.cfg.pad_id),
        }

    def _pad(self, rows):

        width = max(len(r) for r in rows)
        out = [r + [self.cfg.pad_id] * (width - len(r)) for r in rows]
        return torch.tensor(out, dtype=torch.long)
