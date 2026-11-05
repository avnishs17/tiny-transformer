import pickle
from pathlib import Path

from tinytransformer.bpe import BPE

PROCESSED = Path("data/processed")
TOKENIZED = Path("data/tokenized")
BPE_FILE = Path("data/bpe.json")


def summary(lengths):
    lengths = sorted(lengths)
    n = len(lengths)
    return lengths[n // 2], lengths[int(n * 0.99)], lengths[-1]


def run():
    bpe = BPE.load(BPE_FILE)
    TOKENIZED.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        en = (PROCESSED / f"{split}.en").read_text(encoding="utf-8").splitlines()
        hi = (PROCESSED / f"{split}.hi").read_text(encoding="utf-8").splitlines()

        pairs = []
        for i, (e, t) in enumerate(zip(en, hi), 1):
            pairs.append((bpe.encode(e), bpe.encode(t)))
            if i % 5000 == 0:
                print(f"  {split}: {i}/{len(en)}")

        with open(TOKENIZED / f"{split}.pkl", "wb") as f:
            pickle.dump(pairs, f)

        src = summary([len(a) for a, _ in pairs])
        tgt = summary([len(b) for _, b in pairs])
        print(f"{split}: {len(pairs)} pairs  src p50/p99/max {src}  tgt p50/p99/max {tgt}")
