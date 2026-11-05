import random
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-hi.txt.zip"
PREFIX = "OpenSubtitles.en-hi"
ZIP = RAW / "en-hi.txt.zip"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        print("archive already present")
        return
    print("downloading en-hi")
    urllib.request.urlretrieve(URL, ZIP)


def extract():
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW)


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def write(path, rows):
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def devanagari(text):
    return any(0x0900 <= ord(c) <= 0x097F for c in text)


def usable(en, hi):
    if en.lower() == hi.lower():
        return False
    if not devanagari(hi) or devanagari(en):
        return False
    return 0.3 <= len(hi) / len(en) <= 3.0


def split_by_source(pairs, val_size, test_size, seed=0):
    # one English line must land in exactly one split, even with several Hindi targets
    groups = {}
    for en, hi in pairs:
        groups.setdefault(en, []).append((en, hi))

    sources = list(groups)
    random.Random(seed).shuffle(sources)

    test, val, train = [], [], []
    for src in sources:
        rows = groups[src]
        if len(test) < test_size:
            test.extend(rows)
        elif len(val) < val_size:
            val.extend(rows)
        else:
            train.extend(rows)
    return train, val, test


def run():
    download()
    extract()

    en = read_lines(RAW / f"{PREFIX}.en")
    hi = read_lines(RAW / f"{PREFIX}.hi")
    if len(en) != len(hi):
        raise SystemExit(f"misaligned corpus: {len(en)} en vs {len(hi)} hi")

    raw = [(e.strip(), t.strip()) for e, t in zip(en, hi) if e.strip() and t.strip()]
    pairs = sorted(set(raw))
    print(f"{len(raw):,} pairs -> {len(pairs):,} after exact dedupe")

    pairs = [p for p in pairs if usable(*p)]
    print(f"{len(pairs):,} after script and length filters")

    train, val, test = split_by_source(pairs, val_size=1000, test_size=1000)

    src = {name: {e for e, _ in rows}
           for name, rows in (("train", train), ("val", val), ("test", test))}
    assert not src["train"] & src["val"], "source leak train/val"
    assert not src["train"] & src["test"], "source leak train/test"
    assert not src["val"] & src["test"], "source leak val/test"

    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | no cross-split sources")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val), ("test", test)):
        write(PROCESSED / f"{split}.en", [e for e, _ in rows])
        write(PROCESSED / f"{split}.hi", [t for _, t in rows])

    print("next: python main.py bpe")
