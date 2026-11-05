"""Single entry point. Run from the project root: python main.py <cmd>"""

import argparse


def main():
    ap = argparse.ArgumentParser(prog="tinytransformer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="download the corpus and split it")
    sub.add_parser("bpe", help="train byte-level BPE on the train split")
    sub.add_parser("tokenize", help="encode every split with the trained BPE")

    p = sub.add_parser("train", help="train the model")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--patience", type=int, default=3)

    g = sub.add_parser("generate", help="translate the test set")
    g.add_argument("--ckpt", default="ckpt/best.pt")
    g.add_argument("--beam", type=int, default=5)
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--split", default="test")
    g.add_argument("--device", default=None)

    a = ap.parse_args()

    # imported per branch: prepare/bpe/tokenize never pay for torch
    if a.cmd == "prepare":
        from pipeline.prepare_data import run
        run()

    elif a.cmd == "bpe":
        from tinytransformer.bpe import run
        run(["data/processed/train.en", "data/processed/train.hi"])

    elif a.cmd == "tokenize":
        from pipeline.tokenize_data import run
        run()

    elif a.cmd == "train":
        from tinytransformer.train import run
        run(epochs=a.epochs, device=a.device, resume=a.resume, patience=a.patience)

    elif a.cmd == "generate":
        from tinytransformer.generate import run
        run(ckpt=a.ckpt, beam=a.beam, n=a.n, split=a.split, device=a.device)


if __name__ == "__main__":
    main()
