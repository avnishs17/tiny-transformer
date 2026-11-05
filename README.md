# tinytransformer

Encoder-decoder Transformer written from scratch on raw tensors and plain dicts of
parameters. Byte-level BPE, hand-written attention, hand-written AdamW, no `nn.Module`.
English to Hindi translation on OpenSubtitles en-hi: 77,113 train pairs after
dedupe, script/length filters, source-grouped split and the 64-token cap.

8.9M parameters. One epoch takes about 75 seconds on a T4 GPU.

## Setup

```bash
uv sync
```

## Pipeline

Data prep always runs locally. It runs once and produces files that training reads.

```bash
uv run python main.py prepare    # download OpenSubtitles en-hi, split -> data/processed
uv run python main.py bpe        # train byte-level BPE -> data/bpe.json
uv run python main.py tokenize   # encode splits -> data/tokenized/*.pkl
```

## Train on Modal

Training runs on a GPU and writes checkpoints to the `tt-ckpt` Modal volume.

```bash
modal run modal_app.py --epochs 40          # fresh run
modal run modal_app.py --resume latest      # continue from ckpt/last.pt in the volume
```

GPU defaults to T4. Change it per run with `--gpu A10G`, or edit the `gpu=` value in
`modal_app.py`.

The function commits the volume after every epoch. A timeout or preemption keeps the
last completed epoch, so `--resume latest` continues from it.

Fetch checkpoints to this machine for local generation:

```bash
modal volume ls tt-ckpt
modal volume get tt-ckpt best.pt ./ckpt/best.pt
```

## Train locally

The same loop runs on CPU or a local GPU. Use this if Modal credits are unavailable.

```bash
uv run python main.py train --epochs 30 --device cpu
uv run python main.py train --resume ckpt/last.pt
```

Checkpoints go to `ckpt/best.pt` (best val loss) and `ckpt/last.pt` (latest epoch).
Each is about 102 MB: weights plus Adam moments.

## Generate locally

Needs `ckpt/best.pt`, `data/bpe.json` and `data/tokenized/`.

```bash
uv run python main.py generate --ckpt ckpt/best.pt --beam 5 --n 100
uv run python main.py generate --beam 1 --n 5          # greedy, quick look
```

The command prints the first five source/reference/hypothesis triples, then BLEU.
Word and character BLEU are both reported. Hindi uses spaces, so word BLEU is
meaningful here; character BLEU is the more stable of the two.

## Results

OpenSubtitles en-hi, beam 5, first 100 test sentences, seed 0:

| metric | value |
| --- | --- |
| parameters | 8,897,280 |
| best val loss | 3.5037 |
| val token accuracy | 0.528 |
| word BLEU | 25.75 |
| char BLEU | 52.04 |
| epochs trained | 20, early stopped (patience 3) |
| best checkpoint | epoch 17, 20,485 steps |
| steps trained | 24,100 |

`best.pt` is the epoch-17 checkpoint: 20,485 steps, the lowest val loss. Training
then ran to epoch 20 with no further improvement and stopped. All step counts
assume 1,205 batches per epoch (77,113 train pairs, batch 64).

Training config: dropout 0.2, label smoothing 0.1, weight decay 0.05, batch 64,
4 layers, d_model 256, 4 heads. Train loss 2.50 against val 3.51, so the model
overfits. BLEU on 100 sentences carries about ±2 to 3 points of noise.

Sample output:

```
src: Hey... he left his phone here.
ref: अरे ... वह अपने फोन यहाँ छोड़ दिया।
hyp: हे... वह यहाँ अपने फोन छोड़ दिया।

src: - At the hotel.
ref: - होटल में।
hyp: - होटल में.
```

## Layout

```
main.py              single entry point for every command
modal_app.py         Modal image, GPU, volume; calls tinytransformer.train.run
pipeline/            one-time data prep: download, split, BPE encode
src/tinytransformer/
  config.py          hyperparameters
  bpe.py             byte-level BPE: train, encode, decode
  data.py            Vocab and Data, bucketed batches padded to batch max
  layers.py          masks, LayerNorm, FFN, multi-head attention
  model.py           encoder/decoder stack and forward pass
  params.py          parameter allocation, flatten, count
  optim.py           hand-written AdamW with decoupled decay
  train.py           train_step, epoch loop, run
  generate.py        greedy decode, beam search, BLEU
  utils.py           Noam schedule, loss, eval, checkpoint save/load
data/                gitignored: raw archive, processed text, tokenized pkl, bpe.json
ckpt/                gitignored: best.pt, last.pt
```

## Limits

- `max_len` is 64 tokens; 32 train, 0 val and 2 test pairs are dropped.
- Dedupe, script check and length-ratio filter run before splitting. Splits are grouped
  by English source line, so no source sentence appears in two splits.
- Post-norm, ReLU FFN, sinusoidal positions, shared embedding and output matrix.
- One trained epoch produces repetitive output. Train longer before judging quality.
