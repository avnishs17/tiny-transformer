"""Modal wrapper: infrastructure only. Model code stays in src/tinytransformer.

Data prep runs locally (`python main.py prepare|bpe|tokenize`).
Training runs here, writing checkpoints to the `tt-ckpt` volume.
Generation runs locally against a fetched checkpoint (`modal volume get`).
"""

import modal

app = modal.App("tinytransformer")
ckpt = modal.Volume.from_name("tt-ckpt", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")  # PyPI wheel is the CUDA build
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("data/bpe.json", remote_path="/root/data/bpe.json")
    .add_local_dir("data/tokenized", remote_path="/root/data/tokenized")
)


@app.function(image=image, gpu="T4", timeout=4 * 60 * 60, volumes={"/root/ckpt": ckpt})
def train(epochs: int = 30, resume: str | None = None, patience: int = 3):
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    if resume == "latest":
        resume = "ckpt/last.pt" if os.path.exists("ckpt/last.pt") else None
        print(f"resume latest -> {resume}")

    from tinytransformer.train import run

    # commit per epoch: a timeout or preemption still leaves a resumable checkpoint
    run(epochs=epochs, device="cuda", resume=resume, patience=patience,
        on_epoch=lambda step, best: ckpt.commit())
    ckpt.commit()


@app.local_entrypoint()
def main(epochs: int = 30, resume: str | None = None, patience: int = 3, gpu: str = "T4"):
    train.with_options(gpu=gpu).remote(epochs=epochs, resume=resume, patience=patience)
