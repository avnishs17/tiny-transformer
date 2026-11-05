from dataclasses import dataclass


@dataclass
class Config:
    
    data_dir: str = "data"
    
    # tokenizer: byte-level BPE never yields OOV, so there is no <unk>
    vocab_size: int = 6003
    pad_id: int = 6000
    bos_id: int = 6001
    eos_id: int = 6002

    # data
    max_len: int = 64
    batch_size: int = 64

    # model
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 1024
    num_layers: int = 4
    dropout: float = 0.2

    # training
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    weight_decay: float = 0.05