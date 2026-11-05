import math

import torch
import torch.nn.functional as F

from tinytransformer.layers import (
    causal_mask,
    decoder_mask,
    feed_forward,
    layer_norm,
    multi_head_attention,
    padding_mask,
)


def positional_encoding(length, d_model, device):
    pos = torch.arange(length, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def embed(ids, params, cfg):
    x = params["embed"][ids] * math.sqrt(cfg.d_model)
    return x + positional_encoding(ids.shape[1], cfg.d_model, ids.device)


def encoder_layer(x, params, prefix, cfg, mask, training):
    p = cfg.dropout
    a = multi_head_attention(
        x, x,
        params[f"{prefix}.w_q"], params[f"{prefix}.w_k"],
        params[f"{prefix}.w_v"], params[f"{prefix}.w_o"],
        cfg.num_heads, mask,
    )
    x = layer_norm(x + F.dropout(a, p, training),
                   params[f"{prefix}.attn_gamma"], params[f"{prefix}.attn_beta"])

    f = feed_forward(x, params[f"{prefix}.w1"], params[f"{prefix}.b1"],
                     params[f"{prefix}.w2"], params[f"{prefix}.b2"])
    x = layer_norm(x + F.dropout(f, p, training),
                   params[f"{prefix}.ffn_gamma"], params[f"{prefix}.ffn_beta"])
    return x


def encode(src, src_mask, params, cfg, training):
    x = embed(src, params, cfg)
    for i in range(cfg.num_layers):
        x = encoder_layer(x, params, f"enc.{i}", cfg, src_mask, training)
    return x


def decoder_layer(y, enc_out, params, prefix, cfg, self_mask, cross_mask, training):
    p = cfg.dropout

    a = multi_head_attention(
        y, y,
        params[f"{prefix}.self_w_q"], params[f"{prefix}.self_w_k"],
        params[f"{prefix}.self_w_v"], params[f"{prefix}.self_w_o"],
        cfg.num_heads, self_mask,
    )
    y = layer_norm(y + F.dropout(a, p, training),
                   params[f"{prefix}.self_gamma"], params[f"{prefix}.self_beta"])

    c = multi_head_attention(
        y, enc_out,
        params[f"{prefix}.cross_w_q"], params[f"{prefix}.cross_w_k"],
        params[f"{prefix}.cross_w_v"], params[f"{prefix}.cross_w_o"],
        cfg.num_heads, cross_mask,
    )
    y = layer_norm(y + F.dropout(c, p, training),
                   params[f"{prefix}.cross_gamma"], params[f"{prefix}.cross_beta"])

    f = feed_forward(y, params[f"{prefix}.w1"], params[f"{prefix}.b1"],
                     params[f"{prefix}.w2"], params[f"{prefix}.b2"])
    y = layer_norm(y + F.dropout(f, p, training),
                   params[f"{prefix}.ffn_gamma"], params[f"{prefix}.ffn_beta"])
    return y


def decode(tgt_in, enc_out, src_mask, params, cfg, training):
    T = tgt_in.shape[1]
    self_mask = decoder_mask(padding_mask(tgt_in, cfg.pad_id),
                             causal_mask(T).to(tgt_in.device))

    y = embed(tgt_in, params, cfg)
    for i in range(cfg.num_layers):
        y = decoder_layer(y, enc_out, params, f"dec.{i}", cfg, self_mask, src_mask, training)
    return y


def forward(batch, params, cfg, training=True):
    enc_out = encode(batch["src"], batch["src_mask"], params, cfg, training)
    dec_out = decode(batch["tgt_in"], enc_out, batch["src_mask"], params, cfg, training)
    return dec_out @ params["embed"].T
