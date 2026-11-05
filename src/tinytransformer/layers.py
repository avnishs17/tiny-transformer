import math
import torch

def padding_mask(ids, pad_id):
    return ids != pad_id

def causal_mask(length):
    return torch.tril(torch.ones(1, length, length, dtype=torch.bool))

def decoder_mask(pad_mask, causal):
    return pad_mask.unsqueeze(1) & causal

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta

def feed_forward(x, w1, b1, w2, b2):
    return torch.relu(x @ w1 + b1) @ w2 + b2

def split_heads(x, num_heads):
    B, T, C = x.shape
    return x.view(B, T, num_heads, C // num_heads).transpose(1, 2)

def merge_heads(x):
    B, h, T, d = x.shape

    return x.transpose(1, 2).reshape(B, T, h * d)

def attention(q, k, v, mask=None):

    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

    if mask is not None:
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def multi_head_attention(q_in, kv_in, w_q, w_k, w_v, w_o, num_heads, mask=None):
    q = split_heads(q_in @ w_q, num_heads)
    k = split_heads(kv_in @ w_k, num_heads)
    v = split_heads(kv_in @ w_v, num_heads)

    out = attention(q, k, v, mask)
    return merge_heads(out) @ w_o
