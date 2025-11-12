# modules/transformer/transformer_core.py
"""
Full Transformer core implemented with a NumPy/CuPy hybrid backend.

Usage:
    from modules.transformer.transformer_core import TransformerConfig, TransformerModel

Design goals:
- Uses xp (from modules.utils.device_manager) as the numeric backend (NumPy or CuPy).
- Keeps device conversions via to_device/to_cpu.
- Implements Encoder/Decoder Transformer with positional encoding,
  multi-head self-attention, feed-forward, layer normalization, dropout,
  and a simple PhaseInterferenceSafety (quantum-inspired safety stub).
- Deterministic random seed should be set in device_manager (xp.random.seed).
"""

import math
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

# Use the project's device manager for backend abstraction
from modules.utils.device_manager import xp, to_device, to_cpu, DEVICE

# Ensure reproducibility (device_manager should already set seed; we keep safe call)
try:
    xp.random.seed(42)
except Exception:
    # Backend may not support seed() the same way (very unlikely), ignore gracefully
    pass

# ---------- Config and constants ----------
@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            d = json.load(f)
        return cls(**d)


WEIGHT_INIT_FACTOR = 0.02
ATTENTION_MASK_VALUE = -1e9


# ---------- Utilities ----------
def xavier_init(shape: Tuple[int, ...]) -> xp.ndarray:
    """Xavier/Glorot uniform initialization (returns xp.ndarray)."""
    if len(shape) < 2:
        fan_in = shape[0]
        fan_out = shape[0]
    else:
        fan_in = shape[-2]
        fan_out = shape[-1]
    limit = xp.sqrt(6.0 / (fan_in + fan_out))
    return to_device(xp.random.uniform(-limit, limit, size=shape))


def zeros(shape: Tuple[int, ...]) -> xp.ndarray:
    return to_device(xp.zeros(shape))


def ones(shape: Tuple[int, ...]) -> xp.ndarray:
    return to_device(xp.ones(shape))


def dropout(x: xp.ndarray, p: float, training: bool = True) -> Tuple[xp.ndarray, Optional[xp.ndarray]]:
    """Apply dropout. Returns (output, mask) where mask is None if not training or p==0."""
    if not training or p <= 0.0:
        return x, None
    # binomial mask 0/1 then scale to keep expectation
    mask = xp.random.binomial(1, 1 - p, size=x.shape).astype(x.dtype) / (1 - p)
    return x * mask, mask


def stable_softmax(x: xp.ndarray, axis: int = -1) -> xp.ndarray:
    xmax = xp.max(x, axis=axis, keepdims=True)
    ex = xp.exp(x - xmax)
    return ex / xp.sum(ex, axis=axis, keepdims=True)


def create_causal_mask(seq_len: int) -> xp.ndarray:
    """Create causal (lower-triangular) mask for self-attention where True means keep."""
    mask = xp.tril(xp.ones((seq_len, seq_len), dtype=bool))
    return mask


# ---------- LayerNorm ----------
class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = to_device(xp.ones((d_model,)))
        self.beta = to_device(xp.zeros((d_model,)))

        # cached for backward (if needed)
        self.x = None
        self.mean = None
        self.var = None
        self.x_norm = None

        # store last forward output so other modules/training code can read .out
        self.out = None

        # grads (set during backward)
        self.d_gamma = None
        self.d_beta = None

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # x shape: (..., d_model)
        self.x = x
        self.mean = xp.mean(x, axis=-1, keepdims=True)
        self.var = xp.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / xp.sqrt(self.var + self.eps)
        out = self.gamma * self.x_norm + self.beta

        # save forward output for external use (training / projection code expects this)
        self.out = out
        return out

    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        """Analytic backward for layernorm. Assumes last dim is features."""
        x = self.x
        mean = self.mean
        var = self.var
        eps = self.eps
        N = x.shape[-1]

        d_x_norm = d_out * self.gamma
        d_var = xp.sum(d_x_norm * (x - mean) * -0.5 * xp.power(var + eps, -1.5), axis=-1, keepdims=True)
        d_mean = xp.sum(d_x_norm * (-1.0 / xp.sqrt(var + eps)), axis=-1, keepdims=True) + d_var * xp.mean(-2.0 * (x - mean), axis=-1, keepdims=True)

        d_x = d_x_norm / xp.sqrt(var + eps) + d_var * 2.0 * (x - mean) / N + d_mean / N

        # param grads: sum over all dims except last
        axis = tuple(range(d_out.ndim - 1))
        if axis:
            self.d_gamma = xp.sum(d_out * self.x_norm, axis=axis)
            self.d_beta = xp.sum(d_out, axis=axis)
        else:
            self.d_gamma = d_out * self.x_norm
            self.d_beta = d_out
        return d_x

    def __call__(self, x: xp.ndarray) -> xp.ndarray:
        return self.forward(x)

    def get_cpu_weights(self) -> Dict[str, Any]:
        return {"gamma": to_cpu(self.gamma), "beta": to_cpu(self.beta)}


# ---------- MultiHeadAttention ----------
class MultiHeadAttention:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads

        # weights: shape (d_model, d_model) for projections for convenience
        self.W_q = to_device(xavier_init((self.d_model, self.d_model)))
        self.W_k = to_device(xavier_init((self.d_model, self.d_model)))
        self.W_v = to_device(xavier_init((self.d_model, self.d_model)))
        self.W_o = to_device(xavier_init((self.d_model, self.d_model)))

        # caches for backward
        self.Q = None
        self.K = None
        self.V = None
        self.attention_weights = None
        self.attention_output = None
        self.x_input = None
        self.kv_input = None

        # gradients placeholders
        self.d_W_q = None
        self.d_W_k = None
        self.d_W_v = None
        self.d_W_o = None

    def _split_heads(self, x: xp.ndarray) -> xp.ndarray:
        # x: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: xp.ndarray) -> xp.ndarray:
        # x: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch, n_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_heads * d_k)
        return x

    def scaled_dot_product_attention(self, Q: xp.ndarray, K: xp.ndarray, V: xp.ndarray, mask: Optional[xp.ndarray] = None) -> Tuple[xp.ndarray, xp.ndarray]:
        # Q,K,V shapes: (batch, n_heads, seq_len, d_k)
        d_k = Q.shape[-1]
        scores = xp.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)  # (batch, n_heads, seq_q, seq_k)
        if mask is not None:
            # mask shape should be broadcastable to scores
            # where mask == False -> set to large negative
            scores = xp.where(mask, scores, ATTENTION_MASK_VALUE)
        weights = stable_softmax(scores, axis=-1)
        output = xp.matmul(weights, V)
        return output, weights

    def forward(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None, key_value_input: Optional[xp.ndarray] = None) -> xp.ndarray:
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        kv = x if key_value_input is None else key_value_input

        # project
        Q = xp.matmul(x, self.W_q)   # (batch, seq_len, d_model)
        K = xp.matmul(kv, self.W_k)
        V = xp.matmul(kv, self.W_v)

        # split heads
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        self.Q, self.K, self.V = Qh, Kh, Vh
        self.x_input = x
        self.kv_input = kv

        # scaled dot-product attention
        attn_out, weights = self.scaled_dot_product_attention(Qh, Kh, Vh, mask)
        self.attention_weights = weights

        # combine heads
        combined = self._combine_heads(attn_out)  # (batch, seq_len, d_model)
        self.attention_output = combined

        # final linear
        out = xp.matmul(combined, self.W_o)
        return out

    def backward(self, d_out: xp.ndarray) -> Tuple[xp.ndarray, Optional[xp.ndarray]]:
        """
        Very approximate manual backward only to support tests that request grads from layers.
        This returns (d_x, d_kv_input) and sets d_W_* attributes.
        NOTE: This is a functional backward approximation sufficient for unit tests relying
              on shapes and rough gradient propagation. For a full autograd system,
              replacement with a proper autodiff engine is recommended.
        """
        # shapes & cached
        batch, seq_len, d_model = self.x_input.shape
        batch_kv, seq_len_kv, d_model_kv = self.kv_input.shape

        # d_W_o
        self.d_W_o = xp.matmul(self.attention_output.reshape(-1, d_model).T, d_out.reshape(-1, d_model))
        d_combined = xp.matmul(d_out.reshape(-1, d_model), self.W_o.T).reshape(batch, seq_len, d_model)

        # reshape to heads
        d_attn_heads = d_combined.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # gradient wrt attention weights and V
        d_attn_weights = xp.matmul(d_attn_heads, self.V.transpose(0, 1, 3, 2))
        d_V = xp.matmul(self.attention_weights.transpose(0, 1, 3, 2), d_attn_heads)

        # softmax backward (vector-Jacobian product)
        d_scores = d_attn_weights
        # weights * (d - sum(d*weights))
        d_softmax = self.attention_weights * (d_scores - xp.sum(d_scores * self.attention_weights, axis=-1, keepdims=True))

        # grads for Q,K
        d_Q = xp.matmul(d_softmax, self.K)
        d_K = xp.matmul(self.Q.transpose(0, 1, 3, 2), d_softmax)
        d_K = d_K.transpose(0, 1, 3, 2)

        # reshape back to (batch, seq_len, d_model)
        d_Q_input = d_Q.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        d_K_input = d_K.transpose(0, 2, 1, 3).reshape(batch_kv, seq_len_kv, d_model_kv)
        d_V_input = d_V.transpose(0, 2, 1, 3).reshape(batch_kv, seq_len_kv, d_model_kv)

        # weight grads
        self.d_W_q = xp.matmul(self.x_input.transpose(1, 0, 2).reshape(-1, d_model).T, d_Q_input.reshape(-1, d_model))
        self.d_W_k = xp.matmul(self.kv_input.transpose(1, 0, 2).reshape(-1, d_model_kv).T, d_K_input.reshape(-1, d_model_kv))
        self.d_W_v = xp.matmul(self.kv_input.transpose(1, 0, 2).reshape(-1, d_model_kv).T, d_V_input.reshape(-1, d_model_kv))

        # propagate to inputs
        d_x_from_q = xp.matmul(d_Q_input, self.W_q.T)
        d_kv_from_k = xp.matmul(d_K_input, self.W_k.T)
        d_kv_from_v = xp.matmul(d_V_input, self.W_v.T)

        d_x = d_x_from_q
        d_kv = d_kv_from_k + d_kv_from_v

        # if kv is same as x, sum gradients
        if self.kv_input is self.x_input:
            return d_x + d_kv, None
        else:
            return d_x, d_kv


# ---------- FeedForward ----------
class FeedForward:
    def __init__(self, config: TransformerConfig):
        self.W1 = to_device(xavier_init((config.d_model, config.d_ff)))
        self.b1 = to_device(xp.zeros((config.d_ff,)))
        self.W2 = to_device(xavier_init((config.d_ff, config.d_model)))
        self.b2 = to_device(xp.zeros((config.d_model,)))

        # caches for backward
        self.x = None
        self.hidden_pre = None
        self.hidden = None
        self.output = None

        # grads
        self.d_W1 = None
        self.d_b1 = None
        self.d_W2 = None
        self.d_b2 = None

    def relu(self, x: xp.ndarray) -> xp.ndarray:
        return xp.maximum(0, x)

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # x: (batch, seq_len, d_model)
        self.x = x
        self.hidden_pre = xp.matmul(x, self.W1) + self.b1
        self.hidden = self.relu(self.hidden_pre)
        self.output = xp.matmul(self.hidden, self.W2) + self.b2
        return self.output

    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        # shapes
        batch, seq_len, _ = self.x.shape

        # grads for W2/b2
        self.d_W2 = xp.matmul(self.hidden.reshape(-1, self.hidden.shape[-1]).T, d_out.reshape(-1, d_out.shape[-1]))
        self.d_b2 = xp.sum(d_out, axis=(0, 1))

        d_hidden = xp.matmul(d_out, self.W2.T)
        d_hidden_pre = d_hidden * (self.hidden_pre > 0)

        self.d_W1 = xp.matmul(self.x.reshape(-1, self.x.shape[-1]).T, d_hidden_pre.reshape(-1, d_hidden_pre.shape[-1]))
        self.d_b1 = xp.sum(d_hidden_pre, axis=(0, 1))

        d_x = xp.matmul(d_hidden_pre, self.W1.T)
        return d_x


# ---------- PositionalEncoding ----------
class PositionalEncoding:
    def __init__(self, d_model: int, max_seq_len: int = 512):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # precompute in CPU then move to device to avoid weird backend semantics
        pe = xp.zeros((self.max_seq_len, self.d_model))
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                angle = pos / (10000 ** ((2 * i) / self.d_model))
                pe[pos, i] = xp.sin(angle)
                if i + 1 < self.d_model:
                    pe[pos, i + 1] = xp.cos(pos / (10000 ** ((2 * (i + 1)) / self.d_model)))
        self.pos_encoding = to_device(pe)

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # x shape: (batch, seq_len, d_model)
        seq_len = x.shape[1]
        return x + self.pos_encoding[:seq_len, :]

    def __call__(self, x: xp.ndarray) -> xp.ndarray:
        return self.forward(x)


# ---------- Transformer block / Encoder / Decoder ----------
class TransformerBlock:
    def __init__(self, config: TransformerConfig):
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = config.dropout

        # caches for backward
        self.x_input = None
        self.attn_output = None
        self.ff_output = None

    def forward(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        self.x_input = x
        attn_out = self.attention.forward(x, mask)
        self.attn_output = attn_out
        x = x + attn_out  # residual
        x = self.norm1.forward(x)
        x, _ = dropout(x, self.dropout, training=True)

        ff_out = self.feed_forward.forward(x)
        self.ff_output = ff_out
        x = x + ff_out
        x = self.norm2.forward(x)
        x, _ = dropout(x, self.dropout, training=True)
        return x

    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        d_x_after_res2 = self.norm2.backward(d_out)
        d_ff_output = d_x_after_res2
        d_x_norm1 = d_x_after_res2

        d_x_norm1_from_ff = self.feed_forward.backward(d_ff_output)
        d_x_norm1 = d_x_norm1 + d_x_norm1_from_ff

        d_x_after_res1 = self.norm1.backward(d_x_norm1)
        d_attn_output = d_x_after_res1
        d_x_input = d_x_after_res1

        d_x_from_attn, _ = self.attention.backward(d_attn_output)
        d_x = d_x_input + d_x_from_attn
        return d_x

    def __call__(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        return self.forward(x, mask)


class EncoderLayer:
    def __init__(self, config: TransformerConfig):
        self.mha = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.dropout = config.dropout

        # caches
        self.x = None
        self.attn_output = None
        self.ff_output = None
        self.attn_dropout_mask = None
        self.ff_dropout_mask = None

    def forward(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        self.x = x
        attn_out = self.mha.forward(x, mask)
        self.attn_output = attn_out
        x = x + attn_out
        x = self.norm1.forward(x)
        x, self.attn_dropout_mask = dropout(x, self.dropout, training=True)

        ff_out = self.ff.forward(x)
        self.ff_output = ff_out
        x = x + ff_out
        x = self.norm2.forward(x)
        x, self.ff_dropout_mask = dropout(x, self.dropout, training=True)
        return x

    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        d_out = d_out * self.ff_dropout_mask if self.ff_dropout_mask is not None else d_out
        d_ff_output = self.norm2.backward(d_out)
        d_attn_from_ff = d_ff_output
        d_ff_input = self.ff.backward(d_ff_output)

        d_attn_output = self.norm1.backward(d_attn_from_ff + d_ff_input)
        d_attn_output = d_attn_output * self.attn_dropout_mask if self.attn_dropout_mask is not None else d_attn_output
        d_x_from_attn, _ = self.mha.backward(d_attn_output)

        d_x = d_x_from_attn + d_attn_output
        return d_x

    def __call__(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        return self.forward(x, mask)


class DecoderLayer:
    def __init__(self, config: TransformerConfig):
        self.masked_mha = MultiHeadAttention(config)
        self.mha = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.norm3 = LayerNorm(config.d_model)
        self.dropout = config.dropout
        self.phase_safety = PhaseInterferenceSafety(config.d_model)

        # caches
        self.x = None
        self.enc_output = None
        self.masked_attn_output = None
        self.attn_output = None
        self.ff_output = None
        self.masked_attn_dropout_mask = None
        self.attn_dropout_mask = None
        self.ff_dropout_mask = None

    def forward(self, x: xp.ndarray, enc_output: xp.ndarray, look_ahead_mask: Optional[xp.ndarray] = None, padding_mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        self.x = x
        self.enc_output = enc_output

        masked_attn = self.masked_mha.forward(x, look_ahead_mask)
        self.masked_attn_output = masked_attn
        x = x + masked_attn
        x = self.norm1.forward(x)
        x, self.masked_attn_dropout_mask = dropout(x, self.dropout, training=True)

        attn_out = self.mha.forward(x, padding_mask, key_value_input=enc_output)
        self.attn_output = attn_out
        x = x + attn_out
        x = self.norm2.forward(x)
        x, self.attn_dropout_mask = dropout(x, self.dropout, training=True)

        ff_out = self.ff.forward(x)
        self.ff_output = ff_out
        x = x + ff_out
        x = self.norm3.forward(x)

        # phase safety: keep staged and reused to avoid destructive interference
        phase_placeholder = xp.zeros_like(x, dtype=xp.complex128).imag
        x = self.phase_safety.apply_safety(x, phase_placeholder)
        self.phase_safety.quantum_state = x
        current_phase = phase_placeholder
        self.phase_safety.current_phase = current_phase
        x = self.phase_safety.apply_safety(x, current_phase)
        x, self.ff_dropout_mask = dropout(x, self.dropout, training=True)
        return x

    def backward(self, d_out: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        d_out = d_out * self.ff_dropout_mask if self.ff_dropout_mask is not None else d_out
        d_ff_from_phase = self.phase_safety.backward(d_out)
        d_ff = self.norm3.backward(d_ff_from_phase + d_out)
        d_ff_input = self.ff.backward(d_ff)

        d_attn = self.norm2.backward(d_ff + d_ff_input)
        d_attn = d_attn * self.attn_dropout_mask if self.attn_dropout_mask is not None else d_attn
        d_masked_attn_from_attn = d_attn
        d_masked_attn_from_mha, d_enc_from_mha = self.mha.backward(d_attn)
        d_enc = d_enc_from_mha

        d_masked = self.norm1.backward(d_masked_attn_from_attn + d_masked_attn_from_mha)
        d_masked = d_masked * self.masked_attn_dropout_mask if self.masked_attn_dropout_mask is not None else d_masked
        d_x_from_masked, _ = self.masked_mha.backward(d_masked)

        d_x = d_x_from_masked + d_masked
        return d_x, d_enc

    def __call__(self, x: xp.ndarray, enc_output: xp.ndarray, look_ahead_mask: Optional[xp.ndarray] = None, padding_mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        return self.forward(x, enc_output, look_ahead_mask, padding_mask)


# ---------- Embeddings ----------
class Embeddings:
    def __init__(self, config: TransformerConfig):
        self.token_embedding = to_device(xavier_init((config.vocab_size, config.d_model)))
        self.position_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.input_ids = None
        self.token_embeds = None
        self.d_token_embedding = None

    def forward(self, input_ids: xp.ndarray) -> xp.ndarray:
        # input_ids: (batch, seq_len) integer indices
        self.input_ids = input_ids
        # gather
        self.token_embeds = self.token_embedding[input_ids]
        out = self.token_embeds + self.position_encoding.forward(self.token_embeds)
        return out

    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        # d_out shape: (batch, seq_len, d_model)
        d_token_embedding = xp.zeros_like(self.token_embedding)
        # sparse add at indices
        xp.add.at(d_token_embedding, self.input_ids, d_out)
        self.d_token_embedding = d_token_embedding
        return d_out

    def __call__(self, input_ids: xp.ndarray) -> xp.ndarray:
        return self.forward(input_ids)

    def get_cpu_weights(self) -> Dict[str, Any]:
        return {"token_embedding": to_cpu(self.token_embedding)}


# ---------- Full Transformer ----------
class TransformerModel:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.embedding = Embeddings(config)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        # Use EncoderLayer/DecoderLayer objects (call via forward to avoid callable mismatch)
        self.encoder_layers = [EncoderLayer(config) for _ in range(config.n_layers)]
        self.decoder_layers = [DecoderLayer(config) for _ in range(config.n_layers)]
        self.final_norm = LayerNorm(config.d_model)

    def encode(self, src_ids: xp.ndarray, src_mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        # src_ids: (batch, seq_len)
        x = self.embedding.forward(src_ids)
        x = self.positional_encoding.forward(x)
        for layer in self.encoder_layers:
            # call explicit forward method instead of implicit __call__ to avoid any callable mismatch
            x = layer.forward(x, src_mask)
        return self.final_norm.forward(x)

    def decode(self, tgt_ids: xp.ndarray, memory: xp.ndarray, tgt_mask: Optional[xp.ndarray] = None, memory_mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        x = self.embedding.forward(tgt_ids)
        x = self.positional_encoding.forward(x)
        for layer in self.decoder_layers:
            x = layer.forward(x, memory, tgt_mask, memory_mask)
        return self.final_norm.forward(x)

    def forward(self, src_ids: xp.ndarray, tgt_ids: xp.ndarray, src_mask: Optional[xp.ndarray] = None, tgt_mask: Optional[xp.ndarray] = None, memory_mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        memory = self.encode(src_ids, src_mask)
        return self.decode(tgt_ids, memory, tgt_mask, memory_mask)

    # small helper used by text generation in your repo (if needed)
    def create_causal_mask(self, seq_len: int) -> xp.ndarray:
        # Returns boolean mask broadcastable to (batch, n_heads, seq_len, seq_len)
        m = create_causal_mask(seq_len)  # (seq_len, seq_len)
        return m

    def get_cpu_weights(self) -> Dict[str, Any]:
        # convenience for checkpoints: gather main params to CPU
        params = {}
        params.update(self.embedding.get_cpu_weights())
        # encoder/decoder weights would require walking layers; you can extend this if needed
        return params


# Print which device this module will use
try:
    print(f"Using {DEVICE.upper()} for transformer computation")
except Exception:
    print("Using unknown device for transformer computation")
