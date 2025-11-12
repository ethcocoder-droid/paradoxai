from modules.utils.device_manager import xp, to_device, to_cpu, DEVICE
xp.random.seed(42) # For reproducibility
import math
from modules.utils.device_manager import xp
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class TransformerConfig:
    """Configuration for the transformer model"""
    vocab_size: int = 10000
    d_model: int = 512  # Model dimension
    n_heads: int = 8     # Number of attention heads
    n_layers: int = 6    # Number of transformer layers
    d_ff: int = 2048     # Feed-forward dimension
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def save(self, filepath: str):
        """Saves the configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Loads the configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# Constants for weight initialization and attention masking
WEIGHT_INIT_FACTOR = 0.02
ATTENTION_MASK_VALUE = -1e9

def xavier_init(shape: Tuple[int, ...]) -> xp.ndarray:
    """Initializes weights using Xavier (Glorot) initialization.

    Args:
        shape (Tuple[int, ...]): The shape of the weight matrix.

    Returns:
        xp.ndarray: Initialized weight matrix.
    """
    fan_in, fan_out = shape[-2], shape[-1]
    limit = xp.sqrt(6 / (fan_in + fan_out))
    return xp.random.uniform(-limit, limit, size=shape)

def dropout(x: xp.ndarray, p: float, training: bool = True) -> tuple[xp.ndarray, Optional[xp.ndarray]]:
    """Applies dropout to the input.

    Args:
        x (xp.ndarray): Input array.
        p (float): Dropout rate (probability of an element being zeroed).
        training (bool): If True, apply dropout. If False, do nothing.

    Returns:
        tuple[xp.ndarray, Optional[xp.ndarray]]: Output array with dropout applied and dropout mask (None if not training).
    """
    mask = None
    if training:
        mask = xp.random.binomial(1, 1 - p, size=x.shape) / (1 - p)
        return x * mask, mask
    return x, mask


class LayerNorm:
    """Layer normalization - a crucial component for training stability.

    Applies layer normalization over the last dimension of the input. This helps
    stabilize the activations and gradients during training, leading to faster
    convergence and improved model performance.

    Attributes:
        eps (float): A small epsilon value to prevent division by zero.
        gamma (xp.ndarray): Learnable scaling parameter, initialized to ones.
        beta (xp.ndarray): Learnable shifting parameter, initialized to zeros.
        x (xp.ndarray): Stores the input for use in the backward pass.
        mean (xp.ndarray): Stores the mean of the input for the backward pass.
        var (xp.ndarray): Stores the variance of the input for the backward pass.
        x_norm (xp.ndarray): Stores the normalized input for the backward pass.
        out (xp.ndarray): Stores the output of the forward pass for the backward pass.
        d_gamma (xp.ndarray): Gradient for the gamma parameter.
        d_beta (xp.ndarray): Gradient for the beta parameter.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """Initializes the LayerNorm layer.

        Args:
            d_model (int): The dimension of the model (last dimension of the input).
            eps (float): A small epsilon value to prevent division by zero. Defaults to 1e-5.
        """
        self.eps: float = eps
        self.gamma: xp.ndarray = xp.ones(d_model)
        self.beta: xp.ndarray = xp.zeros(d_model)
        
        # Variables to store intermediate values for backward pass
        self.x: Optional[xp.ndarray] = None
        self.mean: Optional[xp.ndarray] = None
        self.var: Optional[xp.ndarray] = None
        self.x_norm: Optional[xp.ndarray] = None
        self.out: Optional[xp.ndarray] = None

        # Gradients
        self.d_gamma: Optional[xp.ndarray] = None
        self.d_beta: Optional[xp.ndarray] = None
        
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        """Applies layer normalization to the input.

        Args:
            x (xp.ndarray): The input array to be normalized.

        Returns:
            xp.ndarray: The normalized output array.
        """
        self.x = x
        self.mean = xp.mean(x, axis=-1, keepdims=True)
        self.var = xp.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / xp.sqrt(self.var + self.eps)
        self.out = self.gamma * self.x_norm + self.beta
        return self.out
    
    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        """Backward pass for layer normalization.

        Computes gradients with respect to the input `x`, and the learnable
        parameters `gamma` and `beta`.

        Args:
            d_out (xp.ndarray): The gradient of the loss with respect to the output
                                of the LayerNorm layer.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input `x`.
        """
        N = self.x.shape[-1] # Dimension of the model

        d_x_norm = d_out * self.gamma
        
        d_var = xp.sum(d_x_norm * (self.x - self.mean) * -0.5 * xp.power(self.var + self.eps, -1.5), axis=-1, keepdims=True)
        d_mean = xp.sum(d_x_norm * -1 / xp.sqrt(self.var + self.eps), axis=-1, keepdims=True) + \
                 d_var * xp.mean(-2 * (self.x - self.mean), axis=-1, keepdims=True)
        
        d_x = d_x_norm / xp.sqrt(self.var + self.eps) + \
              d_mean / N + \
              d_var * 2 * (self.x - self.mean) / N
        
        self.d_gamma = xp.sum(d_out * self.x_norm, axis=tuple(range(d_out.ndim - 1)))
        self.d_beta = xp.sum(d_out, axis=tuple(range(d_out.ndim - 1)))
        
        return d_x
    
    def __call__(self, x: xp.ndarray) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(x)

    def get_cpu_weights(self) -> Dict[str, xp.ndarray]:
        """Returns layer weights converted to CPU for checkpoint saving."""
        return {
            'gamma': to_cpu(self.gamma),
            'beta': to_cpu(self.beta)
        }

class EncoderLayer:
    """Single Encoder Layer in the Transformer.

    Consists of a MultiHeadAttention layer followed by a FeedForward layer,
    with LayerNormalization and residual connections around each sub-layer.

    Attributes:
        mha (MultiHeadAttention): Multi-head attention sub-layer.
        ff (FeedForward): Feed-forward sub-layer.
        norm1 (LayerNorm): Layer normalization for the attention sub-layer.
        norm2 (LayerNorm): Layer normalization for the feed-forward sub-layer.
        dropout (float): Dropout rate.
        x (Optional[xp.ndarray]): Stores input for backward pass.
        attn_output (Optional[xp.ndarray]): Stores attention output for backward pass.
        ff_output (Optional[xp.ndarray]): Stores feed-forward output for backward pass.
    """

    def __init__(self, config: TransformerConfig):
        """Initializes an EncoderLayer.

        Args:
            config (TransformerConfig): Configuration object.
        """
        self.mha: MultiHeadAttention = MultiHeadAttention(config)
        self.ff: FeedForward = FeedForward(config)
        self.norm1: LayerNorm = LayerNorm(config.d_model)
        self.norm2: LayerNorm = LayerNorm(config.d_model)
        self.dropout: float = config.dropout

        # For backward pass
        self.x: Optional[xp.ndarray] = None
        self.attn_output: Optional[xp.ndarray] = None
        self.ff_output: Optional[xp.ndarray] = None
        self.masked_attn_dropout_mask: Optional[xp.ndarray] = None
        self.attn_dropout_mask: Optional[xp.ndarray] = None
        self.ff_dropout_mask: Optional[xp.ndarray] = None
        self.ff_dropout_mask: Optional[xp.ndarray] = None
        self.attn_dropout_mask: Optional[xp.ndarray] = None
        self.ff_dropout_mask: Optional[xp.ndarray] = None

    def forward(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Performs the forward pass for the EncoderLayer.

        Args:
            x (xp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).
            mask (Optional[xp.ndarray]): Attention mask. Defaults to None.

        Returns:
            xp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
        """
        self.x = x

        # Multi-head attention sub-layer
        attn_output = self.mha.forward(x, mask)
        self.attn_output = attn_output
        attn_output = x + attn_output  # Residual connection
        attn_output = self.norm1.forward(attn_output)
        attn_output, self.attn_dropout_mask = dropout(attn_output, self.dropout, training=True)

        # Feed-forward sub-layer
        ff_output = self.ff.forward(attn_output)
        self.ff_output = ff_output
        ff_output = attn_output + ff_output  # Residual connection
        ff_output = self.norm2.forward(ff_output)
        ff_output, self.ff_dropout_mask = dropout(ff_output, self.dropout, training=True)

        return ff_output

    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        """Performs the backward pass for the EncoderLayer.

        Args:
            d_out (xp.ndarray): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            xp.ndarray: Gradient of the loss with respect to the input `x`.
        """
        # Backward through second LayerNorm and residual connection
        d_out = d_out * self.ff_dropout_mask if self.ff_dropout_mask is not None else d_out
        d_ff_output = self.norm2.backward(d_out)
        d_attn_output_from_ff = d_ff_output  # Gradient from residual connection
        d_ff_input = self.ff.backward(d_ff_output)

        # Backward through first LayerNorm and residual connection
        d_attn_output = self.norm1.backward(d_attn_output_from_ff + d_ff_input) # Sum gradients from residual and FF
        d_attn_output = d_attn_output * self.attn_dropout_mask if self.attn_dropout_mask is not None else d_attn_output
        d_x_from_attn = d_attn_output  # Gradient from residual connection
        d_x_from_mha, _ = self.mha.backward(d_attn_output) # We only need d_x from mha

        d_x = d_x_from_attn + d_x_from_mha # Sum gradients from residual and MHA

        return d_x

    def __call__(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(x, mask)


class DecoderLayer:
    """Single Decoder Layer in the Transformer.

    Consists of a Masked MultiHeadAttention layer, a MultiHeadAttention layer
    (for encoder-decoder attention), and a FeedForward layer, with LayerNormalization
    and residual connections around each sub-layer.

    Attributes:
        masked_mha (MultiHeadAttention): Masked multi-head attention sub-layer.
        mha (MultiHeadAttention): Multi-head attention sub-layer (encoder-decoder attention).
        ff (FeedForward): Feed-forward sub-layer.
        norm1 (LayerNorm): Layer normalization for the masked attention sub-layer.
        norm2 (LayerNorm): Layer normalization for the encoder-decoder attention sub-layer.
        norm3 (LayerNorm): Layer normalization for the feed-forward sub-layer.
        dropout (float): Dropout rate.
        x (Optional[xp.ndarray]): Stores input for backward pass.
        enc_output (Optional[xp.ndarray]): Stores encoder output for backward pass.
        masked_attn_output (Optional[xp.ndarray]): Stores masked attention output for backward pass.
        attn_output (Optional[xp.ndarray]): Stores attention output for backward pass.
        ff_output (Optional[xp.ndarray]): Stores feed-forward output for backward pass.
    """

    def __init__(self, config: TransformerConfig):
        """Initializes a DecoderLayer.

        Args:
            config (TransformerConfig): Configuration object.
        """
        self.masked_mha: MultiHeadAttention = MultiHeadAttention(config)
        self.mha: MultiHeadAttention = MultiHeadAttention(config)
        self.ff: FeedForward = FeedForward(config)
        self.norm1: LayerNorm = LayerNorm(config.d_model)
        self.norm2: LayerNorm = LayerNorm(config.d_model)
        self.norm3: LayerNorm = LayerNorm(config.d_model)
        self.dropout: float = config.dropout
        self.phase_safety: PhaseInterferenceSafety = PhaseInterferenceSafety(config.d_model)

        # For backward pass
        self.x: Optional[xp.ndarray] = None
        self.enc_output: Optional[xp.ndarray] = None
        self.masked_attn_output: Optional[xp.ndarray] = None
        self.attn_output: Optional[xp.ndarray] = None
        self.ff_output: Optional[xp.ndarray] = None

    def forward(self, x: xp.ndarray, enc_output: xp.ndarray, look_ahead_mask: Optional[xp.ndarray] = None, padding_mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Performs the forward pass for the DecoderLayer.

        Args:
            x (xp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).
            enc_output (xp.ndarray): Encoder output tensor with shape (batch_size, seq_len, d_model).
            look_ahead_mask (Optional[xp.ndarray]): Look-ahead mask for masked self-attention. Defaults to None.
            padding_mask (Optional[xp.ndarray]): Padding mask for encoder-decoder attention. Defaults to None.

        Returns:
            xp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
        """
        self.x = x
        self.enc_output = enc_output

        # Masked Multi-head attention sub-layer
        masked_attn_output = self.masked_mha.forward(x, look_ahead_mask)
        self.masked_attn_output = masked_attn_output
        masked_attn_output = x + masked_attn_output  # Residual connection
        masked_attn_output = self.norm1.forward(masked_attn_output)
        masked_attn_output, self.masked_attn_dropout_mask = dropout(masked_attn_output, self.dropout, training=True)

        # Multi-head attention sub-layer (Encoder-Decoder Attention)
        attn_output = self.mha.forward(masked_attn_output, padding_mask, enc_output)
        self.attn_output = attn_output
        attn_output = masked_attn_output + attn_output  # Residual connection
        attn_output = self.norm2.forward(attn_output)
        attn_output, self.attn_dropout_mask = dropout(attn_output, self.dropout, training=True)

        # Feed-forward sub-layer
        ff_output = self.ff.forward(attn_output)
        self.ff_output = ff_output
        ff_output = attn_output + ff_output  # Residual connection
        ff_output = self.norm3.forward(ff_output)
        # Apply phase interference safety
        ff_output = self.phase_safety.apply_safety(ff_output, xp.zeros_like(ff_output, dtype=xp.complex128).imag) # Placeholder for current_phase
        # Store quantum_state and current_phase for backward pass
        self.phase_safety.quantum_state = ff_output
        current_phase = xp.zeros_like(ff_output, dtype=xp.complex128).imag # Placeholder for current_phase
        self.phase_safety.current_phase = current_phase
        ff_output = self.phase_safety.apply_safety(ff_output, current_phase)
        ff_output, self.ff_dropout_mask = dropout(ff_output, self.dropout, training=True)

        return ff_output

    def backward(self, d_out: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        """Performs the backward pass for the DecoderLayer.

        Args:
            d_out (xp.ndarray): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            Tuple[xp.ndarray, xp.ndarray]: Gradients with respect to the input `x` and encoder output `enc_output`.
        """
        # Backward through third LayerNorm and residual connection
        d_out = d_out * self.ff_dropout_mask if self.ff_dropout_mask is not None else d_out
        # Backward through phase interference safety
        d_ff_output_from_phase_safety = self.phase_safety.backward(d_out)
        d_ff_output = self.norm3.backward(d_ff_output_from_phase_safety + d_out) # Sum gradients from residual and FF
        d_ff_output = self.norm3.backward(d_out)
        d_attn_output_from_ff = d_ff_output  # Gradient from residual connection
        d_ff_input = self.ff.backward(d_ff_output)

        # Backward through second LayerNorm and residual connection
        d_attn_output = self.norm2.backward(d_attn_output_from_ff + d_ff_input)  # Sum gradients from residual and FF
        d_attn_output = d_attn_output * self.attn_dropout_mask if self.attn_dropout_mask is not None else d_attn_output
        d_masked_attn_output_from_attn = d_attn_output  # Gradient from residual connection
        d_masked_attn_output_from_mha, d_enc_output_from_mha = self.mha.backward(d_attn_output)
        d_enc_output = d_enc_output_from_mha # This should always be present in decoder's second MHA

        # Backward through first LayerNorm and residual connection
        d_masked_attn_output = self.norm1.backward(d_masked_attn_output_from_attn + d_masked_attn_output_from_mha)  # Sum gradients from residual and MHA
        d_masked_attn_output = d_masked_attn_output * self.masked_attn_dropout_mask if self.masked_attn_dropout_mask is not None else d_masked_attn_output
        d_x_from_masked_attn = d_masked_attn_output  # Gradient from residual connection
        d_x_from_masked_mha, _ = self.masked_mha.backward(d_masked_attn_output) # We only need d_x from masked_mha

        d_x = d_x_from_masked_attn + d_x_from_masked_mha  # Sum gradients from residual and masked MHA

        return d_x, d_enc_output

    def __call__(self, x: xp.ndarray, enc_output: xp.ndarray, look_ahead_mask: Optional[xp.ndarray] = None, padding_mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(x, enc_output, look_ahead_mask, padding_mask)


class PhaseInterferenceSafety:
    """
    Manages phase interference for quantum-inspired components to ensure stability.
    This class will apply safety mechanisms to prevent destructive interference
    and maintain coherence in quantum-like computations.
    """
    def __init__(self, d_model: int, num_phases: int = 4):
        """
        Initializes the PhaseInterferenceSafety mechanism.

        Args:
            d_model (int): The dimension of the model.
            num_phases (int): The number of distinct phase states to manage.
        """
        self.d_model = d_model
        self.num_phases = num_phases
        # Example: Initialize a transformation matrix for phase alignment
        self.phase_alignment_matrix = to_device(xavier_init((d_model, d_model)))
        self.bias = to_device(xp.zeros((1, d_model)))

    def apply_safety(self, quantum_state: xp.ndarray, current_phase: xp.ndarray) -> xp.ndarray:
        """
        Applies safety mechanisms to the quantum state based on the current phase.

        Args:
            quantum_state (xp.ndarray): The input quantum-inspired state.
            current_phase (xp.ndarray): The current phase information.

        Returns:
            xp.ndarray: The phase-interference-safe quantum state.
        """
        # Placeholder for actual safety logic
        # This could involve phase rotation, amplitude damping, or other mechanisms
        # For now, a simple linear transformation and phase adjustment
        transformed_state = xp.matmul(quantum_state, self.phase_alignment_matrix) + self.bias
        # Simple phase adjustment (e.g., adding a phase offset based on current_phase)
        safe_state = transformed_state * xp.exp(1j * current_phase)
        return safe_state
    
    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        """
        Performs the backward pass for the PhaseInterferenceSafety layer.
        """
        # Assuming d_out is complex due to the forward pass
        # d(safe_state)/d(transformed_state) = exp(1j * current_phase)
        # d(safe_state)/d(current_phase) = transformed_state * 1j * exp(1j * current_phase)
    
        # Gradient with respect to transformed_state
        d_transformed_state = d_out * xp.exp(-1j * self.current_phase) # Conjugate for backward pass
    
        # Gradient with respect to bias
        self.d_bias = xp.sum(d_transformed_state.real, axis=(0, 1)) # Only real part contributes to real bias
    
        # Gradient with respect to phase_alignment_matrix
        # d(transformed_state)/d(phase_alignment_matrix) = quantum_state.T
        self.d_phase_alignment_matrix = xp.matmul(self.quantum_state.reshape(-1, self.d_model).T, d_transformed_state.real.reshape(-1, self.d_model))
        self.d_bias = xp.sum(d_transformed_state.real, axis=0, keepdims=True)
        d_quantum_state = xp.matmul(d_transformed_state.real, self.phase_alignment_matrix.T)
        return d_quantum_state
    
class MultiHeadAttention:
    """Multi-head self-attention mechanism - the heart of transformers.

    This module applies multiple attention heads in parallel, allowing the model
    to jointly attend to information from different representation subspaces
    at different positions. The outputs are concatenated and linearly transformed
    into the final expected dimension.

    Attributes:
        config (TransformerConfig): Configuration object containing model hyperparameters.
        d_k (int): Dimension of the key and query vectors for each head.
        n_heads (int): Number of attention heads.
        W_q (xp.ndarray): Weight matrix for queries.
        W_k (xp.ndarray): Weight matrix for keys.
        W_v (xp.ndarray): Weight matrix for values.
        W_o (xp.ndarray): Weight matrix for the output linear transformation.
        Q, K, V (Optional[xp.ndarray]): Stores query, key, and value matrices for backward pass.
        mask (Optional[xp.ndarray]): Stores attention mask for backward pass.
        scores (Optional[xp.ndarray]): Stores attention scores for backward pass.
        attention_weights (Optional[xp.ndarray]): Stores attention weights (softmax output) for backward pass.
        attention_output (Optional[xp.ndarray]): Stores concatenated attention output for backward pass.
        output (Optional[xp.ndarray]): Stores final output of the forward pass for backward pass.
        d_W_q, d_W_k, d_W_v, d_W_o (Optional[xp.ndarray]): Gradients for weight matrices.
    """
    
    def __init__(self, config: TransformerConfig):
        """Initializes the MultiHeadAttention layer.

        Args:
            config (TransformerConfig): Configuration object.
        """
        self.config: TransformerConfig = config
        self.d_k: int = config.d_model // config.n_heads
        self.n_heads: int = config.n_heads
        
        # Initialize weight matrices
        self.W_q: xp.ndarray = xavier_init((config.d_model, config.d_model))
        self.W_k: xp.ndarray = xavier_init((config.d_model, config.d_model))
        self.W_v: xp.ndarray = xavier_init((config.d_model, config.d_model))
        self.W_o: xp.ndarray = xavier_init((config.d_model, config.d_model))

    def scaled_dot_product_attention(self, Q: xp.ndarray, K: xp.ndarray, V: xp.ndarray,
                                   mask: Optional[xp.ndarray] = None) -> Tuple[xp.ndarray, xp.ndarray]:
        """Computes scaled dot-product attention.

        Args:
            Q (xp.ndarray): Query matrix.
            K (xp.ndarray): Key matrix.
            V (xp.ndarray): Value matrix.
            mask (Optional[xp.ndarray]): Attention mask to hide future tokens or padding.

        Returns:
            Tuple[xp.ndarray, xp.ndarray]: A tuple containing the attention output and attention weights.
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        d_k = Q.shape[-1]
        scores = xp.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
        self.scores = scores

        # Apply causal mask if provided (for decoder self-attention)
        if mask is not None:
            scores = xp.where(mask, scores, -1e9)

        # Apply softmax to get attention weights
        attention_weights = self.stable_softmax(scores)
        self.attention_weights = attention_weights

        # Apply attention to values
        output = xp.matmul(attention_weights, V)

        return output, attention_weights
    
    def stable_softmax(self, x: xp.ndarray, axis: int = -1) -> xp.ndarray:
        """Applies the stable softmax function to the input array.

        Args:
            x (xp.ndarray): Input array.
            axis (int): The axis along which the softmax is computed.

        Returns:
            xp.ndarray: Output array with stable softmax applied.
        """
        x_max = xp.max(x, axis=axis, keepdims=True)
        exp_x = xp.exp(x - x_max)
        return exp_x / xp.sum(exp_x, axis=axis, keepdims=True)

    def causal_mask(self, seq_len: int) -> xp.ndarray:
        """Generates a causal mask for self-attention.

        Args:
            seq_len (int): The length of the sequence.

        Returns:
            xp.ndarray: A lower-triangular boolean mask of shape (seq_len, seq_len).
        """
        mask = xp.tril(xp.ones((seq_len, seq_len), dtype=bool))
        return mask
    
    def forward(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None, key_value_input: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Performs the forward pass for MultiHeadAttention.

        Args:
            x (xp.ndarray): Input tensor with shape (batch_size, seq_len, d_model) for queries.
            mask (Optional[xp.ndarray]): Attention mask. Defaults to None.
            key_value_input (Optional[xp.ndarray]): Input tensor for keys and values (e.g., encoder output).
                                                 If None, x is used for keys and values (self-attention).

        Returns:
            xp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, d_model = x.shape

        # Determine source for keys and values
        kv_input = x if key_value_input is None else key_value_input
        batch_size_kv, seq_len_kv, d_model_kv = kv_input.shape

        self.x_input = x
        self.kv_input = kv_input

        # Linear transformations for Q, K, V
        Q = xp.matmul(x, self.W_q).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = xp.matmul(kv_input, self.W_k).reshape(batch_size_kv, seq_len_kv, self.n_heads, self.d_k)
        V = xp.matmul(kv_input, self.W_v).reshape(batch_size_kv, seq_len_kv, self.n_heads, self.d_k)

        # Transpose for attention calculation: (batch_size, n_heads, seq_len, d_k)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Calculate attention scores
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply final linear layer
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        self.attention_output = scaled_attention # Store for backward pass
        output = xp.matmul(scaled_attention, self.W_o)
        self.output = output

        return output
    
    def backward(self, d_out: xp.ndarray) -> Tuple[xp.ndarray, Optional[xp.ndarray]]:
        """Backward pass for multi-head attention"""
        batch_size, seq_len, d_model = self.x_input.shape
        batch_size_kv, seq_len_kv, d_model_kv = self.kv_input.shape

        # Gradient for W_o
        self.d_W_o = xp.matmul(self.attention_output.reshape(-1, d_model).T, d_out.reshape(-1, d_model))
        d_attention_output_from_linear = xp.matmul(d_out.reshape(-1, d_model), self.W_o.T).reshape(batch_size, seq_len, d_model)

        # Reshape d_attention_output_from_linear to (batch_size, n_heads, seq_len, d_k)
        d_scaled_attention_for_sdpa = d_attention_output_from_linear.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Gradients for scaled_dot_product_attention
        d_attention_weights = xp.matmul(d_scaled_attention_for_sdpa, self.V.transpose(0, 1, 3, 2))
        d_V = xp.matmul(self.attention_weights.transpose(0, 1, 3, 2), d_scaled_attention_for_sdpa)

        d_scores = d_attention_weights

        # Softmax gradient
        d_softmax = self.attention_weights * (d_scores - xp.sum(d_scores * self.attention_weights, axis=-1, keepdims=True))

        d_Q = xp.matmul(d_softmax, self.K)
        d_K = xp.matmul(self.Q.transpose(0, 1, 3, 2), d_softmax)
        d_K = d_K.transpose(0, 1, 3, 2) # Transpose back for W_k

        d_Q_input = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        d_K_input = d_K.transpose(0, 2, 1, 3).reshape(batch_size_kv, seq_len_kv, d_model_kv)
        d_V_input = d_V.transpose(0, 2, 1, 3).reshape(batch_size_kv, seq_len_kv, d_model_kv)

        self.d_W_q = xp.matmul(self.x_input.transpose(1, 0, 2).reshape(-1, d_model).T, d_Q_input.reshape(-1, d_model))
        self.d_W_k = xp.matmul(self.kv_input.transpose(1, 0, 2).reshape(-1, d_model_kv).T, d_K_input.reshape(-1, d_model_kv))
        self.d_W_v = xp.matmul(self.kv_input.transpose(1, 0, 2).reshape(-1, d_model_kv).T, d_V_input.reshape(-1, d_model_kv))

        d_x = xp.matmul(d_Q_input, self.W_q.T)
        d_kv_input = xp.matmul(d_K_input, self.W_k.T) + xp.matmul(d_V_input, self.W_v.T)

        if self.kv_input is self.x_input:
            return d_x + d_kv_input, None
        else:
            return d_x, d_kv_input
    
    def __call__(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(x, mask)

class FeedForward:
    """Position-wise feed-forward network.

    This module applies two linear transformations with a ReLU activation in between.
    It operates independently on each position.

    Attributes:
        W1 (xp.ndarray): Weight matrix for the first linear transformation.
        b1 (xp.ndarray): Bias vector for the first linear transformation.
        W2 (xp.ndarray): Weight matrix for the second linear transformation.
        b2 (xp.ndarray): Bias vector for the second linear transformation.
        x (Optional[xp.ndarray]): Stores the input for the backward pass.
        hidden_pre_relu (Optional[xp.ndarray]): Stores the output of the first linear transformation before ReLU.
        hidden (Optional[xp.ndarray]): Stores the output of the ReLU activation.
        output (Optional[xp.ndarray]): Stores the final output of the forward pass.
        d_W1, d_b1, d_W2, d_b2 (Optional[xp.ndarray]): Gradients for weight matrices and bias vectors.
    """
    
    def __init__(self, config: TransformerConfig):
        """Initializes the FeedForward layer.

        Args:
            config (TransformerConfig): Configuration object.
        """
        self.W1: xp.ndarray = xavier_init((config.d_model, config.d_ff))
        self.b1: xp.ndarray = xp.zeros(config.d_ff)
        self.W2: xp.ndarray = xavier_init((config.d_ff, config.d_model))
        self.b2: xp.ndarray = xp.zeros(config.d_model)

        # Variables to store intermediate values for backward pass
        self.x: Optional[xp.ndarray] = None
        self.hidden_pre_relu: Optional[xp.ndarray] = None
        self.hidden: Optional[xp.ndarray] = None
        self.output: Optional[xp.ndarray] = None

        # Gradients
        self.d_W1: Optional[xp.ndarray] = None
        self.d_b1: Optional[xp.ndarray] = None
        self.d_W2: Optional[xp.ndarray] = None
        self.d_b2: Optional[xp.ndarray] = None
        
    def relu(self, x: xp.ndarray) -> xp.ndarray:
        """Applies the ReLU activation function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: Output array with ReLU applied.
        """
        return xp.maximum(0, x)
    
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        """Performs the forward pass for the FeedForward network.

        Args:
            x (xp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).

        Returns:
            xp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
        """
        self.x = x
        # First linear transformation + ReLU
        self.hidden_pre_relu = xp.matmul(x, self.W1) + self.b1
        self.hidden = self.relu(self.hidden_pre_relu)
        
        # Second linear transformation
        self.output = xp.matmul(self.hidden, self.W2) + self.b2
        
        return self.output
    
    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        """Performs the backward pass for the FeedForward network.

        Args:
            d_out (xp.ndarray): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            xp.ndarray: Gradient of the loss with respect to the input `x`.
        """
        # Gradient for W2 and b2
        self.d_W2 = xp.matmul(self.hidden.reshape(-1, self.hidden.shape[-1]).T, d_out.reshape(-1, d_out.shape[-1]))
        self.d_b2 = xp.sum(d_out, axis=(0, 1))
        d_hidden = xp.matmul(d_out, self.W2.T)
        
        # Gradient for ReLU
        d_hidden_pre_relu = d_hidden * (self.hidden_pre_relu > 0)
        
        # Gradient for W1 and b1
        self.d_W1 = xp.matmul(self.x.reshape(-1, self.x.shape[-1]).T, d_hidden_pre_relu.reshape(-1, d_hidden_pre_relu.shape[-1]))
        self.d_b1 = xp.sum(d_hidden_pre_relu, axis=(0, 1))
        d_x = xp.matmul(d_hidden_pre_relu, self.W1.T)
        
        return d_x
    
    def __call__(self, x: xp.ndarray) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(x)

class PositionalEncoding:
    """Sinusoidal positional encoding - gives the model information about position.

    This layer generates a matrix of sinusoidal functions of different frequencies
    to inject information about the relative or absolute position of the tokens
    in the sequence. This is crucial for transformer models as they do not inherently
    process sequence order.

    Attributes:
        d_model (int): The dimension of the model (embedding size).
        max_seq_len (int): The maximum sequence length the model can handle.
        pos_encoding (xp.ndarray): The pre-computed positional encoding matrix.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 512):
        """Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimension of the model (embedding size).
            max_seq_len (int): The maximum sequence length. Defaults to 512.
        """
        self.d_model: int = d_model
        self.max_seq_len: int = max_seq_len
        self.pos_encoding: xp.ndarray = self.create_positional_encoding()
        
    def create_positional_encoding(self ) -> xp.ndarray:
        """Creates the sinusoidal positional encoding matrix.

        The positional encoding uses sine and cosine functions of different
        frequencies. Each dimension of the positional encoding corresponds to
        a sinusoid.

        Returns:
            xp.ndarray: A matrix of shape (max_seq_len, d_model) containing
                        the positional encodings.
        """
        pos_encoding = xp.zeros((self.max_seq_len, self.d_model))
        for pos in range (self.max_seq_len):
            for i in range(0, self.d_model, 2 ):
                pos_encoding[pos, i] = xp.sin(pos / (10000 ** ((2  * i) / self.d_model)))
                if i + 1  < self.d_model:
                    pos_encoding[pos, i + 1] = xp.cos(pos / (10000 ** ((2 * (i + 1 )) / self.d_model)))
        return  pos_encoding
    
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        """Adds positional encoding to the input embeddings.

        Args:
            x (xp.ndarray): Input tensor (embeddings) with shape (batch_size, seq_len, d_model).

        Returns:
            xp.ndarray: Input tensor with positional encodings added.
        """
        seq_len: int = x.shape[1]
        return x + self.pos_encoding[:seq_len, :]
    
    def __call__(self, x: xp.ndarray) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(x)
    
class TransformerBlock:
    """A single transformer block with attention and feed-forward.

    This block encapsulates the core components of a transformer layer:
    Multi-Head Attention, followed by a Feed-Forward Network, with residual
    connections and layer normalization applied after each sub-layer.

    Attributes:
        attention (MultiHeadAttention): The multi-head attention sub-layer.
        feed_forward (FeedForward): The position-wise feed-forward sub-layer.
        norm1 (LayerNorm): Layer normalization applied after attention.
        norm2 (LayerNorm): Layer normalization applied after feed-forward.
        x_input (Optional[xp.ndarray]): Stores the input to the block for backward pass.
        attn_output (Optional[xp.ndarray]): Stores the output of the attention sub-layer for backward pass.
        x_norm1_out (Optional[xp.ndarray]): Stores the output after the first layer norm for backward pass.
        ff_output (Optional[xp.ndarray]): Stores the output of the feed-forward sub-layer for backward pass.
        x_norm2_out (Optional[xp.ndarray]): Stores the final output of the block for backward pass.
    """
    
    def __init__(self, config: TransformerConfig):
        """Initializes a TransformerBlock.

        Args:
            config (TransformerConfig): Configuration object.
        """
        self.attention: MultiHeadAttention = MultiHeadAttention(config)
        self.feed_forward: FeedForward = FeedForward(config)
        self.norm1: LayerNorm = LayerNorm(config.d_model, config.layer_norm_eps)
        self.norm2: LayerNorm = LayerNorm(config.d_model, config.layer_norm_eps)
        
        # Variables to store intermediate values for backward pass
        self.x_input: Optional[xp.ndarray] = None
        self.attn_output: Optional[xp.ndarray] = None
        self.x_norm1_out: Optional[xp.ndarray] = None
        self.ff_output: Optional[xp.ndarray] = None
        self.x_norm2_out: Optional[xp.ndarray] = None
        
    def forward(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Performs the forward pass for the TransformerBlock.

        Args:
            x (xp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).
            mask (Optional[xp.ndarray]): Attention mask.

        Returns:
            xp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
        """
        self.x_input = x
        
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        self.attn_output = attn_output
        x_after_res1 = x + attn_output
        x_norm1_out = self.norm1(x_after_res1)
        self.x_norm1_out = x_norm1_out
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x_norm1_out)
        self.ff_output = ff_output
        x_after_res2 = x_norm1_out + ff_output
        x_norm2_out = self.norm2(x_after_res2)
        self.x_norm2_out = x_norm2_out
        
        return x_norm2_out
    
    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        """Performs the backward pass for the TransformerBlock.

        Args:
            d_out (xp.ndarray): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            xp.ndarray: Gradient of the loss with respect to the input `x`.
        """
        # Gradients for the second LayerNorm and residual connection
        d_x_after_res2 = self.norm2.backward(d_out)
        d_ff_output = d_x_after_res2 # Gradient flows directly to ff_output
        d_x_norm1_out_from_res2 = d_x_after_res2 # Gradient flows directly to x_norm1_out

        # Gradients for FeedForward
        d_x_norm1_out_from_ff = self.feed_forward.backward(d_ff_output)
        
        # Combine gradients for x_norm1_out
        d_x_norm1_out = d_x_norm1_out_from_res2 + d_x_norm1_out_from_ff

        # Gradients for the first LayerNorm and residual connection
        d_x_after_res1 = self.norm1.backward(d_x_norm1_out)
        d_attn_output = d_x_after_res1 # Gradient flows directly to attn_output
        d_x_input_from_res1 = d_x_after_res1 # Gradient flows directly to x_input

        # Gradients for MultiHeadAttention
        d_x_input_from_attn, _ = self.attention.backward(d_attn_output)

        # Combine gradients for x_input
        d_x_input = d_x_input_from_res1 + d_x_input_from_attn

        return d_x_input
    
    def __call__(self, x: xp.ndarray, mask: Optional[xp.ndarray] = None) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(x, mask)

class Embeddings:
    """Token and position embeddings.

    This layer combines token embeddings (mapping token IDs to dense vectors)
    with positional encodings, which provide information about the position
    of each token in the sequence.

    Attributes:
        token_embedding (xp.ndarray): Learnable matrix for token embeddings.
        position_encoding (PositionalEncoding): Positional encoding layer.
        input_ids (Optional[xp.ndarray]): Stores input token IDs for backward pass.
        token_embeds (Optional[xp.ndarray]): Stores token embeddings for backward pass.
        d_token_embedding (Optional[xp.ndarray]): Gradients for token embeddings.
    """
    
    def __init__(self, config: TransformerConfig):
        """Initializes the Embeddings layer.

        Args:
            config (TransformerConfig): Configuration object.
        """
        self.token_embedding: xp.ndarray = xavier_init((config.vocab_size, config.d_model))
        self.position_encoding: PositionalEncoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Variables to store intermediate values for backward pass
        self.input_ids: Optional[xp.ndarray] = None
        self.token_embeds: Optional[xp.ndarray] = None

        # Gradients
        self.d_token_embedding: Optional[xp.ndarray] = None
        self.d_position_encoding: Optional[xp.ndarray] = None

    def backward(self, d_out: xp.ndarray) -> xp.ndarray:
        """Performs the backward pass for the Embeddings layer.

        Args:
            d_out (xp.ndarray): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            xp.ndarray: Gradient of the loss with respect to the input `input_ids`.
        """
        d_token_embedding = xp.zeros_like(self.token_embedding)
        # Gradients for token embeddings (this is a sparse update)
        # We need to sum gradients for duplicate indices
        xp.add.at(d_token_embedding, self.input_ids, d_out)
        self.d_token_embedding = d_token_embedding
        return d_out # Positional encoding has no trainable parameters, so just pass gradient through
    
    def __call__(self, input_ids: xp.ndarray) -> xp.ndarray:
        """Alias for the forward method, allowing the object to be called like a function."""
        return self.forward(input_ids)
    def forward(self, input_ids: xp.ndarray) -> xp.ndarray:
        """Performs the forward pass for the Embeddings layer.

        Args:
            input_ids (xp.ndarray): Input tensor of shape (batch_size, seq_len).

        Returns:
            xp.ndarray: Output tensor of shape (batch_size, seq_len, d_model).
        """
        self.input_ids = input_ids
        self.token_embeds = self.token_embedding[input_ids]
        output = self.token_embeds + self.position_encoding.forward(self.token_embeds)
        return output
print(f"Using {DEVICE.upper()} for transformer computation")