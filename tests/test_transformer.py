import pytest
from modules.transformer.transformer_core import TransformerConfig, LayerNorm, MultiHeadAttention, FeedForward, PositionalEncoding, Embeddings, EncoderLayer, DecoderLayer
from modules.utils.device_manager import xp

def test_transformer_config():
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_len=512,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    assert config.vocab_size == 1000
    assert config.max_seq_len == 512
    assert config.d_model == 512
    assert config.n_heads == 8
    assert config.n_layers == 6
    assert config.d_ff == 2048
    assert config.dropout == 0.1

def test_layer_norm():
    d_model = 512
    layer_norm = LayerNorm(d_model)
    
    # Test forward pass
    x = xp.random.rand(1, 10, d_model)  # (batch_size, seq_len, d_model)
    out = layer_norm.forward(x)
    assert out.shape == x.shape
    assert xp.allclose(xp.mean(out, axis=-1), 0.0, atol=1e-6)
    assert xp.allclose(xp.std(out, axis=-1), 1.0, atol=1e-4)

    # Test backward pass
    grad_out = xp.random.rand(1, 10, d_model)
    grad_input = layer_norm.backward(grad_out)
    assert grad_input.shape == x.shape

def test_multi_head_attention():
    d_model = 512
    n_heads = 8
    config = TransformerConfig(vocab_size=1000, max_seq_len=512, d_model=d_model, n_heads=n_heads, n_layers=6, d_ff=2048, dropout=0.1)
    mha = MultiHeadAttention(config)

    batch_size = 1
    seq_len = 10
    x = xp.random.rand(batch_size, seq_len, d_model)
    # Create a causal mask
    causal_mask = xp.triu(xp.ones((seq_len, seq_len)), k=1) * -1e9 # Using -1e9 as ATTENTION_MASK_VALUE
    # Expand dimensions to (1, 1, seq_len, seq_len) for broadcasting
    mask = causal_mask[xp.newaxis, xp.newaxis, :, :]

    # Test forward pass
    out = mha.forward(x, mask)
    assert out.shape == (batch_size, seq_len, d_model)

    # Test backward pass
    grad_out = xp.random.rand(batch_size, seq_len, d_model)
    grad_input, _ = mha.backward(grad_out)
    assert grad_input.shape == x.shape

def test_feed_forward():
    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 10

    config = TransformerConfig(vocab_size=1000, max_seq_len=512, d_model=d_model, n_heads=8, n_layers=6, d_ff=d_ff, dropout=0.1)
    ff = FeedForward(config)

    x = xp.random.rand(batch_size, seq_len, d_model)

    # Test forward pass
    out = ff.forward(x)
    assert out.shape == (batch_size, seq_len, d_model)

    # Test backward pass
    d_out = xp.random.rand(batch_size, seq_len, d_model)
    d_x = ff.backward(d_out)
    assert d_x.shape == (batch_size, seq_len, d_model)

def test_positional_encoding():
    d_model = 512
    max_seq_len = 100
    batch_size = 2
    seq_len = 10

    pe = PositionalEncoding(d_model, max_seq_len)

    x = xp.random.rand(batch_size, seq_len, d_model)

    # Test forward pass
    out = pe.forward(x)
    assert out.shape == (batch_size, seq_len, d_model)
    # Check if positional encoding is actually added (i.e., output is different from input)
    assert not xp.allclose(out, x)

def test_embeddings():
    vocab_size = 1000
    d_model = 512
    max_seq_len = 100
    batch_size = 2
    seq_len = 10

    config = TransformerConfig(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1)
    embeddings = Embeddings(config)

    input_ids = xp.random.randint(0, vocab_size, (batch_size, seq_len))

    # Test forward pass
    out = embeddings.forward(input_ids)
    assert out.shape == (batch_size, seq_len, d_model)

    # Test backward pass
    d_out = xp.random.rand(batch_size, seq_len, d_model)
    d_input_ids = embeddings.backward(d_out)
    assert d_input_ids.shape == (batch_size, seq_len, d_model)

def test_encoder_layer():
    vocab_size = 1000
    d_model = 512
    max_seq_len = 100
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    config = TransformerConfig(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads, n_layers=6, d_ff=d_ff, dropout=dropout)
    encoder_layer = EncoderLayer(config)

    x = xp.random.rand(batch_size, seq_len, d_model)
    causal_mask = xp.triu(xp.ones((seq_len, seq_len)), k=1) * -1e9
    mask = causal_mask[xp.newaxis, xp.newaxis, :, :]

    # Test forward pass
    out = encoder_layer.forward(x, mask)
    assert out.shape == (batch_size, seq_len, d_model)

    # Test backward pass
    d_out = xp.random.rand(batch_size, seq_len, d_model)
    d_x = encoder_layer.backward(d_out)
    assert d_x.shape == (batch_size, seq_len, d_model)

def test_decoder_layer():
    vocab_size = 1000
    d_model = 512
    max_seq_len = 100
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    config = TransformerConfig(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads, n_layers=6, d_ff=d_ff, dropout=dropout)
    decoder_layer = DecoderLayer(config)

    x = xp.random.rand(batch_size, seq_len, d_model)
    enc_output = xp.random.rand(batch_size, seq_len, d_model)

    # Create a look-ahead mask
    look_ahead_mask = xp.triu(xp.ones((seq_len, seq_len)), k=1) * -1e9
    look_ahead_mask = look_ahead_mask[xp.newaxis, xp.newaxis, :, :]

    # Create a padding mask (example: no padding)
    padding_mask = xp.zeros((batch_size, 1, 1, seq_len))

    # Test forward pass
    out = decoder_layer.forward(x, enc_output, look_ahead_mask, padding_mask)
    assert out.shape == (batch_size, seq_len, d_model)

    # Test backward pass
    d_out = xp.random.rand(batch_size, seq_len, d_model)
    d_x, d_enc_output = decoder_layer.backward(d_out)
    assert d_x.shape == (batch_size, seq_len, d_model)
    assert d_enc_output.shape == (batch_size, seq_len, d_model)