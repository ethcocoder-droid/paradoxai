# Transformer Module for Quantum-Like God AI
# Implements modern LLM capabilities with transformer architecture

from .transformer_core import TransformerConfig, LayerNorm, MultiHeadAttention, FeedForward, PositionalEncoding, TransformerBlock, Embeddings
from .transformer_model import TransformerModel, QuantumInspiredTransformer, SimpleTokenizer
from .text_generation import TextGenerator, ConversationalAI
from .training import TransformerTrainer

__all__ = [
    'TransformerConfig',
    'LayerNorm', 
    'MultiHeadAttention',
    'FeedForward',
    'PositionalEncoding',
    'TransformerBlock',
    'Embeddings',
    'TransformerModel',
    'QuantumInspiredTransformer',
    'SimpleTokenizer',
    'TextGenerator',
    'ConversationalAI',
    'TransformerTrainer'
]