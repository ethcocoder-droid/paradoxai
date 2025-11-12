import numpy as np
from typing import Optional, List, Tuple
import re
import json
from .transformer_core import TransformerConfig, TransformerBlock, Embeddings, LayerNorm
from modules.utils.matrix_utils import xavier_init
from modules.utils.device_manager import to_device, xp

class TransformerModel:
    """Complete transformer model for language understanding and generation"""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.embeddings = Embeddings(config)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = LayerNorm(config.d_model, config.layer_norm_eps)
        
        # Output projection to vocabulary
        self.output_projection = to_device(xavier_init((config.d_model, config.vocab_size)))
        
    def create_causal_mask(self, seq_len: int) -> xp.ndarray:
        """Create causal mask for decoder self-attention"""
        mask = to_device(xp.triu(xp.ones((seq_len, seq_len)), k=1))
        return mask
    
    def forward(self, input_ids: xp.ndarray, use_causal_mask: bool = False) -> xp.ndarray:
        """Forward pass through the transformer"""
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            # For now, we'll truncate. In a real application, you might want to log a warning
            # or raise an error, or implement a more sophisticated handling strategy.
            input_ids = input_ids[:, :self.config.max_seq_len]
            seq_len = self.config.max_seq_len

        # Create embeddings
        x = self.embeddings(to_device(input_ids))
        
        # Create causal mask if needed
        mask = None
        if use_causal_mask:
            mask = self.create_causal_mask(seq_len)
        
        # Pass through transformer layers
        intermediate_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            intermediate_outputs.append(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Output projection to vocabulary
        output = xp.matmul(x, self.output_projection)
        
        return output, intermediate_outputs

    def sample_next_token(self, logits: xp.ndarray, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> int:
        """Samples the next token from the logits using temperature, top-k, and top-p sampling."""
        # Apply temperature
        logits = logits / temperature

        # Apply softmax to get probabilities
        probs = xp.exp(logits - xp.max(logits))
        probs = probs / xp.sum(probs)

        # Apply top-k filtering
        if top_k > 0:
            # Get the top_k largest probabilities and their indices
            indices = xp.argsort(probs)[-top_k:]
            top_k_probs = probs[indices]
            # Set other probabilities to zero
            probs = xp.zeros_like(probs)
            probs[indices] = top_k_probs
            # Re-normalize
            probs = probs / xp.sum(probs)

        # Apply top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_probs = xp.sort(probs)[::-1]  # Sort in descending order
            sorted_indices = xp.argsort(probs)[::-1]
            cumulative_probs = xp.cumsum(sorted_probs)

            # Find the smallest set of tokens whose cumulative probability exceeds top_p
            cutoff_index = xp.where(cumulative_probs > top_p)[0][0]
            # If the cutoff index is not 0, include the token at the cutoff index
            # to ensure the cumulative probability is at least top_p
            if cutoff_index > 0:
                cutoff_index += 1

            # Create a mask for tokens to keep
            mask = xp.zeros_like(probs, dtype=bool)
            mask[sorted_indices[:cutoff_index]] = True

            # Set probabilities of tokens outside the top-p nucleus to zero
            probs = probs * mask
            # Re-normalize
            probs = probs / xp.sum(probs)

        # Sample from the distribution
        next_token_id = xp.random.choice(len(probs), p=probs)
        return next_token_id

    def save_checkpoint(self, filepath: str):
        """Saves the model's weights to a file."""
        model_state = {
            "embeddings_token_embedding": self.embeddings.token_embedding,
            "embeddings_position_embedding": self.embeddings.position_embedding,
            "norm_gamma": self.norm.gamma,
            "norm_beta": self.norm.beta,
            "output_projection": self.output_projection,
        }

        for i, layer in enumerate(self.layers):
            # EncoderLayer components
            model_state[f"layer_{i}_mha_Wq"] = layer.mha.W_q
            model_state[f"layer_{i}_mha_Wk"] = layer.mha.W_k
            model_state[f"layer_{i}_mha_Wv"] = layer.mha.W_v
            model_state[f"layer_{i}_mha_Wo"] = layer.mha.W_o
            model_state[f"layer_{i}_ff_W1"] = layer.ff.W1
            model_state[f"layer_{i}_ff_b1"] = layer.ff.b1
            model_state[f"layer_{i}_ff_W2"] = layer.ff.W2
            model_state[f"layer_{i}_ff_b2"] = layer.ff.b2
            model_state[f"layer_{i}_norm1_gamma"] = layer.norm1.gamma
            model_state[f"layer_{i}_norm1_beta"] = layer.norm1.beta
            model_state[f"layer_{i}_norm2_gamma"] = layer.norm2.gamma
            model_state[f"layer_{i}_norm2_beta"] = layer.norm2.beta

        # Convert all arrays in model_state to CPU before saving
        cpu_model_state = {k: to_cpu(v) for k, v in model_state.items()}
        xp.savez(filepath, **cpu_model_state)

    def load_checkpoint(self, filepath: str):
        """Loads the model's weights from a file."""
        model_state = xp.load(filepath)

        self.embeddings.token_embedding = model_state["embeddings_token_embedding"]
        self.embeddings.position_embedding = model_state["embeddings_position_embedding"]
        self.norm.gamma = model_state["norm_gamma"]
        self.norm.beta = model_state["norm_beta"]
        self.output_projection = model_state["output_projection"]

        for i, layer in enumerate(self.layers):
            layer.mha.W_q = model_state[f"layer_{i}_mha_Wq"]
            layer.mha.W_k = model_state[f"layer_{i}_mha_Wk"]
            layer.mha.W_v = model_state[f"layer_{i}_mha_Wv"]
            layer.mha.W_o = model_state[f"layer_{i}_mha_Wo"]
            layer.ff.W1 = model_state[f"layer_{i}_ff_W1"]
            layer.ff.b1 = model_state[f"layer_{i}_ff_b1"]
            layer.ff.W2 = model_state[f"layer_{i}_ff_W2"]
            layer.ff.b2 = model_state[f"layer_{i}_ff_b2"]
            layer.norm1.gamma = model_state[f"layer_{i}_norm1_gamma"]
            layer.norm1.beta = model_state[f"layer_{i}_norm1_beta"]
            layer.norm2.gamma = model_state[f"layer_{i}_norm2_gamma"]
            layer.norm2.beta = model_state[f"layer_{i}_norm2_beta"]

    def decode_tokens(self, token_ids: xp.ndarray, tokenizer: 'SimpleTokenizer') -> list[str]:
        """Decodes a batch of token IDs into a list of human-readable strings."""
        decoded_texts = []
        for ids in token_ids:
            # Filter out padding tokens if any, and then decode
            filtered_ids = [id for id in ids if id != tokenizer.special_tokens['<pad>']]
            decoded_texts.append(tokenizer.decode(filtered_ids))
        return decoded_texts

    def generate_embeddings(self, input_ids: xp.ndarray) -> xp.ndarray:
        """Generate contextual embeddings for input tokens"""
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings
        x = self.embeddings(input_ids)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.norm(x)
        
        return x
    
    def get_attention_weights(self, input_ids: xp.ndarray) -> list:
        """Extract attention weights from all layers for analysis"""
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings
        x = self.embeddings(input_ids)
        
        attention_weights = []
        
        # Pass through transformer layers and collect attention
        for layer in self.layers:
            # Get attention output and weights
            attn_output = layer.attention(x)
            x = layer.norm1(x + attn_output)
            
            # Feed-forward
            ff_output = layer.feed_forward(x)
            x = layer.norm2(x + ff_output)
        
        return attention_weights
    
    def __call__(self, input_ids: xp.ndarray, use_causal_mask: bool = False) -> xp.ndarray:
        logits, _ = self.forward(input_ids, use_causal_mask)
        return logits

class QuantumInspiredTransformer(TransformerModel):
    """Transformer enhanced with quantum-inspired features"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.quantum_superposition = {}
        self.entanglement_strengths = {}
        
    def apply_quantum_interference(self, embeddings: xp.ndarray, query_context: str = "") -> xp.ndarray:
        """Apply quantum-inspired interference to embeddings"""
        # This integrates with your existing quantum interference logic
        if query_context and query_context in self.quantum_superposition:
            interference_pattern = self.quantum_superposition[query_context]
            # Apply phase-based interference
            embeddings = embeddings * (1 + 0.1 * interference_pattern)
        
        return embeddings
    
    def create_entanglement_matrix(self, concepts: list) -> xp.ndarray:
        """Create entanglement matrix for concept relationships"""
        n_concepts = len(concepts)
        entanglement_matrix = xp.eye(n_concepts)
        
        for i, concept_i in enumerate(concepts):
            for j, concept_j in enumerate(concepts):
                if i != j:
                    # Calculate entanglement strength based on semantic similarity
                    # This would integrate with your existing entanglement manager
                    strength = self.calculate_semantic_similarity(concept_i, concept_j)
                    entanglement_matrix[i, j] = strength
                    entanglement_matrix[j, i] = strength
        
        return entanglement_matrix
    
    def calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity between concepts"""
        # This would integrate with your existing vector operations
        # For now, return a placeholder
        return 0.5  # Placeholder - would use your vector operations
    
    def forward_with_quantum_enhancement(self, input_ids: xp.ndarray, 
                                       query_context: str = "") -> xp.ndarray:
        """Forward pass with quantum-inspired enhancements"""
        # Standard transformer forward pass
        logits, _ = self.forward(input_ids, use_causal_mask=True)
        
        # Apply quantum interference if context provided
        if query_context:
            # Get embeddings for interference application
            embeddings = self.generate_embeddings(input_ids)
            enhanced_embeddings = self.apply_quantum_interference(embeddings, query_context)
            
            # Re-compute logits with enhanced embeddings
            enhanced_logits = xp.matmul(enhanced_embeddings, self.output_projection)
            
            # Blend original and enhanced predictions (quantum superposition)
            alpha = 0.7  # Weight for quantum enhancement
            logits = alpha * enhanced_logits + (1 - alpha) * logits
        
        return logits

class SimpleTokenizer:
    """Simple tokenizer for basic text processing"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3
        }
        self.next_id = len(self.special_tokens)
