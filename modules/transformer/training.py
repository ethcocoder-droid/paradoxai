from typing import Optional, List, Dict
from .transformer_core import TransformerConfig, TransformerBlock, Embeddings, LayerNorm
from .transformer_model import TransformerModel
from modules.utils.device_manager import to_device, xp
import numpy as np

class TransformerTrainer:
    """
    Handles the training loop for the TransformerModel.
    """
    def __init__(self, model: TransformerModel, tokenizer, learning_rate: float = 1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.optimizer_params = {} # To store optimizer related parameters like momentum, etc.

    def _compute_loss(self, predictions: xp.ndarray, targets: xp.ndarray) -> xp.ndarray:
        """Compute cross-entropy loss"""
        # Ensure targets are within valid range
        targets = to_device(targets).astype(int)
        predictions = predictions[0] # Take the logits from the tuple
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Flatten predictions and targets for loss calculation
        predictions_flat = predictions.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1) # Assuming targets are token IDs

        # Apply softmax to predictions to get probabilities
        exp_predictions = np.exp(predictions_flat - np.max(predictions_flat, axis=-1, keepdims=True))
        probabilities = exp_predictions / np.sum(exp_predictions, axis=-1, keepdims=True)

        # Select probabilities corresponding to target tokens
        # Add a small epsilon to prevent log(0)
        # Create a mask to ignore padding tokens in loss calculation
        pad_token_id = self.tokenizer.special_tokens['<pad>']
        non_pad_mask = (targets_flat != pad_token_id)

        # Select probabilities corresponding to target tokens, only for non-padding tokens
        target_probabilities = probabilities[np.arange(len(targets_flat)), targets_flat]
        
        # Apply mask and compute loss only for non-padding tokens
        masked_target_probabilities = target_probabilities[non_pad_mask]
        if len(masked_target_probabilities) == 0:
            return 0.0  # No non-padding tokens, so no loss

        loss = -np.sum(np.log(masked_target_probabilities + 1e-9)) / np.sum(non_pad_mask)
        return loss

    def prepare_training_data(self, texts: List[str], tokenizer, max_seq_length: int = 128):
        input_sequences = []
        target_sequences = []

        for text in texts:
            encoded_text = tokenizer.encode(text)
            if len(encoded_text) < 2:  # Need at least one input and one target token
                continue

            # For a given sequence [w1, w2, w3, w4]
            # Input: [SOS, w1, w2, w3]
            # Target: [w1, w2, w3, EOS]
            input_seq = [tokenizer.special_tokens['<sos>']] + encoded_text
            target_seq = encoded_text + [tokenizer.special_tokens['<eos>']]

            # Pad or truncate sequences to max_seq_length
            if len(input_seq) > max_seq_length:
                input_seq = input_seq[:max_seq_length]
                target_seq = target_seq[:max_seq_length]
            else:
                while len(input_seq) < max_seq_length:
                    input_seq.append(tokenizer.special_tokens['<pad>'])
                    target_seq.append(tokenizer.special_tokens['<pad>'])
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        return xp.array(input_sequences), xp.array(target_sequences)

    def _backward(self, predictions: xp.ndarray, targets: xp.ndarray):
        """Compute gradients and update model weights"""
        # This is a simplified backward pass for demonstration
        # In a real scenario, this would involve backpropagation through the entire transformer
        
        # Compute gradients for output projection
        targets = to_device(targets).astype(int)
        predictions = predictions[0] # Take the logits from the tuple
        batch_size, seq_len, vocab_size = predictions.shape

        # Compute gradient of loss with respect to predictions
        d_predictions = xp.copy(predictions)
        d_predictions_flat = d_predictions.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Softmax derivative part
        exp_predictions = xp.exp(d_predictions_flat - xp.max(d_predictions_flat, axis=-1, keepdims=True))
        softmax_output = exp_predictions / xp.sum(exp_predictions, axis=-1, keepdims=True)
        
        # Gradient of cross-entropy loss w.r.t. softmax output
        d_softmax = softmax_output
        d_softmax[xp.arange(len(targets_flat)), targets_flat] -= 1
        d_softmax /= batch_size # Average over batch

        # Backpropagate through the output projection
        d_logits = d_softmax.reshape(batch_size, seq_len, vocab_size)
        
        # Gradient for output_projection
        self.model.d_output_projection = xp.matmul(self.model.norm.out.transpose(1, 0, 2).reshape(-1, self.model.config.d_model).T, d_logits.reshape(-1, vocab_size))
        d_x_norm = xp.matmul(d_logits, self.model.output_projection.T)

        # Backpropagate through final layer norm
        d_x = self.model.norm.backward(d_x_norm)

        # Backpropagate through transformer layers
        for layer in reversed(self.model.layers):
            d_x = layer.backward(d_x)

        # Backpropagate through embeddings
        self.model.embeddings.backward(d_x)

    def _update_params(self) -> None:
        """
        Updates model parameters using computed gradients.
        """
        # Update output projection
        self.model.output_projection -= self.learning_rate * self.model.d_output_projection

        # Update layer norm parameters
        self.model.norm.gamma -= self.learning_rate * self.model.norm.d_gamma
        self.model.norm.beta -= self.learning_rate * self.model.norm.d_beta

        # Update transformer layer parameters
        for layer in self.model.layers:
            # Attention weights
            layer.attention.W_q -= self.learning_rate * layer.attention.d_W_q
            layer.attention.W_k -= self.learning_rate * layer.attention.d_W_k
            layer.attention.W_v -= self.learning_rate * layer.attention.d_W_v
            layer.attention.W_o -= self.learning_rate * layer.attention.d_W_o

            # Feed-forward weights and biases
            layer.feed_forward.W1 -= self.learning_rate * layer.feed_forward.d_W1
            layer.feed_forward.b1 -= self.learning_rate * layer.feed_forward.d_b1
            layer.feed_forward.W2 -= self.learning_rate * layer.feed_forward.d_W2
            layer.feed_forward.b2 -= self.learning_rate * layer.feed_forward.d_b2

            # Layer norm parameters within block
            layer.norm1.gamma -= self.learning_rate * layer.norm1.d_gamma
            layer.norm1.beta -= self.learning_rate * layer.norm1.d_beta
            layer.norm2.gamma -= self.learning_rate * layer.norm2.d_gamma
            layer.norm2.beta -= self.learning_rate * layer.norm2.d_beta

        # Update embeddings
        self.model.embeddings.token_embedding -= self.learning_rate * self.model.embeddings.d_token_embedding

    def train(self, dataset, epochs: int = 10, batch_size: int = 32):
        """
        Runs the training loop.
        """
        num_samples = len(dataset)
        if num_samples == 0:
            print("No training samples available. Skipping training.")
            return

        for epoch in range(epochs):
            total_loss = 0
            steps = 0
            batch_count = 0
            # Manual batching
            for i in range(0, num_samples, batch_size):
                batch_inputs = []
                batch_targets = []
                for j in range(i, min(i + batch_size, num_samples)):
                    input_ids, labels = dataset[j]
                    batch_inputs.append(input_ids)
                    batch_targets.append(labels)
                
                batch_inputs = xp.array(batch_inputs)
                batch_targets = xp.array(batch_targets)

                # Forward pass
                predictions = self.model.forward(batch_inputs, use_causal_mask=True)
                
                # Compute loss
                loss = self._compute_loss(predictions, batch_targets)
                total_loss += loss

                # Backward pass and parameter update
                self._backward(predictions, batch_targets)
                self._update_params()
                steps += 1
                batch_count += 1
                
                if batch_count % 100 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Step {batch_count}, Loss: {loss:.4f}")

            avg_loss = total_loss / max(steps, 1)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def __call__(self, dataset, epochs: int = 10, batch_size: int = 32):
        self.train(dataset, epochs, batch_size)