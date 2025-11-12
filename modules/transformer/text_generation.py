import numpy as np
from typing import List, Optional, Dict
from .transformer_model import TransformerModel, SimpleTokenizer
from modules.utils.device_manager import to_device, xp

class TextGenerator:
    """Text generation using the transformer model"""
    
    def __init__(self, model: TransformerModel, tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.9
        
    def softmax(self, x: xp.ndarray, temperature: float = 1.0) -> xp.ndarray:
        """Apply temperature scaling and softmax"""
        x = x / temperature
        x_max = xp.max(x)
        x_stable = x - x_max
        exp_x = xp.exp(x_stable)
        return exp_x / xp.sum(exp_x)

    def top_k_sampling(self, logits: xp.ndarray, k: int = 50) -> int:
        """Sample from top k most likely tokens"""
        top_k_indices = xp.argsort(logits)[-k:]
        top_k_logits = logits[top_k_indices]
        top_k_probs = self.softmax(top_k_logits)
        
        return xp.random.choice(top_k_indices, p=top_k_probs)
    
    def top_p_sampling(self, logits: xp.ndarray, p: float = 0.9) -> int:
        """Sample from tokens with cumulative probability <= p"""
        sorted_indices = xp.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        sorted_probs = self.softmax(sorted_logits)
        
        cumulative_probs = xp.cumsum(sorted_probs)
        
        # Find where cumulative probability exceeds p
        cutoff_index = xp.where(cumulative_probs > p)[0]
        if len(cutoff_index) > 0:
            cutoff = cutoff_index[0] + 1
        else:
            cutoff = len(sorted_probs)
        
        # Sample from the top tokens
        top_indices = sorted_indices[:cutoff]
        top_probs = sorted_probs[:cutoff]
        top_probs = top_probs / np.sum(top_probs)  # Renormalize
        
        return np.random.choice(top_indices, p=top_probs)
    
    def generate_text(
                     self, 
                     prompt: str, 
                     max_length: int = 100,
                     temperature: float = 1.0,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None) -> str:
        """Generate text from a prompt"""
        
        # Set generation parameters
        self.temperature = temperature
        if top_k is not None:
            self.top_k = top_k
        if top_p is not None:
            self.top_p = top_p
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt)
        generated_ids = list(input_ids)
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Prepare input for model
            current_input = to_device(xp.array(generated_ids[-self.model.config.max_seq_len:]).reshape(1, -1))
            
            # Get model predictions
            logits = self.model(current_input, use_causal_mask=True)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]
            
            # Apply sampling strategy
            if top_k is not None and top_p is not None:
                # Combine top-k and top-p sampling
                next_token_id = self.combined_sampling(next_token_logits)
            elif top_k is not None:
                next_token_id = self.top_k_sampling(next_token_logits, top_k)
            elif top_p is not None:
                next_token_id = self.top_p_sampling(next_token_logits, top_p)
            else:
                # Greedy sampling
                next_token_id = xp.argmax(next_token_logits)
            
            # Add generated token to sequence
            generated_ids.append(next_token_id)
            
            # Check for end token
            if next_token_id == self.tokenizer.special_tokens['<eos>']:
                break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(xp.array(generated_ids))
        
        return generated_text
    
    def combined_sampling(self, logits: xp.ndarray) -> int:
        """Combine top-k and top-p sampling"""
        # First apply top-k filtering
        top_k_indices = xp.argsort(logits)[-self.top_k:]
        top_k_logits = logits[top_k_indices]
        
        # Then apply top-p filtering on the reduced set
        sorted_indices = xp.argsort(top_k_logits)[::-1]
        sorted_probs = self.softmax(top_k_logits[sorted_indices])
        cumulative_probs = xp.cumsum(sorted_probs)
        
        cutoff_index = xp.where(cumulative_probs > self.top_p)[0]
        if len(cutoff_index) > 0:
            cutoff = cutoff_index[0] + 1
        else:
            cutoff = len(sorted_probs)
        
        # Sample from the filtered tokens
        final_indices = top_k_indices[sorted_indices[:cutoff]]
        final_probs = sorted_probs[:cutoff]
        final_probs = final_probs / xp.sum(final_probs)
        
        return xp.random.choice(final_indices, p=final_probs)

class ConversationalAI:
    """Conversational AI that combines transformer with quantum-inspired reasoning"""
    
    def __init__(self, model: TransformerModel, tokenizer: SimpleTokenizer):
        self.generator = TextGenerator(model, tokenizer)
        self.conversation_history = []
        self.context_window = 10  # Keep last 10 exchanges
        
    def add_to_history(self, user_input: str, ai_response: str):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            'user': user_input,
            'ai': ai_response,
            'timestamp': xp.datetime64('now')
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.context_window:
            self.conversation_history = self.conversation_history[-self.context_window:]
    
    def build_context_prompt(self, current_input: str) -> str:
        """Build context-aware prompt from conversation history"""
        if not self.conversation_history:
            return f"User: {current_input}\nAI:"
        
        # Build context from recent history
        context_parts = []
        for exchange in self.conversation_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"AI: {exchange['ai']}")
        
        # Add current input
        context_parts.append(f"User: {current_input}")
        context_parts.append("AI:")
        
        return "\n".join(context_parts)
    
    def generate_response(self, user_input: str, 
                         max_length: int = 150,
                         temperature: float = 0.8,
                         top_k: int = 40,
                         top_p: float = 0.9) -> str:
        """Generate a contextual response"""
        
        # Build context-aware prompt
        context_prompt = self.build_context_prompt(user_input)
        
        # Generate response
        response = self.generator.generate_text(
            prompt=context_prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Clean up response (remove the prompt part)
        if "AI:" in response:
            response = response.split("AI:")[-1].strip()
        
        # Add to conversation history
        self.add_to_history(user_input, response)
        
        return response
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation history."
        
        summary = f"Conversation with {len(self.conversation_history)} exchanges:\n"
        for i, exchange in enumerate(self.conversation_history, 1):
            summary += f"{i}. User: {exchange['user'][:50]}...\n"
            summary += f"   AI: {exchange['ai'][:50]}...\n"
        
        return summary