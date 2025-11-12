#!/usr/bin/env python3
import os
import sys

class DualStream:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

log_file_path = "full_output.log"
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
    log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = DualStream(original_stdout, log_file)
    sys.stderr = DualStream(original_stderr, log_file)

    """
Transformer-based Large Language Model Example
Demonstrates training and text generation with the custom transformer
"""
    import json
    import numpy as np
    from typing import List, Tuple

    # Add the project root to Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from modules.transformer.text_generation import TextGenerator
    from modules.transformer.transformer_core import TransformerConfig
    from modules.transformer.transformer_model import TransformerModel
    from modules.transformer.training import TransformerTrainer
    from modules.transformer.data_loader import SimpleTokenizer, TextDataset

    def demonstrate_transformer_training():
        """Demonstrate transformer training process"""
        print("Demonstrating Transformer Training")
        print("=" * 50)
        
        # Create configuration (initial, vocab_size will be updated)
        config = TransformerConfig(
            vocab_size=500, # Placeholder, will be updated
            d_model=256,      # Increased model size for better learning
            n_heads=4,
            n_layers=2,
            d_ff=1024,        # Increased feed-forward dimension
            max_seq_len=128,
            dropout=0.1
        )

        # Create tokenizer
        tokenizer = SimpleTokenizer()
        gutenberg_text_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'gutenberg_text.txt')

        # Prepare training data using TextDataset, which now builds vocabulary internally
        print("Preparing training data...")
        max_seq_len = config.max_seq_len
        dataset = TextDataset(filepath=gutenberg_text_filepath, tokenizer=tokenizer, seq_len=max_seq_len)

        # Update vocab_size in config after tokenizer has built its vocabulary from the dataset
        config.vocab_size = tokenizer.get_vocab_size()

        # Create model with the updated config
        model = TransformerModel(config)
        
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        print(f"Model parameters: {config.d_model * config.vocab_size * config.n_layers}")
        
        # Create trainer
        trainer = TransformerTrainer(model, tokenizer, learning_rate=1e-3)

        sample_count = len(dataset)
        print(f"Created {sample_count} training samples")

        print("Starting training...")
        if sample_count == 0:
            print("No samples found in training data. Check your data/sample_text.txt or TextDataset implementation.")
        else:
            # Modify training loop to iterate over data_loader
            trainer.train(dataset, epochs=1, batch_size=2)
            print("Training complete.")
        
        # Define a directory to save the trained components
        save_directory = os.path.join(os.path.dirname(__file__), '..', 'trained_model')
        os.makedirs(save_directory, exist_ok=True)

        # Save the configuration, tokenizer, and model
        config.save(os.path.join(save_directory, 'config.json'))
        tokenizer.save(os.path.join(save_directory, 'tokenizer.json'))
        model.save_checkpoint(os.path.join(save_directory, 'model.pth'))

        print(f"Trained model, tokenizer, and configuration saved to {save_directory}")

        return model, tokenizer, trainer

    def demonstrate_text_generation(model=None, tokenizer=None):
        """Demonstrate text generation capabilities"""
        print("Demonstrating Text Generation")
        print("=" * 50)

        # If model or tokenizer are not provided, load them from the saved directory
        if model is None or tokenizer is None:
            print("Loading saved model and tokenizer for text generation...")
            load_directory = os.path.join(os.path.dirname(__file__), '..', 'trained_model')
            
            # Load config
            config_path = os.path.join(load_directory, 'config.json')
            config = TransformerConfig.load(config_path)

            # Load tokenizer
            tokenizer_path = os.path.join(load_directory, 'tokenizer.json')
            tokenizer = SimpleTokenizer.load(tokenizer_path)
            
            # Load model
            model = TransformerModel(config)
            model.load_checkpoint(os.path.join(load_directory, 'model.pth'))
            print("Model and tokenizer loaded successfully.")
        
        # Create text generator
        generator = TextGenerator(model, tokenizer)
        
        # Test prompts
        test_prompts = [
            "Artificial intelligence",
            "The future of",
            "Machine learning",
            "Quantum computing"
        ]
        
        for prompt in test_prompts:
            print(f"Prompt: '{prompt}'")
            
            # Generate text with different methods
            print("Top-k sampling (k=5):")
            generated = generator.generate_text(
                prompt, 
                max_length=20, 
                temperature=0.8,
                top_k=5
            )
            print(f"Generated: '{generated}'")
            
            print("Top-p sampling (p=0.9):")
            generated = generator.generate_text(
                prompt, 
                max_length=20, 
                temperature=0.8,
                top_p=0.9
            )
            print(f"Generated: '{generated}'")

    def demonstrate_conversational_ai(model, tokenizer):
        """Demonstrate conversational capabilities"""
        print("Demonstrating Conversational AI")
        print("=" * 50)
        
        # Create conversational AI
        conversational_ai = ConversationalAI(model, tokenizer)
        
        # Test conversations
        test_inputs = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Tell me about quantum computing.",
            "What are the applications of AI?"
        ]
        
        for user_input in test_inputs:
            print(f"User: {user_input}")
            response = conversational_ai.generate_response(user_input)
            print(f"AI: {response}")
            print(f"   Context used: {len(conversational_ai.conversation_history)} messages")

    def demonstrate_quantum_enhanced_model():
        """Demonstrate quantum-inspired transformer"""
        print("Demonstrating Quantum-Inspired Transformer")
        print("=" * 50)
        
        config = TransformerConfig(
            vocab_size=500,
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=256,
            max_seq_len=16
        )
        
        quantum_model = QuantumInspiredTransformer(config)
        tokenizer = SimpleTokenizer()
        gutenberg_text_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'gutenberg_text.txt')
        
        # Build vocabulary from gutenberg_text.txt
        # For quantum demo, we'll just build vocab from the file directly
        # In a real scenario, you'd use the TextDataset to build vocab and load data
        with open(gutenberg_text_filepath, 'r', encoding='utf-8') as f:
            text_content = f.read()
        tokenizer.build_vocabulary([text_content])
        
        # Update vocab_size in config after tokenizer has built its vocabulary
        config.vocab_size = tokenizer.get_vocab_size()
        
        # Re-initialize quantum_model with updated config
        quantum_model = QuantumInspiredTransformer(config)
        
        # Test quantum-enhanced generation
        test_input = "Quantum computing"
        tokens = tokenizer.encode(test_input)
        input_ids = np.array([tokens])
        
        print(f"Input: '{test_input}'")
        
        # Regular forward pass
        regular_output = quantum_model(input_ids)
        print(f"Regular output shape: {regular_output.shape}")
        
        # Quantum-enhanced forward pass
        quantum_output = quantum_model.forward_with_quantum_enhancement(input_ids)
        print(f"Quantum-enhanced output shape: {quantum_output.shape}")
        
        # Compare outputs
        diff = np.mean(np.abs(quantum_output - regular_output))
        print(f"Mean absolute difference: {diff:.6f}")
        print("Quantum enhancement working!")

    def main():
        _run_demonstrations()
        print(f"All output redirected to full_output.log")

    def _run_demonstrations():
        print("Using CPU for transformer computation")
        print("Quantum-Like God AI - Transformer LLM Demo")
        print("=" * 60)
        print("This demo shows our custom transformer implementation")
        print("with training, text generation, and quantum-inspired features.")
        print("=" * 60)

        try:
            # 1. Demonstrate transformer training
            model, tokenizer, trainer = demonstrate_transformer_training()

            # 2. Demonstrate text generation
            demonstrate_text_generation(model, tokenizer)

            print("All demonstrations completed successfully!")

        except Exception as e:
            print(f"Error during demonstration: {e}")
            import traceback
            traceback.print_exc()

        print("\nYour transformer-based LLM is ready for:")
        print("• Training on larger datasets")
        print("• Fine-tuning for specific tasks")
        print("• Integration with the main AI system")
        print("• Scaling up model size and capabilities")

    if __name__ == "__main__":
        main()

finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr