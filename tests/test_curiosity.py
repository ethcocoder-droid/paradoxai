from modules.curiosity.question_generator import QuestionGenerator, CuriosityConfig
from modules.transformer.transformer_model import TransformerModel
from modules.transformer.data_loader import SimpleTokenizer
from modules.transformer.text_generation import TextGenerator
from modules.utils.device_manager import xp


class MockTransformerConfig:
	def __init__(self):
		self.vocab_size = 1000
		self.max_seq_len = 512
		self.d_model = 512
		self.n_heads = 8
		self.n_layers = 6
		self.d_ff = 2048
		self.dropout = 0.1
		self.layer_norm_eps = 1e-5

def test_question_generator_outputs():
	# Simple probes returning minimal structures
	def kprobe(cid):
		return {"superposition": {"probabilities": [0.5, 0.5]}, "metadata": {"missing_fields": 1}}
	def rprobe(cid):
		return {"conflicts": 0.2}
	
	# Create a mock TransformerConfig
	mock_config = MockTransformerConfig()
	mock_transformer_model = TransformerModel(config=mock_config)
	mock_tokenizer = SimpleTokenizer()
	mock_tokenizer.build_vocabulary(["test tokenizer vocabulary"]) # Build a dummy vocabulary
	mock_text_generator = TextGenerator(model=mock_transformer_model, tokenizer=mock_tokenizer)

	qg = QuestionGenerator(config=CuriosityConfig(uncertainty_threshold=0.0), knowledge_probe=kprobe, reasoning_probe=rprobe, text_generator=mock_text_generator)
	results = qg.generate_questions(["X"])
	assert "internal" in results and "external" in results
	assert len(results["internal"]) > 0
	assert len(results["external"]) > 0


