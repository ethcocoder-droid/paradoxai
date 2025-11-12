import numpy as np
from modules.perception.input_encoder import InputEncoder


def test_encode_text_basic():
	enc = InputEncoder(vector_dim=64, seed=1)
	item = enc.encode_text("hello world")
	assert item.vector.shape == (64,)
	assert np.isfinite(item.vector).all()
	assert abs(np.linalg.norm(item.vector) - 1.0) < 1e-5
	assert "branches" in item.superposition and "probabilities" in item.superposition


def test_encode_text_empty():
	enc = InputEncoder(vector_dim=32)
	item = enc.encode_text("")
	assert np.allclose(item.vector, 0.0)
	assert item.meta["ok"] is False


def test_encode_concept_and_superposition():
	enc = InputEncoder(vector_dim=16, seed=2)
	item = enc.encode_concept("ConceptX")
	assert item.vector.shape == (16,)
	sp = item.superposition
	assert len(sp["branches"]) == 3
	assert np.isclose(sum(sp["probabilities"]), 1.0, atol=1e-6)


