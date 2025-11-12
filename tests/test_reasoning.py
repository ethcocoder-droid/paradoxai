import numpy as np
from modules.reasoning.pathological_logic import PathologicalLogic
from modules.reasoning.interference_engine import InterferenceEngine


def test_pathological_logic_step():
	rl = PathologicalLogic()
	base = np.ones(32, dtype=np.float32)
	superpos = {"branches": [base], "probabilities": [1.0]}
	out = rl.step("C", base, superpos)
	assert "branches" in out and "probabilities" in out
	assert len(out["branches"]) >= 1
	assert np.isclose(sum(out["probabilities"]), 1.0, atol=1e-6)


def test_interference_engine_shapes():
	engine = InterferenceEngine(seed=0)
	branches = [np.ones(16, dtype=np.float32), np.zeros(16, dtype=np.float32)]
	probs = [0.5, 0.5]
	res = engine.apply(branches, probs)
	assert len(res["probabilities"]) == 2
	assert "phases" in res


