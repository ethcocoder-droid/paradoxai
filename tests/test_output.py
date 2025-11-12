import numpy as np
from modules.output.probabilistic_collapse import ProbabilisticCollapse, CollapseConfig


def test_collapse_basic():
	pc = ProbabilisticCollapse(config=CollapseConfig(temperature=0.0, seed=0))
	branches = [np.ones(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]
	probs = [0.7, 0.3]
	emotions = {"Inceptio": 0.6, "Equilibria": 0.5, "Reflexion": 0.5, "Fluxion": 0.4}
	res = pc.collapse({"branches": branches, "probabilities": probs}, emotions=emotions, user_profile={"tone": "neutral"})
	assert "response" in res and isinstance(res["response"], str)
	assert len(res["probabilities"]) == 2


def test_collapse_missing_inputs():
	pc = ProbabilisticCollapse()
	res = pc.collapse({"branches": [], "probabilities": []}, emotions={}, user_profile=None)
	assert "response" in res and "No content" in res["response"]


