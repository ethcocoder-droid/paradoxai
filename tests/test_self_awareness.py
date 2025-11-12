from modules.self_awareness.ai_emotions import AIEmotions
from modules.self_awareness.attention_manager import AttentionManager


def test_emotions_update_and_bounds():
	e = AIEmotions()
	e.update_from_curiosity(1.2)  # clipped
	e.update_from_certainty(-0.5) # clipped
	e.update_from_feedback(0.8)
	state = e.get_state()
	for v in state.values():
		assert 0.0 <= v <= 1.0


def test_attention_allocation_sum_to_one():
	am = AttentionManager()
	state = {"Inceptio": 0.7, "Equilibria": 0.4}
	alloc = am.compute_allocation(curiosity_signal=0.8, correctness_signal=0.5, emotions=state)
	assert abs(alloc["curiosity"] + alloc["correctness"] - 1.0) < 1e-6
	assert 0.0 <= alloc["curiosity"] <= 1.0
	assert 0.0 <= alloc["correctness"] <= 1.0


