import numpy as np
from modules.knowledge.hyper_matrix import HyperMatrixStore, SuperpositionUpdater
from modules.knowledge.entanglement_manager import EntanglementManager


def test_hyper_matrix_upsert_and_tensor():
	store = HyperMatrixStore()
	vec = np.ones(128, dtype=np.float32)
	rec = store.upsert_concept("A", vec, {"branches": [vec], "probabilities": [1.0]}, {"note": "test"})
	tensor = store.get_tensor()
	assert tensor.shape[0] >= 1 and tensor.shape[1] == 128
	assert "note" in rec.metadata


def test_superposition_update():
	store = HyperMatrixStore()
	vec = np.zeros(128, dtype=np.float32)
	store.upsert_concept("B", vec, {"branches": [vec], "probabilities": [1.0]})
	upd = SuperpositionUpdater(store)
	new_branches = [np.eye(1, 128, k=0, dtype=np.float32).flatten()]
	upd.apply_update("B", {"branches": new_branches, "probabilities": [1.0]}, source="test")
	tensor = store.get_tensor()
	assert tensor.shape[2] >= 1


def test_entanglement_manager_roundtrip(tmp_path):
	path = tmp_path / "ent.json"
	em = EntanglementManager(ent_path=path)
	em.set_entanglement("A", "B", strength=0.8, phase=0.3)
	em.decay_all(0.0)
	em.save()
	em2 = EntanglementManager(ent_path=path)
	assert len(em2.neighbors("A")) >= 1


