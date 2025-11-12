"""
Hyper-Matrix Updater
--------------------
Applies developer-encoded payloads to the hyper-matrix store,
supporting upserts and vector adjustments with safe blending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import logging
import numpy as np

from modules.knowledge.hyper_matrix import HyperMatrixStore, SuperpositionUpdater

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


class HyperMatrixUpdater:
	"""Updates the hyper-matrix store based on developer payloads."""

	def __init__(self, store: Optional[HyperMatrixStore] = None):
		self.store = store or HyperMatrixStore()
		self.superpos_updater = SuperpositionUpdater(self.store)

	def upsert_from_payload(self, payload: Dict[str, Any], blend_weight: float = 0.35) -> Dict[str, Any]:
		"""
		Payload shape (from DeveloperInputHandler.encode):
		{
			'concept_id': str,
			'vector': np.ndarray,
			'superposition': {...},
			'meta': {...}
		}
		"""
		concept_id = str(payload.get("concept_id") or "")
		if not concept_id:
			raise ValueError("payload missing concept_id")
		vector = np.asarray(payload.get("vector"), dtype=np.float32)
		superposition = payload.get("superposition") or {}
		meta = payload.get("meta") or {}

		# Upsert base
		existing = self.store.records.get(concept_id)
		if existing is None:
			self.store.upsert_concept(concept_id, vector, superposition, meta)
		else:
			# Blend vectors for stability
			new_vec = (1 - blend_weight) * existing.vector + blend_weight * vector
			self.store.upsert_concept(concept_id, new_vec, superposition, {**existing.metadata, **meta})

		# Update superposition using helper (marks last_update_source)
		self.superpos_updater.apply_update(concept_id, superposition, source=meta.get("source", "developer"))
		self.store.save_to_disk()
		return {
			"status": "ok",
			"concept_id": concept_id,
		}


