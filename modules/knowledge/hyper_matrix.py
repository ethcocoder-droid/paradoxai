"""
Hyper-matrix storage for Quantum-Like God AI.

This module persists concept representations, manages superpositions,
and integrates updates from perception, reasoning, and developer input.

Data Model
----------
- Concepts stored as vectors plus metadata.
- Hyper-matrix represented as a 3D tensor: (concepts, features, branches)
  Each branch corresponds to a superposition branch coming from perception.
- Supports entanglement hooks via callbacks without circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from modules.utils.vector_operations import Vector, cosine_similarity, l2_distance, normalize, random_unit_vector, safe_add, safe_mean, softmax
import json
import logging
from pathlib import Path

import numpy as np

Vector = np.ndarray
Superposition = Dict[str, Any]

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


@dataclass
class HyperMatrixConfig:
	d_model: int = 128 # Dimensionality of concept vectors


@dataclass
class ConceptRecord:
	"""Structured storage for a concept inside the hyper-matrix."""
	concept_id: str
	vector: Vector
	superposition: Superposition
	metadata: Dict[str, Any] = field(default_factory=dict)


class HyperMatrixStore:
	"""
	Manage concept vectors and superpositions within a hyper-matrix tensor.

	Responsibilities:
	- Persist/load concepts from JSON data sources.
	- Maintain in-memory tensor representation for fast operations.
	- Apply superposition updates from reasoning/perception modules.
	- Provide hooks for developer-driven adjustments.
	"""

	def __init__(
		self,
		config: HyperMatrixConfig = HyperMatrixConfig(),
		concepts_path: Path = Path("data/concepts.json"),
	):
		self.config = config
		self.concepts_path = Path(concepts_path)
		self._records: Dict[str, ConceptRecord] = {}
		self._tensor: Optional[np.ndarray] = None  # shape: (N, D, B)
		self._branch_count: int = 0
		self._load_from_disk()

	@property
	def records(self) -> Dict[str, ConceptRecord]:
		return self._records

	def _load_from_disk(self) -> None:
		"""
		Load concept seeds from JSON.
		Each entry should provide concept_id, vector (list), metadata (optional).
		"""
		if not self.concepts_path.exists():
			logger.info("Concept file %s missing; initializing empty store", self.concepts_path)
			self._records = {}
			return

		try:
			content = self.concepts_path.read_text(encoding="utf-8")
			if not content.strip():
				logger.info("Concept file %s empty; skipping load", self.concepts_path)
				self._records = {}
				return
			data = json.loads(content)
			if not isinstance(data, list):
				raise ValueError("Concepts JSON must be a list of entries")
			for entry in data:
				concept_id = str(entry.get("concept_id") or entry.get("id") or "")
				vector_values = entry.get("vector") or []
				if not concept_id:
					logger.warning("Skipping concept entry without valid ID: %s", entry)
					continue
				vector = self._vector_from_list(vector_values)
				metadata = entry.get("metadata") or {}
				superposition = entry.get("superposition") or self._default_superposition(vector)
				self._records[concept_id] = ConceptRecord(
					concept_id=concept_id,
					vector=vector,
					superposition=superposition,
					metadata=metadata,
				)
			self._refresh_tensor()
		except Exception as exc:
			logger.exception("Failed loading concepts: %s", exc)
			self._records = {}

	def save_to_disk(self) -> None:
		"""Persist current records to JSON file."""
		try:
			self.concepts_path.parent.mkdir(parents=True, exist_ok=True)
			payload = []
			for record in self._records.values():
				payload.append(
					{
						"concept_id": record.concept_id,
						"vector": record.vector.tolist(),
						"metadata": record.metadata,
						"superposition": record.superposition,
					}
				)
			self.concepts_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
		except Exception as exc:
			logger.exception("Failed persisting concepts: %s", exc)

	def upsert_concept(
		self,
		concept_id: str,
		vector: Vector,
		superposition: Optional[Superposition] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> ConceptRecord:
		"""
		Create or update a concept record and rebuild tensor view.
		"""
		if not concept_id:
			raise ValueError("concept_id must be provided")
		vector = self._ensure_vector(vector)
		superposition = superposition or self._default_superposition(vector)
		# Coerce superposition to JSON-safe representation
		try:
			branches = superposition.get("branches", [])
			probabilities = superposition.get("probabilities", [])
			branches_safe = [self._ensure_vector(b).tolist() for b in branches]
			if isinstance(probabilities, np.ndarray):
				probabilities = probabilities.tolist()
			superposition = {"branches": branches_safe, "probabilities": list(probabilities)}
		except Exception:
			superposition = self._default_superposition(vector)
		metadata = metadata or {}
		record = ConceptRecord(concept_id, vector, superposition, metadata)
		self._records[concept_id] = record
		self._refresh_tensor()
		return record

	def remove_concept(self, concept_id: str) -> bool:
		"""Remove a concept by ID; returns True if removed."""
		if concept_id in self._records:
			del self._records[concept_id]
			self._refresh_tensor()
			return True
		return False

	def get_tensor(self) -> np.ndarray:
		"""Return tensor view (N concepts x D dims x B branches)."""
		if self._tensor is None:
			self._refresh_tensor()
		assert self._tensor is not None
		return self._tensor

	def update_superposition(
		self,
		concept_id: str,
		branches: Sequence[Vector],
		probabilities: Sequence[float],
		metadata_updates: Optional[Dict[str, Any]] = None,
	) -> None:
		"""
		Update stored superposition for a concept.
		- branches must match vector dimensionality.
		- probabilities must sum to 1 (tolerant to small numerical error).
		"""
		record = self._records.get(concept_id)
		if record is None:
			raise KeyError(f"Concept {concept_id!r} not found")
		branches = [self._ensure_vector(vec) for vec in branches]
		probabilities = np.array(probabilities, dtype=np.float32)
		if branches and len(branches) != probabilities.shape[0]:
			raise ValueError("branches and probabilities length mismatch")
		total = float(probabilities.sum())
		if total <= 0:
			raise ValueError("probabilities must sum to positive value")
		probabilities = probabilities / total
		# Store JSON-safe lists for persistence
		record.superposition["branches"] = [b.tolist() for b in branches]
		record.superposition["probabilities"] = probabilities.tolist()
		if metadata_updates:
			record.metadata.update(metadata_updates)
		self._refresh_tensor()

	def integrate_developer_feedback(
		self,
		concept_id: str,
		feedback: Dict[str, Any],
		weight: float = 0.2,
	) -> ConceptRecord:
		"""
		Apply developer feedback to concept metadata and vector bias.
		- feedback may include 'vector_adjustment' (list) and arbitrary metadata.
		- weight controls blending strength for vector adjustment.
		"""
		record = self._records.get(concept_id)
		if record is None:
			raise KeyError(f"Concept {concept_id!r} not found")
		vector_adjustment = feedback.get("vector_adjustment")
		if vector_adjustment is not None:
			adjustment_vec = self._vector_from_list(vector_adjustment)
			if adjustment_vec.shape != record.vector.shape:
				raise ValueError("vector_adjustment shape mismatch")
			record.vector = self._blend_vectors(record.vector, adjustment_vec, alpha=weight)
		meta_updates = {k: v for k, v in feedback.items() if k != "vector_adjustment"}
		record.metadata.update(meta_updates)
		self._refresh_tensor()
		return record

	def query_concepts(self, top_k: int = 5) -> List[ConceptRecord]:
		"""
		Return top_k concepts sorted by probability mass of first branch.
		Simple heuristic for demonstration/testing.
		"""
		if top_k <= 0:
			return []
		records = list(self._records.values())
		records.sort(
			key=lambda rec: float(rec.superposition.get("probabilities", [0.0])[0] if rec.superposition else 0.0),
			reverse=True,
		)
		return records[:top_k]

	# ---------- Internal helpers ----------

	def _refresh_tensor(self) -> None:
		"""Rebuild tensor representation from records."""
		if not self._records:
			self._tensor = np.zeros((0, self.config.d_model, 0), dtype=np.float32)
			self._branch_count = 0
			return
		vectors = []
		branches = []
		for record in self._records.values():
			vector = self._ensure_vector(record.vector)
			superpos = record.superposition or self._default_superposition(vector)
			branch_vectors = [self._ensure_vector(b) for b in superpos.get("branches", [])]
			if not branch_vectors:
				branch_vectors = [vector]
			vectors.append(vector)
			branches.append(np.stack(branch_vectors, axis=0))
		# Determine max branch count and initialize a padded array
		branch_count = max(branch.shape[0] for branch in branches) if branches else 0
		if branch_count == 0:
			self._tensor = np.zeros((0, self.config.d_model, 0), dtype=np.float32)
			self._branch_count = 0
			return

		padded_branches_array = np.zeros(
			(len(branches), branch_count, self.config.d_model), dtype=np.float32
		)

		for i, branch_stack in enumerate(branches):
			padded_branches_array[i, :branch_stack.shape[0], :] = branch_stack

		tensor = padded_branches_array  # (N, B, D)
		self._tensor = np.transpose(tensor, (0, 2, 1)).astype(np.float32)  # (N, D, B)
		self._branch_count = branch_count

	def _ensure_vector(self, vector: Vector) -> Vector:
		vector = np.asarray(vector, dtype=np.float32)
		if vector.ndim != 1:
			raise ValueError("Vector must be 1-dimensional")
		if vector.shape[0] != self.config.d_model:
			# Pad or trim deterministically
			vector = self._resize_vector(vector, self.config.d_model)
		return normalize(vector)

	def _vector_from_list(self, values: Sequence[float]) -> Vector:
		if not values:
			return np.zeros(self.config.d_model, dtype=np.float32)
		vector = np.array(values, dtype=np.float32)
		return self._ensure_vector(vector)

	def _default_superposition(self, vector: Vector) -> Superposition:
		return {
			"branches": [vector.tolist()],
			"probabilities": [1.0],
		}

	def _resize_vector(self, vector: Vector, target_dim: int) -> Vector:
		if vector.shape[0] == target_dim:
			return vector
		if vector.shape[0] < target_dim:
			padding = np.zeros(target_dim - vector.shape[0], dtype=np.float32)
			return np.concatenate([vector, padding], axis=0)
		return vector[:target_dim]

	def _blend_vectors(self, base: Vector, adjustment: Vector, alpha: float) -> Vector:
		alpha = float(np.clip(alpha, 0.0, 1.0))
		return ((1 - alpha) * base + alpha * adjustment).astype(np.float32)

	def save(self, file_path: Optional[Path] = None) -> None:
		"""Save the hyper-matrix state to a .npz file."""
		_file_path = file_path or self.concepts_path.with_suffix(".npz")
		_file_path.parent.mkdir(parents=True, exist_ok=True)

		records_data = {
			record.concept_id: {
				"vector": record.vector.tolist(),
				"superposition": {
					"branches": [b.tolist() for b in record.superposition["branches"]] if "branches" in record.superposition else [],
					"probabilities": record.superposition["probabilities"],
				},
				"metadata": record.metadata,
			}
			for record in self._records.values()
		}

		np.savez_compressed(
			_file_path,
			records=json.dumps(records_data),
			tensor=self._tensor,
			branch_count=self._branch_count,
		)
		logger.info("HyperMatrix saved to %s", _file_path)

	def load(self, file_path: Optional[Path] = None) -> None:
		"""Load the hyper-matrix state from a .npz file."""
		_file_path = file_path or self.concepts_path.with_suffix(".npz")
		if not _file_path.exists():
			logger.info("HyperMatrix checkpoint %s not found; initializing from concepts.json", _file_path)
			self._load_from_disk()
			return

		data = np.load(_file_path, allow_pickle=True)
		records_data = json.loads(data["records"])

		self._records = {}
		for concept_id, record_dict in records_data.items():
			vector = np.array(record_dict["vector"], dtype=np.float32)
			superposition_branches = [np.array(b, dtype=np.float32) for b in record_dict["superposition"]["branches"]]
			superposition_probabilities = record_dict["superposition"]["probabilities"]
			superposition = {
				"branches": superposition_branches,
				"probabilities": superposition_probabilities,
			}
			metadata = record_dict["metadata"]
			self._records[concept_id] = ConceptRecord(concept_id, vector, superposition, metadata)

		self._tensor = data["tensor"]
		self._branch_count = data["branch_count"].item() # .item() to get scalar from 0-d array
		logger.info("HyperMatrix loaded from %s", _file_path)

		self._refresh_tensor() # Ensure tensor consistency after loading


# Hook type for entanglement updates
EntanglementUpdateHook = Optional[Any]


class SuperpositionUpdater:
	"""
	Helper class to apply superposition updates from external modules.
	Encapsulates update strategy for easier testing/mocking.
	"""

	def __init__(self, store: HyperMatrixStore, entanglement_hook: EntanglementUpdateHook = None):
		self.store = store
		self.entanglement_hook = entanglement_hook

	def apply_update(
		self,
		concept_id: str,
		new_superposition: Dict[str, Any],
		source: str = "unknown",
	) -> ConceptRecord:
		branches = new_superposition.get("branches")
		if branches is None:
			branches = []
		probabilities = new_superposition.get("probabilities")
		# Safely coerce numpy arrays or other sequences to list
		if probabilities is None:
			probabilities = []
		else:
			try:
				import numpy as _np  # local import to avoid global dependency
				if isinstance(probabilities, _np.ndarray):
					probabilities = probabilities.tolist()
			except Exception:
				pass
		self.store.update_superposition(concept_id, branches, probabilities, metadata_updates={"last_update_source": source})

		if callable(self.entanglement_hook):
			try:
				self.entanglement_hook(concept_id, new_superposition)
			except Exception as exc:  # pragma: no cover
				logger.exception("Entanglement hook failed: %s", exc)

		return self.store.records[concept_id]


