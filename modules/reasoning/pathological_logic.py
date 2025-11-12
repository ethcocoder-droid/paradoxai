"""
Pathological Logic Engine
-------------------------
Generates multiple "pathological" perspectives on an input representation
and updates its superposition accordingly. This simulates reasoning that
explores contradictory, counterfactual, and edge-case viewpoints.

The engine is modular and testable:
- Accepts a base vector and current superposition
- Produces refined branches and probability distribution
- Optionally consults entanglement/knowledge signals via injected callbacks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np

Vector = np.ndarray

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


# Optional callback signatures to decouple from other modules
EntanglementSignalFn = Optional[Callable[[str], Dict[str, float]]]
KnowledgeLookupFn = Optional[Callable[[str], Dict[str, Any]]]


@dataclass
class ReasoningConfig:
	num_perspectives: int = 4
	contrast_scale: float = 0.15
	random_scale: float = 0.05
	max_prob_shift: float = 0.25  # cap how much probabilities can move per step
	seed: Optional[int] = 7


class PathologicalLogic:
	"""
	Multi-perspective reasoning over a superposition.
	Produces new branches via contrastive and noisy transformations, and
	rebalances probabilities with optional entanglement-informed priors.
	"""

	def __init__(
		self,
		config: ReasoningConfig = ReasoningConfig(),
		entanglement_signal: EntanglementSignalFn = None,
		knowledge_lookup: KnowledgeLookupFn = None,
	):
		self.config = config
		self.entanglement_signal = entanglement_signal
		self.knowledge_lookup = knowledge_lookup
		self.rng = np.random.default_rng(self.config.seed if self.config.seed is not None else 0)

	def step(
		self,
		concept_id: str,
		base_vector: Vector,
		superposition: Dict[str, Any],
	) -> Dict[str, Any]:
		"""
		Take one reasoning step: generate perspectives and updated probabilities.
		Returns a new superposition dict with 'branches' and 'probabilities'.
		"""
		branches: List[Vector] = [np.asarray(b, dtype=np.float32) for b in superposition.get("branches", [])]
		if not branches:
			branches = [np.asarray(base_vector, dtype=np.float32)]
		probabilities = np.array(superposition.get("probabilities", [1.0]), dtype=np.float32)
		probabilities = self._normalize_probs(probabilities)

		# 1) Generate new perspective branches
		new_branches = self._generate_perspectives(base_vector, branches)
		# 2) Merge with existing branches (keep diversity while bounded)
		merged = self._merge_branches(branches, new_branches, max_total=6)
		# 3) Update probabilities with entanglement-informed priors
		new_probs = self._rebalance_probabilities(concept_id, merged, probabilities)
		return {
			"branches": [b.astype(np.float32) for b in merged],
			"probabilities": new_probs.astype(np.float32).tolist(),
		}

	# ---------- internals ----------
	def _generate_perspectives(self, base: Vector, current: Sequence[Vector]) -> List[Vector]:
		"""
		Create contrastive and random variants simulating alternative perspectives:
		- contrast branch: move away from base along difference from current branch
		- inversion branch: invert salient dimensions
		- random exploration: gaussian noise
		"""
		persp: List[Vector] = []
		for i in range(self.config.num_perspectives):
			ref = current[i % len(current)]
			diff = ref - base
			contrast = self._normalize(ref + self.config.contrast_scale * diff)
			inversion = self._normalize(base - self.config.contrast_scale * diff)
			noise = self._normalize(base + self.rng.normal(0.0, self.config.random_scale, size=base.shape).astype(np.float32))
			persp.extend([contrast, inversion, noise])
		return persp

	def _merge_branches(self, old: Sequence[Vector], new: Sequence[Vector], max_total: int) -> List[Vector]:
		"""
		Keep up to max_total diverse branches using farthest-point sampling.
		"""
		all_branches = [self._normalize(b) for b in list(old) + list(new)]
		selected: List[Vector] = []
		if not all_branches:
			return []
		# Start with the highest-norm branch for stability
		if not all_branches:
			return []
		
		all_branches_np = np.array([self._normalize(b) for b in list(old) + list(new)], dtype=np.float32)
		if all_branches_np.size == 0:
			return []

		norms = np.linalg.norm(all_branches_np, axis=1)
		selected_indices = [int(np.argmax(norms))]
		selected: List[Vector] = [all_branches_np[selected_indices[0]]]

		while len(selected) < min(max_total, len(all_branches_np)):
			# Calculate distances from all unselected branches to the current set of selected branches
			selected_np = np.array(selected, dtype=np.float32)
			
			# Compute cosine similarity between all_branches_np and selected_np
			similarities = np.dot(all_branches_np, selected_np.T)
			norms_all = np.linalg.norm(all_branches_np, axis=1, keepdims=True)
			norms_selected = np.linalg.norm(selected_np, axis=1, keepdims=True).T
			
			denominators = np.maximum(norms_all * norms_selected, 1e-12) # Avoid division by zero
			cosine_distances = 1.0 - (similarities / denominators)
			
			# Find the minimum distance of each candidate branch to any of the selected branches
			min_distances = np.min(cosine_distances, axis=1)
			
			# Exclude already selected branches from consideration
			min_distances[selected_indices] = -1 # Set to -1 so they are not chosen
			
			next_idx = int(np.argmax(min_distances))
			
			if min_distances[next_idx] < 0: # No more suitable candidates
				break
			
			selected.append(all_branches_np[next_idx])
			selected_indices.append(next_idx)
		return selected
		return selected

	def _rebalance_probabilities(
		self,
		concept_id: str,
		branches: Sequence[Vector],
		old_probs: np.ndarray,
	) -> np.ndarray:
		"""
		Rebalance probabilities using:
		- Diversity prior: encourage spread across distinct branches
		- Entanglement prior: optional bias from external graph
		- Bounded update magnitude (max_prob_shift)
		"""
		n = len(branches)
		if n == 0:
			return old_probs
		# Diversity prior: higher average distance -> higher prior weight
		if n > 0:
			normed_branches = np.asarray([b / (np.linalg.norm(b) + 1e-12) for b in branches], dtype=np.float32)
			dist_matrix = 1.0 - np.matmul(normed_branches, normed_branches.T)
			diversity = 1e-6 + dist_matrix.mean(axis=1)
			diversity = diversity / diversity.sum()
		else:
			diversity = np.array([], dtype=np.float32)

		# Entanglement prior (if available)
		ent_prior = np.ones(n, dtype=np.float32) / float(n)
		if callable(self.entanglement_signal):
			try:
				# For now, just a normalized scalar bias applied uniformly
				scores = self.entanglement_signal(concept_id) or {}
				if scores:
					scale = float(np.clip(np.mean(list(scores.values())), -1.0, 1.0))
					ent_prior = self._normalize_probs(np.maximum(1e-6, ent_prior + 0.1 * scale))
			except Exception as exc:  # pragma: no cover
				logger.exception("entanglement_signal failed: %s", exc)

		raw = 0.6 * diversity + 0.4 * ent_prior
		new_probs = self._normalize_probs(raw)

		# Bound per-step shift
		if old_probs.shape[0] != n:
			old_probs = np.ones(n, dtype=np.float32) / float(n)
		shift = np.clip(new_probs - old_probs, -self.config.max_prob_shift, self.config.max_prob_shift)
		return self._normalize_probs(old_probs + shift)

	def _normalize(self, v: Vector, eps: float = 1e-12) -> Vector:
		v = np.asarray(v, dtype=np.float32)
		n = float(np.linalg.norm(v))
		if n < eps:
			return np.zeros_like(v, dtype=np.float32)
		return (v / n).astype(np.float32)

	def _normalize_probs(self, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
		p = np.asarray(p, dtype=np.float32)
		s = float(p.sum())
		if s < eps:
			return np.ones_like(p, dtype=np.float32) / float(len(p))
		return (p / s).astype(np.float32)

	def _cosine_distance(self, a: Vector, b: Vector, eps: float = 1e-12) -> float:
		den = float(np.linalg.norm(a) * np.linalg.norm(b))
		if den < eps:
			return 1.0
		sim = float(np.dot(a, b) / den)
		return float(1.0 - np.clip(sim, -1.0, 1.0))


