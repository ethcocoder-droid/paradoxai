"""
Interference Engine
-------------------
Applies quantum-like interference across superposition branches by
assigning phases and computing constructive/destructive effects on
resultant probabilities.

Outputs an updated superposition suitable for storage in the hyper-matrix.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np

Vector = np.ndarray

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


class InterferenceEngine:
	"""
	Computes amplitude-level interference using simple complex amplitudes.
	Each branch i has amplitude a_i = r_i * exp(j * phi_i).
	Probabilities are derived from |sum_i a_i * v_i|, approximated by
	phase-weighted combination of branch probabilities.
	"""

	def __init__(self, seed: Optional[int] = 11):
		self.rng = np.random.default_rng(seed if seed is not None else 0)

	def apply(
		self,
		branches: Sequence[Vector],
		probabilities: Sequence[float],
		phases: Optional[Sequence[float]] = None,
	) -> Dict[str, Any]:
		"""
		Apply interference and return updated superposition with adjusted probabilities.
		- branches: list of state vectors
		- probabilities: prior probabilities (sum to 1)
		- phases: optional phases per branch; random if not provided
		"""
		if not branches:
			return {"branches": [], "probabilities": []}
		B = len(branches)
		p = np.array(probabilities, dtype=np.float32)
		if p.size != B:
			p = np.ones(B, dtype=np.float32) / float(B)
		else:
			p = self._normalize_probs(p)
		if phases is None:
			phi = self.rng.uniform(-np.pi, np.pi, size=B).astype(np.float32)
		else:
			phi = np.array(phases, dtype=np.float32)
			if phi.size != B:
				phi = np.resize(phi, B)

		# Compute interference matrix M_ij ~ cos(phi_i - phi_j) * similarity(v_i, v_j)
		sim = np.eye(B, dtype=np.float32)
		for i in range(B):
			for j in range(i + 1, B):
				s = 1.0 - self._cosine_distance(branches[i], branches[j])
				sim[i, j] = sim[j, i] = float(np.clip(s, -1.0, 1.0))

		phase_mat = np.zeros((B, B), dtype=np.float32)
		for i in range(B):
			for j in range(B):
				phase_mat[i, j] = np.cos(float(phi[i] - phi[j]))

		M = sim * phase_mat
		# Effective energy per branch: e_i = sum_j M_ij * p_j
		e = M @ p
		# Convert energies into multiplicative factors on p
		shift = 0.25 * (e - e.mean())  # small bounded adjustment
		new_p = self._normalize_probs(np.maximum(1e-6, p + shift))

		return {
			"branches": [np.asarray(b, dtype=np.float32) for b in branches],
			"probabilities": new_p.astype(np.float32).tolist(),
			"phases": phi.tolist(),
		}

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


