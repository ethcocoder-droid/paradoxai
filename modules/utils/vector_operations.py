"""
Vector operations with robust error handling.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple
import numpy as np

Vector = np.ndarray


def check_finite(vec: Vector) -> bool:
	"""Return True if the vector has no NaN/Inf."""
	arr = np.asarray(vec)
	return np.all(np.isfinite(arr))


def normalize(vec: Vector, eps: float = 1e-12) -> Vector:
	"""L2-normalize; return zero vector on tiny norms to avoid NaNs."""
	v = np.asarray(vec, dtype=np.float32)
	n = float(np.linalg.norm(v))
	if n < eps:
		return np.zeros_like(v, dtype=np.float32)
	return (v / n).astype(np.float32)


def cosine_similarity(a: Vector, b: Vector, eps: float = 1e-12) -> float:
	"""Cosine similarity in [-1,1], robust to zeros."""
	a = np.asarray(a, dtype=np.float32)
	b = np.asarray(b, dtype=np.float32)
	den = float(np.linalg.norm(a) * np.linalg.norm(b))
	if den < eps:
		return 0.0
	return float(np.clip(np.dot(a, b) / den, -1.0, 1.0))


def l2_distance(a: Vector, b: Vector) -> float:
	"""Euclidean distance."""
	a = np.asarray(a, dtype=np.float32)
	b = np.asarray(b, dtype=np.float32)
	return float(np.linalg.norm(a - b))


def softmax(x: Vector, temperature: float = 1.0) -> Vector:
	"""Temperature-scaled softmax with numerical stability."""
	x = np.asarray(x, dtype=np.float32)
	t = max(1e-6, float(temperature))
	z = (x - float(np.max(x))) / t
	e = np.exp(z)
	return (e / float(np.sum(e))).astype(np.float32)


def random_unit_vector(dim: int, seed: Optional[int] = None) -> Vector:
	"""Generate a random unit vector."""
	rng = np.random.default_rng(seed if seed is not None else 0)
	v = rng.standard_normal(dim).astype(np.float32)
	return normalize(v)


def pad_or_trim(vec: Vector, target_dim: int) -> Vector:
	"""Resize vector to target_dim by trimming or zero-padding."""
	v = np.asarray(vec, dtype=np.float32)
	if v.shape[0] == target_dim:
		return v
	if v.shape[0] < target_dim:
		return np.concatenate([v, np.zeros(target_dim - v.shape[0], dtype=np.float32)], axis=0)
	return v[:target_dim]


def safe_add(a: Vector, b: Vector) -> Vector:
	"""Elementwise add with shape alignment by padding/trim to the larger size."""
	a = np.asarray(a, dtype=np.float32)
	b = np.asarray(b, dtype=np.float32)
	target = max(a.shape[0], b.shape[0])
	a2 = pad_or_trim(a, target)
	b2 = pad_or_trim(b, target)
	return (a2 + b2).astype(np.float32)


def safe_mean(vectors: Iterable[Vector], eps: float = 1e-12) -> Vector:
	"""Mean of vectors with automatic resizing to max dimension."""
	vecs = [np.asarray(v, dtype=np.float32) for v in vectors]
	if not vecs:
		return np.zeros(0, dtype=np.float32)
	target = max(v.shape[0] for v in vecs)
	stack = np.stack([pad_or_trim(v, target) for v in vecs], axis=0)
	m = np.mean(stack, axis=0)
	return np.where(np.isfinite(m), m, 0.0).astype(np.float32)


