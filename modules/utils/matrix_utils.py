"""
Matrix utilities with safe operations and projections.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple
import numpy as np

Matrix = np.ndarray
Vector = np.ndarray


def ensure_2d(x: np.ndarray) -> Matrix:
	"""Ensure input is 2D (N,D)."""
	arr = np.asarray(x, dtype=np.float32)
	if arr.ndim == 1:
		return arr.reshape(1, -1)
	if arr.ndim == 2:
		return arr
	raise ValueError("Input must be 1D or 2D array")


def xavier_init(shape: Tuple[int, ...], seed: Optional[int] = None) -> Matrix:
	"""Xavier initialization for weight matrices (2D shape required)."""
	if len(shape) != 2:
		raise ValueError("Xavier initialization requires a 2D shape tuple (fan_in, fan_out)")
	fan_in, fan_out = shape
	rng = np.random.default_rng(seed if seed is not None else 0)
	limit = np.sqrt(6.0 / (fan_in + fan_out))
	return rng.uniform(low=-limit, high=limit, size=shape).astype(np.float32)


def matmul_safe(a: np.ndarray, b: np.ndarray) -> Matrix:
	"""Matrix multiplication with type/shape checks."""
	A = ensure_2d(a)
	B = ensure_2d(b)
	if A.shape[1] != B.shape[0]:
		raise ValueError(f"Incompatible shapes for matmul: {A.shape} x {B.shape}")
	return (A @ B).astype(np.float32)


def random_projection_matrix(source_dim: int, target_dim: int, seed: Optional[int] = None) -> Matrix:
	"""Sparse Achlioptas-style random projection matrix."""
	rng = np.random.default_rng(seed if seed is not None else 0)
	M = rng.choice([-1.0, 0.0, 1.0], size=(target_dim, source_dim), p=[1/6, 2/3, 1/6]).astype(np.float32)
	return M


def stack_pad(rows: Iterable[Vector]) -> Matrix:
	"""Stack 1D vectors after padding to the largest dimension."""
	row_list = [np.asarray(r, dtype=np.float32) for r in rows]
	if not row_list:
		return np.zeros((0, 0), dtype=np.float32)
	target = max(r.shape[0] for r in row_list)
	stack = np.stack([pad_or_trim(r, target) for r in row_list], axis=0)
	return stack


def pad_or_trim(vec: Vector, target_dim: int) -> Vector:
	v = np.asarray(vec, dtype=np.float32)
	if v.shape[0] == target_dim:
		return v
	if v.shape[0] < target_dim:
		return np.concatenate([v, np.zeros(target_dim - v.shape[0], dtype=np.float32)], axis=0)
	return v[:target_dim]


def normalize_rows(M: Matrix, eps: float = 1e-12) -> Matrix:
	"""Row-wise L2 normalization with zero-row safeguard."""
	X = ensure_2d(M)
	norms = np.linalg.norm(X, axis=1, keepdims=True)
	mask = norms < eps
	norms[mask] = 1.0
	Y = X / norms
	Y[mask] = 0.0
	return Y.astype(np.float32)


def covariance(M: Matrix) -> Matrix:
	"""Compute covariance matrix (D x D) given rows as samples."""
	X = ensure_2d(M)
	if X.shape[0] < 2:
		return np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
	Xc = X - np.mean(X, axis=0, keepdims=True)
	C = (Xc.T @ Xc) / float(Xc.shape[0] - 1)
	return C.astype(np.float32)


def safe_inverse(M: Matrix, eps: float = 1e-6) -> Matrix:
	"""Stable pseudo-inverse using Tikhonov regularization."""
	X = np.asarray(M, dtype=np.float32)
	D = X.shape[0]
	reg = eps * np.eye(D, dtype=np.float32)
	# If not square, fall back to pinv
	if X.shape[0] != X.shape[1]:
		return np.linalg.pinv(X).astype(np.float32)
	return np.linalg.inv(X + reg).astype(np.float32)


