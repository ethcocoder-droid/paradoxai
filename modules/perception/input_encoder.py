"""
Input Encoder
-------------
Converts heterogeneous inputs (text, images, and pre-defined concepts)
into multi-dimensional vectors, normalizes them, and prepares
superposition-ready states for downstream modules.

Design goals:
- Deterministic, testable transformations with clear interfaces
- Robust handling of missing values and malformed inputs
- Minimal dependencies (NumPy; Pillow optional for image support)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import hashlib
import logging

import numpy as np

try:
	# Pillow is optional; image encoding gracefully degrades if unavailable.
	from PIL import Image
	PIL_AVAILABLE = True
except Exception:  # pragma: no cover - environment may lack Pillow
	PIL_AVAILABLE = False


# Configure module-level logger (library-friendly: no handlers if root configured)
logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


Vector = np.ndarray


@dataclass(frozen=True)
class EncodedItem:
	"""
	Container for an encoded input, including:
	- vector: normalized feature vector (unit L2 norm or zero vector on failure)
	- meta: metadata describing the encoding
	- superposition: data structure ready for quantum-like operations
	"""
	vector: Vector
	meta: Dict[str, Any]
	superposition: Dict[str, Any]


class InputEncoder:
	"""
	Encodes text, images, and pre-defined concepts to vectors and prepares
	superposition-ready states.

	Parameters
	----------
	vector_dim : int
		Target dimensionality for output vectors.
	seed : Optional[int]
		Seed for deterministic hashing to vector projections.
	"""

	def __init__(self, vector_dim: int = 128, seed: Optional[int] = 42):
		if not isinstance(vector_dim, int) or vector_dim <= 0:
			raise ValueError("vector_dim must be a positive integer")
		self.vector_dim = vector_dim
		self.seed = seed

	def encode_text(self, text: Optional[str]) -> EncodedItem:
		"""
		Encode text to a fixed-size normalized vector using deterministic hashing.
		- Robust to None/empty input: returns zero vector and informative metadata.
		"""
		if text is None or not isinstance(text, str) or text.strip() == "":
			logger.warning("encode_text received empty or invalid text")
			vec = self._zero_vector()
			return self._package(vec, kind="text", raw=text, ok=False, reason="empty_or_invalid")

		hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
		vec = self._hash_bytes_to_vector(hash_bytes)
		vec = self._normalize(vec)
		return self._package(vec, kind="text", raw=text, ok=True)

	def encode_concept(self, concept_id: Optional[Union[str, int]]) -> EncodedItem:
		"""
		Encode a concept identifier into a vector.
		- Works for string or integer identifiers; robust to None.
		"""
		if concept_id is None or (isinstance(concept_id, str) and concept_id.strip() == ""):
			logger.warning("encode_concept received empty or invalid concept_id")
			vec = self._zero_vector()
			return self._package(vec, kind="concept", raw=concept_id, ok=False, reason="empty_or_invalid")

		serialized = str(concept_id)
		hash_bytes = hashlib.blake2b(serialized.encode("utf-8"), digest_size=32).digest()
		vec = self._hash_bytes_to_vector(hash_bytes)
		vec = self._normalize(vec)
		return self._package(vec, kind="concept", raw=serialized, ok=True)

	def encode_image(self, image: Optional[Union[str, "Image.Image"]]) -> EncodedItem:
		"""
		Encode an image into a vector.
		- Accepts a PIL Image or a filesystem path string.
		- If Pillow is unavailable or input invalid, returns a zero vector with metadata.
		- Uses simple downsample + grayscale histogram as a lightweight, deterministic embedding.
		"""
		if not PIL_AVAILABLE:
			logger.warning("encode_image called but Pillow is not available")
			vec = self._zero_vector()
			return self._package(vec, kind="image", raw=None, ok=False, reason="pillow_not_available")

		if image is None:
			logger.warning("encode_image received None")
			return self._package(self._zero_vector(), kind="image", raw=None, ok=False, reason="none_input")

		try:
			if isinstance(image, str):
				img = Image.open(image).convert("L")
			elif isinstance(image, Image.Image):
				img = image.convert("L")
			else:
				logger.warning("encode_image received unsupported type: %s", type(image))
				return self._package(self._zero_vector(), kind="image", raw=None, ok=False, reason="unsupported_type")

			# Lightweight deterministic embedding: resize + histogram
			img_small = img.resize((16, 16))  # 256 pixels
			hist = np.array(img_small.histogram(), dtype=np.float32)  # 256 bins for grayscale
			# Project histogram to target dimension deterministically
			vec = self._project(hist, target_dim=self.vector_dim)
			vec = self._normalize(vec)
			return self._package(vec, kind="image", raw="pil_image", ok=True)
		except Exception as exc:  # pragma: no cover - depends on filesystem/images
			logger.exception("encode_image failed: %s", exc)
			return self._package(self._zero_vector(), kind="image", raw=None, ok=False, reason="exception")

	def encode_batch(self, items: Iterable[Tuple[str, Any]]) -> List[EncodedItem]:
		"""
		Encode a batch of (kind, payload) pairs where kind in {'text','image','concept'}.
		Returns a list of EncodedItem in the same order.
		"""
		results: List[EncodedItem] = []
		for kind, payload in items:
			if kind == "text":
				results.append(self.encode_text(payload))
			elif kind == "image":
				results.append(self.encode_image(payload))
			elif kind == "concept":
				results.append(self.encode_concept(payload))
			else:
				logger.warning("encode_batch encountered unknown kind=%s", kind)
				results.append(self._package(self._zero_vector(), kind="unknown", raw=payload, ok=False, reason="unknown_kind"))
		return results

	# ---------- Internal utilities ----------

	def _hash_bytes_to_vector(self, data: bytes) -> Vector:
		"""
		Map a byte digest deterministically to a real-valued vector using a seeded RNG.
		"""
		seed_material = int.from_bytes(hashlib.sha256(data).digest()[:8], byteorder="big")
		seed = (seed_material if self.seed is None else (seed_material ^ int(self.seed))) & 0xFFFFFFFF
		rng = np.random.default_rng(seed)
		return rng.standard_normal(self.vector_dim).astype(np.float32)

	def _project(self, vec: Vector, target_dim: int) -> Vector:
		"""
		Project a vector to target_dim using a deterministic random projection matrix.
		"""
		if vec.ndim != 1:
			raise ValueError("Input vector must be 1D")
		source_dim = vec.shape[0]
		seed = (source_dim if self.seed is None else (source_dim ^ int(self.seed))) & 0xFFFFFFFF
		rng = np.random.default_rng(seed)
		# Achlioptas-style sparse random projection for efficiency
		proj = rng.choice([-1.0, 0.0, 1.0], size=(target_dim, source_dim), p=[1/6, 2/3, 1/6]).astype(np.float32)
		projected = proj @ vec
		return projected

	def _normalize(self, vec: Vector, eps: float = 1e-12) -> Vector:
		"""
		L2-normalize a vector; returns zero vector if norm is ~0 to avoid NaNs.
		"""
		norm = float(np.linalg.norm(vec))
		if norm < eps:
			logger.warning("Normalization produced near-zero norm; returning zero vector")
			return np.zeros_like(vec, dtype=np.float32)
		return (vec / norm).astype(np.float32)

	def _zero_vector(self) -> Vector:
		return np.zeros(self.vector_dim, dtype=np.float32)

	def _init_superposition(self, base_vector: Vector, num_branches: int = 3) -> Dict[str, Any]:
		"""
		Initialize a simple superposition structure:
		- 'branches': list of state vectors (perturbations around base_vector)
		- 'amplitudes': complex-like amplitudes stored as real 2D (re, im)
		- 'probabilities': derived from amplitude magnitudes (sum to 1.0)
		This structure is intentionally simple and will be consumed by the
		reasoning/interference engines in later chunks.
		"""
		if num_branches <= 0:
			raise ValueError("num_branches must be positive")

		rng = np.random.default_rng(self.seed if self.seed is not None else 0)
		branches: List[Vector] = []
		for _ in range(num_branches):
			noise = rng.normal(loc=0.0, scale=0.05, size=base_vector.shape).astype(np.float32)
			state = self._normalize(base_vector + noise)
			branches.append(state)

		# Random amplitudes (re, im), then normalized to probabilities
		amps = rng.normal(size=(num_branches, 2)).astype(np.float32)
		probs = np.sum(amps**2, axis=1)
		if probs.sum() <= 0:
			# Fallback to uniform
			probs = np.ones(num_branches, dtype=np.float32) / float(num_branches)
		else:
			probs = (probs / probs.sum()).astype(np.float32)

		return {
			"branches": branches,
			"amplitudes": amps,
			"probabilities": probs.tolist(),
		}

	def _package(self, vec: Vector, *, kind: str, raw: Any, ok: bool, reason: Optional[str] = None) -> EncodedItem:
		"""
		Helper to create EncodedItem with superposition attached.
		"""
		superpos = self._init_superposition(vec, num_branches=3)
		meta: Dict[str, Any] = {
			"kind": kind,
			"vector_dim": self.vector_dim,
			"ok": ok,
			"reason": reason,
		}
		return EncodedItem(vector=vec, meta=meta, superposition=superpos)


# ----------- Minimal connectivity hook to knowledge module -----------
def to_knowledge_payload(encoded: EncodedItem) -> Dict[str, Any]:
	"""
	Translate EncodedItem into a generic dictionary expected by the knowledge module.
	This avoids a hard dependency before the knowledge module exists.
	"""
	return {
		"vector": encoded.vector.astype(np.float32),
		"meta": encoded.meta,
		"superposition": encoded.superposition,
	}


