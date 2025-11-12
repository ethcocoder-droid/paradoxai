"""
Probabilistic Collapse
----------------------
Collapses a superposition into a single state while factoring AI-emotions,
temperature, and user profile preferences. Produces a structured response
with trace metadata for downstream logging and UI display.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import logging

import numpy as np

Vector = np.ndarray

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


@dataclass
class CollapseConfig:
	temperature: float = 0.7
	max_response_length: int = 256
	seed: Optional[int] = 17
	# Emotion influences
	inceptio_boost: float = 0.15
	equilibria_smoothing: float = 0.2
	reflexion_consistency: float = 0.1
	fluxion_diversity: float = 0.1


class ProbabilisticCollapse:
	"""
	Perform emotion-aware probabilistic collapse of a superposition and generate
	a lightweight user-adaptive response with metadata.
	"""

	def __init__(self, config: CollapseConfig = CollapseConfig()):
		self.config = config
		self.rng = np.random.default_rng(self.config.seed if self.config.seed is not None else 0)

	def collapse(
		self,
		superposition: Dict[str, Any],
		emotions: Dict[str, float],
		user_profile: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""
		Returns a dict: {
			"chosen_index": int,
			"chosen_vector": np.ndarray,
			"probabilities": List[float],
			"response": str,
			"meta": {...}
		}
		Handles missing/degenerate inputs robustly.
		"""
		branches = [np.asarray(b, dtype=np.float32) for b in superposition.get("branches", [])]
		probabilities = np.array(superposition.get("probabilities", []), dtype=np.float32)
		if not branches:
			logger.warning("collapse received empty branches; returning default response")
			return self._default_response("No content to collapse.", probabilities=[1.0])
		B = len(branches)
		if probabilities.size != B or float(probabilities.sum()) <= 0:
			probabilities = np.ones(B, dtype=np.float32) / float(B)
		else:
			probabilities = (probabilities / probabilities.sum()).astype(np.float32)

		# Apply emotion-aware adjustments
		adj_p = self._adjust_probabilities(probabilities, branches, emotions)
		# Sample or pick argmax based on temperature
		choice = self._select_index(adj_p, temperature=float(self.config.temperature))
		chosen_vec = branches[choice]

		response_text = self._render_response(chosen_vec, emotions, user_profile)
		return {
			"chosen_index": int(choice),
			"chosen_vector": chosen_vec,
			"probabilities": adj_p.astype(np.float32).tolist(),
			"response": response_text,
			"meta": {
				"temperature": float(self.config.temperature),
				"emotions": {k: float(v) for k, v in emotions.items()},
				"user_profile": (user_profile or {}),
			},
		}

	def _adjust_probabilities(
		self,
		probabilities: np.ndarray,
		branches: Sequence[Vector],
		emotions: Dict[str, float],
	) -> np.ndarray:
		p = probabilities.astype(np.float32, copy=True)
		inceptio = float(np.clip(emotions.get("Inceptio", 0.5), 0.0, 1.0))
		equilibria = float(np.clip(emotions.get("Equilibria", 0.5), 0.0, 1.0))
		reflexion = float(np.clip(emotions.get("Reflexion", 0.5), 0.0, 1.0))
		fluxion = float(np.clip(emotions.get("Fluxion", 0.5), 0.0, 1.0))

		# Inceptio: boost high-energy branches (larger norm)
		norms = np.array([float(np.linalg.norm(b)) for b in branches], dtype=np.float32)
		if norms.sum() > 0:
			norms = norms / norms.sum()
			p = self._mix(p, self._softmax(norms / 0.5), self.config.inceptio_boost * inceptio)

		# Equilibria: smooth probabilities towards uniform
		uniform = np.ones_like(p) / float(len(p))
		p = self._mix(p, uniform, self.config.equilibria_smoothing * equilibria)

		# Reflexion: reduce volatility (shrink extremes)
		center = 1.0 / float(len(p))
		p = center + (p - center) * (1.0 - self.config.reflexion_consistency * reflexion)

		# Fluxion: encourage diversity by entropy-boost via temperature-like spread
		entropy_boost = 1.0 + self.config.fluxion_diversity * fluxion
		p = self._softmax(np.log(np.maximum(1e-8, p)) * entropy_boost)
		return p

	def _select_index(self, probabilities: np.ndarray, temperature: float) -> int:
		if temperature <= 1e-3:
			return int(np.argmax(probabilities))
		# Temperature sampling via power transform
		temp_p = probabilities ** (1.0 / max(1e-6, temperature))
		temp_p = temp_p / temp_p.sum()
		return int(self.rng.choice(len(temp_p), p=temp_p))

	def _render_response(
		self,
		vector: Vector,
		emotions: Dict[str, float],
		user_profile: Optional[Dict[str, Any]],
	) -> str:
		"""
		Generate a simple natural-language response in English, shaped by tone/style.
		Optionally uses user_profile['prompt'] to reference the user's query.
		"""
		prefs = (user_profile or {})
		tone = str(prefs.get("tone", "neutral"))
		style = str(prefs.get("style", "concise"))
		prompt = str(prefs.get("prompt", "")).strip()

		# Map tone/style to phrasing
		tone_prefix = {
			"friendly": "Sure! ",
			"formal": "Certainly. ",
			"neutral": "",
		}.get(tone, "")
		style_suffix = {
			"concise": "",
			"detailed": " If you'd like, I can expand with more details.",
		}.get(style, "")

		# Lightweight summary guided by vector magnitude and emotions
		confidence = max(0.0, min(1.0, 1.0 - float(np.clip(np.linalg.norm(vector), 0.0, 1.0)) * 0.1))
		em_summary = self._emotion_summary(emotions)

		if prompt:
			text = f"{tone_prefix}Here's a clear English answer about “{prompt}”. This build focuses on reasoning dynamics rather than a large knowledge base, so I’ll keep it straightforward. {style_suffix} (Signals: {em_summary})"
		else:
			text = f"{tone_prefix}Here's a clear English response. {style_suffix} (Signals: {em_summary})"

		# Cap length
		if len(text) > self.config.max_response_length:
			text = text[: self.config.max_response_length - 3] + "..."
		return text

	def _emotion_summary(self, emotions: Dict[str, float]) -> str:
		keys = ["Inceptio", "Equilibria", "Reflexion", "Fluxion"]
		parts = [f"{k}={float(np.clip(emotions.get(k, 0.5),0,1)):.2f}" for k in keys]
		return " | ".join(parts)

	def _softmax(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=np.float32)
		x = x - float(np.max(x))
		exp_x = np.exp(x)
		return exp_x / float(np.sum(exp_x))

	def _mix(self, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
		alpha = float(np.clip(alpha, 0.0, 1.0))
		return (1 - alpha) * a + alpha * b

	def _default_response(self, text: str, probabilities: Optional[Sequence[float]] = None) -> Dict[str, Any]:
		if probabilities is None:
			probabilities = [1.0]
		return {
			"chosen_index": 0,
			"chosen_vector": np.zeros(1, dtype=np.float32),
			"probabilities": list(probabilities),
			"response": text,
			"meta": {"temperature": float(self.config.temperature), "emotions": {}, "user_profile": {}},
		}


