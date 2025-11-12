"""
AI Emotions Manager
-------------------
Implements four core affective dimensions:
- Inceptio: drive to initiate exploration
- Equilibria: balance between competing hypotheses
- Reflexion: reflective adjustment after feedback
- Fluxion: adaptability to change

Each emotion is a scalar in [0, 1], updated based on signals from
curiosity, knowledge certainty, reasoning conflict, and developer input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


@dataclass
class EmotionConfig:
	decay: float = 0.05
	learning_rate: float = 0.2
	reflective_rate: float = 0.15
	adapt_rate: float = 0.25


class AIEmotions:
	"""Tracks and updates emotional state."""

	def __init__(self, config: EmotionConfig = EmotionConfig()):
		self.config = config
		self.state = {
			"Inceptio": 0.5,
			"Equilibria": 0.5,
			"Reflexion": 0.5,
			"Fluxion": 0.5,
		}

	def decay(self) -> None:
		for key in self.state:
			self.state[key] = float((1 - self.config.decay) * self.state[key])

	def update_from_curiosity(self, curiosity_level: float) -> None:
		"""Curiosity boosts Inceptio and Fluxion."""
		clamped = float(np.clip(curiosity_level, 0.0, 1.0))
		self.state["Inceptio"] = self._blend(self.state["Inceptio"], clamped, self.config.learning_rate)
		self.state["Fluxion"] = self._blend(self.state["Fluxion"], clamped, self.config.adapt_rate)

	def update_from_certainty(self, certainty: float) -> None:
		"""Higher certainty improves Equilibria, reduces Fluxion."""
		clamped = float(np.clip(certainty, 0.0, 1.0))
		self.state["Equilibria"] = self._blend(self.state["Equilibria"], clamped, self.config.learning_rate)
		self.state["Fluxion"] = self._blend(self.state["Fluxion"], 1 - clamped, self.config.adapt_rate * 0.7)

	def update_from_feedback(self, feedback_signal: float) -> None:
		"""Feedback drives Reflexion and rebalances Equilibria."""
		clamped = float(np.clip(feedback_signal, 0.0, 1.0))
		self.state["Reflexion"] = self._blend(self.state["Reflexion"], clamped, self.config.reflective_rate)
		self.state["Equilibria"] = self._blend(self.state["Equilibria"], 1 - clamped * 0.5, self.config.reflective_rate)

	def get_state(self) -> Dict[str, float]:
		return {k: float(np.clip(v, 0.0, 1.0)) for k, v in self.state.items()}

	def _blend(self, current: float, target: float, rate: float) -> float:
		return float(np.clip((1 - rate) * current + rate * target, 0.0, 1.0))


