"""
Attention Manager
-----------------
Balances attention allocation between curiosity-driven exploration and
correctness-driven consolidation, mediated by AI emotions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


@dataclass
class AttentionConfig:
	curiosity_weight: float = 0.6
	correctness_weight: float = 0.4
	emotion_influence: float = 0.3
	min_allocation: float = 0.1


class AttentionManager:
	"""Compute attention weights for curiosity vs correctness pathways."""

	def __init__(self, config: AttentionConfig = AttentionConfig()):
		self.config = config

	def compute_allocation(
		self,
		curiosity_signal: float,
		correctness_signal: float,
		emotions: Dict[str, float],
	) -> Dict[str, float]:
		"""
		Return attention weights with keys: curiosity, correctness.
		Signals expected in [0, 1]; emotions should include Inceptio and Equilibria.
		"""
		c_signal = float(np.clip(curiosity_signal, 0.0, 1.0))
		k_signal = float(np.clip(correctness_signal, 0.0, 1.0))
		em_inceptio = float(np.clip(emotions.get("Inceptio", 0.5), 0.0, 1.0))
		em_equilibria = float(np.clip(emotions.get("Equilibria", 0.5), 0.0, 1.0))

		curiosity_score = self.config.curiosity_weight * c_signal + self.config.emotion_influence * em_inceptio
		correctness_score = self.config.correctness_weight * k_signal + self.config.emotion_influence * em_equilibria

		total = curiosity_score + correctness_score
		if total <= 0:
			return {"curiosity": 0.5, "correctness": 0.5}

		curiosity_alloc = curiosity_score / total
		correctness_alloc = correctness_score / total

		min_alloc = self.config.min_allocation
		curiosity_alloc = float(np.clip(curiosity_alloc, min_alloc, 1 - min_alloc))
		correctness_alloc = float(np.clip(correctness_alloc, min_alloc, 1 - min_alloc))

		# Renormalize after clipping
		norm = curiosity_alloc + correctness_alloc
		return {
			"curiosity": curiosity_alloc / norm,
			"correctness": correctness_alloc / norm,
		}


