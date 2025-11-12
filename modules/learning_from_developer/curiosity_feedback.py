"""
Curiosity Feedback
------------------
Generates clarification prompts from developer updates and integrates
with AI-emotions to steer future exploration intensity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import numpy as np

from modules.self_awareness.ai_emotions import AIEmotions

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


class CuriosityFeedback:
	"""Produces clarification prompts and updates emotions based on uncertainty."""

	def __init__(self, emotions: Optional[AIEmotions] = None):
		self.emotions = emotions or AIEmotions()

	def generate(self, concept_id: str, uncertainty: float, missing_fields: int = 0) -> Dict[str, Any]:
		"""
		Create a compact set of prompts for the developer and adjust emotions.
		"""
		u = float(np.clip(uncertainty, 0.0, 2.0))
		missing = int(max(0, missing_fields))
		# Update emotion signals
		self.emotions.update_from_curiosity(min(1.0, u))
		self.emotions.update_from_feedback(min(1.0, missing / 5.0))

		prompts: List[str] = []
		if u > 0.5:
			prompts.append(f"What key constraints define {concept_id}?")
		if missing > 0:
			prompts.append(f"Please provide {missing} missing attribute(s) for {concept_id}.")
		if not prompts:
			prompts.append(f"Any examples to solidify understanding of {concept_id}?")

		return {
			"concept_id": concept_id,
			"uncertainty": u,
			"missing_fields": missing,
			"prompts": prompts,
			"emotions": self.emotions.get_state(),
		}


