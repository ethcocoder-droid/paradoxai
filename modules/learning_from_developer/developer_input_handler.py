"""
Developer Input Handler
-----------------------
Validates and encodes developer input to a unified payload suitable
for updating the hyper-matrix and guiding curiosity generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

import numpy as np

from modules.perception.input_encoder import InputEncoder, to_knowledge_payload

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


@dataclass
class DevInputConfig:
	vector_dim: int = 128


class DeveloperInputHandler:
	"""Encodes text/concept developer inputs into normalized vectors and superpositions."""

	def __init__(self, config: DevInputConfig = DevInputConfig()):
		self.config = config
		self.encoder = InputEncoder(vector_dim=config.vector_dim)

	def encode(self, concept_id: str, text: Optional[str] = None, concept_hint: Optional[str] = None) -> Dict[str, Any]:
		"""
		Accepts developer text or concept hint and returns a normalized payload.
		At least one of text or concept_hint should be provided.
		"""
		if not concept_id:
			raise ValueError("concept_id must be provided")
		if (text is None or str(text).strip() == "") and (concept_hint is None or str(concept_hint).strip() == ""):
			logger.warning("DeveloperInputHandler.encode received empty content")
			text = ""
		# Prefer text; fall back to concept hint
		item = self.encoder.encode_text(text) if (text and str(text).strip() != "") else self.encoder.encode_concept(concept_hint)
		payload = to_knowledge_payload(item)
		payload["concept_id"] = concept_id
		payload["meta"]["source"] = "developer"
		return payload


