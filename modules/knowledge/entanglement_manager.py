"""
Entanglement management for Quantum-Like God AI.

Handles relationships between concepts stored in the hyper-matrix.
Each entanglement expresses bi-directional influence with weights and phase.

Key responsibilities:
- Persist entanglement graph to disk (JSON).
- Provide query/update operations for reasoning modules.
- Integrate developer feedback to reinforce or damp specific links.
- Offer utilities for propagating superposition changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


@dataclass
class Entanglement:
	source: str
	target: str
	strength: float
	phase: float  # simplified scalar phase for interference computation
	metadata: Dict[str, float]


class EntanglementManager:
	"""
	Manage entanglement graph among concepts.

	Graph stored as adjacency dictionary:
	{source: {target: Entanglement}}
	Entanglements are symmetric by default; operations ensure both directions sync.
	"""

	def __init__(self, ent_path: Path = Path("data/entanglements.json")):
		self.ent_path = Path(ent_path)
		self._graph: Dict[str, Dict[str, Entanglement]] = {}
		self._load_from_disk()

	def neighbors(self, concept_id: str) -> List[Entanglement]:
		return list(self._graph.get(concept_id, {}).values())

	def set_entanglement(
		self,
		source: str,
		target: str,
		strength: float,
		phase: float = 0.0,
		metadata: Optional[Dict[str, float]] = None,
		symmetric: bool = True,
	) -> None:
		"""
		Create/update entanglement between source and target.
		Strength constrained to [-1, 1] for stability.
		Phase wraps into [-pi, pi].
		"""
		if source == target:
			raise ValueError("Self-entanglement is not permitted")
		strength = float(np.clip(strength, -1.0, 1.0))
		phase = float(np.arctan2(np.sin(phase), np.cos(phase)))  # wrap to [-pi, pi]
		metadata = metadata or {}
		self._graph.setdefault(source, {})[target] = Entanglement(source, target, strength, phase, metadata)
		if symmetric:
			self._graph.setdefault(target, {})[source] = Entanglement(target, source, strength, phase, metadata)

	def decay_all(self, rate: float = 0.01) -> None:
		"""Apply exponential decay to all entanglement strengths."""
		decay = float(np.clip(rate, 0.0, 1.0))
		for neighbors in self._graph.values():
			for ent in neighbors.values():
				ent.strength *= (1.0 - decay)

	def reinforce(self, source: str, target: str, delta: float) -> None:
		"""Adjust entanglement strength; Positive delta reinforces, negative weakens."""
		if target not in self._graph.get(source, {}):
			self.set_entanglement(source, target, strength=delta)
			return
		ent = self._graph[source][target]
		self.set_entanglement(source, target, ent.strength + delta, ent.phase, ent.metadata)

	def apply_superposition_influence(
		self,
		concept_id: str,
		superposition_probabilities: Iterable[float],
		scaling: float = 0.1,
	) -> Dict[str, float]:
		"""
		Propagate superposition probabilities onto neighbors.
		Returns influence scores per neighbor for higher-level modules.
		"""
		probs = np.array(list(superposition_probabilities), dtype=np.float32)
		if probs.size == 0 or float(probs.sum()) <= 0:
			return {}
		probs = probs / probs.sum()
		neighbor_scores: Dict[str, float] = {}
		for neighbor in self.neighbors(concept_id):
			influence = float(np.dot(probs, neighbor.strength)) * scaling
			neighbor_scores[neighbor.target] = influence
		return neighbor_scores

	def integrate_developer_guidance(
		self,
		updates: List[Dict[str, float]],
		default_phase: float = 0.0,
	) -> None:
		"""
		Apply list of developer-specified entanglement adjustments.
		Each update expects keys: source, target, strength_delta (optional), phase (optional).
		"""
		for update in updates:
			try:
				source = str(update["source"])
				target = str(update["target"])
			except KeyError as exc:
				logger.warning("Skipping update missing key: %s", exc)
				continue
			delta = float(update.get("strength_delta", 0.1))
			new_phase = float(update.get("phase", default_phase))
			self.reinforce(source, target, delta)
			self.set_entanglement(source, target, self._graph[source][target].strength, new_phase)

	def save(self) -> None:
		try:
			self.ent_path.parent.mkdir(parents=True, exist_ok=True)
			payload = []
			for source, targets in self._graph.items():
				for ent in targets.values():
					payload.append(
						{
							"source": source,
							"target": ent.target,
							"strength": ent.strength,
							"phase": ent.phase,
							"metadata": ent.metadata,
						}
					)
			self.ent_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
		except Exception as exc:
			logger.exception("Failed to save entanglements: %s", exc)

	def _load_from_disk(self) -> None:
		if not self.ent_path.exists():
			logger.info("Entanglement file %s missing; starting empty graph", self.ent_path)
			return
		try:
			content = self.ent_path.read_text(encoding="utf-8")
			if not content.strip():
				return
			data = json.loads(content)
			if not isinstance(data, list):
				raise ValueError("Entanglements JSON must be a list")
			for record in data:
				self.set_entanglement(
					source=str(record["source"]),
					target=str(record["target"]),
					strength=float(record.get("strength", 0.0)),
					phase=float(record.get("phase", 0.0)),
					metadata=record.get("metadata") or {},
					symmetric=False,
				)
		except Exception as exc:
			logger.exception("Failed to load entanglements: %s", exc)


