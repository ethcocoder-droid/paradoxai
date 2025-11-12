"""
Curiosity Question Generator
----------------------------
Evaluates knowledge and reasoning footprints to detect uncertainty hotspots
and generate internal (self-reflective) and external (developer-facing)
questions to drive exploration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np
import os

from modules.transformer.text_generation import TextGenerator
from modules.transformer.transformer_model import TransformerModel, SimpleTokenizer

logger = logging.getLogger(__name__)
if not logger.handlers:
	logger.addHandler(logging.NullHandler())


KnowledgeProbeFn = Callable[[str], Dict[str, Any]]
ReasoningProbeFn = Callable[[str], Dict[str, Any]]
DeveloperFeedbackFn = Callable[[Dict[str, Any]], None]


@dataclass
class CuriosityConfig:
	uncertainty_threshold: float = 0.35
	max_questions: int = 5
	min_probability_mass: float = 0.05
	curiosity_decay: float = 0.05  # reduces curiosity over time to avoid runaway loops
	temperature: float = 0.7
	top_k: int = 50
	top_p: float = 0.9


class CuriosityState:
	"""
	Tracks curiosity levels per concept, including decay and reinforcement.
	"""

	def __init__(self, config: CuriosityConfig = CuriosityConfig()):
		self.config = config
		self.levels: Dict[str, float] = {}

	def update(self, concept_id: str, uncertainty: float) -> float:
		"""
		Blend current curiosity with observed uncertainty.
		"""
		prev = self.levels.get(concept_id, 0.0)
		new_level = max(
			0.0,
			(1 - self.config.curiosity_decay) * prev + uncertainty,
		)
		self.levels[concept_id] = new_level
		return new_level

	def get(self, concept_id: str) -> float:
		return self.levels.get(concept_id, 0.0)


class QuestionGenerator:
	"""
	Aggregates signals to produce curiosity-driven questions.
	"""

	def __init__(
		self,
		config: CuriosityConfig = CuriosityConfig(),
		knowledge_probe: Optional[KnowledgeProbeFn] = None,
		reasoning_probe: Optional[ReasoningProbeFn] = None,
		developer_feedback_hook: Optional[DeveloperFeedbackFn] = None,
		text_generator: Optional[TextGenerator] = None,
	):
		self.config = config
		self.knowledge_probe = knowledge_probe
		self.reasoning_probe = reasoning_probe
		self.developer_feedback_hook = developer_feedback_hook
		
		if text_generator is None:
			# Initialize a default TransformerModel and SimpleTokenizer
			# This configuration should ideally be loaded from a central config or passed in
			from modules.transformer.transformer_core import TransformerConfig # Import here to avoid circular dependency
			default_transformer_config = TransformerConfig(
				vocab_size=5000,
				d_model=256,
				n_heads=4,
				n_layers=2,
				d_ff=1024,
				max_seq_len=128,
				dropout=0.1
			)
			default_model = TransformerModel(default_transformer_config)
			default_tokenizer = SimpleTokenizer()
			# Build a dummy vocabulary for the default tokenizer
			default_tokenizer.build_vocabulary(["<pad> <unk> <sos> <eos>"])
			# Note: In a real scenario, the tokenizer would need to be trained/loaded with a vocabulary
			self.text_generator = TextGenerator(default_model, default_tokenizer)
		else:
			self.text_generator = text_generator
		self.curiosity_state = CuriosityState(config)

	def evaluate_concept(self, concept_id: str) -> Dict[str, Any]:
		"""
		Inspect knowledge + reasoning signals.
		Returns dictionary with uncertainty metrics.
		"""
		knowledge = self.knowledge_probe(concept_id) if callable(self.knowledge_probe) else {}
		reasoning = self.reasoning_probe(concept_id) if callable(self.reasoning_probe) else {}

		superposition = knowledge.get("superposition") or {}
		probs = np.array(superposition.get("probabilities") or [], dtype=np.float32)
		entropy = self._shannon_entropy(probs) if probs.size else 0.0

		reasoning_conflicts = reasoning.get("conflicts", 0.0)
		missing = float(knowledge.get("metadata", {}).get("missing_fields", 0))

		uncertainty = float(entropy + 0.2 * reasoning_conflicts + 0.1 * missing)
		curiosity_level = self.curiosity_state.update(concept_id, uncertainty)

		return {
			"concept_id": concept_id,
			"uncertainty": uncertainty,
			"entropy": entropy,
			"reasoning_conflicts": reasoning_conflicts,
			"missing": missing,
			"curiosity_level": curiosity_level,
		}

	def generate_questions(self, concept_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
		"""
		Produce internal and external questions for provided concepts.
		Returns: {"internal": [...], "external": [...]}
		"""
		internal: List[Dict[str, Any]] = []
		external: List[Dict[str, Any]] = []

		for cid in concept_ids:
			report = self.evaluate_concept(cid)
			if report["uncertainty"] < self.config.uncertainty_threshold:
				continue
			internal.extend(self._internal_questions(report))
			external.extend(self._external_questions(report))

		internal = internal[: self.config.max_questions]
		external = external[: self.config.max_questions]

		if callable(self.developer_feedback_hook):
			for question in external:
				try:
					self.developer_feedback_hook(question)
					self._store_q_a_pair(question) # Store generated external questions
				except Exception as exc:  # pragma: no cover
					logger.exception("developer_feedback_hook failed: %s", exc)

		for question in internal:
			self._store_q_a_pair(question) # Store generated internal questions

		return {"internal": internal, "external": external}

	def _store_q_a_pair(self, question: Dict[str, Any], answer: str = ""):
		"""
		Stores a generated question and its potential answer (if available) into data/training.md.
		"""
		data_dir = "data"
		file_path = os.path.join(data_dir, "training.md")

		if not os.path.exists(data_dir):
			os.makedirs(data_dir)

		with open(file_path, "a", encoding="utf-8") as f:
			f.write(f"## Question: {question.get('question', question.get('prompt'))}\n")
			f.write(f"### Concept ID: {question.get('concept_id')}\n")
			if answer:
				f.write(f"**Answer:** {answer}\n")
			f.write("---\n\n")

	def _internal_questions(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
		base_prompt_entropy = f"What hidden assumptions drive high entropy ({report['entropy']:.2f}) for {report['concept_id']}?"
		base_prompt_conflicts = f"How can I reconcile reasoning conflicts ({report['reasoning_conflicts']:.2f}) for {report['concept_id']}?"
		return [
			{
				"type": "internal",
				"concept_id": report["concept_id"],
				"prompt": self.generate_question(base_prompt_entropy),
				"priority": report["curiosity_level"],
			},
			{
				"type": "internal",
				"concept_id": report["concept_id"],
				"prompt": self.generate_question(base_prompt_conflicts),
				"priority": report["curiosity_level"] * 0.9,
			},
		]

	def _external_questions(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
		base_question_missing = f"Could you clarify missing data points ({report['missing']}) for {report['concept_id']}?"
		base_question_example = f"What real-world example best fits {report['concept_id']} to reduce entropy?"
		return [
			{
				"type": "external",
				"concept_id": report["concept_id"],
				"question": self.generate_question(base_question_missing),
				"priority": report["curiosity_level"] * 0.8,
			},
			{
				"type": "external",
				"concept_id": report["concept_id"],
				"question": self.generate_question(base_question_example),
				"priority": report["curiosity_level"] * 0.7,
			},
		]

	def _shannon_entropy(self, probs: np.ndarray, eps: float = 1e-9) -> float:
		p = probs + eps
		p = p / p.sum()
		return float(-np.sum(p * np.log2(p)))

	def generate_question(self, context: str) -> str:
		"""
		Generates a question string using the TextGenerator.
		"""
		if self.text_generator is None:
			raise ValueError("TextGenerator not provided to QuestionGenerator.")
		return self.text_generator.generate_text(
			context,
			temperature=self.config.temperature,
			top_k=self.config.top_k,
			top_p=self.config.top_p,
		)


