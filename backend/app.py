"""
Flask backend for Paradox AI (by Ethco Coders).
"""
from __future__ import annotations

import sys
from pathlib import Path
# Ensure project root is importable when running this file directly from the backend/ directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from typing import Any, Dict, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from modules.perception.input_encoder import InputEncoder
from modules.reasoning.pathological_logic import PathologicalLogic, ReasoningConfig
from modules.reasoning.interference_engine import InterferenceEngine
from modules.output.probabilistic_collapse import ProbabilisticCollapse, CollapseConfig
from modules.knowledge.hyper_matrix import HyperMatrixStore, SuperpositionUpdater
from modules.knowledge.entanglement_manager import EntanglementManager
from modules.learning_from_developer.developer_input_handler import DeveloperInputHandler
from modules.learning_from_developer.hyper_matrix_updater import HyperMatrixUpdater
from modules.self_awareness.ai_emotions import AIEmotions

import json
import hashlib
import secrets


class AppState:
	def __init__(self):
		self.encoder = InputEncoder(vector_dim=128)
		self.store = HyperMatrixStore()
		self.superpos_updater = SuperpositionUpdater(self.store)
		self.entanglement = EntanglementManager()
		self.reasoner = PathologicalLogic(config=ReasoningConfig())
		self.interference = InterferenceEngine()
		self.collapse = ProbabilisticCollapse(config=CollapseConfig())
		self.emotions = AIEmotions()
		self.dev_input = DeveloperInputHandler()
		self.hm_updater = HyperMatrixUpdater(self.store)
		# simple JSON-backed admin data (rooted to project /data)
		self.users_path = _ROOT / "data/users.json"
		self.contacts_path = _ROOT / "data/contacts.json"
		self.posts_path = _ROOT / "data/posts.json"
		self.tokens: dict[str, dict] = {}  # token -> {"username":..., "role":...}
		self.training_path = _ROOT / "data/training.json"
		self._ensure_files()
		self._maybe_bootstrap_admin()

	def _ensure_files(self):
		for p in [self.users_path, self.contacts_path, self.posts_path, self.training_path]:
			p.parent.mkdir(parents=True, exist_ok=True)
			if not p.exists():
				p.write_text("[]", encoding="utf-8")

	def _read_json(self, path: Path):
		try:
			txt = path.read_text(encoding="utf-8")
			return json.loads(txt) if txt.strip() else []
		except Exception:
			return []

	def _write_json(self, path: Path, data):
		path.write_text(json.dumps(data, indent=2), encoding="utf-8")

	def _maybe_bootstrap_admin(self):
		users = self._read_json(self.users_path)
		if not users:
			users.append({"username": "admin", "password_hash": hashlib.sha256(b"admin").hexdigest(), "role": "developer"})
			self._write_json(self.users_path, users)

	def upsert_training_concept(self, concept_id: str, text_value: str):
		"""Create/update a concept representing a training item to enrich the hyper-matrix."""
		if not text_value:
			return
		item = self.encoder.encode_text(text_value)
		self.store.upsert_concept(concept_id, item.vector, item.superposition, {"source": "training"})

	def update_entanglements_for_training(self, new_id: str, qa_list: list):
		"""
		For a new QA concept, connect it to other QA concepts with cosine-based strength.
		"""
		try:
			# Build vector map
			vecs = []
			ids = []
			for it in qa_list:
				if it.get("id") and isinstance(it.get("q_vec"), list):
					ids.append(it["id"])
					vecs.append(np.asarray(it["q_vec"], dtype=np.float32))
			if not vecs or new_id not in ids:
				return
			idx_new = ids.index(new_id)
			v_new = vecs[idx_new]
			for j, v in enumerate(vecs):
				if j == idx_new:
					continue
				num = float(np.dot(v_new, v))
				den = float(np.linalg.norm(v_new) * np.linalg.norm(v) + 1e-9)
				sim = num / den if den > 0 else 0.0
				if sim >= 0.7:
					# Map similarity to strength [-1,1] -> here [0,1]
					strength = float(min(1.0, max(0.0, sim)))
					self.entanglement.set_entanglement(new_id, ids[j], strength=strength, phase=0.0, symmetric=True)
					self.entanglement.save()
		except Exception:
			pass


def _hash_pw(pw: str) -> str:
	return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def _require_dev(state: AppState, token: Optional[str]) -> bool:
	if not token or token not in state.tokens:
		return False
	return state.tokens[token].get("role") == "developer"


app = Flask("Paradox AI (by Ethco Coders)")
CORS(app, resources={r"/api/*": {"origins": "*"}})
STATE = AppState()


@app.get("/")
def root():
	return jsonify({
		"service": "Paradox AI (Flask)",
		"status": "ok",
		"message": "Backend is running. Use /api/* endpoints.",
		"examples": ["/api/health", "/api/query", "/api/debug/hyper-matrix", "/api/debug/emotions"]
	})


@app.get("/api/health")
def api_health():
	return jsonify({"status": "ok"})


@app.post("/api/query")
def api_query():
	payload = request.get_json(force=True, silent=True) or {}
	text = str(payload.get("query") or "").strip()
	if not text:
		return jsonify({"response": "Empty query.", "probabilities": [1.0], "chosen_index": 0, "meta": {"error": "empty_query"}})
	parameters = payload.get("parameters") or {}
	temperature = float(parameters.get("temperature", 0.7))
	user_profile = payload.get("user_profile") or {}

	# Training retrieval (exact/substring + vector similarity)
	training = STATE._read_json(STATE.training_path)
	qa_list = training.get("qa", []) if isinstance(training, dict) else []
	lower_text = text.lower()
	# exact/substring
	for item in qa_list:
		q = str(item.get("question",""))
		if not q:
			continue
		ql = q.lower()
		if ql == lower_text or ql in lower_text or lower_text in ql:
			ans = str(item.get("answer",""))
			if ans:
				return jsonify({"response": ans, "meta": {"source":"training_retrieval"}})
	# vector similarity (and optional biasing)
	biased_base_vec = None
	if qa_list:
		in_vec = STATE.encoder.encode_text(text).vector
		sims: list[tuple[float, dict]] = []
		for item in qa_list:
			qv = item.get("q_vec")
			if not isinstance(qv, list):
				continue
			try:
				qv_np = np.asarray(qv, dtype=np.float32)
				num = float(np.dot(in_vec, qv_np))
				den = float(np.linalg.norm(in_vec) * np.linalg.norm(qv_np) + 1e-9)
				sim = num / den if den > 0 else -1.0
				sims.append((sim, item))
			except Exception:
				continue
		sims.sort(key=lambda x: x[0], reverse=True)
		if sims:
			best_sim, best_item = sims[0]
			best_ans = str(best_item.get("answer",""))
			if best_sim >= 0.85 and best_ans:
				return jsonify({"response": best_ans, "meta": {"source":"training_similarity","similarity":best_sim}})
			# multi-hop blend for moderate matches
			topk = sims[:3]
			weighted = None
			total_w = 0.0
			for sim, it in topk:
				if sim < 0.6:
					continue
				try:
					avec = STATE.encoder.encode_text(str(it.get("answer",""))).vector
					w = float(max(0.0, sim))
					weighted = (avec * w) if weighted is None else (weighted + avec * w)
					total_w += w
				except Exception:
					continue
			if weighted is not None and total_w > 0:
				blend = (weighted / total_w).astype(np.float32)
				alpha = float(min(0.5, max(0.15, (min(0.85, best_sim) - 0.6) * 2.0)))  # 0.15..0.5
				biased_base_vec = ((1 - alpha) * in_vec + alpha * blend).astype(np.float32)

	# Encode
	encoded = STATE.encoder.encode_text(text)
	base_vec = biased_base_vec if biased_base_vec is not None else encoded.vector
	superpos = encoded.superposition if biased_base_vec is None else STATE.encoder._init_superposition(base_vec, num_branches=3)

	# Reasoning
	updated = STATE.reasoner.step("ad_hoc_query", base_vec, superpos)
	after_interf = STATE.interference.apply(updated["branches"], updated["probabilities"])

	# Emotion updates
	probs = np.array(after_interf["probabilities"], dtype=np.float32)
	entropy = float(-np.sum((probs + 1e-9) * np.log2(probs + 1e-9)))
	certainty = float(1.0 / (1.0 + entropy))
	STATE.emotions.update_from_curiosity(min(1.0, entropy / 2.0))
	STATE.emotions.update_from_certainty(min(1.0, certainty))
	# small entanglement decay per query to avoid stale over-connection
	try:
		STATE.entanglement.decay_all(rate=0.002)
		STATE.entanglement.save()
	except Exception:
		pass

	# Collapse
	result = STATE.collapse.collapse(
		{"branches": after_interf["branches"], "probabilities": after_interf["probabilities"]},
		emotions=STATE.emotions.get_state(),
		user_profile={**user_profile, "temperature": temperature, "prompt": text},
	)

	# If entropy is still high and we didn't hit training retrieval, return clarifying suggestion
	if entropy > 2.2:
		clarify = f"Could you clarify the specific aspect of “{text}” you want? For example: definition, example, or steps."
		if isinstance(result.get("response"), str) and len(result["response"]) < 40:
			result["response"] = clarify
		else:
			result["response"] = f"{result['response']}  {clarify}"

	return jsonify({
		"response": result["response"],
		"probabilities": result["probabilities"],
		"chosen_index": int(result["chosen_index"]),
		"meta": result["meta"],
		"emotions": STATE.emotions.get_state(),
	})


@app.post("/api/developer_input")
def api_developer_input():
	payload = request.get_json(force=True, silent=True) or {}
	concept_id = str(payload.get("concept_id") or "")
	text = payload.get("text")
	concept_hint = payload.get("concept_hint")
	encoded = STATE.dev_input.encode(concept_id=concept_id, text=text, concept_hint=concept_hint)
	STATE.hm_updater.upsert_from_payload(encoded)
	return jsonify({"status": "ok", "concept_id": concept_id})


@app.get("/api/debug/hyper-matrix")
def api_debug_hyper_matrix():
	tensor = STATE.store.get_tensor()
	return jsonify({
		"tensor_shape": list(tensor.shape),
		"concept_ids": list(STATE.store.records.keys()),
	})


@app.get("/api/debug/emotions")
def api_debug_emotions():
	return jsonify(STATE.emotions.get_state())


@app.get("/api/debug/entanglements")
def api_debug_entanglements():
	edges = []
	for source, targets in getattr(STATE.entanglement, "_graph", {}).items():
		for target, ent in targets.items():
			edges.append({"source": source, "target": target, "strength": ent.strength, "phase": ent.phase})
	return jsonify({"edges": edges})


@app.post("/api/auth/register")
def api_register():
	payload = request.get_json(force=True, silent=True) or {}
	username = str(payload.get("username") or "").strip()
	password = str(payload.get("password") or "")
	role = str(payload.get("role") or "user").lower()
	admin_code = str(payload.get("admin_code") or "")
	users = STATE._read_json(STATE.users_path)
	if any(u.get("username") == username for u in users):
		return jsonify({"status": "error", "message": "username_taken"})
	if role == "developer":
		if admin_code != "ETHCO_ADMIN":
			return jsonify({"status": "error", "message": "invalid_admin_code"})
	else:
		role = "user"
	users.append({"username": username, "password_hash": _hash_pw(password), "role": role})
	STATE._write_json(STATE.users_path, users)
	token = secrets.token_hex(16)
	STATE.tokens[token] = {"username": username, "role": role}
	return jsonify({"status": "ok", "message": "registered", "token": token, "role": role})


@app.post("/api/auth/login")
def api_login():
	payload = request.get_json(force=True, silent=True) or {}
	username = str(payload.get("username") or "").strip()
	password = str(payload.get("password") or "")
	users = STATE._read_json(STATE.users_path)
	for u in users:
		if u.get("username") == username and u.get("password_hash") == _hash_pw(password):
			token = secrets.token_hex(16)
			STATE.tokens[token] = {"username": username, "role": u.get("role", "user")}
			return jsonify({"status": "ok", "message": "logged_in", "token": token, "role": u.get("role", "user")})
	return jsonify({"status": "error", "message": "invalid_credentials"})

@app.get("/api/auth/me")
def api_me():
	token = request.args.get("token")
	if not token or token not in STATE.tokens:
		return jsonify({"authenticated": False})
	info = STATE.tokens[token]
	return jsonify({"authenticated": True, "username": info.get("username"), "role": info.get("role","user")})

@app.post("/api/auth/logout")
def api_logout():
	payload = request.get_json(force=True, silent=True) or {}
	token = payload.get("token")
	if token in STATE.tokens:
		del STATE.tokens[token]
	return jsonify({"status":"ok"})
@app.get("/api/users")
def api_users_list():
	token = request.args.get("token")
	if not _require_dev(STATE, token):
		return jsonify({"status": "error", "message": "forbidden"}), 403
	users = STATE._read_json(STATE.users_path)
	return jsonify([{"username": u.get("username")} for u in users])


@app.get("/api/contacts")
def api_contacts_list():
	token = request.args.get("token")
	if not _require_dev(STATE, token):
		return jsonify({"status": "error", "message": "forbidden"}), 403
	return jsonify(STATE._read_json(STATE.contacts_path))


@app.post("/api/contacts")
def api_contacts_create():
	payload = request.get_json(force=True, silent=True) or {}
	name = str(payload.get("name") or "")
	email = str(payload.get("email") or "")
	message = str(payload.get("message") or "")
	contacts = STATE._read_json(STATE.contacts_path)
	new = {"id": len(contacts) + 1, "name": name, "email": email, "message": message}
	contacts.append(new)
	STATE._write_json(STATE.contacts_path, contacts)
	return jsonify({"status": "ok", "contact": new})

# ---------- Feedback ----------
@app.post("/api/feedback")
def api_feedback():
	"""
	Payload: { qa_id?: str, question?: str, correct: bool }
	- If qa_id provided, adjust training metadata and entanglement strengths slightly.
	- Else if question provided, try match qa_id via similarity.
	"""
	payload = request.get_json(force=True, silent=True) or {}
	correct = bool(payload.get("correct", True))
	current = STATE._read_json(STATE.training_path)
	if not isinstance(current, dict):
		return jsonify({"status":"error","message":"no_training"}), 400
	qa_list = current.get("qa", [])
	target_idx = None
	qid = payload.get("qa_id")
	if qid:
		for i, it in enumerate(qa_list):
			if it.get("id") == qid:
				target_idx = i
				break
	elif payload.get("question"):
		text = str(payload.get("question"))
		in_vec = STATE.encoder.encode_text(text).vector
		best_sim, best_i = -1.0, None
		for i, it in enumerate(qa_list):
			qv = it.get("q_vec")
			if not isinstance(qv, list): 
				continue
			qv_np = np.asarray(qv, dtype=np.float32)
			num = float(np.dot(in_vec, qv_np))
			den = float(np.linalg.norm(in_vec) * np.linalg.norm(qv_np) + 1e-9)
			sim = num / den if den > 0 else -1.0
			if sim > best_sim:
				best_sim, best_i = sim, i
		if best_i is not None:
			target_idx = best_i
	if target_idx is None:
		return jsonify({"status":"error","message":"qa_not_found"}), 404
	qa = qa_list[target_idx]
	qa.setdefault("stats", {"pos":0,"neg":0})
	if correct:
		qa["stats"]["pos"] += 1
		# reinforce entanglements
		for it in qa_list:
			if it is qa or not it.get("id"):
				continue
			try:
				STATE.entanglement.reinforce(qa["id"], it["id"], delta=0.02)
			except Exception:
				pass
	else:
		qa["stats"]["neg"] += 1
		# slight decay
		try:
			STATE.entanglement.decay_all(rate=0.01)
		except Exception:
			pass
	STATE._write_json(STATE.training_path, current)
	STATE.entanglement.save()
	return jsonify({"status":"ok","qa": qa.get("id"), "stats": qa.get("stats")})

# ---------- Training endpoints ----------
@app.post("/api/train/qa")
def api_train_qa():
	payload = request.get_json(force=True, silent=True) or {}
	q = str(payload.get("question") or "").strip()
	a = str(payload.get("answer") or "").strip()
	if not q or not a:
		return jsonify({"status":"error","message":"missing_fields"}), 400
	current = STATE._read_json(STATE.training_path)
	if isinstance(current, dict):
		dset = current.get("qa", [])
		current["qa"] = dset
	else:
		current = {"qa": [], "user_text": []}
	dset = current["qa"]
	# store vector for question to enable semantic retrieval
	q_vec = STATE.encoder.encode_text(q).vector.tolist()
	rec_id = f"qa:{len(dset)+1}"
	dset.append({"question": q, "answer": a, "q_vec": q_vec, "id": rec_id})
	STATE._write_json(STATE.training_path, current)
	# also upsert into hyper-matrix for downstream reasoning context
	try:
		STATE.upsert_training_concept(rec_id, a or q)
		STATE.update_entanglements_for_training(rec_id, dset)
	except Exception:
		pass
	return jsonify({"status":"ok","added":1,"size":len(dset)})


@app.get("/api/train/qa")
def api_train_qa_list():
	current = STATE._read_json(STATE.training_path)
	if isinstance(current, dict):
		return jsonify(current.get("qa", []))
	return jsonify([])


@app.post("/api/train/user_text")
def api_train_user_text():
	payload = request.get_json(force=True, silent=True) or {}
	text = str(payload.get("text") or "").strip()
	meta = payload.get("meta") or {}
	if not text:
		return jsonify({"status":"error","message":"missing_text"}), 400
	current = STATE._read_json(STATE.training_path)
	if isinstance(current, dict):
		ut = current.get("user_text", [])
		current["user_text"] = ut
	else:
		current = {"qa": [], "user_text": []}
	ut = current["user_text"]
	ut.append({"text": text, "meta": meta})
	STATE._write_json(STATE.training_path, current)
	return jsonify({"status":"ok","added":1,"size":len(ut)})

@app.post("/api/train/upload")
def api_train_upload():
	if 'file' not in request.files:
		return jsonify({"status":"error","message":"no_file"}), 400
	file = request.files['file']
	filename = (file.filename or "").lower()
	raw = file.read().decode('utf-8', errors='ignore')
	current = STATE._read_json(STATE.training_path)
	if not isinstance(current, dict):
		current = {"qa": [], "user_text": []}
	added = 0
	if filename.endswith('.csv'):
		import csv
		reader = csv.DictReader(raw.splitlines())
		for row in reader:
			q = (row.get('question') or '').strip()
			a = (row.get('answer') or '').strip()
			if q and a:
				q_vec = STATE.encoder.encode_text(q).vector.tolist()
				rec_id = f"qa:{len(current['qa'])+1}"
				current["qa"].append({"question": q, "answer": a, "q_vec": q_vec, "id": rec_id})
				try:
					STATE.upsert_training_concept(rec_id, a or q)
					STATE.update_entanglements_for_training(rec_id, current["qa"])
				except Exception:
					pass
				added += 1
	elif filename.endswith('.md') or filename.endswith('.markdown') or filename.endswith('.txt'):
		lines = raw.splitlines()
		q, a, collecting_a = None, "", False
		for ln in lines:
			low = ln.strip().lower()
			if low.startswith('q:') or low.startswith('### q:'):
				if q and a:
					q_vec = STATE.encoder.encode_text(q.strip()).vector.tolist()
					rec_id = f"qa:{len(current['qa'])+1}"
					current["qa"].append({"question": q.strip(), "answer": a.strip(), "q_vec": q_vec, "id": rec_id})
					try:
						STATE.upsert_training_concept(rec_id, (a or q).strip())
						STATE.update_entanglements_for_training(rec_id, current["qa"])
					except Exception:
						pass
					added += 1
				q = ln.split(':',1)[1]
				a = ""
				collecting_a = False
			elif low.startswith('a:') or low.startswith('### a:'):
				a = ln.split(':',1)[1]
				collecting_a = True
			else:
				if collecting_a:
					a += ("\n" + ln)
		if q and a:
			q_vec = STATE.encoder.encode_text(q.strip()).vector.tolist()
			rec_id = f"qa:{len(current['qa'])+1}"
			current["qa"].append({"question": q.strip(), "answer": a.strip(), "q_vec": q_vec, "id": rec_id})
			try:
				STATE.upsert_training_concept(rec_id, (a or q).strip())
				STATE.update_entanglements_for_training(rec_id, current["qa"])
			except Exception:
				pass
			added += 1
	else:
		return jsonify({"status":"error","message":"unsupported_format"}), 400
	STATE._write_json(STATE.training_path, current)
	return jsonify({"status":"ok","added":added,"total":len(current['qa'])})
@app.get("/api/posts")
def api_posts_list():
	return jsonify(STATE._read_json(STATE.posts_path))


@app.post("/api/posts")
def api_posts_create():
	token = request.args.get("token")
	if not _require_dev(STATE, token):
		return jsonify({"status": "error", "message": "forbidden"}), 403
	payload = request.get_json(force=True, silent=True) or {}
	title = str(payload.get("title") or "")
	content = str(payload.get("content") or "")
	author = str(payload.get("author") or "anon")
	posts = STATE._read_json(STATE.posts_path)
	new = {"id": len(posts) + 1, "title": title, "content": content, "author": author}
	posts.append(new)
	STATE._write_json(STATE.posts_path, posts)
	return jsonify({"status": "ok", "post": new})


if __name__ == "__main__":
	app.run(host="127.0.0.1", port=8000, debug=False)
