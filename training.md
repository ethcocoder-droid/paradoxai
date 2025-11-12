Q: What is Paradox AI?
A: Paradox AI is a modular, quantum‑inspired system that answers questions using trained Q&A, retrieval, and reasoning.
Q: What is a qubit?
A: A qubit is a quantum bit that can exist in superposition until measured.
Q: What is superposition?
A: Superposition is the coexistence of multiple possible states until collapse on observation.
Q: What is entanglement?
A: Entanglement is a strong correlation between quantum systems that persists across distance.
Q: What is interference in this AI?
A: Interference blends and reweights branch probabilities based on phase and similarity to refine outcomes.
Q: What is a hyper‑matrix?
A: A hyper‑matrix stores concept vectors and their superposition branches for fast retrieval and updates.
Q: What is a vector embedding?
A: It is a numeric representation of text where semantic similarity is reflected by geometric closeness.
Q: What does “biasing the base vector” mean?
A: It means mixing the input vector with retrieved answer vectors to steer reasoning towards learned content.
Q: What is cosine similarity?
A: Cosine similarity measures the angle between vectors; 1.0 is identical direction, 0 is orthogonal.
Q: What is a clarifying question?
A: A clarifying question is a follow‑up prompt generated under high uncertainty to narrow the user’s intent.
Q: What is retrieval?
A: Retrieval is the process of searching trained Q&A and selecting the closest matches to the user query.
Q: What does ≥ 0.85 similarity do?
A: It returns the trained answer directly because the match is strong.
Q: What does 0.75–0.85 similarity do?
A: It biases the reasoning vector toward the answer without returning it verbatim.
Q: How does feedback work?
A: Feedback marks Q&A as correct or incorrect, reinforcing or decaying entanglements for future routing.
Q: What is Inceptio?
A: Inceptio models the drive to initiate exploration in the self‑awareness module.
Q: What is Equilibria?
A: Equilibria models balance between competing hypotheses and stabilizes outputs.
Q: What is Reflexion?
A: Reflexion models reflective adaptation after feedback or new evidence.
Q: What is Fluxion?
A: Fluxion models adaptability and willingness to change paths.
Q: What does “collapse” do?
A: Collapse selects a concrete response from a probability distribution informed by emotions and branches.
Q: Where is training stored?
A: Training is stored in data/training.json at the project root.
Q: How do I upload CSV training?
A: Use POST /api/train/upload with a file containing headers “question,answer”.
Q: How do I upload Markdown training?
A: Use POST /api/train/upload with a .md file that has well‑formed Q: and A: blocks.
Q: What is the Q: line?
A: It is a line that starts with “Q:” followed by the question text.
Q: What is the A: line?
A: It is a line that starts with “A:” followed by the answer text.
Q: Can I add multiple Q&A in one Markdown?
A: Yes, repeat Q: then A: blocks; each pair becomes a training item.
Q: Can the same question have multiple answers?
A: Yes; include variants with different contexts or tags in the answer text.
Q: How does the hyper‑matrix update on training?
A: Each Q&A adds a concept (qa:N) using the answer vector and updates entanglements to similar items.
Q: What is an entanglement edge?
A: It is a weighted connection between two QA concepts indicating relatedness.
Q: Why do edges decay slightly per query?
A: To prevent stale over‑connection; active paths strengthen via feedback.
Q: How do I see entanglements?
A: Call GET /api/debug/entanglements to list edges.
Q: How do I see emotions?
A: Call GET /api/debug/emotions for current emotion levels.
Q: How do I see the hyper‑matrix shape?
A: Call GET /api/debug/hyper-matrix to get tensor shape and concept IDs.
Q: What is “opt‑in user text training”?
A: It logs user prompts when the checkbox is enabled to help propose future Q&A.
Q: Where do user texts go?
A: They are appended to user_text in data/training.json with metadata.
Q: What is multi‑hop recall?
A: A blending of several top answer vectors to guide reasoning when matches are moderate.
Q: How many answers are blended?
A: Up to three top answers above a similarity floor are blended with similarity weights.
Q: Why return a clarifying prompt?
A: To reduce uncertainty when retrieval is weak and entropy remains high after reasoning.
Q: Can I personalize tone and style?
A: Yes; set tone (neutral, friendly, formal) and style (concise, detailed).
Q: What is the role system?
A: Users and developers; developers can upload bulk training and manage data.
Q: How does login work?
A: POST /api/auth/login returns a token and role; UI saves them in localStorage.
Q: How do I check my session?
A: GET /api/auth/me?token=... returns authentication and role.
Q: How do I logout?
A: POST /api/auth/logout with the token removes the session server‑side.
Q: Can I train on definitions?
A: Yes; add Q&A with definitions, which will be used in retrieval and biasing.
Q: Can I train on procedures?
A: Yes; write step‑by‑step answers; they are treated like any other text.
Q: Can I train on examples?
A: Yes; answers with examples help retrieval and improve similarity signals.
Q: What does “project root data” mean?
A: The data directory at QuantumGodAI/data where training.json and others persist now.
Q: Does the server reinitialize files on restart?
A: Only if files are missing; existing files are preserved.
Q: Why use cosine similarity?
A: It is scale‑invariant and robust for comparing embeddings.
Q: What happens on low similarity?
A: The system favors reasoning and may ask for clarification rather than guessing.
Q: How do I enable bulk upload from the UI?
A: Use the Developer Dashboard → Bulk Upload (CSV or Markdown).
Q: What is the minimum for Markdown upload?
A: At least one Q: line followed by one A: line; multiple pairs are allowed.
Q: Do headers matter in Markdown?
A: No; plain “Q:” and “A:” are sufficient; “### Q:” and “### A:” are also accepted.
Q: Does order matter in Q&A blocks?
A: Yes; each “Q:” should be followed by its corresponding “A:” block.
Q: Can I use punctuation in Q: lines?
A: Yes; the parser uses the “Q:” prefix and reads the rest of the line.
Q: Can answers span multiple lines?
A: Yes; lines after “A:” are appended until the next “Q:” block or EOF.
Q: Is there a size limit?
A: Keep files moderate; large uploads work but may be slower without an index.
Q: How do I correct a bad answer?
A: Use /api/feedback with qa_id and correct=false; then upload a better variant.
Q: Does feedback change retrieval immediately?
A: It adjusts entanglements and stats; retrieval will gradually favor better items.
Q: Can I delete a QA?
A: Not yet via API; you can edit data/training.json and restart.
Q: Can I export the training set?
A: Yes; copy data/training.json and associated data/ files.
Q: Why do I get “added: 0” on upload?
A: The file had no valid “Q:”/“A:” pairs; check formatting and retry.
Q: What is a good format for CSV?
A: A header “question,answer” and one Q&A per row.
Q: How does the system handle ties?
A: It blends top answers or uses reasoning collapse to arbitrate.
Q: Will it learn automatically from usage?
A: With opt‑in logs and feedback, it adapts by biasing and adjusting entanglements.
Q: Can I restrict training to developers?
A: Yes; only developers should use the upload interface.
Q: What is the next upgrade path?
A: Add FAISS/pgvector indexing for scalable similarity search.
Q: Does the system call an external LLM?
A: Not by default; it uses local retrieval and reasoning. An LLM adapter can be added.
Q: What if my domain is specialized?
A: Upload a tailored Q&A set; retrieval and biasing will align answers with your corpus.
Q: Can answers include lists?
A: Yes; Markdown lists are fine within the A: block.
Q: Are HTML tags supported?
A: They are treated as text; keep answers simple Markdown for consistency.
Q: What is the purpose of emotions in output?
A: Emotions shape smoothing, diversity, and how the collapse balances alternatives.
Q: Can I inspect probabilities?
A: Yes; use the API directly to see probabilities and metadata.
Q: Does the clarifying prompt always appear?
A: Only when uncertainty is high and no strong retrieval match is found.
Q: Can I train in multiple languages?
A: Yes; provide Q&A in the target language; retrieval uses embeddings, not keywords only.
Q: What is the role of entanglement among QAs?
A: It connects related items so future retrieval and biasing can traverse learned structure.
Q: Does the system forget?
A: Entanglements decay slightly per query; rarely used links weaken over time.
Q: Can I import multiple Markdown files?
A: Yes; upload them one after another; they will be appended to training.json.
Q: How do I ensure ordering?
A: Each upload appends; include IDs or timestamps in your own records if needed.
Q: Is there a limit on Q&A length?
A: No strict limit; very long answers may be truncated in UI display but stored fully.
Q: How do I view the dataset size?
A: GET /api/train/qa returns the list; your upload response also shows totals.
Q: Are duplicate questions allowed?
A: Yes; variants are distinguished by answer text and later by stats/feedback.
Q: How do I tune selection among variants?
A: Provide clearer context in the question or use feedback to guide preference.
Q: Can I attach tags to answers?
A: For now include tags textually in the answer like [beginner], [advanced].
Q: What happens if I upload the same file twice?
A: The system will append duplicates; you can deduplicate manually later.
Q: Where do I file issues?
A: Use your project tracker; include sample Q:/A: that failed and the file.
Q: How do I back up?
A: Copy the /data directory; it contains the entire learned state for this system.

