// Minimal front-end client for Quantum-Like God AI
// Degrades gracefully if backend is unavailable.
(function () {
	const statusEl = document.getElementById('status');
	const responseEl = document.getElementById('responseArea');
	const inputEl = document.getElementById('userInput');
	const tempEl = document.getElementById('temperature');
	const toneEl = document.getElementById('tone');
	const styleEl = document.getElementById('style');

	function getApiBase() {
		const saved = (window.localStorage.getItem('qgai_api_base') || '').trim();
		return saved || 'http://127.0.0.1:8000';
	}

	function setStatus(s) {
		if (statusEl) statusEl.textContent = s;
	}

	async function callApi(endpoint, body) {
		const base = getApiBase();
		const res = await fetch(base + endpoint, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
		});
		if (!res.ok) {
			throw new Error(`API error: ${res.status}`);
		}
		return await res.json();
	}

	async function sendQuery() {
		const text = (inputEl && inputEl.value || '').trim();
		const temperature = parseFloat(tempEl && tempEl.value || '0.7');
		const tone = (toneEl && toneEl.value) || 'neutral';
		const style = (styleEl && styleEl.value) || 'concise';
		const optIn = !!document.getElementById('optInTrain')?.checked;
		if (!text) {
			responseEl.textContent = 'Please enter a question.';
			return;
		}
		setStatus('Sending...');
		try {
			// Placeholder payload for future backend integration
			const payload = {
				query: text,
				user_profile: { tone, style },
				parameters: { temperature }
			};
			let data;
			try {
				data = await callApi('/api/query', payload);
			} catch (e) {
				// Graceful fallback while backend not ready
				data = {
					response: `Simulated response: "${text.slice(0, 60)}" ... Tone=${tone}, Style=${style}, Temp=${temperature}`,
					meta: { simulated: true }
				};
			}
			if (data && typeof data === 'object' && 'response' in data) {
				responseEl.textContent = data.response;
			} else {
				responseEl.textContent = typeof data === 'object' ? JSON.stringify(data, null, 2) : String(data);
			}
			// optional user text training
			if(optIn){
				try{
					const base = getApiBase();
					await fetch(base + '/api/train/user_text', {
						method:'POST',
						headers:{'Content-Type':'application/json'},
						body: JSON.stringify({ text, meta: { tone, style } })
					});
				}catch(_e){}
			}
			setStatus('Done');
		} catch (err) {
			responseEl.textContent = `Error: ${err.message || err}`;
			setStatus('Error');
		}
	}

	async function submitQA(){
		const q = (document.getElementById('teachQ')?.value||'').trim();
		const a = (document.getElementById('teachA')?.value||'').trim();
		const out = document.getElementById('teachStatus');
		if(!q || !a){ out.textContent='Please provide both question and answer.'; return; }
		setStatus('Submitting Q&A...');
		try{
			const base = getApiBase();
			const res = await fetch(base + '/api/train/qa', {
				method:'POST',
				headers:{'Content-Type':'application/json'},
				body: JSON.stringify({ question:q, answer:a })
			});
			out.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){
			out.textContent = 'Error: ' + (e.message||e);
			setStatus('Error');
		}
	}

	async function loadPosts(){
		setStatus('Loading posts...');
		try{
			if(!API_BASE) throw new Error('Backend API not configured yet');
			const res = await fetch(API_BASE + '/api/posts');
			if(!res.ok) throw new Error('API error: ' + res.status);
			const posts = await res.json();
			const container = document.getElementById('blogFeed');
			if(Array.isArray(posts) && posts.length){
				container.innerHTML = posts.map(p => `<div class="panel"><h3>${p.title}</h3><p>${p.content}</p><small>by ${p.author||'anon'}</small></div>`).join('');
			}else{
				container.textContent = 'No posts yet.';
			}
			setStatus('Done');
		}catch(e){
			const container = document.getElementById('blogFeed');
			container.textContent = 'Error: ' + (e.message || e);
			setStatus('Error');
		}
	}

	window.QGAI = {
		sendQuery,
		loadPosts,
		submitQA
	};
})();


