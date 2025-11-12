// Developer dashboard wiring with graceful fallback if backend is absent.
(function () {
	const statusEl = document.getElementById('status');
	const apiBaseEl = document.getElementById('apiBase');
	const saveBtn = document.getElementById('saveSettings');
	const matrixBtn = document.getElementById('refreshMatrix');
	const emotionsBtn = document.getElementById('refreshEmotions');
	const entBtn = document.getElementById('refreshEntanglements');
	const matrixView = document.getElementById('matrixView');
	const emotionsView = document.getElementById('emotionsView');
	const entView = document.getElementById('entanglementsView');
	// Admin elements
	const userName = document.getElementById('userName');
	theUserPass = document.getElementById('userPass');
	const usersView = document.getElementById('usersView');
	const contactsView = document.getElementById('contactsView');
	const postTitle = document.getElementById('postTitle');
	const postContent = document.getElementById('postContent');
	const postsView = document.getElementById('postsView');
	const teachConcept = document.getElementById('teachConcept');
	const teachText = document.getElementById('teachText');
	const teachView = document.getElementById('teachView');
	// Training elements
	const qaQ = document.getElementById('qaQ');
	const qaA = document.getElementById('qaA');
	const qaView = document.getElementById('qaView');
	// Upload
	const trainFile = document.getElementById('trainFile');
	const uploadView = document.getElementById('uploadView');
	// Session
	const roleBadge = document.getElementById('roleBadge');

	function setStatus(s) { if (statusEl) statusEl.textContent = s; }

	function getBase() {
		const saved = (window.localStorage.getItem('qgai_api_base') || '').trim();
		return saved || 'http://127.0.0.1:8000';
	}
	function setBase(v) {
		window.localStorage.setItem('qgai_api_base', v || '');
	}

	async function fetchJson(path) {
		const base = getBase();
		if (!base) throw new Error('API base not configured.');
		const headers = {};
		const token = localStorage.getItem('qgai_token');
		if(token){ headers['Authorization'] = 'Bearer ' + token; }
		// token is passed as query param for simple backend handler
		const url = path.includes('?') ? `${base}${path}&token=${encodeURIComponent(token||'')}` : `${base}${path}?token=${encodeURIComponent(token||'')}`;
		const res = await fetch(url, { headers });
		if (!res.ok) throw new Error('Fetch failed: ' + res.status);
		return await res.json();
	}

	saveBtn && saveBtn.addEventListener('click', () => {
		setBase(apiBaseEl && apiBaseEl.value || '');
		setStatus('Saved API base.');
	});

	matrixBtn && matrixBtn.addEventListener('click', async () => {
		setStatus('Loading matrix...');
		try {
			let data;
			try {
				data = await fetchJson('/api/debug/hyper-matrix');
			} catch (e) {
				data = { simulated: true, note: 'Backend not ready', tensor_shape: [0,0,0] };
			}
			matrixView.textContent = JSON.stringify(data, null, 2);
			setStatus('Done');
		} catch (e) {
			matrixView.textContent = 'Error: ' + (e.message || e);
			setStatus('Error');
		}
	});

	emotionsBtn && emotionsBtn.addEventListener('click', async () => {
		setStatus('Loading emotions...');
		try {
			let data;
			try {
				data = await fetchJson('/api/debug/emotions');
			} catch (e) {
				data = { simulated: true, Inceptio: 0.5, Equilibria: 0.5, Reflexion: 0.5, Fluxion: 0.5 };
			}
			emotionsView.textContent = JSON.stringify(data, null, 2);
			setStatus('Done');
		} catch (e) {
			emotionsView.textContent = 'Error: ' + (e.message || e);
			setStatus('Error');
		}
	});

	entBtn && entBtn.addEventListener('click', async () => {
		setStatus('Loading entanglements...');
		try {
			let data;
			try {
				data = await fetchJson('/api/debug/entanglements');
			} catch (e) {
				data = { simulated: true, edges: [] };
			}
			entView.textContent = JSON.stringify(data, null, 2);
			setStatus('Done');
		} catch (e) {
			entView.textContent = 'Error: ' + (e.message || e);
			setStatus('Error');
		}
	});

	// Initialize field from storage
	if (apiBaseEl) apiBaseEl.value = (window.localStorage.getItem('qgai_api_base') || 'http://127.0.0.1:8000');

	// Show role
	(async () => {
		try{
			const token = localStorage.getItem('qgai_token')||'';
			if(!token || !roleBadge) return;
			const res = await fetch(getBase() + '/api/auth/me?token=' + encodeURIComponent(token));
			const data = await res.json();
			if(data && data.authenticated){ roleBadge.textContent = data.role; }
		}catch(e){ /* ignore */ }
	})();

	document.getElementById('logoutBtn')?.addEventListener('click', async () => {
		try{
			const token = localStorage.getItem('qgai_token')||'';
			await fetch(getBase() + '/api/auth/logout', {
				method:'POST', headers:{'Content-Type':'application/json'},
				body: JSON.stringify({ token })
			});
			localStorage.removeItem('qgai_token');
			localStorage.removeItem('qgai_role');
			setStatus('Logged out');
		}catch(e){ setStatus('Logout error'); }
	});

	// User management
	document.getElementById('registerUser')?.addEventListener('click', async () => {
		setStatus('Registering user...');
		try{
			const res = await fetch(getBase() + '/api/auth/register', {
				method:'POST',
				headers:{'Content-Type':'application/json'},
				body: JSON.stringify({ username: (userName?.value||'').trim(), password: (theUserPass?.value||'') })
			});
			usersView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ usersView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});
	document.getElementById('listUsers')?.addEventListener('click', async () => {
		setStatus('Loading users...');
		try{
			const token = localStorage.getItem('qgai_token')||'';
			const res = await fetch(getBase() + '/api/users?token=' + encodeURIComponent(token));
			usersView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ usersView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});

	// Contacts
	document.getElementById('listContacts')?.addEventListener('click', async () => {
		setStatus('Loading contacts...');
		try{
			const token = localStorage.getItem('qgai_token')||'';
			const res = await fetch(getBase() + '/api/contacts?token=' + encodeURIComponent(token));
			contactsView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ contactsView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});

	// Blog posts
	document.getElementById('createPost')?.addEventListener('click', async () => {
		setStatus('Creating post...');
		try{
			const token = localStorage.getItem('qgai_token')||'';
			const res = await fetch(getBase() + '/api/posts?token=' + encodeURIComponent(token), {
				method:'POST',
				headers:{'Content-Type':'application/json'},
				body: JSON.stringify({ title: (postTitle?.value||'').trim(), content: (postContent?.value||'').trim(), author: (localStorage.getItem('qgai_user')||'anon') })
			});
			postsView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ postsView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});
	document.getElementById('listPosts')?.addEventListener('click', async () => {
		setStatus('Loading posts...');
		try{
			const res = await fetch(getBase() + '/api/posts');
			postsView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ postsView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});

	// Teach AI
	document.getElementById('teachBtn')?.addEventListener('click', async () => {
		setStatus('Teaching AI...');
		try{
			const res = await fetch(getBase() + '/api/developer_input', {
				method:'POST',
				headers:{'Content-Type':'application/json'},
				body: JSON.stringify({ concept_id: (teachConcept?.value||'concept'), text: (teachText?.value||'') })
			});
			teachView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ teachView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});

	// Training Q&A
	document.getElementById('qaAdd')?.addEventListener('click', async () => {
		setStatus('Adding Q&A...');
		try{
			const res = await fetch(getBase() + '/api/train/qa', {
				method:'POST',
				headers:{'Content-Type':'application/json'},
				body: JSON.stringify({ question: (qaQ?.value||'').trim(), answer:(qaA?.value||'').trim() })
			});
			qaView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ qaView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});
	document.getElementById('qaList')?.addEventListener('click', async () => {
		setStatus('Listing Q&A...');
		try{
			const res = await fetch(getBase() + '/api/train/qa');
			qaView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){ qaView.textContent='Error: '+(e.message||e); setStatus('Error'); }
	});

	// Upload training file
	document.getElementById('uploadTrain')?.addEventListener('click', async () => {
		setStatus('Uploading...');
		try{
			if(!trainFile?.files?.length){ uploadView.textContent='Select a file first.'; setStatus('Idle'); return; }
			const fd = new FormData();
			fd.append('file', trainFile.files[0]);
			const res = await fetch(getBase() + '/api/train/upload', { method:'POST', body: fd });
			uploadView.textContent = JSON.stringify(await res.json(), null, 2);
			setStatus('Done');
		}catch(e){
			uploadView.textContent = 'Error: ' + (e.message||e);
			setStatus('Error');
		}
	});
})(); 


