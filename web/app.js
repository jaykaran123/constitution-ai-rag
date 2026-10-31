'use strict';
const $ = s => document.querySelector(s);
const chat = $('#chat'), messages = $('#messages'), welcome = $('#welcome'), input = $('#input');
const TOKEN_KEY = 'constitution-ai-session-token', LEGACY_KEY = 'constitution-ai-session';
let sessionToken = null, sessionPromise = null, conversationId = null, busy = false, controller = null, currentUser = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}
function store(key, value) { try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch {} }
function load(key) { try { return localStorage.getItem(key); } catch { return null; } }

/* ------------------------------------------------------------------ session */
async function ensureSession() {
  if (sessionToken) return sessionToken;
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const existing = load(TOKEN_KEY), legacy = load(LEGACY_KEY);
      const url = '/api/session' + (!existing && legacy ? '?legacy_session_id=' + encodeURIComponent(legacy) : '');
      let r = await fetch(url, {method: 'POST', headers: existing ? {'X-Session-Token': existing} : {}});
      if (!r.ok && existing?.startsWith('u_')) {
        store(TOKEN_KEY, null);
        r = await fetch('/api/session', {method: 'POST'});
      }
      if (!r.ok) throw Error('Could not start a session.');
      const data = await r.json();
      if (!data.session_token) throw Error('The server did not issue a session token.');
      sessionToken = data.session_token;
      store(TOKEN_KEY, sessionToken);
      return sessionToken;
    })();
  }
  try { return await sessionPromise; }
  finally { sessionPromise = null; }
}
async function api(path, options = {}) {
  const token = await ensureSession();
  const headers = {...(options.headers || {}), 'X-Session-Token': token};
  let r = await fetch(path, {...options, headers});
  if (r.status === 401) {               // stale token (e.g. secret changed): renew once and retry
    if (token.split('.')[0].startsWith('u_')) {
      store(TOKEN_KEY, null); sessionToken = null; currentUser = null; updateAuthButton();
      showAuth('login');
      throw Error('Your sign-in expired. Please sign in again.');
    }
    if (sessionToken === token) { store(TOKEN_KEY, null); sessionToken = null; }
    const freshToken = await ensureSession();
    r = await fetch(path, {...options, headers: {...headers, 'X-Session-Token': freshToken}});
  }
  return r;
}

function updateAuthButton() {
  const button = $('#authButton');
  if (!button) return;
  button.textContent = currentUser ? 'Sign out' : 'Log in';
  button.title = currentUser ? `Signed in as ${currentUser.email}` : 'Sign in or create an account';
}
function setAuthMode(mode) {
  const registering = mode === 'register';
  $('#authDialog').dataset.mode = mode;
  $('#authTitle').textContent = registering ? 'Create account' : 'Sign in';
  $('#authSubmit').textContent = registering ? 'Create account' : 'Sign in';
  $('#authHelp').textContent = registering ? 'Use at least 8 characters for your password.' : 'Sign in to access your saved conversations.';
  $('#authPassword').minLength = registering ? 8 : 1;
  $('#authPassword').autocomplete = registering ? 'new-password' : 'current-password';
  $('#showLogin').classList.toggle('active', !registering);
  $('#showRegister').classList.toggle('active', registering);
  $('#authError').classList.add('hidden');
}
function showAuth(mode = 'login') {
  setAuthMode(mode);
  if (!$('#authDialog').open) $('#authDialog').showModal();
}
async function loadAccount() {
  try {
    const r = await api('/api/auth/me');
    const data = r.ok ? await r.json() : {authenticated: false};
    currentUser = data.authenticated ? data.user : null;
    updateAuthButton();
  } catch {}
}
async function submitAuth(event) {
  event.preventDefault();
  const mode = $('#authDialog').dataset.mode || 'login';
  const submit = $('#authSubmit'), error = $('#authError');
  submit.disabled = true; error.classList.add('hidden');
  try {
    const token = await ensureSession();
    const response = await fetch(`/api/auth/${mode}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Session-Token': token},
      body: JSON.stringify({email: $('#authEmail').value, password: $('#authPassword').value}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw Error(data.detail || `Account request failed (${response.status}).`);
    sessionToken = data.session_token;
    store(TOKEN_KEY, sessionToken);
    currentUser = data.user;
    updateAuthButton();
    $('#authDialog').close();
    $('#authForm').reset();
    conversationId = null; messages.replaceChildren(); welcome.classList.remove('hidden');
    await loadHistory();
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove('hidden');
  } finally { submit.disabled = false; }
}

/* ------------------------------------------------------------------ markdown (safe subset) */
function inline(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])_(.+?)_(?=[\s.,;:)]|$)/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[((?:[CSU]\d+)(?:\s*,\s*[CSU]\d+)*)\]/g, (_, ids) =>
      ids.split(/\s*,\s*/).map(id => `<a href="#" class="cite" data-ref="${id}">${id}</a>`).join(''));
}
function renderMarkdown(md) {
  const lines = String(md || '').replace(/\r/g, '').split('\n');
  let html = '', list = null, quote = [];
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const closeQuote = () => { if (quote.length) { html += `<blockquote>${quote.map(inline).join('<br>')}</blockquote>`; quote = []; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if (/^>\s?/.test(line.trim())) { closeList(); quote.push(line.trim().replace(/^>\s?/, '')); continue; }
    closeQuote();
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { closeList(); html += `<h4>${inline(m[2])}</h4>`; }
    else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { if (list !== 'ul') { closeList(); html += '<ul>'; list = 'ul'; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { if (list !== 'ol') { closeList(); html += '<ol>'; list = 'ol'; } html += `<li>${inline(m[1])}</li>`; }
    else if (!line.trim()) { closeList(); }
    else { closeList(); html += `<p>${inline(line)}</p>`; }
  }
  closeQuote(); closeList();
  return html;
}

/* ------------------------------------------------------------------ rendering */
const STATUS = {
  answered: ['ok', 'Citations verified'],
  answered_with_warnings: ['warn', 'Citations verified with warnings'],
  unverified: ['bad', 'AI draft failed citation checks; showing source-only answer'],
  degraded: ['warn', 'AI unavailable; answer built only from verified sources'],
  abstained: ['info', 'More information needed'],
  error: ['bad', 'Error'],
};
const VIOLATION = {
  unlikely_on_stated_facts: 'Constitutional violation unlikely on these facts',
  possible_but_cannot_be_determined_without_more_facts: 'Possible, but needs more facts',
  depends_on_whether_the_actor_is_the_state: "Depends on whether the actor is 'the State'",
  not_a_violation_question: 'General information',
  cannot_be_determined: 'Cannot be determined',
};
const DOMAIN = {ordinary_law: 'Primarily ordinary law', constitutional: 'Constitutional', mixed: 'Constitutional and ordinary law', general_information: 'Information'};
const ACTOR = {state: 'State actor', private: 'Private actor', mixed: 'State and private actors', unknown: 'Actor unclear'};

function citationHtml(c) {
  const verified = c.verification_status === 'verified';
  const where = [c.page ? `PDF p. ${c.page}` : '', c.printed_page ? `printed p. ${c.printed_page}` : ''].filter(Boolean).join(' · ');
  const amend = (c.amendments || []).filter(a => a.amendment_act).map(a => `${esc(a.action || 'Amended')} by ${esc(a.amendment_act)}${a.effective_from ? ' (w.e.f. ' + esc(a.effective_from) + ')' : ''}`).join('<br>');
  const statusTag = c.status === 'omitted' ? '<span class="tag bad">Omitted provision</span>' : c.status === 'amended' ? '<span class="tag">Amended</span>' : '';
  return `<div class="citation${c.cited_in_answer ? ' cited' : ''}" id="cite-${esc(c.id)}" data-ref="${esc(c.id)}">
    <b><span class="cid">${esc(c.id)}</span> ${esc(c.source_location || c.source_title)}${c.title ? ' — ' + esc(c.title) : ''}</b>
    <div class="meta">${esc(c.source_title)}${c.version ? ' · ' + esc(c.version) : ''}${where ? ' · ' + where : ''}</div>
    <div class="tags"><span class="tag ${verified ? 'ok' : 'warn'}">${verified ? 'Verified source text' : c.kind === 'upload' ? 'Your upload (not authoritative)' : 'Text not verified'}</span>
      <span class="tag">${c.quote_type === 'exact_source_text' ? 'Exact source text' : 'Document text'}</span>${statusTag}</div>
    <div class="excerpt">${esc(c.excerpt)}</div>
    ${amend ? `<div class="meta amend">${amend}</div>` : ''}
    <div class="meta">${esc(c.relevance_reason || '')}</div>
    ${c.source_url ? `<a href="${esc(c.source_url)}" target="_blank" rel="noreferrer">Official source ↗</a>` : ''}
  </div>`;
}

function renderAssistant(wrap, data) {
  const [kind, label] = STATUS[data.status] || ['info', data.status || ''];
  const a = data.assessment || {};
  const chips = [];
  if (a.constitutional_violation && VIOLATION[a.constitutional_violation]) chips.push(VIOLATION[a.constitutional_violation]);
  if (a.primary_domain && DOMAIN[a.primary_domain] && a.primary_domain !== 'general_information') chips.push(DOMAIN[a.primary_domain]);
  if (a.actor_type && ACTOR[a.actor_type]) chips.push(ACTOR[a.actor_type]);
  const conf = data.confidence && data.confidence.level ? `<span class="chip conf ${esc(data.confidence.level)}" title="${esc(data.confidence.basis || '')}">Confidence: ${esc(data.confidence.level)}</span>` : '';
  const citations = data.citations || [];
  const cited = citations.filter(c => c.cited_in_answer), other = citations.filter(c => !c.cited_in_answer);
  let html = `<div class="answer-label">CONSTITUTION AI</div>
    <div class="status-row"><span class="status ${kind}">${esc(label)}</span>${chips.map(c => `<span class="chip">${esc(c)}</span>`).join('')}${conf}</div>
    <div class="bubble md">${renderMarkdown(data.content)}</div>`;
  if (data.unverified_draft) html += `<details class="draft"><summary>Show the unverified AI draft (failed citation checks)</summary><div class="md">${renderMarkdown(data.unverified_draft)}</div></details>`;
  if (data.validation && (data.validation.errors || []).length && data.status !== 'unverified') html += `<div class="validation-note">${esc(data.validation.errors.join(' '))}</div>`;
  if (citations.length) {
    html += `<details class="citations" ${cited.length ? 'open' : ''}><summary>${cited.length || citations.length} cited source${(cited.length || citations.length) === 1 ? '' : 's'}${other.length && cited.length ? ` · ${other.length} more retrieved` : ''}</summary>
      ${(cited.length ? cited : citations).map(citationHtml).join('')}
      ${cited.length && other.length ? `<details class="more"><summary>Other retrieved evidence</summary>${other.map(citationHtml).join('')}</details>` : ''}</details>`;
    html += `<div class="answer-actions"><button class="copy">Copy answer</button><button class="graph-btn">View legal connections</button></div><div class="graph hidden"></div>`;
  }
  wrap.innerHTML = html;
  wrap.querySelectorAll('.cite').forEach(link => link.addEventListener('click', e => {
    e.preventDefault();
    const box = wrap.querySelector('.citations'); if (box) box.open = true;
    const target = wrap.querySelector(`.citation[data-ref="${link.dataset.ref}"]`);
    if (target) { target.closest('details.more')?.setAttribute('open', ''); target.scrollIntoView({behavior: 'smooth', block: 'center'}); target.classList.add('flash'); setTimeout(() => target.classList.remove('flash'), 1400); }
  }));
  wrap.querySelector('.copy')?.addEventListener('click', () => navigator.clipboard.writeText(data.content));
  wrap.querySelector('.graph-btn')?.addEventListener('click', async () => {
    const box = wrap.querySelector('.graph');
    if (!box.classList.contains('hidden')) { box.classList.add('hidden'); return; }
    const ids = [...new Set(citations.filter(c => c.kind === 'constitution' && (c.cited_in_answer || !cited.length)).map(c => c.provision_id))].slice(0, 3);
    try {
      const r = await fetch('/api/knowledge-graph/related?provisions=' + encodeURIComponent(ids.join(',')));
      box.innerHTML = renderLegalGraph(await r.json());
    } catch { box.textContent = 'Legal connections could not be loaded.'; }
    box.classList.remove('hidden');
  });
}

function addMessage(role, text, data) {
  welcome.classList.add('hidden');
  const wrap = document.createElement('div');
  wrap.className = 'message ' + role;
  if (role === 'assistant') renderAssistant(wrap, {...(data || {}), content: text});
  else wrap.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  messages.append(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

/* ------------------------------------------------------------------ chat (SSE streaming) */
async function send(text = input.value) {
  text = text.trim();
  if (!text || busy) return;
  busy = true; $('#send').disabled = true;
  $('#notice').classList.add('hidden');
  addMessage('user', text);
  input.value = ''; input.style.height = 'auto';
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.innerHTML = '<div class="answer-label">CONSTITUTION AI</div><div class="loading">Understanding your question…</div><div class="bubble md streaming hidden"></div>';
  messages.append(wrap); chat.scrollTop = chat.scrollHeight;
  const loading = wrap.querySelector('.loading'), bubble = wrap.querySelector('.bubble');
  let draft = '', finished = false;
  const stageText = {understanding: 'Understanding your question…', retrieved: 'Reading the relevant constitutional text…', generating: 'Writing an answer…', validating: 'Checking every citation against the source…'};
  try {
    controller = new AbortController();
    const r = await api('/api/chat/stream', {method: 'POST', signal: controller.signal, headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, conversation_id: conversationId})});
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw Error(d.detail || `Request failed (${r.status}).`); }
    const reader = r.body.getReader(), decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let index;
      while ((index = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, index); buffer = buffer.slice(index + 2);
        const event = (block.match(/^event: (.*)$/m) || [])[1], payload = (block.match(/^data: (.*)$/m) || [])[1];
        if (!event || !payload) continue;
        const data = JSON.parse(payload);
        if (event === 'meta') conversationId = data.conversation_id;
        else if (event === 'stage') loading.textContent = stageText[data.stage] || loading.textContent;
        else if (event === 'token') {
          draft += data.text; bubble.classList.remove('hidden');
          bubble.innerHTML = renderMarkdown(draft) + '<div class="draft-flag">Draft — citations are being checked</div>';
          chat.scrollTop = chat.scrollHeight;
        } else if (event === 'final') {
          finished = true; conversationId = data.conversation_id; renderAssistant(wrap, data);
        } else if (event === 'error') throw Error(data.detail || 'The answer could not be completed.');
      }
    }
    if (!finished) throw Error('The connection closed before the answer was validated. Please try again.');
    loadHistory();
  } catch (e) {
    if (!finished) {
      wrap.remove();
      $('#notice').textContent = e.name === 'AbortError' ? 'Request cancelled.' : e.message;
      $('#notice').classList.remove('hidden');
    }
  } finally {
    busy = false; $('#send').disabled = false; controller = null;
    chat.scrollTop = chat.scrollHeight;
  }
}

/* ------------------------------------------------------------------ history, status, uploads */
async function loadHistory() {
  try {
    const r = await api('/api/conversations');
    if (!r.ok) return;
    const rows = await r.json(), holder = $('#history');
    holder.replaceChildren();
    if (!rows.length) { holder.innerHTML = '<div class="empty-history">Your conversations will appear here.</div>'; return; }
    for (const row of rows) {
      const b = document.createElement('button');
      b.className = 'history-item'; b.textContent = row.title; b.title = row.title;
      b.onclick = async () => {
        const detail = await (await api('/api/conversations/' + encodeURIComponent(row.id))).json();
        conversationId = detail.id; messages.replaceChildren();
        welcome.classList.toggle('hidden', detail.messages.length > 0);
        for (const m of detail.messages) addMessage(m.role, m.content, m);
        $('#sidebar').classList.remove('open'); $('#scrim').classList.remove('show');
      };
      holder.append(b);
    }
  } catch {}
}

async function status() {
  try {
    const s = await (await fetch('/api/sources/status')).json();
    const d = s.details || {};
    $('#sideStatus').textContent = s.is_available ? (s.is_verified ? 'Verified official edition' : 'Installed · verification pending') : 'Source PDF not installed';
    $('#sideStatus').previousElementSibling?.classList?.toggle('ready', s.is_available);
    document.querySelectorAll('.source-pill .dot, .source-mini .dot').forEach(dot => dot.classList.toggle('ready', s.is_available && s.is_verified));
    $('#sourceDetails').innerHTML = `<b>${esc(s.name)}</b><br>Status: ${esc(d.status)}<br>Edition: ${esc(d.version || 'Not verified')}<br>
      Publisher: ${esc(d.source_authority || '')}<br>Verified: ${s.is_verified ? 'Yes' : 'No'}${d.last_verified_at ? ' (checked ' + esc(d.last_verified_at.slice(0, 10)) + ')' : ''}<br>
      Articles indexed: ${esc(d.articles)} · provisions: ${esc(d.provisions)} · passages: ${esc(s.total_chunks)} · with embeddings: ${esc(d.embedded_chunks)}<br>
      Amendment status: ${esc(d.amendment_status || 'unknown')}<br>
      Checksum (SHA-256): <code>${esc((d.checksum_sha256 || '').slice(0, 16))}…</code><br><br>${esc(d.setup)}`;
    $('#notice').textContent = s.is_available ? (s.is_verified ? '' : 'Constitution text is installed, but its edition could not be verified.') :
      'The Constitution source is not installed yet. Answers require installed source text.';
    $('#notice').classList.toggle('hidden', s.is_available && s.is_verified);
  } catch { $('#sideStatus').textContent = 'Backend unavailable'; }
}

$('#send').onclick = () => send();
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 150) + 'px'; });
document.querySelectorAll('.prompt').forEach(b => b.onclick = () => { input.value = b.dataset.prompt; send(); });
$('#newChat').onclick = () => { conversationId = null; messages.innerHTML = ''; welcome.classList.remove('hidden'); input.focus(); };
$('#authButton').onclick = async () => {
  if (!currentUser) { showAuth('login'); return; }
  try { await api('/api/auth/logout', {method: 'POST'}); } catch {}
  store(TOKEN_KEY, null); sessionToken = null; sessionPromise = null; currentUser = null;
  conversationId = null; messages.replaceChildren(); welcome.classList.remove('hidden');
  updateAuthButton();
  try { await ensureSession(); await loadHistory(); }
  catch (err) { $('#notice').textContent = err.message; $('#notice').classList.remove('hidden'); }
};
$('#showLogin').onclick = () => setAuthMode('login');
$('#showRegister').onclick = () => setAuthMode('register');
$('#authForm').addEventListener('submit', submitAuth);
$('#closeAuth').onclick = () => $('#authDialog').close();
$('#settingsBtn').onclick = () => $('#settings').showModal();
$('#sourcePill').onclick = () => $('#settings').showModal();
$('#closeSettings').onclick = () => $('#settings').close();
$('#refreshStatus').onclick = status;
$('#menuBtn').onclick = () => { $('#sidebar').classList.add('open'); $('#scrim').classList.add('show'); };
$('#closeSide').onclick = $('#scrim').onclick = () => { $('#sidebar').classList.remove('open'); $('#scrim').classList.remove('show'); };
$('#upload').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  $('#attachName').textContent = 'Uploading ' + f.name + '…';
  const data = new FormData(); data.append('file', f);
  try {
    const r = await api('/api/documents/upload', {method: 'POST', body: data}), d = await r.json();
    if (!r.ok) throw Error(d.detail || 'Upload failed.');
    $('#attachName').textContent = `${d.title} · ${d.ingestion_status} · not treated as authoritative`;
    input.value = `Please consider my uploaded document (${d.title}) along with relevant constitutional provisions.`;
  } catch (err) { $('#attachName').textContent = err.message; }
  e.target.value = '';
};
ensureSession().then(() => { loadHistory(); loadAccount(); }).catch(() => {});
updateAuthButton();
status();
