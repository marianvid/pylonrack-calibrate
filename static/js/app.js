/*  Model Calibrate UI — single page app.
    Communicates with the PylonRack slot over WebSocket.
    Server is on the same host & port that served this page. */

// ============================ STATE ============================

const state = {
  ws:               null,
  connected:        false,
  reconnectAttempts: 0,

  models:           [],
  selectedPaths:    new Set(),
  selectedProfiles: new Set(['single', 'throughput']),
  budget:           'standard',
  mode:             'auto',
  draftMap:         {},   // { modelPath: draftPath | null }

  available_gb:     null,
  total_gb:         null,
  warnings:         [],
  blockers:         [],

  currentSuiteId:   null,
  totalRuns:        0,
  completedRuns:    0,
  startedAt:        null,
  etaSeconds:       0,
  runs:             [],
  currentSpec:      null,

  history:          [],
};

// ============================ HELPERS ============================

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

function shortModelName(path) {
  if (!path) return '?';
  const m = state.models.find(mm => mm.full_path === path);
  if (m) return m.display_name;
  return path.split('/').pop();
}

function formatSeconds(s) {
  if (s == null || s < 0) return '—';
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function elapsedSec() {
  if (!state.startedAt) return 0;
  return Math.floor((Date.now() - state.startedAt) / 1000);
}

// ============================ CONNECTION ============================

function connect() {
  // ws_port is injected by the server-rendered config endpoint
  const wsPort = state._wsPort || 8767;
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  state.ws = new WebSocket(`${proto}//${window.location.hostname}:${wsPort}`);

  state.ws.onopen = () => {
    state.connected = true;
    state.reconnectAttempts = 0;
    setConnectionStatus('connected');
    state.ws.send(JSON.stringify({ type: 'manifest' }));
    requestAction('get_models');
    requestAction('get_resources');
    requestAction('get_history');
    if (!state._resourceTimer) {
      state._resourceTimer = setInterval(() => requestAction('get_resources', resourcesPayload()), 10000);
    }
  };

  state.ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    handleMessage(msg);
  };

  state.ws.onclose = () => {
    state.connected = false;
    setConnectionStatus('disconnected');
    if (state._resourceTimer) {
      clearInterval(state._resourceTimer);
      state._resourceTimer = null;
    }
    const delay = Math.min(1000 * Math.pow(2, state.reconnectAttempts), 10000);
    state.reconnectAttempts++;
    setTimeout(connect, delay);
  };

  state.ws.onerror = () => setConnectionStatus('error');
}

function setConnectionStatus(s) {
  const el = document.getElementById('connection-status');
  el.className = `tab-status ${s === 'connected' ? 'connected' : (s === 'error' ? 'error' : '')}`;
  el.textContent = s === 'connected'    ? 'connected'
                  : s === 'disconnected' ? 'reconnecting…'
                  : s === 'error'        ? 'error'
                  : 'connecting…';
}

function requestAction(control_id, payload) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const msg = { type: 'action', control_id };
  if (payload) msg.payload = payload;
  state.ws.send(JSON.stringify(msg));
}

// ============================ MESSAGE DISPATCH ============================

function handleMessage(msg) {
  if (msg.type === 'action_result') {
    const action = msg.action;
    const data   = msg.data || {};
    if (action === 'models')        return onModels(data);
    if (action === 'resources')     return onResources(data);
    if (action === 'history')       return onHistory(data);
    if (action === 'suite')         return onSuiteFull(data);
    if (action === 'delete_suite')  return onSuiteDeleted(data);
    if (action === 'start_suite')   return onStartSuiteResult(data);
    if (action === 'suite_event')   return onSuiteEvent(data);
  }
}

// ============================ MODELS ============================

function onModels(data) {
  state.models = data.items || [];
  renderModelsList();
}

function renderModelsList() {
  const container = document.getElementById('models-list');
  if (state.models.length === 0) {
    container.innerHTML = '<div class="empty">No GGUF models found in cache.</div>';
    return;
  }
  const sorted = [...state.models].sort((a, b) => a.size_gb - b.size_gb);
  const singleProfileActive = state.selectedProfiles.has('single');

  container.innerHTML = sorted.map(m => {
    const selected = state.selectedPaths.has(m.full_path);
    const fit = computeFit(m);
    // Draft dropdown only shown when (a) this model is selected, (b)
    // 'single' profile is active, and (c) at least one smaller candidate
    // exists in the cache. Throughput runs ignore the draft setting.
    const showDraftPicker = selected && singleProfileActive;
    const draftPickerHtml = showDraftPicker ? renderDraftPicker(m) : '';
    return `
      <div class="model-item ${selected ? 'selected' : ''} ${fit.cls}"
           data-path="${escapeAttr(m.full_path)}">
        <div class="model-checkbox">${selected ? '✓' : ''}</div>
        <div class="model-name">${escapeHtml(m.display_name)}</div>
        <div class="model-size">${m.size_gb.toFixed(1)} GB</div>
        <div class="model-status ${fit.cls}">${fit.label}</div>
      </div>
      ${draftPickerHtml}
    `;
  }).join('');

  container.querySelectorAll('.model-item').forEach(el => {
    el.addEventListener('click', () => {
      if (el.classList.contains('blocked')) return;
      const path = el.getAttribute('data-path');
      if (state.selectedPaths.has(path)) {
        state.selectedPaths.delete(path);
        delete state.draftMap[path];   // clear draft when deselecting
      } else {
        state.selectedPaths.add(path);
      }
      renderModelsList();
      renderSelectionSummary();
      requestAction('get_resources', resourcesPayload());
      renderEtaPreview();
    });
  });

  // Wire up draft dropdowns. Listen on change, do not propagate to the
  // parent .model-item click handler (which would toggle selection).
  container.querySelectorAll('.draft-picker select').forEach(sel => {
    sel.addEventListener('click', (e) => e.stopPropagation());
    sel.addEventListener('change', (e) => {
      e.stopPropagation();
      const path = sel.getAttribute('data-for');
      const val  = sel.value;
      if (val) state.draftMap[path] = val;
      else     delete state.draftMap[path];
    });
  });

  renderSelectionSummary();
}

function draftCandidatesFor(model) {
  // A model is a viable draft if it's strictly smaller (heuristic: under
  // 50% of main size). Tokenizer compatibility is the real constraint at
  // runtime — we surface all small candidates and let llama-server reject
  // mismatches (loud failure is better than silent exclusion).
  const threshold = model.size_gb * 0.5;
  return state.models.filter(m =>
    m.full_path !== model.full_path && m.size_gb < threshold
  ).sort((a, b) => a.size_gb - b.size_gb);
}

function renderDraftPicker(model) {
  const candidates = draftCandidatesFor(model);
  if (candidates.length === 0) return '';
  const current = state.draftMap[model.full_path] || '';
  const options = ['<option value="">None (no speculative decoding)</option>']
    .concat(candidates.map(c =>
      `<option value="${escapeAttr(c.full_path)}" ${c.full_path === current ? 'selected' : ''}>` +
      `${escapeHtml(c.display_name)} — ${c.size_gb.toFixed(1)} GB</option>`
    ));
  return `
    <div class="draft-picker" data-for="${escapeAttr(model.full_path)}">
      <span class="draft-label">Draft (single only):</span>
      <select data-for="${escapeAttr(model.full_path)}">${options.join('')}</select>
    </div>
  `;
}

function computeFit(model) {
  if (state.available_gb == null) return { cls: '', label: '' };
  const kv = 0.06 * model.size_gb;
  const total = model.size_gb + kv + 2.0;
  if (state.available_gb < total) return { cls: 'block', label: 'no fit' };
  if (state.available_gb - total < 4.0) return { cls: 'warn',  label: 'tight' };
  return { cls: 'ok', label: 'fits' };
}

function renderSelectionSummary() {
  const total = [...state.selectedPaths].reduce((sum, p) => {
    const m = state.models.find(mm => mm.full_path === p);
    return sum + (m ? m.size_gb : 0);
  }, 0);
  const count = state.selectedPaths.size;
  const el = document.getElementById('selection-summary');
  if (count === 0) {
    el.innerHTML = '<span class="hint">No models selected.</span>';
  } else {
    el.innerHTML = `Selected: <span class="count">${count}</span> model${count > 1 ? 's' : ''}, total ${total.toFixed(1)} GB`;
  }
}

// ============================ RESOURCES ============================

function onResources(data) {
  state.available_gb = data.available_gb;
  state.total_gb     = data.memory ? data.memory.total_gb : null;
  state.warnings     = data.warnings || [];
  state.blockers     = data.blockers || [];
  renderResourcesBar();
  if (state.models.length > 0) renderModelsList();
}

function renderResourcesBar() {
  document.getElementById('res-available').textContent =
    `${state.available_gb != null ? state.available_gb.toFixed(1) : '—'} GB available` +
    (state.total_gb != null ? ` / ${state.total_gb.toFixed(0)} GB total` : '');
  const pct = state.total_gb ? (state.available_gb / state.total_gb) * 100 : 0;
  document.getElementById('res-bar-fill').style.width = `${pct}%`;
  const warnEl = document.getElementById('res-warning');
  warnEl.textContent = state.warnings.length > 0 ? state.warnings[0] : '';
}

function resourcesPayload() {
  return {
    selected_models: [...state.selectedPaths].map(p => {
      const m = state.models.find(mm => mm.full_path === p);
      return { full_path: p, size_gb: m ? m.size_gb : 5.0 };
    }),
    ctx_size: 32768,
    parallel: 24,
  };
}

// ============================ PROFILES / BUDGET / MODE ============================

function initProfileCards() {
  document.querySelectorAll('.profile-card').forEach(card => {
    card.addEventListener('click', () => {
      const p = card.getAttribute('data-profile');
      if (state.selectedProfiles.has(p)) {
        if (state.selectedProfiles.size > 1) state.selectedProfiles.delete(p);
      } else state.selectedProfiles.add(p);
      card.classList.toggle('selected', state.selectedProfiles.has(p));
      // Re-render models list so the draft picker appears/disappears
      // depending on whether 'single' profile is now active.
      renderModelsList();
      renderEtaPreview();
    });
  });
}

function initBudgetRadios() {
  document.querySelectorAll('input[name="budget"]').forEach(r => {
    r.addEventListener('change', () => {
      if (r.checked) {
        state.budget = r.value;
        renderEtaPreview();
      }
    });
  });
}

function renderEtaPreview() {
  const el = document.getElementById('eta-preview');
  const nModels = state.selectedPaths.size;
  if (nModels === 0) { el.textContent = ''; return; }
  const combosPerProfile = state.budget === 'quick' ? 2
                          : state.budget === 'standard' ? 4.5
                          : 7;
  const profileCount  = state.selectedProfiles.size;
  const totalCombos   = Math.round(nModels * profileCount * combosPerProfile);
  const eta           = totalCombos * 30;
  el.textContent = `≈ ${totalCombos} runs · ETA ${formatSeconds(eta)}`;
}

// ============================ START / STOP ============================

function initStartButton() {
  document.getElementById('btn-start-suite').addEventListener('click', () => {
    if (state.selectedPaths.size === 0) { showToast('Select at least one model.'); return; }
    if (state.selectedProfiles.size === 0) { showToast('Select at least one profile.'); return; }
    const selected_models = [...state.selectedPaths].map(p => {
      const m = state.models.find(mm => mm.full_path === p);
      return { full_path: p, size_gb: m ? m.size_gb : 5.0 };
    });
    // Only include draft entries for models that are currently selected
    // (defensive — stale entries should already be cleared on deselect).
    const draft_map = {};
    for (const p of state.selectedPaths) {
      if (state.draftMap[p]) draft_map[p] = state.draftMap[p];
    }
    requestAction('start_suite', {
      selected_models,
      profiles:  [...state.selectedProfiles],
      budget:    state.budget,
      mode:      state.mode,
      draft_map,
    });
  });
}

function initStopButton() {
  const btn = document.getElementById('btn-stop-suite');
  btn.addEventListener('click', () => {
    // Two-step arm pattern — same reason as history-delete: WKWebView
    // does not surface confirm() dialogs without a UIDelegate.
    if (btn.classList.contains('armed')) {
      clearTimeout(btn._armTimer);
      btn.classList.remove('armed');
      btn.textContent = btn._originalLabel || 'Stop suite';
      requestAction('stop_suite');
      return;
    }
    btn._originalLabel = btn.textContent;
    btn.textContent = 'Click again to confirm';
    btn.classList.add('armed');
    btn._armTimer = setTimeout(() => {
      btn.classList.remove('armed');
      btn.textContent = btn._originalLabel || 'Stop suite';
    }, 4000);
  });
}

// ToToast — small ephemeral message at the top of the screen. WKWebView
// suppresses alert() without a UIDelegate, so we draw our own. Self-removes
// after 3 seconds.
function showToast(message, kind = 'info') {
  let host = document.getElementById('toast-host');
  if (!host) {
    host = document.createElement('div');
    host.id = 'toast-host';
    host.style.cssText = 'position:fixed;top:14px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;flex-direction:column;gap:6px;pointer-events:none;';
    document.body.appendChild(host);
  }
  const el = document.createElement('div');
  el.textContent = message;
  const bg = kind === 'error' ? '#5b1f1f' : '#1f3a2a';
  const fg = kind === 'error' ? '#ff9999' : '#9bdcb1';
  el.style.cssText = `background:${bg};color:${fg};padding:8px 14px;border-radius:6px;font-size:0.85rem;border:1px solid ${fg}33;box-shadow:0 4px 12px rgba(0,0,0,0.35);`;
  host.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function onStartSuiteResult(data) {
  if (!data.ok) {
    if (data.error === 'resources_insufficient' && data.feasibility) {
      showToast('Cannot start: ' + (data.feasibility.blockers || []).join(' '), 'error');
    } else {
      showToast('Failed to start suite: ' + (data.error || 'unknown'), 'error');
    }
    return;
  }
  state.currentSuiteId = data.suite_id;
  state.totalRuns      = data.total_runs;
  state.etaSeconds     = data.eta_seconds;
  state.startedAt      = Date.now();
  state.runs           = [];
  state.completedRuns  = 0;
  state.currentSpec    = null;
  document.getElementById('winners-card').style.display = 'none';
  renderLiveSummary();
  renderRunsTable();
  switchTab('live');
  startLiveTimer();
  document.getElementById('btn-stop-suite').disabled = false;
}

// ============================ SUITE EVENTS ============================

function onSuiteEvent(event) {
  const t = event.type;
  const d = event.data || {};
  if (t === 'suite_started') {
    state.totalRuns  = d.total_runs;
    state.etaSeconds = d.eta_seconds;
    renderLiveSummary();
  }
  else if (t === 'suite_progress') {
    state.currentSpec = d.current;
    state.completedRuns = d.run_index;
    renderLiveSummary();
    renderRunsTable();
  }
  else if (t === 'run_complete') {
    state.runs.push(d.run);
    state.completedRuns = state.runs.length;
    state.currentSpec = null;
    renderLiveSummary();
    renderRunsTable();
  }
  else if (t === 'suite_complete') {
    state.currentSpec = null;
    state.completedRuns = state.totalRuns;
    renderLiveSummary();
    renderWinners(d.winners || {});
    document.getElementById('btn-stop-suite').disabled = true;
    stopLiveTimer();
    requestAction('get_history');
  }
  else if (t === 'suite_aborted') {
    state.currentSpec = null;
    document.getElementById('btn-stop-suite').disabled = true;
    stopLiveTimer();
    document.getElementById('live-progress-text').textContent =
      `Aborted: ${d.error || 'stopped by user'}`;
    requestAction('get_history');
  }
}

// ============================ LIVE RENDERING ============================

function renderLiveSummary() {
  document.getElementById('live-suite-id').textContent = state.currentSuiteId || '—';
  document.getElementById('live-counter-progress').textContent =
    `${state.completedRuns} / ${state.totalRuns || '—'}`;
  document.getElementById('live-counter-eta').textContent =
    formatSeconds(Math.max(0, state.etaSeconds - elapsedSec()));
  document.getElementById('live-counter-elapsed').textContent =
    formatSeconds(elapsedSec());

  const pct = state.totalRuns ? (state.completedRuns / state.totalRuns) * 100 : 0;
  document.getElementById('progress-bar-fill').style.width = `${pct}%`;

  let txt = 'Idle';
  if (state.currentSpec) {
    txt = `${shortModelName(state.currentSpec.model_path)} · ${state.currentSpec.profile} · ${state.currentSpec.label}`;
  } else if (state.completedRuns >= state.totalRuns && state.totalRuns > 0) {
    txt = 'Complete';
  } else if (state.totalRuns > 0) {
    txt = 'Running…';
  }
  document.getElementById('live-progress-text').textContent = txt;
}

function renderRunsTable() {
  const tbody = document.getElementById('runs-tbody');
  const rows = state.runs.map((r, i) => renderRunRow(i, r, false));
  if (state.currentSpec && state.runs.length < state.totalRuns) {
    rows.push(renderRunRow(state.runs.length, { ...state.currentSpec, status: 'running' }, true));
  }
  tbody.innerHTML = rows.length > 0 ? rows.join('') :
    '<tr><td colspan="9" class="empty">No runs yet.</td></tr>';
}

function renderRunRow(i, r, isRunning) {
  const cls = isRunning ? 'row-running'
            : r.status === 'ok' ? 'row-ok' : 'row-fail';
  const agg = r.aggregate || {};
  const model = r.model || r.model_path || '?';
  const cellPrefill = agg.prefill_tok_s !== undefined ? `${agg.prefill_tok_s} t/s` : '—';
  const cellDecode  = agg.decode_tok_s !== undefined  ? `${agg.decode_tok_s} t/s`  : '—';
  const cellTtft    = agg.ttft_ms !== undefined ? `${Math.round(agg.ttft_ms)} ms`
                     : agg.median_ttft_ms !== undefined ? `${Math.round(agg.median_ttft_ms)} ms`
                     : '—';
  const cellAgg     = agg.aggregate_tok_s !== undefined ? `${agg.aggregate_tok_s} t/s` : '—';
  let statusHtml;
  if (isRunning)              statusHtml = '<span class="status-pill run">running</span>';
  else if (r.status === 'ok') statusHtml = '<span class="status-pill ok">ok</span>';
  else                         statusHtml = `<span class="status-pill fail">${r.status || 'fail'}</span>`;
  return `
    <tr class="${cls}">
      <td class="num">${i + 1}</td>
      <td class="model-cell" title="${escapeAttr(model)}">${escapeHtml(shortModelName(model))}</td>
      <td>${r.profile || '—'}</td>
      <td class="params">${escapeHtml(r.label || '—')}</td>
      <td class="num">${cellPrefill}</td>
      <td class="num">${cellDecode}</td>
      <td class="num">${cellTtft}</td>
      <td class="num">${cellAgg}</td>
      <td>${statusHtml}</td>
    </tr>
  `;
}

function renderWinners(winners) {
  const grid   = document.getElementById('winners-grid');
  const cardEl = document.getElementById('winners-card');
  if (!winners || Object.keys(winners).length === 0) {
    cardEl.style.display = 'none';
    return;
  }
  cardEl.style.display = 'block';
  const cards = [];
  for (const [modelPath, profiles] of Object.entries(winners)) {
    for (const [profile, w] of Object.entries(profiles)) {
      cards.push(renderWinnerCard(modelPath, profile, w));
    }
  }
  grid.innerHTML = cards.join('');
  wireCopyButtons(grid);
}

function renderWinnerCard(modelPath, profile, winner) {
  const agg = winner.aggregate || {};
  let metrics;
  if (profile === 'single') {
    metrics = `
      <div class="winner-metric">
        <div class="metric-name">Decode</div>
        <div class="metric-value">${agg.decode_tok_s || 0} t/s</div>
      </div>
      <div class="winner-metric">
        <div class="metric-name">Prefill</div>
        <div class="metric-value">${agg.prefill_tok_s || 0} t/s</div>
      </div>
      <div class="winner-metric">
        <div class="metric-name">TTFT</div>
        <div class="metric-value">${Math.round(agg.ttft_ms || 0)} ms</div>
      </div>
    `;
  } else {
    metrics = `
      <div class="winner-metric">
        <div class="metric-name">Aggregate</div>
        <div class="metric-value">${agg.aggregate_tok_s || 0} t/s</div>
      </div>
      <div class="winner-metric">
        <div class="metric-name">Per request</div>
        <div class="metric-value">${agg.per_request_decode || 0} t/s</div>
      </div>
      <div class="winner-metric">
        <div class="metric-name">TTFT (median)</div>
        <div class="metric-value">${Math.round(agg.median_ttft_ms || 0)} ms</div>
      </div>
    `;
  }
  return `
    <div class="winner-card">
      <div class="winner-head">
        <div>
          <div class="winner-profile">${profile === 'single' ? 'Single-use' : 'Throughput'}</div>
          <div class="winner-model">${escapeHtml(shortModelName(modelPath))}</div>
        </div>
      </div>
      <div class="winner-metrics">${metrics}</div>
      <div class="winner-actions">
        <button class="btn btn-copy-cmd" data-cmd="${escapeAttr(winner.command || '')}">Copy command</button>
      </div>
      <div class="winner-command">${escapeHtml(winner.command || '')}</div>
    </div>
  `;
}

function wireCopyButtons(scope) {
  scope.querySelectorAll('.btn-copy-cmd').forEach(btn => {
    btn.addEventListener('click', () => {
      const cmd = btn.getAttribute('data-cmd');
      navigator.clipboard.writeText(cmd).then(() => {
        const old = btn.textContent;
        btn.textContent = 'Copied';
        setTimeout(() => btn.textContent = old, 1500);
      });
    });
  });
}

// ============================ LIVE TIMER ============================

function startLiveTimer() {
  stopLiveTimer();
  state._liveTimer = setInterval(renderLiveSummary, 1000);
}
function stopLiveTimer() {
  if (state._liveTimer) { clearInterval(state._liveTimer); state._liveTimer = null; }
}

// ============================ HISTORY ============================

function initHistoryButtons() {
  document.getElementById('btn-refresh-history').addEventListener('click', () => {
    requestAction('get_history');
  });
  document.getElementById('btn-close-detail').addEventListener('click', () => {
    document.getElementById('history-detail').style.display = 'none';
  });
}

function onHistory(data) {
  state.history = data.suites || [];
  renderHistoryList();
}

function renderHistoryList() {
  const container = document.getElementById('history-list');
  if (state.history.length === 0) {
    container.innerHTML = '<div class="empty">No suites yet.</div>';
    return;
  }
  const rows = [...state.history].reverse().map(s => {
    const dur     = s.duration_sec ? formatSeconds(s.duration_sec) : '—';
    const started = s.started_at ? new Date(s.started_at).toLocaleString() : '—';
    return `
      <div class="history-row" data-id="${escapeAttr(s.id)}">
        <div class="history-id">${escapeHtml(s.id)}</div>
        <div class="history-meta">${escapeHtml(started)}</div>
        <div class="history-meta">${s.runs_count} runs · ${dur}</div>
        <div class="history-meta">${(s.profiles || []).join(', ')}</div>
        <button class="history-delete" data-id="${escapeAttr(s.id)}" title="Delete">×</button>
      </div>
    `;
  });
  container.innerHTML = rows.join('');

  container.querySelectorAll('.history-row').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.classList.contains('history-delete')) return;
      requestAction('get_suite', { suite_id: el.getAttribute('data-id') });
    });
  });
  container.querySelectorAll('.history-delete').forEach(el => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = el.getAttribute('data-id');
      // Two-step inline confirmation — do not use confirm() since WKWebView
      // suppresses native dialogs without a UIDelegate. First click arms
      // the button (shows checkmark, red bg); second click within 4s
      // performs the delete; clicking anywhere else cancels.
      if (el.classList.contains('armed')) {
        clearTimeout(el._armTimer);
        el.classList.remove('armed');
        requestAction('delete_suite', { suite_id: id });
        return;
      }
      el.classList.add('armed');
      const original = el.textContent;
      el.textContent = '✓';
      el._armTimer = setTimeout(() => {
        el.classList.remove('armed');
        el.textContent = original;
      }, 4000);
    });
  });
}

function onSuiteFull(data) {
  const suite = data.suite;
  if (!suite) return;
  const detail = document.getElementById('history-detail');
  detail.style.display = 'block';
  document.getElementById('history-detail-title').textContent =
    `${suite.id} · ${suite.runs.length} runs · ${formatSeconds(suite.duration_sec || 0)}`;

  const container = document.getElementById('history-detail-content');
  let html = '';

  if (suite.winners && Object.keys(suite.winners).length > 0) {
    const cards = [];
    for (const [modelPath, profiles] of Object.entries(suite.winners)) {
      for (const [profile, w] of Object.entries(profiles)) {
        cards.push(renderWinnerCard(modelPath, profile, w));
      }
    }
    html += '<div class="winners-grid">' + cards.join('') + '</div>';
  }

  html += '<div class="runs-table-wrap" style="margin-top:14px;"><table class="runs-table"><thead>' +
          '<tr><th>#</th><th>Model</th><th>Profile</th><th>Params</th><th>Prefill</th>' +
          '<th>Decode</th><th>TTFT</th><th>Aggregate</th><th>Status</th></tr></thead><tbody>';
  suite.runs.forEach((r, i) => {
    html += renderRunRow(i, r, false);
  });
  html += '</tbody></table></div>';

  container.innerHTML = html;
  wireCopyButtons(container);
}

function onSuiteDeleted(data) {
  if (data.success) {
    document.getElementById('history-detail').style.display = 'none';
    requestAction('get_history');
  }
}

// ============================ TABS ============================

function initTabs() {
  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => switchTab(t.getAttribute('data-tab')));
  });
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.getAttribute('data-tab') === name));
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('active', p.id === `tab-${name}`));
  if (name === 'history') requestAction('get_history');
}

// ============================ INIT ============================

window.addEventListener('DOMContentLoaded', async () => {
  // Fetch ws_port from the HTTP server (which knows it)
  try {
    const r = await fetch('/config');
    if (r.ok) {
      const cfg = await r.json();
      state._wsPort = cfg.ws_port;
    }
  } catch {}
  initTabs();
  initProfileCards();
  initBudgetRadios();
  initStartButton();
  initStopButton();
  initHistoryButtons();
  connect();
});
