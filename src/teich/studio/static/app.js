/* Teich Studio frontend */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function api(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  let payload = null;
  try { payload = await response.json(); } catch (e) { /* empty body */ }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

let toastTimer = null;
function toast(message, type = "") {
  const node = $("#toast");
  node.textContent = message;
  node.className = type;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 3800);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleString();
}

function get(obj, path, fallback) {
  let current = obj;
  for (const key of path.split(".")) {
    if (current == null || typeof current !== "object") return fallback;
    current = current[key];
  }
  return current == null ? fallback : current;
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

const state = {
  status: null,
  config: {},
  prompts: [],
  promptsDirty: false,
  session: null,
  sessionSource: null,
  sessionProvider: "pi",
  term: null,
  termSocket: null,
  jobSource: null,
  job: null,
  traces: [],
  selectedTrace: null,
};

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function showView(name) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  if (name === "output") loadTracesQuiet();
  if (name === "generate") refreshRunSummary();
  if (name === "interactive" && state.term) requestAnimationFrame(fitTerminal);
}

$$(".nav-item").forEach((item) => item.addEventListener("click", () => showView(item.dataset.view)));
document.addEventListener("click", (event) => {
  const goto = event.target.closest("[data-goto]");
  if (goto) { event.preventDefault(); showView(goto.dataset.goto); }
});

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async function loadStatus() {
  try {
    state.status = await api("GET", "/api/status");
  } catch (err) {
    toast(`Failed to load status: ${err.message}`, "error");
    return;
  }
  const s = state.status;
  const pathEl = $("#project-path");
  pathEl.textContent = s.project_dir;
  pathEl.title = s.project_dir;

  const docker = $("#status-docker");
  docker.classList.toggle("ok", s.docker.available);
  docker.classList.toggle("bad", !s.docker.available);
  docker.title = s.docker.available ? `Docker ${s.docker.detail}` : (s.docker.detail || "Docker unavailable");

  const key = $("#status-key");
  key.classList.toggle("ok", s.api_key_present);
  key.classList.toggle("bad", !s.api_key_present);
  key.title = s.api_key_present ? "An API key is configured" : "No API key found in config or environment";

  const badge = $("#nav-prompts-count");
  badge.hidden = !(s.prompts_count > 0);
  badge.textContent = s.prompts_count > 0 ? s.prompts_count : "";

  $("#api-key-hint").innerHTML = s.api_key_present
    ? "✓ An API key was found (config or environment)."
    : "Tip: leave empty and set <code>TEICH_API_KEY</code> / <code>OPENROUTER_API_KEY</code> in your environment instead.";
}

// ---------------------------------------------------------------------------
// Setup view
// ---------------------------------------------------------------------------

const PROVIDER_TAGS = { agent: "Docker", chat: "API only" };

function renderProviderCards() {
  const grid = $("#provider-grid");
  grid.innerHTML = "";
  const selected = get(state.config, "agent.provider", "pi");
  for (const provider of state.status.providers) {
    const card = el("div", "provider-card" + (provider.id === selected ? " selected" : ""));
    const title = el("h3", null, provider.label);
    title.appendChild(el("span", "provider-tag", PROVIDER_TAGS[provider.kind] || ""));
    card.appendChild(title);
    card.appendChild(el("p", null, provider.description));
    card.addEventListener("click", () => {
      state.config.agent = state.config.agent || {};
      state.config.agent.provider = provider.id;
      renderProviderCards();
      syncApiProviderOptions();
    });
    grid.appendChild(card);
  }
}

const API_DEFAULT_URLS = {
  openrouter: "https://openrouter.ai/api/v1",
  openai: "https://api.openai.com/v1",
  anthropic: "https://api.anthropic.com",
};
const DIRECT_ANTHROPIC_BASE_URL = "https://api.anthropic.com";

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "").toLowerCase();
}

function syncApiProviderOptions() {
  const select = $("#cfg-api-provider");
  const anthropicOption = select.querySelector('option[value="anthropic"]');
  const agentProvider = String(get(state.config, "agent.provider", "pi")).trim().toLowerCase();
  const isChat = agentProvider === "chat";
  if (anthropicOption) {
    anthropicOption.disabled = isChat;
    anthropicOption.hidden = isChat;
  }
  if (isChat && select.value === "anthropic") {
    select.value = "openrouter";
    $("#cfg-base-url").value = API_DEFAULT_URLS.openrouter;
    delete select.dataset.customValue;
  }
}

function fillConfigForm() {
  const c = state.config;
  renderProviderCards();
  $("#cfg-model").value = get(c, "model.model", "");
  $("#cfg-reasoning").value = get(c, "model.reasoning_effort", "") || "";
  const apiProvider = get(c, "api.provider", "openrouter");
  const select = $("#cfg-api-provider");
  select.value = ["openrouter", "openai", "anthropic"].includes(apiProvider) ? apiProvider : "custom";
  if (select.value === "custom") select.dataset.customValue = apiProvider;
  $("#cfg-base-url").value = get(c, "api.base_url", "") || "";
  syncApiProviderOptions();
  $("#cfg-api-key").value = get(c, "api.api_key", "") || "";
  $("#cfg-concurrency").value = get(c, "max_concurrency", 1);
  $("#cfg-timeout").value = get(c, "timeout_seconds", 600);
  $("#cfg-traces-dir").value = get(c, "output.traces_dir", "./output");
  $("#cfg-pretty-name").value = get(c, "output.pretty_name", "");
  $("#cfg-dev-instructions").value = get(c, "developer_instructions", "") || "";
  $("#cfg-repo-id").value = get(c, "publish.repo_id", "") || "";
  $("#cfg-private").checked = Boolean(get(c, "publish.private", false));
}

$("#cfg-api-provider").addEventListener("change", (event) => {
  const value = event.target.value;
  if (value !== "custom") $("#cfg-base-url").value = API_DEFAULT_URLS[value] || "";
  syncApiProviderOptions();
});

function collectConfigUpdates() {
  const apiProviderSelect = $("#cfg-api-provider");
  const apiProvider = apiProviderSelect.value === "custom"
    ? (apiProviderSelect.dataset.customValue || "openai")
    : apiProviderSelect.value;
  return {
    agent: { provider: get(state.config, "agent.provider", "pi") },
    model: {
      model: $("#cfg-model").value.trim(),
      reasoning_effort: $("#cfg-reasoning").value || null,
    },
    api: {
      provider: apiProvider,
      base_url: $("#cfg-base-url").value.trim() || null,
      api_key: $("#cfg-api-key").value.trim() || null,
    },
    output: {
      traces_dir: $("#cfg-traces-dir").value.trim() || "./output",
      pretty_name: $("#cfg-pretty-name").value.trim() || "My Agent Traces",
    },
    publish: {
      repo_id: $("#cfg-repo-id").value.trim() || null,
      private: $("#cfg-private").checked,
    },
    max_concurrency: Math.max(1, parseInt($("#cfg-concurrency").value, 10) || 1),
    timeout_seconds: Math.max(30, parseInt($("#cfg-timeout").value, 10) || 600),
    developer_instructions: $("#cfg-dev-instructions").value.trim() || null,
  };
}

function isDirectAnthropicChatConfig(updates) {
  const provider = String(updates.api.provider || "").trim().toLowerCase();
  const agentProvider = String(updates.agent.provider || "").trim().toLowerCase();
  return agentProvider === "chat"
    && (provider === "anthropic" || normalizeBaseUrl(updates.api.base_url) === DIRECT_ANTHROPIC_BASE_URL);
}

$("#btn-save-config").addEventListener("click", async () => {
  const note = $("#save-note");
  note.textContent = "";
  const updates = collectConfigUpdates();
  if (!updates.model.model) {
    toast("Please set a model ID first", "error");
    return;
  }
  if (isDirectAnthropicChatConfig(updates)) {
    toast("Chat runs need an OpenAI-compatible API. Use OpenRouter, OpenAI, or a compatible custom base URL.", "error");
    return;
  }
  try {
    const result = await api("PUT", "/api/config", { config: updates });
    state.config = result.config;
    note.textContent = "Saved to config.yaml ✓";
    note.classList.remove("error");
    setTimeout(() => { note.textContent = ""; }, 3000);
    loadStatus();
    renderProviderSeg();
  } catch (err) {
    note.textContent = err.message;
    note.classList.add("error");
  }
});

async function loadConfig() {
  try {
    const result = await api("GET", "/api/config");
    state.config = result.config || {};
    fillConfigForm();
  } catch (err) {
    toast(`Failed to load config: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Prompts view
// ---------------------------------------------------------------------------

function markPromptsDirty() {
  state.promptsDirty = true;
  $("#prompts-actions").hidden = false;
}

function renderPrompts() {
  const list = $("#prompts-list");
  list.innerHTML = "";
  $("#prompts-empty").hidden = state.prompts.length > 0;
  $("#prompts-actions").hidden = !state.promptsDirty && state.prompts.length === 0;

  state.prompts.forEach((prompt, index) => {
    const card = el("div", "prompt-card");
    const head = el("div", "prompt-head");
    head.appendChild(el("span", "prompt-index", `#${index + 1}`));
    const badges = el("div", "prompt-badges");
    if (prompt.system) badges.appendChild(el("span", "badge", "system"));
    if (prompt.github_repo) badges.appendChild(el("span", "badge", prompt.github_repo));
    if (prompt.follow_up_prompts && prompt.follow_up_prompts.length) {
      badges.appendChild(el("span", "badge", `${prompt.follow_up_prompts.length} follow-ups`));
    }
    head.appendChild(badges);
    head.appendChild(el("div", "grow"));

    const toggleBtn = el("button", "link-btn", "details");
    const deleteBtn = el("button", "link-btn danger", "delete");
    deleteBtn.addEventListener("click", () => {
      state.prompts.splice(index, 1);
      markPromptsDirty();
      renderPrompts();
    });
    head.appendChild(toggleBtn);
    head.appendChild(deleteBtn);
    card.appendChild(head);

    const textArea = el("textarea", "prompt-text");
    textArea.rows = 2;
    textArea.placeholder = "What should the agent do?";
    textArea.value = prompt.prompt || "";
    textArea.addEventListener("input", () => { prompt.prompt = textArea.value; markPromptsDirty(); });
    card.appendChild(textArea);

    const extra = el("div", "prompt-extra");
    extra.hidden = !(prompt.system || prompt.github_repo || (prompt.follow_up_prompts || []).length);

    const systemInput = el("input");
    systemInput.type = "text";
    systemInput.placeholder = "System prompt (optional)";
    systemInput.value = prompt.system || "";
    systemInput.addEventListener("input", () => {
      if (systemInput.value.trim()) prompt.system = systemInput.value; else delete prompt.system;
      markPromptsDirty();
    });

    const repoInput = el("input");
    repoInput.type = "text";
    repoInput.placeholder = "GitHub repo: owner/repo (optional)";
    repoInput.value = prompt.github_repo || "";
    repoInput.addEventListener("input", () => {
      if (repoInput.value.trim()) prompt.github_repo = repoInput.value.trim(); else delete prompt.github_repo;
      markPromptsDirty();
    });

    const followUps = el("textarea", "span-2");
    followUps.rows = 2;
    followUps.placeholder = "Follow-up prompts — one per line (optional)";
    followUps.value = (prompt.follow_up_prompts || []).join("\n");
    followUps.addEventListener("input", () => {
      const lines = followUps.value.split("\n").map((line) => line.trim()).filter(Boolean);
      if (lines.length) prompt.follow_up_prompts = lines; else delete prompt.follow_up_prompts;
      markPromptsDirty();
    });

    extra.appendChild(systemInput);
    extra.appendChild(repoInput);
    extra.appendChild(followUps);
    card.appendChild(extra);

    toggleBtn.addEventListener("click", () => { extra.hidden = !extra.hidden; });
    list.appendChild(card);
  });
}

$("#btn-add-prompt").addEventListener("click", () => {
  state.prompts.push({ prompt: "" });
  markPromptsDirty();
  renderPrompts();
  const areas = $$("#prompts-list .prompt-text");
  if (areas.length) areas[areas.length - 1].focus();
});

$("#btn-save-prompts").addEventListener("click", async () => {
  const note = $("#prompts-save-note");
  note.textContent = "";
  const rows = state.prompts.filter((p) => (p.prompt || "").trim());
  try {
    const result = await api("PUT", "/api/prompts", { prompts: rows });
    state.prompts = result.prompts;
    state.promptsDirty = false;
    renderPrompts();
    note.textContent = "Saved ✓";
    note.classList.remove("error");
    setTimeout(() => { note.textContent = ""; }, 3000);
    loadStatus();
  } catch (err) {
    note.textContent = err.message;
    note.classList.add("error");
  }
});

$("#btn-upload-prompts").addEventListener("click", () => $("#prompts-file-input").click());
$("#prompts-file-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  event.target.value = "";
  try {
    const result = await api("POST", "/api/prompts/import", {
      text,
      replace: $("#upload-replace").checked,
    });
    state.prompts = result.prompts;
    state.promptsDirty = false;
    renderPrompts();
    toast(`Imported — now ${state.prompts.length} prompts`, "success");
    loadStatus();
  } catch (err) {
    toast(`Import failed: ${err.message}`, "error");
  }
});

async function loadPrompts() {
  try {
    const result = await api("GET", "/api/prompts");
    state.prompts = result.prompts || [];
    $("#prompts-path").textContent = result.path || "";
    renderPrompts();
  } catch (err) {
    toast(`Failed to load prompts: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Generate view
// ---------------------------------------------------------------------------

function refreshRunSummary() {
  const summary = $("#run-summary");
  summary.innerHTML = "";
  const stats = [
    [get(state.config, "agent.provider", "—"), "Agent"],
    [get(state.config, "model.model", "—"), "Model"],
    [String(state.prompts.length), "Prompts"],
    [String(get(state.config, "max_concurrency", 1)), "Parallel"],
  ];
  for (const [value, label] of stats) {
    const stat = el("div", "stat");
    stat.appendChild(el("b", null, value));
    stat.appendChild(el("span", null, label));
    summary.appendChild(stat);
  }
}

function renderJob() {
  const job = state.job;
  const card = $("#run-progress-card");
  if (!job) { card.hidden = true; return; }
  card.hidden = false;

  const running = job.status === "running" || job.status === "starting";
  $("#btn-start-run").hidden = running;
  $("#btn-stop-run").hidden = !running;

  const statusEl = $("#run-status");
  statusEl.innerHTML = "";
  const chip = el("span", `chip ${job.status}`, job.status);
  if (running) chip.classList.add("pulsing");
  statusEl.appendChild(chip);
  statusEl.appendChild(el("span", null, job.message || job.error || ""));

  const rows = $("#run-rows");
  rows.innerHTML = "";
  for (const prompt of job.prompts || []) {
    const row = el("div", "run-row");
    row.appendChild(el("span", `chip ${prompt.status}`, prompt.status));
    row.appendChild(el("span", "prompt-preview", prompt.prompt_preview || ""));
    const metaParts = [];
    if (prompt.metrics && prompt.metrics.total_tokens != null) metaParts.push(`${prompt.metrics.total_tokens} tok`);
    if (prompt.metrics && prompt.metrics.total_cost != null) metaParts.push(`$${prompt.metrics.total_cost.toFixed(4)}`);
    if (prompt.trace) metaParts.push(prompt.trace);
    if (prompt.error) metaParts.push(prompt.error);
    row.appendChild(el("span", "meta", metaParts.join(" · ")));
    rows.appendChild(row);
  }
}

function connectJobEvents() {
  if (state.jobSource) state.jobSource.close();
  const source = new EventSource("/api/generate/events?after=0");
  state.jobSource = source;
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.kind === "job_status") {
      state.job = state.job || { prompts: [] };
      state.job.status = data.status;
      state.job.message = data.text || "";
      state.job.error = data.error || null;
      if (["completed", "failed", "stopped"].includes(data.status)) loadTracesQuiet();
    } else if (data.kind === "prompt_update") {
      state.job = state.job || { prompts: [] };
      const prompts = state.job.prompts;
      const existing = prompts.findIndex((p) => p.prompt_id === data.prompt_id);
      if (existing >= 0) prompts[existing] = data; else prompts.push(data);
      prompts.sort((a, b) => (a.prompt_index || 0) - (b.prompt_index || 0));
    }
    renderJob();
  };
  source.addEventListener("end", () => source.close());
}

$("#btn-start-run").addEventListener("click", async () => {
  if (!state.prompts.length) {
    toast("Add some prompts first", "error");
    showView("prompts");
    return;
  }
  try {
    state.job = await api("POST", "/api/generate", { resume: $("#run-resume").checked });
    state.job.prompts = state.job.prompts || [];
    renderJob();
    connectJobEvents();
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#btn-stop-run").addEventListener("click", async () => {
  try {
    await api("POST", "/api/generate/stop");
    toast("Stopping run — completed traces are kept");
  } catch (err) {
    toast(err.message, "error");
  }
});

async function loadCurrentJob() {
  try {
    const result = await api("GET", "/api/generate");
    if (result.job) {
      state.job = result.job;
      renderJob();
      if (["running", "starting"].includes(result.job.status)) connectJobEvents();
    }
  } catch (err) { /* no job yet */ }
}

// ---------------------------------------------------------------------------
// Interactive — session start
// ---------------------------------------------------------------------------

function renderProviderSeg() {
  const seg = $("#sess-provider-seg");
  seg.innerHTML = "";
  for (const provider of state.status.providers) {
    const button = el("button", provider.id === state.sessionProvider ? "selected" : "", provider.label);
    button.addEventListener("click", () => {
      state.sessionProvider = provider.id;
      renderProviderSeg();
      const isChat = provider.id === "chat";
      $("#sess-system-field").hidden = !isChat;
      $("#sess-repo-field").hidden = isChat;
    });
    seg.appendChild(button);
  }
  $("#sess-model").placeholder = get(state.config, "model.model", "Use configured model");
}

$("#btn-start-session").addEventListener("click", async () => {
  const provider = state.sessionProvider;
  const body = {
    provider,
    model: $("#sess-model").value.trim() || null,
    github_repo: provider === "chat" ? null : $("#sess-repo").value.trim() || null,
    system: provider === "chat" ? $("#sess-system").value.trim() || null : null,
  };
  $("#sess-note").textContent = "Launching…";
  try {
    const session = await api("POST", "/api/sessions", body);
    state.session = session;
    $("#sess-note").textContent = "";
    openSessionView(session);
  } catch (err) {
    $("#sess-note").textContent = "";
    toast(err.message, "error");
  }
});

function openSessionView(session) {
  $("#session-start").hidden = true;
  $("#session-live").hidden = false;
  const isTerminal = session.mode === "terminal";
  $("#term-window").hidden = !isTerminal;
  $("#chat-window").hidden = isTerminal;
  $("#term-banner").hidden = true;
  $("#chat-messages").innerHTML = "";
  $("#term-title-text").textContent = `${session.provider} · ${session.model} — /workspace`;
  setSessionStatus(session.status, "");
  connectSessionEvents(session.id);
  if (isTerminal) mountTerminal(session);
}

function setSessionStatus(status, message) {
  $("#sess-status-dot").className = `status-dot ${status}`;
  $("#sess-status-text").textContent = message || status;
  if (state.session) state.session.status = status;
  const busy = ["saving"].includes(status);
  $("#btn-save-session").disabled = busy;
  // chat composer state
  const sendDisabled = ["running", "starting", "saving", "finished", "discarded", "error"].includes(status);
  const sendBtn = $("#btn-send");
  if (sendBtn) sendBtn.disabled = sendDisabled;
  showTyping(["running"].includes(status) && state.session && state.session.mode === "chat");
}

// ---------------------------------------------------------------------------
// Interactive — native terminal
// ---------------------------------------------------------------------------

function fitTerminal() {
  if (state.term && state.term._fit) {
    try { state.term._fit.fit(); } catch (e) { /* not attached yet */ }
  }
}

function mountTerminal(session) {
  disposeTerminal();
  const container = $("#terminal");
  container.innerHTML = "";
  const term = new Terminal({
    fontFamily: "'Cascadia Code', 'SF Mono', Consolas, monospace",
    fontSize: 13,
    lineHeight: 1.25,
    cursorBlink: true,
    scrollback: 8000,
    theme: {
      background: "#101013",
      foreground: "#e8e8ec",
      cursor: "#f25c1a",
      cursorAccent: "#101013",
      selectionBackground: "rgba(242, 92, 26, 0.35)",
      black: "#1c1c21",
      red: "#ef5a5f",
      green: "#3ecf8e",
      yellow: "#e7bb4a",
      blue: "#6cb2f7",
      magenta: "#c792ea",
      cyan: "#5fd7d7",
      white: "#d6d6dc",
      brightBlack: "#5b5b66",
      brightRed: "#ff8085",
      brightGreen: "#62e2a8",
      brightYellow: "#ffd479",
      brightBlue: "#94c9ff",
      brightMagenta: "#dfb3ff",
      brightCyan: "#8af0f0",
      brightWhite: "#ffffff",
    },
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term._fit = fit;
  term.open(container);
  state.term = term;
  requestAnimationFrame(() => {
    fit.fit();
    connectTerminalSocket(session, term);
  });
  window.addEventListener("resize", fitTerminal);
  term.writeln("\x1b[38;5;208m⌁ teich studio\x1b[0m — starting container and launching \x1b[1m" + session.provider + "\x1b[0m …");
  term.writeln("");
}

function connectTerminalSocket(session, term) {
  const cols = term.cols || 120;
  const rows = term.rows || 32;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/api/sessions/${session.id}/term?cols=${cols}&rows=${rows}`);
  state.termSocket = socket;
  $("#term-conn").textContent = "connecting…";

  socket.onopen = () => { $("#term-conn").textContent = `${cols}×${rows} · connected`; };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "stdout") {
      term.write(message.data);
    } else if (message.type === "exit") {
      term.writeln(`\r\n\x1b[38;5;208m${message.detail || "Session ended."}\x1b[0m`);
    }
  };
  socket.onclose = () => { $("#term-conn").textContent = "disconnected"; };
  socket.onerror = () => { $("#term-conn").textContent = "connection error"; };

  term.onData((data) => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stdin", data }));
    }
  });
}

function disposeTerminal() {
  if (state.termSocket) { try { state.termSocket.close(); } catch (e) {} state.termSocket = null; }
  if (state.term) { try { state.term.dispose(); } catch (e) {} state.term = null; }
  window.removeEventListener("resize", fitTerminal);
}

function showTermBanner(text, kind) {
  const banner = $("#term-banner");
  banner.textContent = text;
  banner.className = `term-banner ${kind || ""}`;
  banner.hidden = false;
}

// ---------------------------------------------------------------------------
// Interactive — session events (SSE) + chat mode
// ---------------------------------------------------------------------------

function connectSessionEvents(sessionId) {
  if (state.sessionSource) state.sessionSource.close();
  const source = new EventSource(`/api/sessions/${sessionId}/events?after=0`);
  state.sessionSource = source;
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleSessionEvent(data);
  };
  source.addEventListener("end", () => source.close());
}

function handleSessionEvent(data) {
  const session = state.session;
  if (!session) return;
  if (data.kind === "session_status") {
    setSessionStatus(data.status, data.text || "");
    if (session.mode === "terminal") {
      if (data.status === "exited") showTermBanner(data.text || "Agent CLI exited — save the trace or discard the session.", "");
      if (data.status === "error") showTermBanner(data.text || "Session failed.", "error");
    } else if (data.text && ["starting", "ready"].includes(data.status)) {
      appendChatNode(el("div", "msg status-line", data.text));
    }
    return;
  }
  if (data.kind === "session_saved") {
    if (session.mode === "terminal") {
      showTermBanner(`Trace saved to your dataset: ${data.text}`, "success");
    } else {
      appendChatNode(el("div", "msg saved-line", `Trace saved: ${data.text}`));
    }
    return;
  }
  if (session.mode === "chat") {
    const node = renderDisplayEvent(data);
    if (node) appendChatNode(node);
  } else if (data.kind === "error") {
    showTermBanner(data.text || "Error", "error");
  }
}

function appendChatNode(node) {
  showTyping(false);
  $("#chat-messages").appendChild(node);
  if (state.session && state.session.status === "running") showTyping(true);
  scrollChat();
}

function showTyping(show) {
  let typing = $("#chat-messages .typing");
  if (show && !typing) {
    typing = el("div", "typing");
    typing.innerHTML = 'agent is working <span class="dots"><span>●</span><span>●</span><span>●</span></span>';
    $("#chat-messages").appendChild(typing);
    scrollChat();
  } else if (!show && typing) {
    typing.remove();
  }
}

function scrollChat() {
  const box = $("#chat-messages");
  box.scrollTop = box.scrollHeight;
}

function renderDisplayEvent(data) {
  const kind = data.kind;
  if (kind === "user" || kind === "assistant") {
    const msg = el("div", `msg ${kind}`);
    msg.appendChild(el("div", "msg-body", data.text));
    return msg;
  }
  if (kind === "thinking" || kind === "tool_call" || kind === "tool_result") {
    const msg = el("div", `msg block ${kind}`);
    const details = el("details");
    const summary = el("summary");
    if (kind === "thinking") {
      summary.appendChild(el("span", null, "💭 thinking"));
    } else if (kind === "tool_call") {
      summary.appendChild(el("span", null, "⚒"));
      summary.appendChild(el("span", "tool-name", data.name || "tool"));
    } else {
      summary.appendChild(el("span", null, "↳ result"));
      if (data.name) summary.appendChild(el("span", "tool-name", data.name));
    }
    const previewText = (data.text || "").replace(/\s+/g, " ").slice(0, 90);
    summary.appendChild(el("span", "muted", previewText));
    details.appendChild(summary);
    details.appendChild(el("div", "block-body", data.text || ""));
    msg.appendChild(details);
    return msg;
  }
  if (kind === "system") {
    const msg = el("div", "msg block system");
    const details = el("details");
    const summary = el("summary");
    summary.appendChild(el("span", null, "system prompt"));
    details.appendChild(summary);
    details.appendChild(el("div", "block-body", data.text || ""));
    msg.appendChild(details);
    return msg;
  }
  if (kind === "status" || kind === "log") return el("div", "msg status-line", data.text || "");
  if (kind === "error") return el("div", "msg error-line", data.text || "");
  if (kind === "session_saved") return el("div", "msg saved-line", `Trace saved: ${data.text}`);
  return null;
}

async function sendMessage() {
  const input = $("#composer-input");
  const text = input.value.trim();
  if (!text || !state.session) return;
  try {
    await api("POST", `/api/sessions/${state.session.id}/message`, { text });
    input.value = "";
  } catch (err) {
    toast(err.message, "error");
  }
}

$("#btn-send").addEventListener("click", sendMessage);
$("#composer-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

// ---------------------------------------------------------------------------
// Interactive — save / discard
// ---------------------------------------------------------------------------

$("#btn-save-session").addEventListener("click", async () => {
  if (!state.session) return;
  if (state.session.mode === "terminal" && !confirm("Save this session as a trace? The agent CLI will be stopped and the trace exported to your dataset.")) return;
  try {
    const result = await api("POST", `/api/sessions/${state.session.id}/save`);
    toast(`Trace saved: ${result.trace}`, "success");
    setTimeout(resetSessionView, 2200);
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#btn-discard-session").addEventListener("click", async () => {
  if (!state.session) return;
  if (!confirm("Discard this session? Nothing will be saved to your dataset.")) return;
  try {
    await api("POST", `/api/sessions/${state.session.id}/discard`);
  } catch (err) { /* already gone */ }
  resetSessionView();
});

function resetSessionView() {
  if (state.sessionSource) { state.sessionSource.close(); state.sessionSource = null; }
  disposeTerminal();
  state.session = null;
  $("#session-live").hidden = true;
  $("#session-start").hidden = false;
  loadTracesQuiet();
}

// ---------------------------------------------------------------------------
// Output view
// ---------------------------------------------------------------------------

async function loadTracesQuiet() { try { await loadTraces(); } catch (e) { /* ignore */ } }

async function loadTraces() {
  const result = await api("GET", "/api/traces");
  state.traces = result.traces || [];
  const badge = $("#nav-traces-count");
  badge.hidden = !state.traces.length;
  badge.textContent = state.traces.length || "";
  const list = $("#trace-list");
  list.innerHTML = "";
  if (!state.traces.length) {
    const empty = el("div", "empty-state small");
    empty.appendChild(el("div", "empty-icon", "▤"));
    empty.appendChild(el("p", null, "No traces yet. Generate a batch or save an interactive session."));
    list.appendChild(empty);
    return;
  }
  for (const trace of state.traces) {
    const item = el("button", "trace-item" + (state.selectedTrace === trace.name ? " selected" : ""));
    item.appendChild(el("div", "trace-name", trace.name));
    item.appendChild(el("div", "trace-meta", `${formatBytes(trace.size_bytes)} · ${formatTime(trace.modified_at)}`));
    item.addEventListener("click", () => previewTrace(trace.name));
    list.appendChild(item);
  }
}

async function previewTrace(name) {
  state.selectedTrace = name;
  $$("#trace-list .trace-item").forEach((item) => {
    item.classList.toggle("selected", item.querySelector(".trace-name").textContent === name);
  });
  const preview = $("#trace-preview");
  preview.innerHTML = "";
  preview.appendChild(el("div", "msg status-line", "Loading…"));
  try {
    const result = await api("GET", `/api/traces/preview?name=${encodeURIComponent(name)}`);
    preview.innerHTML = "";
    const head = el("div", "preview-head");
    head.appendChild(el("span", "badge", result.provider));
    head.appendChild(el("span", "muted", name));
    preview.appendChild(head);
    if (!result.display.length) {
      preview.appendChild(el("div", "msg status-line", "No displayable messages in this trace."));
    }
    for (const eventData of result.display) {
      const node = renderDisplayEvent(eventData);
      if (node) preview.appendChild(node);
    }
    if (result.truncated) preview.appendChild(el("div", "msg status-line", "… preview truncated"));
  } catch (err) {
    preview.innerHTML = "";
    preview.appendChild(el("div", "msg error-line", err.message));
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  await loadStatus();
  await loadConfig();
  await loadPrompts();
  state.sessionProvider = (() => {
    const provider = get(state.config, "agent.provider", "pi");
    if (["claude", "claude_code"].includes(provider)) return "claude-code";
    if (["hermes-agent", "hermes_agent"].includes(provider)) return "hermes";
    return provider;
  })();
  renderProviderSeg();
  $("#sess-system-field").hidden = state.sessionProvider !== "chat";
  $("#sess-repo-field").hidden = state.sessionProvider === "chat";
  refreshRunSummary();
  loadCurrentJob();
  loadTracesQuiet();
  setInterval(loadStatus, 60000);
}

init();
