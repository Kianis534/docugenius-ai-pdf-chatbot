/*
 * DocuGenius frontend.
 * Vanilla JavaScript with no build step. Every API response is rendered through
 * escapeHtml() or DOMPurify, so untrusted text (PDF content, Reddit posts, model
 * output) can never inject markup.
 */
(() => {
  "use strict";

  // ------------------------------------------------------------------------
  // Configuration
  // ------------------------------------------------------------------------
  // Same-origin by default. When index.html is opened from disk, fall back to a
  // local server. Set window.DOCUGENIUS_API_BASE before this script to override.
  const API_BASE = String(
    window.DOCUGENIUS_API_BASE ?? (location.protocol === "file:" ? "http://localhost:7860" : "")
  ).replace(/\/+$/, "");

  const STORAGE_KEYS = { theme: "docugenius-theme", doc: "docugenius-document" };
  const MAX_CHAT_HISTORY = 8;
  const GHOST_LIMIT = 5000;

  const state = { doc: null, chatHistory: [], tables: [], lastCapture: "", maxUploadMb: 20 };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  // ------------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------------
  const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);

  const storage = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* ignore */ } },
    remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
  };

  if (window.marked) window.marked.use({ gfm: true, breaks: true });
  if (window.DOMPurify) {
    window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
      if (node.tagName === "A") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      }
    });
  }

  function renderMarkdown(text) {
    const source = String(text ?? "");
    if (window.marked && window.DOMPurify) {
      return window.DOMPurify.sanitize(window.marked.parse(source));
    }
    return escapeHtml(source).replace(/\n/g, "<br>");
  }

  function debounce(fn, wait) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function describeError(detail, status) {
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail) && detail.length) {
      return detail
        .map((item) => {
          const field = (item.loc || []).filter((part) => part !== "body").join(".");
          return field ? `${field}: ${item.msg}` : item.msg;
        })
        .join("; ");
    }
    return `Request failed (HTTP ${status}).`;
  }

  async function api(path, { method = "GET", json, form, timeout = 90_000 } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    let response;
    try {
      response = await fetch(API_BASE + path, {
        method,
        headers: json ? { "Content-Type": "application/json" } : undefined,
        body: json ? JSON.stringify(json) : form,
        signal: controller.signal,
      });
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The request timed out. Please try again.");
      throw new Error("Can't reach the DocuGenius server. Check that it is running.");
    } finally {
      clearTimeout(timer);
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(describeError(data.detail, response.status));
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function toast(message, variant = "success") {
    const element = document.createElement("div");
    element.className = `toast align-items-center text-bg-${variant} border-0`;
    element.setAttribute("role", variant === "danger" ? "alert" : "status");
    element.setAttribute("aria-live", variant === "danger" ? "assertive" : "polite");
    const closeClass = variant === "warning" || variant === "light" ? "" : "btn-close-white";
    element.innerHTML = `<div class="d-flex"><div class="toast-body"></div>
      <button type="button" class="btn-close ${closeClass} me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div>`;
    element.querySelector(".toast-body").textContent = message;
    $("#toastContainer").appendChild(element);
    element.addEventListener("hidden.bs.toast", () => element.remove());
    window.bootstrap.Toast.getOrCreateInstance(element, { delay: 4000 }).show();
  }

  async function withBusy(button, task) {
    if (button.disabled) return undefined;
    const original = button.innerHTML;
    button.style.minWidth = `${button.offsetWidth}px`;
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" aria-hidden="true"></span><span class="visually-hidden">Working…</span>';
    try {
      return await task();
    } finally {
      button.disabled = false;
      button.innerHTML = original;
      button.style.minWidth = "";
    }
  }

  // ------------------------------------------------------------------------
  // Output panels
  // ------------------------------------------------------------------------
  const ACTIONS_HTML = `<div class="output-actions">
      <button type="button" class="btn btn-sm btn-outline-secondary" data-action="copy" title="Copy"><i class="bi bi-clipboard"></i></button>
      <button type="button" class="btn btn-sm btn-outline-secondary" data-action="clear" title="Clear"><i class="bi bi-x-lg"></i></button>
    </div>`;

  function setLoading(output, text) {
    delete output.dataset.raw;
    output.innerHTML = `<div class="loading-line"><span class="spinner-border spinner-border-sm" aria-hidden="true"></span>${escapeHtml(text)}</div>`;
  }

  function setError(output, message) {
    delete output.dataset.raw;
    output.innerHTML = `<div class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(message)}</div>`;
  }

  function setContent(output, { markdown, text, raw, extra = "" }) {
    const body = markdown !== undefined
      ? `<div class="markdown">${renderMarkdown(markdown)}</div>`
      : `<div style="white-space: pre-wrap">${escapeHtml(text)}</div>`;
    output.innerHTML = ACTIONS_HTML + body + extra;
    output.dataset.raw = raw ?? markdown ?? text ?? "";
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    toast("Copied to clipboard");
  }

  document.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]");
    if (action) {
      const output = action.closest(".output");
      if (!output) return;
      if (action.dataset.action === "copy") copyText(output.dataset.raw || output.innerText);
      if (action.dataset.action === "clear") {
        output.innerHTML = "";
        delete output.dataset.raw;
      }
      return;
    }
    const tableButton = event.target.closest("[data-copy-table]");
    if (tableButton) copyText(state.tables[Number(tableButton.dataset.copyTable)]?.markdown || "");
  });

  function addChatNote(text) {
    const messages = $("#chatMessages");
    if (!messages) return;
    const note = document.createElement("div");
    note.className = "chat-message bot chat-note";
    note.textContent = text;
    messages.appendChild(note);
    messages.scrollTop = messages.scrollHeight;
  }

  // Citations: pages for PDFs, sections for Word, text and Markdown documents.
  const locationLabel = (source) => (source.page ? `p. ${source.page}` : source.section ? `§ ${source.section}` : "excerpt");

  function citationSummary(sources) {
    return [...new Set((sources || []).filter((s) => s.page || s.section).map(locationLabel))].join(", ");
  }

  function sourcesHtml(sources) {
    if (!sources || !sources.length) return "";
    const cited = citationSummary(sources);
    const label = cited ? `Sources · ${cited}` : "Sources";
    const items = sources
      .map((s) => `<div class="source-item"><span class="page-chip">${escapeHtml(locationLabel(s))}</span>${escapeHtml(s.snippet)}</div>`)
      .join("");
    return `<details class="sources"><summary>${escapeHtml(label)}</summary>${items}</details>`;
  }

  // ------------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------------
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-bs-theme", theme);
    const icon = $("#themeToggle i");
    if (icon) icon.className = theme === "dark" ? "bi bi-sun" : "bi bi-moon-stars";
  }

  function initTheme() {
    applyTheme(document.documentElement.getAttribute("data-bs-theme") || "light");
    $("#themeToggle").addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      storage.set(STORAGE_KEYS.theme, next);
    });
    const params = new URLSearchParams(location.search);
    if (params.get("theme") === "dark" || params.get("theme") === "light") applyTheme(params.get("theme"));
  }

  // ------------------------------------------------------------------------
  // Service status
  // ------------------------------------------------------------------------
  async function refreshStatus() {
    const badge = $("#statusBadge");
    const text = $(".status-text", badge);
    badge.classList.remove("ok", "warn", "down");
    try {
      const health = await api("/health", { timeout: 10_000 });
      $("#appVersion").textContent = health.version;
      if (health.limits && health.limits.max_upload_mb) {
        state.maxUploadMb = health.limits.max_upload_mb;
        if (!state.doc) $("#dropzoneHint").textContent = uploadHint();
      }
      const gemini = health.services.gemini;
      if (gemini.configured) {
        badge.classList.add("ok");
        text.textContent = "Gemini enabled";
        badge.title = `Models: ${gemini.models.join(", ")}`;
      } else {
        badge.classList.add("warn");
        text.textContent = "API key missing";
        badge.title = "Set GOOGLE_API_KEY to enable AI features";
        $("#configAlert").classList.remove("d-none");
      }
    } catch {
      badge.classList.add("down");
      text.textContent = "Server offline";
      badge.title = "The backend could not be reached";
    }
  }

  // ------------------------------------------------------------------------
  // Documents
  // ------------------------------------------------------------------------
  const uploadHint = () => `PDF, DOCX, TXT or MD up to ${state.maxUploadMb} MB`;
  const SUPPORTED_FILE = /\.(pdf|docx|txt|md|markdown)$/i;

  function clearDocumentOutputs() {
    ["#askOutput", "#summaryOutput", "#tablesOutput"].forEach((selector) => {
      const output = $(selector);
      output.innerHTML = "";
      delete output.dataset.raw;
    });
    state.tables = [];
  }

  function setDocument(doc) {
    const switched = Boolean(state.doc) && state.doc.doc_id !== doc.doc_id;
    if (!state.doc || switched) {
      state.chatHistory = [];
      state.tables = [];
    }
    if (switched) {
      // Answers about the previous document must not linger next to the new one.
      clearDocumentOutputs();
      addChatNote(`Switched to “${doc.filename}”. I'll answer from this document now.`);
    }
    state.doc = doc;
    storage.set(STORAGE_KEYS.doc, JSON.stringify(doc));
    const badge = $("#docBadge");
    const unit = !doc.kind || doc.kind === "pdf" ? "page" : "section";
    badge.innerHTML = `<i class="bi bi-check-circle-fill"></i><span class="doc-name">${escapeHtml(doc.filename)}</span>
      <span>${doc.pages} ${unit}${doc.pages === 1 ? "" : "s"} · ${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}</span>
      <button type="button" class="doc-remove" id="removeDocBtn" title="Remove this document from the server"
              aria-label="Remove document"><i class="bi bi-x-lg"></i></button>`;
    badge.title = doc.filename;
    badge.classList.remove("d-none");
    $("#dropzoneHint").textContent = "Drop another file to switch documents";
    $("#chatMode").textContent = `Answering from ${doc.filename}`;
  }

  function clearDocument() {
    state.doc = null;
    storage.remove(STORAGE_KEYS.doc);
    $("#docBadge").classList.add("d-none");
    $("#dropzoneHint").textContent = uploadHint();
    $("#chatMode").textContent = "General assistant";
  }

  function requireDocument() {
    if (state.doc) return true;
    toast("Upload a document first.", "warning");
    $("#dropzone").scrollIntoView({ behavior: "smooth", block: "center" });
    return false;
  }

  function handleDocumentError(error) {
    if (error.status === 404) {
      clearDocument();
      toast("That document is no longer on the server. Please upload it again.", "warning");
    }
  }

  async function uploadDocument(file) {
    if (!SUPPORTED_FILE.test(file.name) && file.type !== "application/pdf") {
      toast("Please choose a PDF, Word (.docx), text (.txt) or Markdown (.md) file.", "warning");
      $("#pdfUpload").value = "";
      return;
    }
    if (file.size > state.maxUploadMb * 1024 * 1024) {
      toast(`“${file.name}” is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${state.maxUploadMb} MB.`, "warning");
      $("#pdfUpload").value = "";
      return;
    }
    const progress = $("#uploadProgress");
    progress.classList.remove("d-none");
    $("#dropzoneHint").textContent = `Uploading and indexing ${file.name}…`;
    const form = new FormData();
    form.append("file", file);
    try {
      const doc = await api("/api/documents", { method: "POST", form, timeout: 300_000 });
      setDocument(doc);
      toast(`Indexed “${doc.filename}”. Ask away!`);
    } catch (error) {
      toast(error.message, "danger");
      if (state.doc) setDocument(state.doc); else clearDocument();
    } finally {
      progress.classList.add("d-none");
      $("#pdfUpload").value = "";
    }
  }

  async function removeDocument() {
    if (!state.doc) return;
    const { doc_id: docId, filename } = state.doc;
    if (!window.confirm(`Remove “${filename}” and its search index from the server?`)) return;
    try {
      await api(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE", timeout: 30_000 });
      clearDocumentOutputs();
      clearDocument();
      state.chatHistory = [];
      addChatNote(`Removed “${filename}” from the server.`);
      toast("Document removed");
    } catch (error) {
      if (error.status === 404) {
        clearDocument();
        toast("That document was already removed.", "warning");
      } else {
        toast(error.message, "danger");
      }
    }
  }

  async function restoreDocument() {
    const saved = storage.get(STORAGE_KEYS.doc);
    if (!saved) return;
    try {
      const { doc_id: docId } = JSON.parse(saved);
      setDocument(await api(`/api/documents/${encodeURIComponent(docId)}`, { timeout: 10_000 }));
    } catch {
      clearDocument();
    }
  }

  function initUpload() {
    const input = $("#pdfUpload");
    const zone = $("#dropzone");
    input.addEventListener("change", () => input.files[0] && uploadDocument(input.files[0]));
    $("#docBadge").addEventListener("click", (event) => {
      if (event.target.closest("#removeDocBtn")) removeDocument();
    });
    ["dragenter", "dragover"].forEach((type) =>
      zone.addEventListener(type, (event) => { event.preventDefault(); zone.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((type) =>
      zone.addEventListener(type, (event) => { event.preventDefault(); zone.classList.remove("dragover"); }));
    zone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (file) uploadDocument(file);
    });
    restoreDocument();
  }

  function initDocumentTools() {
    $("#askForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const question = $("#questionInput").value.trim();
      const output = $("#askOutput");
      if (!question) return toast("Type a question first.", "warning");
      if (!requireDocument()) return undefined;
      return withBusy($("#askBtn"), async () => {
        setLoading(output, "Retrieving relevant passages…");
        try {
          const result = await api(`/api/documents/${state.doc.doc_id}/query`, { method: "POST", json: { question } });
          setContent(output, { markdown: result.answer, extra: sourcesHtml(result.sources) });
        } catch (error) {
          handleDocumentError(error);
          setError(output, error.message);
        }
      });
    });

    $("#summarizeBtn").addEventListener("click", (event) => {
      const output = $("#summaryOutput");
      if (!requireDocument()) return;
      withBusy(event.currentTarget, async () => {
        setLoading(output, "Reading the whole document…");
        try {
          const result = await api(`/api/documents/${state.doc.doc_id}/summary`, { method: "POST", timeout: 120_000 });
          setContent(output, { markdown: result.summary });
        } catch (error) {
          handleDocumentError(error);
          setError(output, error.message);
        }
      });
    });

    $("#tablesBtn").addEventListener("click", (event) => {
      const output = $("#tablesOutput");
      if (!requireDocument()) return;
      withBusy(event.currentTarget, async () => {
        setLoading(output, "Scanning pages for tables…");
        try {
          const { tables } = await api(`/api/documents/${state.doc.doc_id}/tables`, { timeout: 120_000 });
          state.tables = tables;
          if (!tables.length) {
            setContent(output, { text: "No ruled tables were detected in this document.", raw: "" });
            return;
          }
          const html = tables.map((table, index) => {
            const [header, ...rows] = table.rows;
            return `<div class="table-block">
              <div class="d-flex justify-content-between align-items-center mb-1">
                <span class="page-chip">Table ${index + 1}${table.page ? ` · p. ${table.page}` : ""}</span>
                <button type="button" class="btn btn-sm btn-link p-0" data-copy-table="${index}">Copy as Markdown</button>
              </div>
              <div class="table-responsive"><table class="table table-sm table-bordered align-middle">
                <thead><tr>${header.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
                <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`).join("")}</tbody>
              </table></div></div>`;
          }).join("");
          output.innerHTML = ACTIONS_HTML + html;
          output.dataset.raw = tables.map((t) => t.markdown).join("\n\n");
        } catch (error) {
          handleDocumentError(error);
          setError(output, error.message);
        }
      });
    });
  }

  // ------------------------------------------------------------------------
  // Ghostwriter
  // ------------------------------------------------------------------------
  function initGhostwriter() {
    const input = $("#ghostInput");
    const counter = $("#ghostCount");
    const output = $("#ghostOutput");
    const updateCount = () => {
      const limit = $("#ghostStyle").value === "twitter" ? 280 : null;
      counter.textContent = `${input.value.length} / ${GHOST_LIMIT}${limit ? ` · target ${limit}` : ""}`;
    };
    input.addEventListener("input", updateCount);
    $("#ghostStyle").addEventListener("change", updateCount);
    updateCount();

    $("#ghostForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const text = input.value.trim();
      const style = $("#ghostStyle").value;
      const engine = $("#ghostEngine").value;
      if (!text) return toast("Enter some text to rewrite.", "warning");
      return withBusy($("#ghostBtn"), async () => {
        setLoading(output, "Rewriting…");
        try {
          const result = await api("/api/ghostwriter", { method: "POST", json: { text, style, engine } });
          const warnings = (result.warnings || [])
            .map((w) => `<div class="small text-body-secondary mt-2"><i class="bi bi-info-circle me-1"></i>${escapeHtml(w)}</div>`)
            .join("");
          setContent(output, { text: result.rewritten, extra: warnings });
          const badge = $("#engineBadge");
          badge.textContent = result.engine === "gemini" ? "Gemini" : "Offline engine";
          badge.classList.remove("d-none");
        } catch (error) {
          setError(output, error.message);
        }
      });
    });

    // Selecting text inside any answer sends it to the Ghostwriter.
    document.addEventListener("selectionchange", debounce(() => {
      const selection = window.getSelection();
      const text = selection ? selection.toString().trim() : "";
      if (text.length < 15 || text === state.lastCapture) return;
      const anchor = selection.anchorNode && (selection.anchorNode.nodeType === 1 ? selection.anchorNode : selection.anchorNode.parentElement);
      if (!anchor || !anchor.closest(".output, .chat-message") || anchor.closest("#ghostOutput")) return;
      state.lastCapture = text;
      input.value = text.slice(0, GHOST_LIMIT);
      updateCount();
      toast("Selection sent to the Ghostwriter", "info");
    }, 800));
  }

  // ------------------------------------------------------------------------
  // Translate, transcribe, search
  // ------------------------------------------------------------------------
  function initAssistantTools() {
    $("#translateForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const text = $("#translateInput").value.trim();
      const language = $("#languageInput").value.trim();
      const output = $("#translateOutput");
      if (!text || !language) return toast("Enter the text and a target language.", "warning");
      return withBusy($("#translateBtn"), async () => {
        setLoading(output, `Translating into ${language}…`);
        try {
          const result = await api("/api/translate", { method: "POST", json: { text, target_language: language } });
          setContent(output, { text: result.translation });
        } catch (error) {
          setError(output, error.message);
        }
      });
    });

    $("#transcribeForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const file = $("#audioUpload").files[0];
      const output = $("#transcribeOutput");
      if (!file) return toast("Choose an audio file first.", "warning");
      const form = new FormData();
      form.append("file", file);
      form.append("language", $("#audioLanguage").value);
      return withBusy($("#transcribeBtn"), async () => {
        setLoading(output, "Transcribing…");
        try {
          const result = await api("/api/transcribe", { method: "POST", form, timeout: 180_000 });
          setContent(output, { text: result.text });
        } catch (error) {
          setError(output, error.message);
        }
      });
    });

    $("#searchForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const query = $("#searchInput").value.trim();
      const output = $("#searchOutput");
      if (query.length < 2) return toast("Type a search query.", "warning");
      return withBusy($("#searchBtn"), async () => {
        setLoading(output, "Searching…");
        try {
          const result = await api("/api/search", { method: "POST", json: { query } });
          let extra = "";
          if (result.sources && result.sources.length) {
            extra = `<details class="sources" open><summary>Web sources</summary>${result.sources
              .map((s) => `<div class="source-item"><a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.title)}</a></div>`)
              .join("")}</details>`;
          } else if (!result.grounded) {
            extra = '<div class="small text-body-secondary mt-2"><i class="bi bi-info-circle me-1"></i>Answered from the model\'s own knowledge. Live Google Search grounding needs a paid Gemini tier.</div>';
          }
          setContent(output, { markdown: result.answer, extra });
        } catch (error) {
          setError(output, error.message);
        }
      });
    });
  }

  // ------------------------------------------------------------------------
  // Reddit sentiment
  // ------------------------------------------------------------------------
  function redditHtml(data) {
    const chart = typeof data.chart === "string" && data.chart.startsWith("data:image/png;base64,")
      ? `<img class="reddit-chart" src="${data.chart}" alt="Sentiment distribution chart" />` : "";
    const subreddits = Object.entries(data.top_subreddits || {})
      .map(([name, count]) => `<span class="subreddit-chip">r/${escapeHtml(name)} · ${count}</span>`).join("");
    const posts = (data.sample_posts || []).map((post) => {
      const url = /^https:\/\/www\.reddit\.com\//.test(post.url) ? post.url : null;
      return `<div class="reddit-post">
        <div class="reddit-post-meta">
          <span>r/${escapeHtml(post.subreddit)}</span><span>▲ ${post.upvotes}</span>
          <span class="text-${escapeHtml(post.sentiment)}">${escapeHtml(post.sentiment)}</span>
          <span>${post.is_submission ? "post" : "comment"}</span>
          ${url ? `<a class="ms-auto" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">View</a>` : ""}
        </div>
        <div>${escapeHtml(post.text)}</div>
      </div>`;
    }).join("");
    return `
      <div class="d-flex flex-wrap justify-content-between align-items-center gap-2">
        <strong>${escapeHtml(data.overall)}</strong>
        <span class="meta-chip">${data.total_posts} posts &amp; comments · ${escapeHtml(data.time_filter)}</span>
      </div>
      <div class="stat-grid">
        <div class="stat"><div class="stat-value text-positive">${data.positive}%</div><div class="stat-label">Positive</div></div>
        <div class="stat"><div class="stat-value text-neutral">${data.neutral}%</div><div class="stat-label">Neutral</div></div>
        <div class="stat"><div class="stat-value text-negative">${data.negative}%</div><div class="stat-label">Negative</div></div>
      </div>
      ${chart}
      ${subreddits ? `<div class="mt-2">${subreddits}</div>` : ""}
      <h3 class="h6 mt-3">${data.analysis_engine === "gemini" ? "AI insights" : "Summary"}</h3>
      <div class="markdown">${renderMarkdown(data.analysis)}</div>
      ${posts ? `<h3 class="h6 mt-3">Top posts</h3>${posts}` : ""}`;
  }

  function initReddit() {
    $("#redditForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const subject = $("#redditSubject").value.trim();
      const output = $("#redditOutput");
      if (subject.length < 2) return toast("Enter a topic to analyse.", "warning");
      return withBusy($("#redditBtn"), async () => {
        setLoading(output, `Collecting Reddit discussions about “${subject}”… this can take up to a minute.`);
        try {
          const data = await api("/api/reddit/sentiment", {
            method: "POST",
            json: { subject, time_filter: $("#redditTime").value },
            timeout: 100_000,
          });
          output.innerHTML = ACTIONS_HTML + redditHtml(data);
          output.dataset.raw = data.analysis;
        } catch (error) {
          setError(output, error.message);
        }
      });
    });
  }

  // ------------------------------------------------------------------------
  // Sketch assistant
  // ------------------------------------------------------------------------
  function initSketch() {
    const canvas = $("#sketchCanvas");
    const ctx = canvas.getContext("2d");
    const status = $("#sketchStatus");
    let strokes = [];
    let redoStack = [];
    let current = null;

    const currentTool = () => ($('input[name="sketchTool"]:checked') || {}).value || "pen";
    const pointFrom = (event) => {
      const rect = canvas.getBoundingClientRect();
      return { x: Math.round((event.clientX - rect.left) * 10) / 10, y: Math.round((event.clientY - rect.top) * 10) / 10 };
    };

    function drawStroke(stroke) {
      const pts = stroke.points;
      if (!pts.length) return;
      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = stroke.width;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      if (stroke.tool === "line" && pts[1]) {
        ctx.moveTo(pts[0].x, pts[0].y);
        ctx.lineTo(pts[1].x, pts[1].y);
      } else if (stroke.tool === "rectangle" && pts[1]) {
        ctx.rect(pts[0].x, pts[0].y, pts[1].x - pts[0].x, pts[1].y - pts[0].y);
      } else if (stroke.tool === "circle" && pts[1]) {
        const radius = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
        ctx.arc(pts[0].x, pts[0].y, radius, 0, Math.PI * 2);
      } else if (pts.length === 1) {
        ctx.arc(pts[0].x, pts[0].y, stroke.width / 2, 0, Math.PI * 2);
      } else if (stroke.tool === "path") {
        // Cleaned-up shapes from the server: straight segments keep corners sharp.
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
      } else {
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length - 1; i += 1) {
          const midX = (pts[i].x + pts[i + 1].x) / 2;
          const midY = (pts[i].y + pts[i + 1].y) / 2;
          ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
        }
        ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
      }
      ctx.stroke();
    }

    function redraw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      strokes.forEach(drawStroke);
      if (current) drawStroke(current);
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      redraw();
    }

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return; // ignore right/middle clicks
      canvas.setPointerCapture(event.pointerId);
      const point = pointFrom(event);
      current = {
        tool: currentTool(),
        color: $("#sketchColor").value,
        width: Number($("#sketchWidth").value),
        points: [point],
      };
      redraw();
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!current) return;
      const point = pointFrom(event);
      if (current.tool === "pen") current.points.push(point);
      else current.points[1] = point;
      redraw();
    });

    const finish = () => {
      if (!current) return;
      const pts = current.points;
      const meaningful = current.tool === "pen" ? pts.length >= 1
        : pts[1] && Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y) > 2;
      if (meaningful) {
        strokes.push(current);
        redoStack = [];
      }
      current = null;
      redraw();
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);

    const undo = () => { if (strokes.length) { redoStack.push(strokes.pop()); redraw(); } };
    const redo = () => { if (redoStack.length) { strokes.push(redoStack.pop()); redraw(); } };
    $("#undoBtn").addEventListener("click", undo);
    $("#redoBtn").addEventListener("click", redo);
    document.addEventListener("keydown", (event) => {
      if (!(event.ctrlKey || event.metaKey) || /input|textarea|select/i.test(document.activeElement.tagName)) return;
      if (event.key.toLowerCase() === "z" && !event.shiftKey) { event.preventDefault(); undo(); }
      if (event.key.toLowerCase() === "y" || (event.key.toLowerCase() === "z" && event.shiftKey)) { event.preventDefault(); redo(); }
    });

    $("#clearSketchBtn").addEventListener("click", () => {
      if (!strokes.length) return;
      redoStack = [];
      strokes = [];
      status.textContent = "";
      redraw();
    });

    $("#saveSketchBtn").addEventListener("click", () => {
      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;
      const exportCtx = exportCanvas.getContext("2d");
      exportCtx.fillStyle = getComputedStyle(canvas).backgroundColor || "#ffffff";
      exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      exportCtx.drawImage(canvas, 0, 0);
      const link = document.createElement("a");
      link.download = "docugenius-sketch.png";
      link.href = exportCanvas.toDataURL("image/png");
      link.click();
    });

    $("#cleanBtn").addEventListener("click", (event) => {
      const freehand = strokes.map((stroke, index) => ({ stroke, index })).filter(({ stroke }) => stroke.tool === "pen" && stroke.points.length > 1);
      if (!freehand.length) {
        toast("Draw something freehand first, then clean it up.", "warning");
        return;
      }
      withBusy(event.currentTarget, async () => {
        status.textContent = "Analysing strokes…";
        try {
          const result = await api("/api/sketch/enhance", {
            method: "POST",
            json: { strokes: freehand.map(({ stroke }) => ({ points: stroke.points })) },
          });
          const next = strokes.slice();
          result.strokes.forEach((cleaned, i) => {
            const { stroke, index } = freehand[i];
            next[index] = { tool: "path", color: stroke.color, width: stroke.width, points: cleaned.points };
          });
          redoStack = [];
          strokes = next;
          redraw();
          status.textContent = result.message;
        } catch (error) {
          status.textContent = error.message;
          toast(error.message, "danger");
        }
      });
    });

    window.addEventListener("resize", debounce(resize, 150));
    resize();
  }

  // ------------------------------------------------------------------------
  // Chat widget
  // ------------------------------------------------------------------------
  function initChat() {
    const widget = $("#chatWidget");
    const toggle = $("#chatToggle");
    const messages = $("#chatMessages");
    const input = $("#chatInput");
    let busy = false;

    const setOpen = (open) => {
      widget.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", String(open));
      if (open) setTimeout(() => input.focus(), 200);
    };
    toggle.addEventListener("click", () => setOpen(!widget.classList.contains("open")));

    const addMessage = (role, html, extraClass = "") => {
      const bubble = document.createElement("div");
      bubble.className = `chat-message ${role} ${extraClass}`.trim();
      bubble.innerHTML = html;
      messages.appendChild(bubble);
      messages.scrollTop = messages.scrollHeight;
      return bubble;
    };

    $("#chatForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message || busy) return;
      busy = true;
      input.value = "";
      addMessage("user", escapeHtml(message));
      const typing = addMessage("bot", '<span class="typing" aria-label="DocuGenius is typing"><span></span><span></span><span></span></span>');
      try {
        const result = await api("/api/chat", {
          method: "POST",
          json: { message, doc_id: state.doc ? state.doc.doc_id : null, history: state.chatHistory.slice(-MAX_CHAT_HISTORY) },
        });
        const cited = citationSummary(result.sources);
        typing.innerHTML = renderMarkdown(result.reply)
          + (cited ? `<div class="chat-sources">Sources: ${escapeHtml(cited)}</div>` : "");
        state.chatHistory.push({ role: "user", content: message.slice(0, 4000) }, { role: "assistant", content: result.reply.slice(0, 4000) });
        state.chatHistory = state.chatHistory.slice(-MAX_CHAT_HISTORY * 2);
      } catch (error) {
        if (error.status === 404) handleDocumentError(error);
        typing.classList.add("error");
        typing.textContent = error.message;
      } finally {
        busy = false;
        messages.scrollTop = messages.scrollHeight;
      }
    });
  }

  // ------------------------------------------------------------------------
  // Sidebar, info modal, misc
  // ------------------------------------------------------------------------
  function initInfoModal() {
    const titles = { architecture: "System architecture", guide: "User guide", faq: "Frequently asked questions" };
    const modal = new window.bootstrap.Modal("#infoModal");
    $$("[data-info]").forEach((item) => {
      item.addEventListener("click", () => {
        const key = item.dataset.info;
        window.bootstrap.Offcanvas.getOrCreateInstance("#sidebar").hide();
        $("#infoModalTitle").textContent = titles[key];
        const body = $("#infoModalBody");
        body.replaceChildren($(`#tpl-${key}`).content.cloneNode(true));
        modal.show();
      });
    });
  }

  function initLogo() {
    const container = $("#lottie-logo");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!window.lottie) {
      container.textContent = "🤖";
      return;
    }
    window.lottie.loadAnimation({
      container,
      renderer: "svg",
      loop: !reduceMotion,
      autoplay: !reduceMotion,
      path: `${API_BASE}/assets/animations/ai_animation.json`,
    });
  }

  function initScrollTop() {
    const button = $("#scrollTop");
    window.addEventListener("scroll", () => button.classList.toggle("show", window.scrollY > 400), { passive: true });
    button.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  }

  function initShortcuts() {
    // Ctrl/Cmd + Enter submits the form a textarea belongs to.
    $$("textarea").forEach((area) => area.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && area.form) {
        event.preventDefault();
        area.form.requestSubmit();
      }
    }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initLogo();
    refreshStatus();
    initUpload();
    initDocumentTools();
    initGhostwriter();
    initAssistantTools();
    initReddit();
    initSketch();
    initChat();
    initInfoModal();
    initScrollTop();
    initShortcuts();
  });
})();
