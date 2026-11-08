/* Machaon local console: no framework, no build step, no remote requests. */
(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const LOCALES = window.MACHAON_I18N || {};
  const SUPPORTED_LOCALES = ["en", "zh", "ja", "ko"];
  const state = {
    token: "",
    locale: "en",
    session: null,
    proposal: null,
    lastJob: null,
    events: [],
    busy: false,
    lastTensors: null,
    lastCheckpointSummary: null,
    connectionKind: "checking",
    connectionKey: "conn.checking",
    lastMetrics: null,
    lastGate: null,
  };

  const el = {
    connection: $("#connectionPill"),
    version: $("#versionPill"),
    tokenButton: $("#tokenButton"),
    guideButton: $("#guideButton"),
    onboardingGuideButton: $("#onboardingGuideButton"),
    guideDialog: $("#guideDialog"),
    guideClose: $("#guideClose"),
    guideCloseX: $("#guideCloseX"),
    languageSelect: $("#languageSelect"),
    dialog: $("#tokenDialog"),
    tokenInput: $("#tokenInput"),
    checkpointPath: $("#checkpointPath"),
    allowedRoots: $("#allowedRoots"),
    modelsList: $("#modelsList"),
    checkpointSummary: $("#checkpointSummary"),
    checkpointBadge: $("#checkpointBadge"),
    tensorRows: $("#tensorRows"),
    sessionBadge: $("#sessionBadge"),
    proposalCard: $("#proposalCard"),
    gateCard: $("#gateCard"),
    stateCard: $("#stateCard"),
    timeline: $("#timeline"),
    eventCount: $("#eventCount"),
    rawEvents: $("#rawEvents"),
    jobPill: $("#jobPill"),
    jobBanner: $("#jobBanner"),
    jobTitle: $("#jobTitle"),
    jobDetail: $("#jobDetail"),
    apply: $("#applyButton"),
    evaluate: $("#evaluateButton"),
    rollback: $("#rollbackButton"),
    save: $("#saveButton"),
    dtype: $("#dtype"),
    quantize: $("#quantize"),
  };

  function detectLocale() {
    const params = new URLSearchParams(window.location.search);
    const requested = (params.get("lang") || "").toLowerCase().split("-")[0];
    if (SUPPORTED_LOCALES.includes(requested)) return requested;
    const stored = (localStorage.getItem("machaon-web-locale") || "").toLowerCase().split("-")[0];
    if (SUPPORTED_LOCALES.includes(stored)) return stored;
    const browser = (navigator.language || "en").toLowerCase().split("-")[0];
    return SUPPORTED_LOCALES.includes(browser) ? browser : "en";
  }

  function t(key, vars = {}) {
    const table = LOCALES[state.locale] || LOCALES.en || {};
    const fallback = (LOCALES.en || {})[key] || key;
    return String(table[key] || fallback).replace(/\{(\w+)\}/g, (_match, name) =>
      Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : "{" + name + "}"
    );
  }

  function applyStaticLocale() {
    document.documentElement.lang = state.locale;
    document.title = t("meta.title");
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-html]").forEach((node) => {
      node.innerHTML = t(node.dataset.i18nHtml);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
    });
    document.querySelectorAll("[data-i18n-aria-label]").forEach((node) => {
      node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel));
    });
    const count = document.getElementById("eventCount");
    if (count && state.events.length === 0) {
      count.textContent = t("common.events", { n: 0 });
    }
    if (el.languageSelect) el.languageSelect.value = state.locale;
  }

  function setLocale(locale, { persist = true } = {}) {
    if (!SUPPORTED_LOCALES.includes(locale)) return;
    state.locale = locale;
    if (persist) localStorage.setItem("machaon-web-locale", locale);
    const params = new URLSearchParams(window.location.search);
    params.set("lang", locale);
    const query = params.toString();
    history.replaceState(null, "", window.location.pathname + (query ? "?" + query : ""));
    applyStaticLocale();
    renderConnection();
    renderSessionState();
    renderProposal(state.proposal);
    renderTensors(state.lastTensors || []);
    renderCheckpoint(state.lastCheckpointSummary || null);
    if (state.lastMetrics) renderMetrics(state.lastMetrics, state.lastGate);
    else renderGate(state.lastGate);
    updateEventCount();
    if (state.lastJob) {
      el.jobTitle.textContent = t("job.default");
      el.jobDetail.textContent = t("job.streaming", { id: state.lastJob });
    }
  }

  function renderConnection() {
    setConnection(
      state.connectionKind || "checking",
      t(state.connectionKey || "conn.checking"),
      state.connectionKey || "conn.checking"
    );
  }

  function readToken() {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get("token");
    if (fromUrl) {
      sessionStorage.setItem("machaon-web-token", fromUrl);
      params.delete("token");
      const query = params.toString();
      history.replaceState(null, "", window.location.pathname + (query ? "?" + query : ""));
      return fromUrl;
    }
    return sessionStorage.getItem("machaon-web-token") || "";
  }

  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (state.token) headers["X-Machaon-Token"] = state.token;
    if (options.body !== undefined && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(path, Object.assign({}, options, { headers }));
    if (response.status === 401) {
      openTokenDialog(t("errors.tokenMissing"));
      throw new Error("unauthorised");
    }
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(body.detail || ("HTTP " + response.status));
    }
    return body;
  }

  function openTokenDialog(message) {
    if (message) {
      const note = el.dialog.querySelector("p:not(.eyebrow)");
      if (note) note.textContent = message;
    }
    el.tokenInput.value = state.token || "";
    if (typeof el.dialog.showModal === "function" && !el.dialog.open) {
      el.dialog.showModal();
    }
  }

  function saveToken() {
    state.token = el.tokenInput.value.trim();
    if (state.token) sessionStorage.setItem("machaon-web-token", state.token);
    else sessionStorage.removeItem("machaon-web-token");
    el.dialog.close();
    boot();
  }

  function openGuide() {
    if (!el.guideDialog) return;
    if (typeof el.guideDialog.showModal === "function" && !el.guideDialog.open) {
      el.guideDialog.showModal();
    }
  }

  function closeGuide() {
    if (el.guideDialog && el.guideDialog.open) el.guideDialog.close();
    localStorage.setItem("machaon-web-guide-seen", "1");
  }

  function maybeShowGuide() {
    if (localStorage.getItem("machaon-web-guide-seen") === "1") return;
    openGuide();
  }

  function fmt(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return String(value);
      if (Math.abs(value) >= 1000) return value.toLocaleString();
      if (Math.abs(value) > 0 && Math.abs(value) < 0.001) return value.toExponential(3);
      return Number(value.toFixed(5)).toString();
    }
    if (typeof value === "boolean") return value ? t("common.yes") : t("common.no");
    return String(value);
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
  }

  function metric(k, v, accent = false) {
    return '<div class="metric"><span class="k">' + escapeHtml(k) +
      '</span><span class="v' + (accent ? " accent" : "") + '">' +
      escapeHtml(fmt(v)) + "</span></div>";
  }

  function showBusy(label, detail) {
    state.busy = true;
    el.jobBanner.classList.remove("hidden");
    el.jobTitle.textContent = label || t("job.default");
    el.jobDetail.textContent = detail || t("evidence.runningDetail");
    el.jobPill.textContent = t("status.running");
    el.jobPill.className = "badge live";
    setActionState();
  }

  function stopBusy(status) {
    state.busy = false;
    el.jobBanner.classList.add("hidden");
    el.jobPill.textContent = t("status." + (status || "idle"));
    el.jobPill.className = "badge " + (status === "done" ? "live" : status === "error" ? "quiet" : "quiet");
    setActionState();
  }

  function setActionState() {
    const ready = !!state.session;
    const canApply = ready && (!!state.proposal || !!state.session.has_pending) && !state.busy;
    const canEvaluate = ready && !state.busy;
    el.apply.disabled = !canApply;
    el.evaluate.disabled = !canEvaluate;
    el.rollback.disabled = !ready || state.busy;
    el.save.disabled = !ready || !(state.session.edited || []).length || state.busy;
  }

  function setConnection(kind, text, key = null) {
    state.connectionKind = kind;
    if (key) state.connectionKey = key;
    el.connection.className = "pill " + kind;
    el.connection.innerHTML = "<i></i> " + escapeHtml(text);
  }

  async function boot() {
    setConnection("", t("conn.checking"), "conn.checking");
    try {
      const health = await api("/api/health");
      el.version.textContent = "v" + health.version;
      el.allowedRoots.textContent = (health.allowed_roots || []).join(" · ");
      const session = await api("/api/session/state");
      state.session = session && session.ready ? session : null;
      setConnection("live", t("conn.live"), "conn.live");
      renderSessionState();
      if (state.session) el.checkpointPath.value = state.session.model || el.checkpointPath.value;
      maybeShowGuide();
    } catch (err) {
      setConnection("error", t("conn.offline"), "conn.offline");
      if (err.message !== "unauthorised") {
        el.version.textContent = "v—";
        openTokenDialog(t("errors.serverDown"));
      }
    }
  }

  function renderCheckpoint(summary) {
    state.lastCheckpointSummary = summary;
    if (!summary) {
      el.checkpointSummary.className = "summary-grid empty-state";
      el.checkpointSummary.textContent = t("checkpoint.emptySummary");
      el.checkpointBadge.textContent = t("badge.noCheckpoint");
      return;
    }
    el.checkpointSummary.className = "summary-grid";
    el.checkpointSummary.innerHTML = [
      metric(t("metadata.tensors"), summary.tensors, true),
      metric(t("metadata.parameters"), summary.parameters),
      metric(t("metadata.shards"), summary.shards),
      metric(t("metadata.dtypes"), Object.keys(summary.dtypes || {}).join(" · ")),
      metric(t("metadata.path"), summary.path),
    ].join("");
    el.checkpointBadge.textContent = t("badge.inspected");
    el.checkpointBadge.className = "badge live";
  }

  function renderTensors(rows) {
    state.lastTensors = rows || [];
    if (!rows || !rows.length) {
      el.tensorRows.className = "table-wrap empty-state";
      el.tensorRows.textContent = t("tensor.empty");
      return;
    }
    el.tensorRows.className = "table-wrap";
    el.tensorRows.innerHTML = "<table><thead><tr><th>" + escapeHtml(t("table.tensor")) +
      "</th><th>" + escapeHtml(t("table.shape")) + "</th><th>" +
      escapeHtml(t("table.dtype")) + "</th></tr></thead><tbody>" +
      rows.map((row) => "<tr><td title=\"" + escapeHtml(row.name) + "\">" +
        escapeHtml(row.name) + "</td><td>" + escapeHtml(JSON.stringify(row.shape)) +
        "</td><td>" + escapeHtml(row.dtype) + "</td></tr>").join("") +
      "</tbody></table>";
  }

  function renderSessionState() {
    const s = state.session;
    if (!s) {
      el.sessionBadge.textContent = t("badge.notOpen");
      el.sessionBadge.className = "badge";
      el.stateCard.className = "evidence-card empty-state";
      el.stateCard.textContent = t("evidence.sessionEmpty");
      setActionState();
      return;
    }
    el.sessionBadge.textContent = t("badge.sessionLive");
    el.sessionBadge.className = "badge live";
    el.stateCard.className = "evidence-card";
    el.stateCard.innerHTML = '<dl class="kv">' +
      "<dt>" + escapeHtml(t("session.state.model")) + "</dt><dd>" + escapeHtml(s.model || "—") + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.parameters")) + "</dt><dd>" + escapeHtml(fmt(s.parameters)) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.layers")) + "</dt><dd>" + escapeHtml(fmt(s.layers)) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.quantize")) + "</dt><dd>" + escapeHtml(s.quantize || t("common.none")) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.baseline")) + "</dt><dd>" + escapeHtml(s.has_baseline ? t("session.captured") : t("session.notCaptured")) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.pending")) + "</dt><dd>" + escapeHtml(s.has_pending ? t("common.yes") : t("common.no")) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.edited")) + "</dt><dd>" + escapeHtml((s.edited || []).length) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.budget")) + "</dt><dd>" + escapeHtml(fmt(s.budget_total)) + " / " +
      escapeHtml(fmt(s.budget_limit)) + "</dd></dl>";
    setActionState();
  }

  function renderProposal(p) {
    if (!p) {
      state.proposal = null;
      el.proposalCard.className = "evidence-card empty-state";
      el.proposalCard.textContent = t("evidence.proposalEmpty");
      return;
    }
    state.proposal = p;
    el.proposalCard.className = "evidence-card";
    el.proposalCard.innerHTML = '<div class="verdict pass"><strong>' + escapeHtml(t("proposal.solved")) + '</strong><span class="tiny">' +
      escapeHtml(p.solver || "") + "</span></div><dl class=\"kv\">" +
      "<dt>" + escapeHtml(t("proposal.module")) + "</dt><dd>" + escapeHtml(p.module) + "</dd>" +
      "<dt>" + escapeHtml(t("proposal.layer")) + "</dt><dd>" + escapeHtml(p.layer) + "</dd>" +
      "<dt>" + escapeHtml(t("proposal.relativeDelta")) + "</dt><dd>" + escapeHtml(fmt(p.relative_delta)) + "</dd>" +
      "<dt>" + escapeHtml(t("proposal.targetProbability")) + "</dt><dd>" + escapeHtml(fmt(p.target_probability)) + "</dd>" +
      "<dt>" + escapeHtml(t("proposal.steps")) + "</dt><dd>" + escapeHtml(fmt(p.steps)) + "</dd>" +
      "<dt>" + escapeHtml(t("proposal.budgetAfter")) + "</dt><dd>" + escapeHtml(fmt(p.budget_after)) + "</dd></dl>";
    setActionState();
  }

  function renderGate(gate) {
    state.lastGate = gate;
    if (!gate) {
      el.gateCard.className = "evidence-card empty-state";
      el.gateCard.textContent = t("evidence.gateEmpty");
      return;
    }
    const passed = !!gate.passed;
    el.gateCard.className = "evidence-card";
    const failures = (gate.failures || []).concat(gate.missing || []);
    el.gateCard.innerHTML = '<div class="verdict ' + (passed ? "pass" : "fail") +
      '"><strong>' + escapeHtml(passed ? t("gate.pass") : t("gate.fail")) +
      '</strong><span class="tiny">' + escapeHtml(t("gate.failClosed")) + "</span></div>" +
      '<dl class="kv"><dt>' + escapeHtml(t("gate.gates")) + "</dt><dd>" + escapeHtml(fmt(gate.n_gates)) + "</dd>" +
      "<dt>" + escapeHtml(t("gate.failures")) + "</dt><dd>" + escapeHtml(fmt(failures.length)) + "</dd></dl>" +
      (failures.length ? '<div class="failures">' +
        failures.map((failure) => escapeHtml(localizedFailure(failure))).join(" · ") +
        "</div>" : "");
  }

  function renderMetrics(metrics, gate) {
    state.lastMetrics = metrics;
    state.lastGate = gate;
    if (metrics) {
      const entries = Object.entries(metrics).filter(([key]) => key !== "n");
      el.gateCard.className = "evidence-card";
      el.gateCard.innerHTML = '<div class="verdict ' + (gate && gate.passed ? "pass" : "fail") +
        '"><strong>' + escapeHtml(gate ? (gate.passed ? t("gate.pass") : t("gate.fail")) : t("gate.measured")) +
        '</strong><span class="tiny">' + escapeHtml(t("evidence.metrics")) + "</span></div><dl class=\"kv\">" +
        entries.map(([key, value]) => "<dt>" + escapeHtml(metricLabel(key)) + "</dt><dd>" +
          escapeHtml(fmt(value)) + "</dd>").join("") + "</dl>" +
        (gate && (gate.failures || []).length ? '<div class="failures">' +
          gate.failures.map((failure) => escapeHtml(localizedFailure(failure))).join(" · ") + "</div>" : "");
    }
    if (!metrics) renderGate(gate);
  }

  function addTimelineEvent(type, text, detail, kind) {
    if (el.timeline.classList.contains("empty-state")) {
      el.timeline.className = "timeline";
      el.timeline.innerHTML = "";
    }
    const item = document.createElement("div");
    item.className = "event " + (kind || "");
    item.innerHTML = '<span class="event-mark"></span><div class="event-body"><strong>' +
      escapeHtml(text) + "</strong>" + (detail ? "<small>" + escapeHtml(detail) + "</small>" : "") +
      "</div>";
    el.timeline.prepend(item);
    updateEventCount();
  }

  function updateEventCount() {
    el.eventCount.textContent = t("common.events", { n: state.events.length });
  }

  function metricLabel(key) {
    const translated = t("metrics." + key);
    return translated === "metrics." + key ? key : translated;
  }

  function localizedFailure(text) {
    return String(text).replace(
      /\b(rewrite_acc|paraphrase_acc|unrelated_acc|kl_p95|kl_max|mmlu_delta|ppl_ratio|longform_contradictions)\b/g,
      (key) => metricLabel(key)
    );
  }

  function jobLabel(kind) {
    const key = "job.kind." + kind;
    const value = t(key);
    return value === key ? String(kind).replaceAll("_", " ") : value;
  }

  function recordEvent(event) {
    state.events.push(event);
    if (state.events.length > 250) state.events.shift();
    el.rawEvents.textContent = JSON.stringify(state.events.slice(-80), null, 2);
    updateEventCount();
    const data = event.data || {};
    if (event.type === "status") {
      addTimelineEvent("status", t("status." + (data.status || "idle")),
        jobLabel(event.kind || state.lastJob || ""),
        data.status === "done" ? "ok" : data.status === "error" ? "err" : "");
    } else if (event.type === "result") {
      addTimelineEvent("result", t("timeline.result"), "", "ok");
    } else if (event.type === "error") {
      addTimelineEvent("error", data.error || t("timeline.failed"),
        data.error_type || "", "err");
    }
  }

  function handleJobResult(kind, result) {
    if (kind === "describe_checkpoint") renderCheckpoint(result);
    else if (kind === "list_tensors") renderTensors(result);
    else if (kind === "open_session" || kind === "close_session") {
      state.session = result && result.ready ? result : null;
      state.proposal = null;
      renderProposal(null);
      renderSessionState();
    } else if (kind === "baseline") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      addTimelineEvent("baseline", t("timeline.baseline"),
        JSON.stringify(result.baseline || {}), "ok");
    } else if (kind === "propose_edit") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      renderProposal(result && result.proposal);
      if (result && result.densified !== undefined) {
      addTimelineEvent("densify", t("timeline.densify"),
          t("timeline.densified", { n: result.densified }), "ok");
      }
    } else if (kind === "apply_edit") {
      if (result && result.state) state.session = result.state;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("apply", t("timeline.apply"), t("timeline.charged"), "ok");
    } else if (kind === "evaluate") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      renderMetrics(result.metrics, result.gate);
    } else if (kind === "rollback") {
      state.session = result && result.state ? result.state : null;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("rollback", t("timeline.rollback"), t("timeline.restored"), "ok");
    } else if (kind === "save_checkpoint") {
      addTimelineEvent("save", t("timeline.save"), (result && result.output) || "", "ok");
    } else if (kind === "audit_checkpoint_edit" || kind === "apply_delta") {
      addTimelineEvent(kind, t("timeline.checkpoint"), JSON.stringify(result), "ok");
    }
  }

  function attachJob(jobId, kind) {
    state.lastJob = jobId;
    showBusy(jobLabel(kind), t("job.streaming", { id: jobId }));
    const source = new EventSource("/api/jobs/" + encodeURIComponent(jobId) +
      "/events?token=" + encodeURIComponent(state.token));
    source.onmessage = (message) => {
      let event;
      try { event = JSON.parse(message.data); } catch (_) { return; }
      recordEvent(event);
      if (event.type === "result") handleJobResult(kind, event.data.result);
      if (event.type === "error") {
        addTimelineEvent("error", event.data.error || "job failed", "", "err");
        stopBusy("error");
      }
    };
    source.addEventListener("end", (message) => {
      source.close();
      let final = {};
      try { final = JSON.parse(message.data); } catch (_) {}
      stopBusy(final.status || "done");
    });
    source.onerror = () => {
      source.close();
      stopBusy("done");
    };
  }

  async function startJob(path, body, label) {
    if (state.busy) return;
    try {
      const response = await api(path, { method: "POST", body: JSON.stringify(body || {}) });
      attachJob(response.job, response.kind || label);
    } catch (err) {
      stopBusy("error");
      addTimelineEvent("error", localizedJobError(err.message), path, "err");
    }
  }

  function localizedJobError(message) {
    if (!state.locale || state.locale === "en") return message;
    const key = Object.keys(LOCALES.en || {}).find((candidate) =>
      LOCALES.en[candidate] === message
    );
    return key ? t(key) : message;
  }

  function tensorRequest() {
    const body = { path: el.checkpointPath.value.trim(), limit: 200 };
    const query = $("#tensorQuery").value.trim();
    const layer = $("#tensorLayer").value.trim();
    if (query) body.contains = query;
    if (layer !== "") body.layer = Number(layer);
    return body;
  }

  function preserveLines() {
    return $("#preservePrompts").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  }

  function proposalRequest() {
    const max = $("#maxRelNorm").value.trim();
    return {
      prompt: $("#prompt").value,
      target: $("#target").value,
      layer: Number($("#layer").value),
      preserve_prompts: preserveLines(),
      solver: $("#solver").value,
      steps: Number($("#steps").value),
      lr: Number($("#lr").value),
      module_suffix: $("#moduleSuffix").value.trim() || "mlp.down_proj",
      max_rel_norm: max === "" ? null : Number(max),
      densify_from_layer: $("#densify").checked,
    };
  }

  async function findModels() {
    try {
      const result = await api("/api/models");
      el.modelsList.classList.remove("hidden");
      el.modelsList.innerHTML = "";
      if (!result.models.length) {
        el.modelsList.innerHTML = '<span class="empty-state">' +
          escapeHtml(t("model.empty")) + "</span>";
        return;
      }
      result.models.forEach((model) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "model-option";
        button.innerHTML = escapeHtml(model.name) + "<small>" + escapeHtml(model.path) + "</small>";
        button.addEventListener("click", () => {
          el.checkpointPath.value = model.path;
          el.modelsList.classList.add("hidden");
        });
        el.modelsList.appendChild(button);
      });
    } catch (err) {
      addTimelineEvent("error", err.message, t("errors.modelDiscovery"), "err");
    }
  }

  el.tokenButton.addEventListener("click", () => openTokenDialog());
  if (el.guideButton) el.guideButton.addEventListener("click", openGuide);
  if (el.onboardingGuideButton) el.onboardingGuideButton.addEventListener("click", openGuide);
  if (el.guideClose) el.guideClose.addEventListener("click", closeGuide);
  if (el.guideCloseX) el.guideCloseX.addEventListener("click", closeGuide);
  if (el.guideDialog) {
    el.guideDialog.addEventListener("close", () => {
      localStorage.setItem("machaon-web-guide-seen", "1");
    });
  }
  $("#tokenSave").addEventListener("click", saveToken);
  $("#tokenCancel").addEventListener("click", () => el.dialog.close());
  $("#modelsButton").addEventListener("click", findModels);
  $("#describeButton").addEventListener("click", () => {
    const path = el.checkpointPath.value.trim();
    if (!path) return addTimelineEvent("error", t("errors.pathRequired"), "", "err");
    startJob("/api/checkpoints/describe", { path }, "describe_checkpoint");
  });
  $("#tensorsButton").addEventListener("click", () => {
    const body = tensorRequest();
    if (!body.path) return addTimelineEvent("error", t("errors.pathRequired"), "", "err");
    startJob("/api/checkpoints/tensors", body, "list_tensors");
  });
  $("#openButton").addEventListener("click", () => {
    const path = el.checkpointPath.value.trim();
    if (!path) return addTimelineEvent("error", t("errors.pathRequired"), "", "err");
    const body = {
      path,
      dtype: $("#dtype").value,
      quantize: $("#quantize").value || null,
      device: $("#device").value.trim() || null,
      budget_total: Number($("#budget").value),
    };
    if (body.quantize && body.dtype !== "auto") {
      return addTimelineEvent("error", t("errors.quantizeDtype"), "", "err");
    }
    startJob("/api/session/open", body, "open_session");
  });
  $("#proposeButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    startJob("/api/edit/propose", proposalRequest(), "propose_edit");
  });
  $("#applyButton").addEventListener("click", () => {
    if (!state.session) return;
    startJob("/api/edit/apply", {}, "apply_edit");
  });
  $("#baselineButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    const prompts = preserveLines();
    startJob("/api/session/baseline", { neutral_prompts: prompts, corpus_texts: [] }, "baseline");
  });
  $("#evaluateButton").addEventListener("click", () => {
    if (!state.session) return;
    startJob("/api/edit/evaluate", {
      prompts: [$("#prompt").value],
      targets: [$("#target").value],
    }, "evaluate");
  });
  $("#rollbackButton").addEventListener("click", () => {
    if (!state.session) return;
    state.proposal = null;
    startJob("/api/edit/rollback", {}, "rollback");
  });
  $("#saveButton").addEventListener("click", () => {
    if (!state.session) return;
    const out = $("#savePath").value.trim();
    if (!out) return addTimelineEvent("error", t("errors.outputRequired"), "", "err");
    startJob("/api/session/save", { out_dir: out }, "save_checkpoint");
  });
  el.dtype.addEventListener("change", () => {
    if (el.quantize.value) el.dtype.value = "auto";
  });
  el.quantize.addEventListener("change", () => {
    if (el.quantize.value) el.dtype.value = "auto";
  });

  state.locale = detectLocale();
  applyStaticLocale();
  if (el.languageSelect) {
    el.languageSelect.addEventListener("change", () => {
      setLocale(el.languageSelect.value);
    });
  }
  state.token = readToken();
  if (!state.token) {
    setConnection("error", t("conn.tokenRequired"), "conn.tokenRequired");
    openTokenDialog(t("errors.tokenPaste"));
  } else {
    boot();
  }
})();
