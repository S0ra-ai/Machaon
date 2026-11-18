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
    evidence: null,
    lastTeach: null,
    lastTeachMany: null,
    allowedRoots: [],
    capabilities: {},
    pendingAfterBusy: null,
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
    teachCard: $("#teachCard"),
    teachManyCard: $("#teachManyCard"),
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

  function wspLabel(value) {
    if (!value) return "—";
    const key = "wsp.label." + value;
    const translated = t(key);
    return translated === key ? String(value) : translated;
  }

  function mechanismLabel(value) {
    if (!value) return "—";
    const key = "wsp.mechanism." + value;
    const translated = t(key);
    return translated === key ? String(value) : translated;
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
    if (state.lastTeach) {
      if (state.lastTeach.result) renderTeachResult(state.lastTeach);
      else renderTeachPreview(state.lastTeach);
    }
    if (state.lastTeachMany) renderTeachManyResult(state.lastTeachMany);
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

  function renderCapabilities() {
    const banner = $("#capabilityBanner");
    if (!banner) return;
    const missing = ["torch", "transformers", "safetensors"].filter(
      (name) => state.capabilities[name] === false
    );
    if (!missing.length) {
      banner.classList.add("hidden");
      banner.innerHTML = "";
      return;
    }
    banner.classList.remove("hidden");
    banner.innerHTML = escapeHtml(t("capabilities.missing", {
      list: missing.join(", "),
    })) + ' <code>pip install "machaon[hf]"</code>';
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
    const pending = state.pendingAfterBusy;
    state.pendingAfterBusy = null;
    if (pending) setTimeout(pending, 0);
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
      state.allowedRoots = health.allowed_roots || [];
      state.capabilities = health.capabilities || {};
      el.allowedRoots.textContent = state.allowedRoots.join(" · ");
      renderCapabilities();
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
    const isBatch = Object.prototype.hasOwnProperty.call(p, "exact") ||
      Array.isArray(p.per_key_error);
    const perKey = isBatch && Array.isArray(p.per_key_error) ? p.per_key_error : [];
    el.proposalCard.className = "evidence-card";
    el.proposalCard.innerHTML = '<div class="verdict pass"><strong>' + escapeHtml(t("proposal.solved")) + '</strong><span class="tiny">' +
      escapeHtml(isBatch ? t("proposal.batch") : (p.solver || "")) + "</span></div><dl class=\"kv\">" +
      "<dt>" + escapeHtml(t("proposal.module")) + "</dt><dd>" + escapeHtml(p.module) + "</dd>" +
      "<dt>" + escapeHtml(t("proposal.layer")) + "</dt><dd>" + escapeHtml(p.layer) + "</dd>" +
      "<dt>" + escapeHtml(t("proposal.relativeDelta")) + "</dt><dd>" + escapeHtml(fmt(p.relative_delta)) + "</dd>" +
      (isBatch
        ? "<dt>" + escapeHtml(t("proposal.exact")) + "</dt><dd>" + escapeHtml(p.exact ? t("common.yes") : t("common.no")) + "</dd>" +
          "<dt>" + escapeHtml(t("proposal.clamped")) + "</dt><dd>" + escapeHtml(p.clamped ? t("common.yes") : t("common.no")) + "</dd>" +
          "<dt>" + escapeHtml(t("proposal.targetError")) + "</dt><dd>" + escapeHtml(fmt(p.target_error)) + "</dd>" +
          "<dt>" + escapeHtml(t("proposal.rank")) + "</dt><dd>" + escapeHtml(fmt(p.rank)) + "</dd>"
        : "<dt>" + escapeHtml(t("proposal.targetProbability")) + "</dt><dd>" + escapeHtml(fmt(p.target_probability)) + "</dd>" +
          "<dt>" + escapeHtml(t("proposal.steps")) + "</dt><dd>" + escapeHtml(fmt(p.steps)) + "</dd>") +
      "<dt>" + escapeHtml(t("proposal.budgetAfter")) + "</dt><dd>" + escapeHtml(fmt(p.budget_after)) + "</dd></dl>" +
      (perKey.length
        ? '<details class="per-key"><summary>' + escapeHtml(t("proposal.perKey")) +
          " (" + perKey.length + ')</summary><div class="per-key-scroll"><table><thead><tr><th>' +
          escapeHtml(t("proposal.key")) + "</th><th>" + escapeHtml(t("proposal.error")) +
          "</th></tr></thead><tbody>" +
          perKey.map((value, index) => "<tr><td>" + (index + 1) + "</td><td>" +
            escapeHtml(fmt(value)) + "</td></tr>").join("") +
          "</tbody></table></div></details>"
        : "");
    setActionState();
  }

  function renderTeachPreview(payload) {
    const card = el.teachCard;
    if (!card) return;
    if (!payload) {
      card.className = "evidence-card empty-state";
      card.textContent = t("teach.empty");
      return;
    }
    card.className = "evidence-card teach-result";
    const notes = (payload.notes || [])
      .map((note) => "<li>" + escapeHtml(note) + "</li>")
      .join("");
    const command = payload.command
      ? '<p class="mono tiny">' + escapeHtml(payload.command) + "</p>"
      : "";
    card.innerHTML =
      '<div class="teach-status"><strong>' + escapeHtml(t("teach.previewTitle")) +
      '</strong><span class="badge quiet">' + escapeHtml(payload.confidence || "high") +
      "</span></div>" +
      '<div class="teach-grid">' +
      "<span>" + escapeHtml(t("teach.prompt")) + "<br><strong>" +
      escapeHtml(payload.prompt) + "</strong></span>" +
      "<span>" + escapeHtml(t("teach.answer")) + "<br><strong>" +
      escapeHtml(payload.answer) + "</strong></span>" +
      "</div>" +
      (notes ? '<ul class="teach-notes">' + notes + "</ul>" : "") +
      command;
  }

  function renderTeachManyResult(payload) {
    const card = el.teachManyCard;
    if (!card) return;
    if (!payload || !payload.result) {
      card.className = "evidence-card empty-state";
      card.textContent = t("teachMany.empty");
      return;
    }
    const result = payload.result;
    const accepted = !!result.accepted;
    const steps = (result.steps || []).map((step) => {
      const ok = step.status === "applied" || step.status === "already_true";
      return '<span class="check-chip ' + (ok ? "ok" : "bad") + '">' +
        escapeHtml(step.index + " " + step.status) + "</span>";
    }).join("");
    const report = result.report_path
      ? '<p class="mono tiny">' + escapeHtml(result.report_path) + "</p>" : "";
    const useFinal = result.final
      ? '<button id="teachManyUseFinal" class="ghost small" type="button">' +
        escapeHtml(t("teachMany.useFinal")) + "</button>" : "";
    card.className = "evidence-card teach-result";
    card.innerHTML =
      '<div class="teach-status"><strong>' + escapeHtml(teachLabel(result.status)) +
      '</strong><span class="badge ' + (accepted ? "live" : "quiet") + '">' +
      escapeHtml(result.status || "") + "</span></div>" +
      '<p class="teach-summary">' + escapeHtml(result.summary || "") + "</p>" +
      '<div class="teach-grid">' +
      "<span>" + escapeHtml(t("teachMany.finalLabel")) + "<br><strong>" +
      escapeHtml(result.final || "—") + "</strong></span>" +
      "<span>" + escapeHtml(t("teachMany.cumulative")) + "<br><strong>" +
      ((result.cumulative_relative_delta || 0) * 100).toFixed(2) + "%</strong></span>" +
      "</div>" +
      (steps ? '<div class="check-list">' + steps + "</div>" : "") +
      report + useFinal;
    const button = card.querySelector("#teachManyUseFinal");
    if (button && result.final) {
      button.addEventListener("click", () => {
        el.checkpointPath.value = result.final;
        addTimelineEvent("teach_many", t("teachMany.selectedFinal"),
          result.final, "ok");
        $("#openButton").click();
      });
    }
  }

  function teachLabel(status) {
    const key = "teach.status." + status;
    const value = t(key);
    return value === key ? String(status || "") : value;
  }

  function renderTeachResult(payload) {
    const card = el.teachCard;
    if (!card) return;
    if (!payload || !payload.result) {
      renderTeachPreview(null);
      return;
    }
    const result = payload.result;
    const accepted = !!result.accepted;
    const checks = (result.checks || [])
      .map((check) => '<span class="check-chip ' + (check.passed ? "ok" : "bad") +
        '" title="' + escapeHtml(check.requirement || "") + '">' +
        escapeHtml(check.name) + "</span>")
      .join("");
    const warnings = (result.warnings || [])
      .map((warning) => '<p class="teach-warning">' + escapeHtml(warning) + "</p>")
      .join("");
    const location = result.tensor
      ? escapeHtml(result.tensor) +
        (result.best_layer === null ? "" : " · layer " + result.best_layer)
      : escapeHtml(t("teach.noEdit"));
    card.className = "evidence-card teach-result";
    card.innerHTML =
      '<div class="teach-status"><strong>' + escapeHtml(teachLabel(result.status)) +
      '</strong><span class="badge ' + (accepted ? "live" : "quiet") + '">' +
      escapeHtml(result.status || "") + "</span></div>" +
      '<p class="teach-summary">' + escapeHtml(result.summary || "") + "</p>" +
      '<div class="teach-grid">' +
      "<span>" + escapeHtml(t("teach.location")) + "<br><strong>" + location +
      "</strong></span>" +
      "<span>" + escapeHtml(t("teach.delta")) + "<br><strong>" +
      (result.relative_delta === null || result.relative_delta === undefined
        ? "—" : (100 * result.relative_delta).toFixed(2) + "%") +
      "</strong></span>" +
      (result.base_prediction_text
        ? "<span>" + escapeHtml(t("teach.baseline")) + "<br><strong>" +
          escapeHtml(result.base_prediction_text) + "</strong></span>"
        : "") +
      "</div>" +
      (checks ? '<div class="check-list">' + checks + "</div>" : "") +
      warnings;
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

  function renderWsp(report) {
    const card = $("#wspCard");
    if (!report) {
      card.className = "evidence-card empty-state";
      card.textContent = t("wsp.empty");
      return;
    }
    const accepted = !!report.accepted;
    const classification = report.classification || {};
    const mechanism = report.mechanism || null;
    card.className = "evidence-card";
    card.innerHTML = '<div class="verdict ' + (accepted ? "pass" : "fail") +
      '"><strong>' + escapeHtml(wspLabel(report.label)) + "</strong><span class=\"tiny\">" +
      escapeHtml(accepted ? t("wsp.accepted") : t("wsp.rejected")) + "</span></div>" +
      '<dl class="kv"><dt>' + escapeHtml(t("wsp.reason")) + "</dt><dd>" +
      escapeHtml(report.reason || "—") + "</dd>" +
      "<dt>" + escapeHtml(t("wsp.controlGap")) + "</dt><dd>" +
      escapeHtml(fmt(classification.control_gap)) + "</dd>" +
      "<dt>" + escapeHtml(t("wsp.headroomGap")) + "</dt><dd>" +
      escapeHtml(fmt(classification.headroom_gap)) + "</dd>" +
      (mechanism ? "<dt>" + escapeHtml(t("wsp.mechanism")) + "</dt><dd>" +
        escapeHtml((mechanism.passed ? t("wsp.mechanismPass") : t("wsp.mechanismFail")) +
          " · " + mechanismLabel(mechanism.label)) + "</dd>" : "") +
      "</dl>";
  }

  function renderWspEvidence(result) {
    const node = $("#wspEvidence");
    if (!result) {
      state.evidence = null;
      node.textContent = "";
      return;
    }
    state.evidence = result;
    node.textContent = "SHA-256 " + (result.evidence_sha256 || "—") +
      " · " + (result.path || "");
  }

  function renderOpenwspBinding(result) {
    const node = $("#openwspBinding");
    if (!result) {
      node.textContent = "";
      return;
    }
    const manifest = result.manifest || {};
    const linked = manifest.machaon && manifest.machaon.evidence_sha256;
    node.textContent = "edit " + (manifest.edit_sha256 || "—") +
      " · evidence " + (linked || "not bound") +
      " · " + (result.output || "");
  }

  function renderBinding(result) {
    const card = $("#bindingCard");
    if (!result) {
      card.className = "evidence-card empty-state";
      card.textContent = t("wsp.bindingEmpty");
      return;
    }
    const passed = !!result.passed;
    card.className = "evidence-card";
    card.innerHTML = '<div class="verdict ' + (passed ? "pass" : "fail") +
      '"><strong>' + escapeHtml(passed ? t("wsp.bindingPassed") : t("wsp.bindingFailed")) +
      '</strong><span class="tiny">' + escapeHtml(wspLabel(result.label)) + "</span></div>" +
      '<dl class="kv"><dt>edit_sha256</dt><dd>' + escapeHtml(result.edit_sha256 || "—") + "</dd>" +
      "<dt>evidence_sha256</dt><dd>" + escapeHtml(result.evidence_sha256 || "—") + "</dd>" +
      "<dt>" + escapeHtml(t("wsp.bindingErrors")) + "</dt><dd>" +
      escapeHtml((result.errors || []).join(" · ") || t("common.none")) + "</dd></dl>";
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
    else if (kind === "fetch_model") {
      const path = result && result.dest ? result.dest : "";
      if (path) el.checkpointPath.value = path;
      addTimelineEvent("fetch", t("timeline.fetch"),
        path + " · " + ((result && result.parameters) || 0).toLocaleString() + " params", "ok");
    }
    else if (kind === "open_session" || kind === "close_session") {
      state.session = result && result.ready ? result : null;
      state.proposal = null;
      state.lastTeach = null;
      renderProposal(null);
      renderTeachPreview(null);
      renderSessionState();
      if (state.session && result && result.ready) {
        const teach = $("#teachSentence");
        if (teach && !teach.value) teach.focus();
      }
    } else if (kind === "baseline") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      addTimelineEvent("baseline", t("timeline.baseline"),
        JSON.stringify(result.baseline || {}), "ok");
    } else if (kind === "propose_edit" || kind === "propose_batch") {
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
      state.lastTeach = null;
      renderTeachPreview(null);
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("apply", t("timeline.apply"), t("timeline.charged"), "ok");
    } else if (kind === "evaluate") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      renderMetrics(result.metrics, result.gate);
    } else if (kind === "teach_many") {
      if (result && result.state) state.session = result.state;
      state.lastTeachMany = result;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      renderTeachManyResult(result);
      $("#teachManyPanel").open = true;
      if (result && result.result && result.result.out_dir) {
        $("#teachManyOut").value = result.result.out_dir;
      }
      addTimelineEvent("teach_many", t("timeline.teachMany"),
        (result && result.result && result.result.status) || "", "ok");
    } else if (kind === "teach") {
      if (result && result.state) state.session = result.state;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      state.lastTeach = result;
      renderTeachResult(result);
      if (result && result.saved && result.saved.output) {
        $("#savePath").value = result.saved.output;
      }
      addTimelineEvent("teach", t("timeline.teach"),
        (result && result.result && result.result.status) || "", "ok");
    } else if (kind === "rollback") {
      state.session = result && result.state ? result.state : null;
      state.proposal = null;
      state.lastTeach = null;
      renderTeachPreview(null);
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("rollback", t("timeline.rollback"), t("timeline.restored"), "ok");
    } else if (kind === "save_checkpoint") {
      addTimelineEvent("save", t("timeline.save"), (result && result.output) || "", "ok");
    } else if (kind === "wsp_audit") {
      renderWsp(result);
      addTimelineEvent("wsp", t("timeline.wsp"), (result && result.label) || "", "ok");
    } else if (kind === "wsp_session") {
      renderWsp(result);
      addTimelineEvent("wsp", t("timeline.wsp"), (result && result.label) || "", "ok");
    } else if (kind === "wsp_session_evidence") {
      renderWsp(result && result.report);
      renderWspEvidence(result && result.evidence);
      addTimelineEvent("wsp", t("wsp.saved"), (result && result.evidence && result.evidence.evidence_sha256) || "", "ok");
    } else if (kind === "save_wsp_evidence") {
      renderWspEvidence(result);
      addTimelineEvent("wsp", t("wsp.saved"), (result && result.evidence_sha256) || "", "ok");
    } else if (kind === "export_openwsp") {
      renderOpenwspBinding(result);
      addTimelineEvent("openwsp", t("wsp.exported"),
        (result && result.manifest && result.manifest.edit_sha256) || "", "ok");
    } else if (kind === "verify_binding") {
      renderBinding(result);
      addTimelineEvent("binding", t("wsp.bindingChecked"), (result && result.passed) ? "PASS" : "FAIL", result && result.passed ? "ok" : "err");
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

  function batchRequest() {
    const prompts = $("#batchPrompts").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!prompts.length) throw new Error(t("batch.invalidPrompts"));
    let values;
    try {
      values = JSON.parse($("#batchValues").value);
    } catch (_) {
      throw new Error(t("batch.invalidValues"));
    }
    if (!Array.isArray(values) || values.some((row) =>
      !Array.isArray(row) || row.some((value) => !Number.isFinite(Number(value)))
    )) {
      throw new Error(t("batch.invalidValues"));
    }
    if (values.length !== prompts.length) {
      throw new Error(t("batch.rowMismatch"));
    }
    const max = $("#batchMaxRelNorm").value.trim();
    return {
      prompts,
      values,
      layer: Number($("#batchLayer").value),
      preserve_prompts: $("#batchPreserve").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean),
      max_rel_norm: max === "" ? null : Number(max),
      clamp: $("#batchClamp").checked,
      ridge: 0.0,
      module_suffix: $("#moduleSuffix").value.trim() || "mlp.down_proj",
    };
  }

  function sessionWspRequest() {
    const batchPrompts = $("#batchPrompts").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const prompts = batchPrompts.length ? batchPrompts : [$("#prompt").value];
    const headroom = $("#wspHeadroom").value.trim();
    return {
      prompts,
      targets: prompts.map(() => $("#target").value),
      headroom: headroom === "" ? null : { accuracy: Number(headroom) },
      target_metric: "accuracy",
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
  $("#fetchButton").addEventListener("click", () => {
    const option = $("#starterModel").selectedOptions[0];
    if (!option || !state.allowedRoots.length) {
      return addTimelineEvent("error", t("checkpoint.fetchNoRoot"), "", "err");
    }
    const name = option.dataset.name || option.value;
    const root = String(state.allowedRoots[0]).replace(/\/+$/, "");
    startJob("/api/models/fetch", {
      starter: option.value,
      dest: root + "/" + name,
    }, "fetch_model");
  });
  $("#teachPreviewButton").addEventListener("click", async () => {
    const sentence = $("#teachSentence").value.trim();
    if (!sentence) return addTimelineEvent("error", t("teach.sentenceRequired"), "", "err");
    try {
      const payload = await api("/api/teach/preview", {
        method: "POST",
        body: JSON.stringify({ sentence }),
      });
      state.lastTeach = payload;
      renderTeachPreview(payload);
      $("#prompt").value = payload.prompt;
      $("#target").value = payload.answer;
      addTimelineEvent("teach", t("timeline.teachPreview"),
        payload.source || "", "ok");
    } catch (err) {
      addTimelineEvent("error", err.message, t("teach.invalid"), "err");
    }
  });
  $("#teachManyButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    const facts = $("#teachManyFacts").value.split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"));
    if (!facts.length) return addTimelineEvent("error", t("teachMany.factsRequired"), "", "err");
    let out = $("#teachManyOut").value.trim();
    if (!out && state.session.model) {
      out = String(state.session.model).replace(/\/+$/, "") + "-machaon-many";
    }
    if (!out) return addTimelineEvent("error", t("errors.outputRequired"), "", "err");
    startJob("/api/teach_many", {
      facts,
      out_dir: out,
      budget: Number($("#teachManyBudget").value),
      max_rel_norm: Number($("#teachManyMaxNorm").value),
      keep_steps: $("#teachManyKeep").checked,
      continue_on_reject: $("#teachManyContinue").checked,
    }, "teach_many");
  });
  $("#teachButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    const sentence = $("#teachSentence").value.trim();
    if (!sentence) return addTimelineEvent("error", t("teach.sentenceRequired"), "", "err");
    const out = $("#teachOut").value.trim();
    startJob("/api/teach", {
      sentence,
      out_dir: out || null,
      solver: "direction",
      max_rel_norm: $("#maxRelNorm").value.trim() === ""
        ? 0.02 : Number($("#maxRelNorm").value),
    }, "teach");
  });
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
  $("#batchProposeButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    let body;
    try {
      body = batchRequest();
    } catch (err) {
      return addTimelineEvent("error", err.message, "", "err");
    }
    startJob("/api/edit/propose_batch", body, "propose_batch");
  });
  $("#wspAuditButton").addEventListener("click", () => {
    let report;
    try {
      report = JSON.parse($("#wspReport").value);
    } catch (_) {
      return addTimelineEvent("error", t("wsp.invalid"), "", "err");
    }
    startJob("/api/wsp/audit", { report }, "wsp_audit");
  });
  $("#wspSessionButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    startJob("/api/wsp/session", sessionWspRequest(), "wsp_session");
  });
  $("#wspSessionSaveButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    const out = $("#wspSavePath").value.trim();
    if (!out) return addTimelineEvent("error", t("wsp.pathRequired"), "", "err");
    startJob("/api/wsp/session_evidence",
      Object.assign(sessionWspRequest(), { out_dir: out }),
      "wsp_session_evidence");
  });
  $("#wspSaveButton").addEventListener("click", () => {
    let report;
    try {
      report = JSON.parse($("#wspReport").value);
    } catch (_) {
      return addTimelineEvent("error", t("wsp.invalid"), "", "err");
    }
    const out = $("#wspSavePath").value.trim();
    if (!out) return addTimelineEvent("error", t("wsp.pathRequired"), "", "err");
    startJob("/api/wsp/save", { report, out_dir: out }, "save_wsp_evidence");
  });
  $("#openwspExportButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    const out = $("#openwspOut").value.trim();
    if (!out) return addTimelineEvent("error", t("wsp.openwspPathRequired"), "", "err");
    startJob("/api/session/export_openwsp", {
      out_dir: out,
      evidence_sha256: state.evidence ? state.evidence.evidence_sha256 : null,
    }, "export_openwsp");
  });
  $("#bindingVerifyButton").addEventListener("click", () => {
    const edit = $("#bindingEdit").value.trim();
    const evidence = $("#bindingEvidence").value.trim();
    if (!edit || !evidence) return addTimelineEvent("error", t("wsp.bindingPathRequired"), "", "err");
    startJob("/api/binding/verify", { edit_dir: edit, evidence_dir: evidence }, "verify_binding");
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
