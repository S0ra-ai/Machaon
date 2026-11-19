/* Machaon local console: no framework, no build step, no remote requests. */
(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const LOCALES = window.MACHAON_I18N || {};
  const SUPPORTED_LOCALES = ["en", "zh", "ja", "ko"];
  const LAST_CHECKPOINT_KEY = "machaon-last-checkpoint";
  const PROGRAM_HISTORY_KEY = "machaon-program-history";
  const PROGRAM_RUNS_KEY = "machaon-program-runs";
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
    devices: null,
    pendingAfterBusy: null,
    firstEdit: null,
    firstEditStatusKey: "firstEdit.ready",
    firstEditStatusVars: {},
    firstEditStatusKind: "",
    firstEditEvidence: null,
    program: null,
    programHistory: [],
    programResult: null,
    programRuns: [],
    programReportText: "",
    programReportName: "",
    programVerifyCheckModel: false,
    programVerification: null,
  };

  const el = {
    connection: $("#connectionPill"),
    version: $("#versionPill"),
    tokenButton: $("#tokenButton"),
    languageSelect: $("#languageSelect"),
    dialog: $("#tokenDialog"),
    tokenInput: $("#tokenInput"),
    checkpointPath: $("#checkpointPath"),
    starterModel: $("#starterModel"),
    allowedRoots: $("#allowedRoots"),
    modelsList: $("#modelsList"),
    checkpointSummary: $("#checkpointSummary"),
    checkSentence: $("#checkSentence"),
    checkButton: $("#checkButton"),
    checkCard: $("#checkCard"),
    programPanel: $("#programPanel"),
    programBase: $("#programBase"),
    programText: $("#programText"),
    programPlanButton: $("#programPlanButton"),
    programRunButton: $("#programRunButton"),
    programPrefillButton: $("#programPrefillButton"),
    programDownloadButton: $("#programDownloadButton"),
    programFileButton: $("#programFileButton"),
    programFile: $("#programFile"),
    programTemplateSmol: $("#programTemplateSmol"),
    programTemplateTwo: $("#programTemplateTwo"),
    programTemplateBilingual: $("#programTemplateBilingual"),
    programFacts: $("#programFacts"),
    programBuildButton: $("#programBuildButton"),
    programHistory: $("#programHistory"),
    programHistoryLoad: $("#programHistoryLoad"),
    programRuns: $("#programRuns"),
    programRunHistoryLoad: $("#programRunHistoryLoad"),
    programCopyButton: $("#programCopyButton"),
    programReportButton: $("#programReportButton"),
    programUseOutputButton: $("#programUseOutputButton"),
    programReportOut: $("#programReportOut"),
    programVerifyButton: $("#programVerifyButton"),
    programVerifyModelButton: $("#programVerifyModelButton"),
    programVerifyLastButton: $("#programVerifyLastButton"),
    programVerifyMatch: $("#programVerifyMatch"),
    programVerifyCurrent: $("#programVerifyCurrent"),
    programReportFile: $("#programReportFile"),
    programVerifyCard: $("#programVerifyCard"),
    programCard: $("#programCard"),
    checkpointBadge: $("#checkpointBadge"),
    tensorRows: $("#tensorRows"),
    sessionBadge: $("#sessionBadge"),
    closeSessionButton: $("#closeSessionButton"),
    proposalCard: $("#proposalCard"),
    gateCard: $("#gateCard"),
    teachCard: $("#teachCard"),
    teachManyCard: $("#teachManyCard"),
    firstEditCard: $("#firstEditCard"),
    firstEditSentence: $("#firstEditSentence"),
    firstEditStarter: $("#firstEditStarter"),
    firstEditDevice: $("#firstEditDevice"),
    firstEditButton: $("#firstEditButton"),
    firstEditSaveRow: $("#firstEditSaveRow"),
    firstEditOut: $("#firstEditOut"),
    firstEditSave: $("#firstEditSave"),
    firstEditStatus: $("#firstEditStatus"),
    firstEditEvidence: $("#firstEditEvidence"),
    firstEditNext: $("#firstEditNext"),
    firstEditOpen: $("#firstEditOpen"),
    firstEditList: $("#firstEditList"),
    firstEditProgram: $("#firstEditProgram"),
    stateCard: $("#stateCard"),
    timeline: $("#timeline"),
    eventCount: $("#eventCount"),
    rawEvents: $("#rawEvents"),
    jobPill: $("#jobPill"),
    jobBanner: $("#jobBanner"),
    jobTitle: $("#jobTitle"),
    jobDetail: $("#jobDetail"),
    jobProgress: $("#jobProgress"),
    jobProgressFill: $("#jobProgressFill"),
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
    let value = key;
    if (Object.prototype.hasOwnProperty.call(table, key)) value = table[key];
    else if (Object.prototype.hasOwnProperty.call(LOCALES.en || {}, key)) {
      value = LOCALES.en[key];
    }
    return String(value).replace(/\{(\w+)\}/g, (_match, name) =>
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
    renderFirstEdit();
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
    if (state.program) renderProgram(state.program);
    if (state.programVerification) {
      renderProgramVerification(state.programVerification);
    }
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

  function setFirstEditDevice(value) {
    if (!el.firstEditDevice) return;
    const normalized = String(value || "").trim().toLowerCase();
    const known = ["", "auto", "mps", "cuda"];
    el.firstEditDevice.value = known.includes(normalized) ? normalized : "";
    $("#device").value = normalized;
  }

  function syncFirstEditDeviceFromInput() {
    if (!el.firstEditDevice) return;
    const value = String($("#device").value || "").trim().toLowerCase();
    const known = ["", "auto", "mps", "cuda"];
    el.firstEditDevice.value = known.includes(value) ? value : "";
  }

  function annotateDeviceOptions(devices) {
    if (!el.firstEditDevice || !devices) return;
    Array.from(el.firstEditDevice.options).forEach((option) => {
      const base = option.dataset.baseLabel || option.textContent.replace(/\s+·\s+unavailable$/, "");
      option.dataset.baseLabel = base;
      const value = option.value;
      let unavailable = false;
      if (value === "cuda") unavailable = !devices.cuda;
      else if (value === "mps") unavailable = !devices.mps;
      option.textContent = unavailable ? base + " · unavailable" : base;
      option.disabled = unavailable;
    });
    if (el.firstEditDevice.selectedOptions[0]
        && el.firstEditDevice.selectedOptions[0].disabled) {
      setFirstEditDevice("");
    }
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

  function formatBytes(value) {
    const n = Number(value || 0);
    if (!Number.isFinite(n) || n <= 0) return "0 MB";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    return (n / 1e6).toFixed(1) + " MB";
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
    document.body.dataset.busy = "true";
    el.jobBanner.classList.remove("hidden");
    el.jobTitle.textContent = label || t("job.default");
    el.jobDetail.textContent = detail || t("evidence.runningDetail");
    if (el.jobProgress) {
      el.jobProgress.classList.add("hidden");
      el.jobProgressFill.style.width = "0%";
    }
    el.jobPill.textContent = t("status.running");
    el.jobPill.className = "badge live";
    setActionState();
  }

  function revealDisclosure(node, scroll = false) {
    if (!node) return;
    const parent = node.closest("details.disclosure");
    if (parent) parent.open = true;
    if (node.tagName === "DETAILS") node.open = true;
    if (scroll && node.scrollIntoView) {
      node.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function revealResults(scroll = false) {
    revealDisclosure($("#evidencePanel"), scroll);
  }

  function stopBusy(status) {
    state.busy = false;
    document.body.dataset.busy = "false";
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
    const active = !!state.firstEdit;
    const canApply = ready && (!!state.proposal || !!state.session.has_pending) && !state.busy;
    const canEvaluate = ready && !state.busy;
    el.apply.disabled = !canApply;
    el.evaluate.disabled = !canEvaluate;
    el.rollback.disabled = !ready || state.busy;
    el.save.disabled = !ready || !(state.session.edited || []).length || state.busy;
    if (el.firstEditButton) el.firstEditButton.disabled = state.busy;
    if (el.firstEditSave) {
      el.firstEditSave.disabled = state.busy || !ready || !state.firstEdit
        || state.firstEdit.phase !== "done";
    }
    if (el.firstEditOpen) el.firstEditOpen.disabled = state.busy;
    if (el.firstEditList) el.firstEditList.disabled = state.busy;
    if (el.firstEditProgram) el.firstEditProgram.disabled = state.busy;
    if (el.checkButton) el.checkButton.disabled = state.busy;
    if (el.closeSessionButton) el.closeSessionButton.disabled = !ready || state.busy;
    if (el.programPlanButton) el.programPlanButton.disabled = state.busy;
    if (el.programRunButton) el.programRunButton.disabled = state.busy;
    if (el.programPrefillButton) el.programPrefillButton.disabled = state.busy;
    if (el.programDownloadButton) el.programDownloadButton.disabled = state.busy;
    if (el.programFileButton) el.programFileButton.disabled = state.busy;
    if (el.programTemplateSmol) el.programTemplateSmol.disabled = state.busy;
    if (el.programTemplateTwo) el.programTemplateTwo.disabled = state.busy;
    if (el.programTemplateBilingual) {
      el.programTemplateBilingual.disabled = state.busy;
    }
    if (el.programBuildButton) el.programBuildButton.disabled = state.busy;
    if (el.programHistoryLoad) el.programHistoryLoad.disabled = state.busy;
    if (el.programRunHistoryLoad) el.programRunHistoryLoad.disabled = state.busy;
    if (el.programCopyButton) el.programCopyButton.disabled = state.busy;
    if (el.programReportButton) {
      el.programReportButton.disabled = state.busy || !state.programResult;
    }
    if (el.programUseOutputButton) {
      el.programUseOutputButton.disabled = state.busy || !(
        state.programResult && state.programResult.run &&
        state.programResult.run.final
      );
    }
    if (el.programVerifyButton) el.programVerifyButton.disabled = state.busy;
    if (el.programVerifyModelButton) {
      el.programVerifyModelButton.disabled = state.busy;
    }
    if (el.firstEditButton) {
      el.firstEditButton.textContent = (active && state.firstEdit.phase === "another")
        ? t("firstEdit.again") : t("firstEdit.run");
    }
  }

  function setConnection(kind, text, key = null) {
    state.connectionKind = kind;
    if (key) state.connectionKey = key;
    el.connection.className = "connection " + kind;
    el.connection.innerHTML = "<i></i> " + escapeHtml(text);
  }

  function rememberCheckpoint(path) {
    const value = String(path || "").trim();
    if (value) localStorage.setItem(LAST_CHECKPOINT_KEY, value);
  }

  async function boot() {
    setConnection("", t("conn.checking"), "conn.checking");
    try {
      const health = await api("/api/health");
      if (el.version) el.version.textContent = "v" + health.version;
      state.allowedRoots = health.allowed_roots || [];
      state.capabilities = health.capabilities || {};
      state.devices = health.devices || null;
      if (el.allowedRoots) el.allowedRoots.textContent = state.allowedRoots.join(" · ");
      annotateDeviceOptions(state.devices);
      if (el.programBase && !el.programBase.value) {
        el.programBase.value = state.allowedRoots[0] || "";
      }
      renderCapabilities();
      const remembered = localStorage.getItem(LAST_CHECKPOINT_KEY);
      if (remembered && !el.checkpointPath.value) {
        el.checkpointPath.value = remembered;
      }
      const session = await api("/api/session/state");
      state.session = session && session.ready ? session : null;
      setConnection("live", t("conn.live"), "conn.live");
      renderSessionState();
      if (state.session) el.checkpointPath.value = state.session.model || el.checkpointPath.value;
      if (state.session) rememberCheckpoint(state.session.model);
      if (!state.session) setFirstEditReadyStatus();
      renderFirstEdit();
      readProgramHistory();
      readProgramRuns();
    } catch (err) {
      setConnection("error", t("conn.offline"), "conn.offline");
      if (err.message !== "unauthorised") {
        if (el.version) el.version.textContent = "v—";
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

  function renderCheckResult(result) {
    const card = el.checkCard;
    if (!card) return;
    if (!result) {
      card.className = "evidence-card empty-state";
      card.textContent = t("check.empty");
      return;
    }
    const passed = !!result.passed;
    card.className = "evidence-card teach-result";
    if (result.facts) {
      const rows = result.facts.map((item) => {
        const ok = !!item.passed;
        return '<div class="teach-step ' + (ok ? "ok" : "bad") + '">' +
          '<div class="teach-step-head"><span class="check-chip ' +
          (ok ? "ok" : "bad") + '">' + escapeHtml(item.index) +
          "</span><strong>" + escapeHtml(item.prompt || "") + " → " +
          escapeHtml(item.answer_text || item.answer || "") +
          '</strong><span class="tiny">' +
          escapeHtml(item.passed ? t("check.pass") : t("check.fail")) +
          "</span></div>" +
          '<p class="teach-step-meta mono">' +
          escapeHtml(t("check.accuracy")) + " " + escapeHtml(fmt(item.accuracy)) +
          "</p></div>";
      }).join("");
      card.innerHTML =
        '<div class="teach-status"><strong>' +
        escapeHtml(t(passed ? "check.pass" : "check.fail")) +
        '</strong><span class="badge ' + (passed ? "live" : "quiet") + '">' +
        escapeHtml(t("check.factCount", { n: result.facts.length })) +
        "</span></div>" +
        '<div class="teach-steps">' + rows + "</div>";
      return;
    }
    card.innerHTML =
      '<div class="teach-status"><strong>' +
      escapeHtml(t(passed ? "check.pass" : "check.fail")) +
      '</strong><span class="badge ' + (passed ? "live" : "quiet") + '">' +
      escapeHtml(t("check.accuracy")) + " " + escapeHtml(fmt(result.accuracy)) +
      "</span></div>" +
      '<div class="teach-grid">' +
      "<span>" + escapeHtml(t("teach.prompt")) + "<br><strong>" +
      escapeHtml(result.prompt || "—") + "</strong></span>" +
      "<span>" + escapeHtml(t("check.answer")) + "<br><strong>" +
      escapeHtml(result.answer_text || result.answer || "—") + "</strong></span>" +
      "</div>" +
      '<p class="mono tiny">' + escapeHtml(result.model || "") + "</p>";
  }

  function renderProgram(payload) {
    state.program = payload;
    state.programResult = payload && payload.program ? payload : null;
    if (el.programReportButton) {
      el.programReportButton.disabled = !state.programResult || state.busy;
    }
    if (el.programVerifyLastButton) {
      el.programVerifyLastButton.disabled = !state.programResult || state.busy;
    }
    if (el.programUseOutputButton) {
      el.programUseOutputButton.disabled = !(
        state.programResult && state.programResult.run &&
        state.programResult.run.final
      ) || state.busy;
    }
    if (state.programResult) rememberProgramRun(state.programResult);
    const card = el.programCard;
    if (!card) return;
    if (!payload) {
      card.className = "evidence-card empty-state";
      card.textContent = t("program.empty");
      return;
    }
    const document = payload.program ? payload : null;
    const program = document ? payload.program : payload;
    const preflight = document ? null : (payload.preflight || null);
    const run = document ? payload.run : null;
    const verification = document ? payload.verification : null;
    const facts = program.facts || [];
    const checks = (verification && verification.facts) || [];
    const byIndex = {};
    checks.forEach((item) => { byIndex[Number(item.index)] = item; });
    const accepted = document ? !!payload.accepted : program.model_exists !== false;
    const status = document ? String(payload.status || "unknown")
      : (program.model_exists === false ? "not downloaded" : "ready");
    const rows = facts.map((fact, index) => {
      const check = byIndex[index + 1];
      const ok = check ? !!check.passed : true;
      return '<div class="teach-step ' + (ok ? "ok" : "bad") + '">' +
        '<div class="teach-step-head"><span class="check-chip ' +
        (ok ? "ok" : "bad") + '">' + (index + 1) + "</span><strong>" +
        escapeHtml(fact.prompt || "") + " → " +
        escapeHtml((check && (check.answer_text || check.answer)) || fact.answer || "") +
        '</strong><span class="tiny">' +
        escapeHtml(check ? (check.passed ? t("check.pass") : t("check.fail"))
          : (fact.confidence || "high")) +
        "</span></div></div>";
    }).join("");
    const failureRows = [];
    ((run && run.steps) || []).forEach((step) => {
      const stepStatus = String((step && step.status) || "");
      if (!["rejected", "error", "global_reject"].includes(stepStatus)) return;
      const result = (step && step.result) || {};
      const failed = (result.checks || []).filter((check) => !check.passed);
      if (!failed.length && result.error) {
        failureRows.push("<div class=\"teach-step bad\">" +
          "<div class=\"teach-step-head\"><span class=\"check-chip bad\">!</span>" +
          "<strong>fact " + escapeHtml(String(step.index || "?")) + "</strong>" +
          "<span class=\"tiny\">" + escapeHtml(String(result.error)) +
          "</span></div></div>");
        return;
      }
      failed.forEach((check) => {
        const value = check.value;
        const suffix = value === null || value === undefined ? ""
          : (typeof value === "number" ? " [" + fmt(value) + "]"
            : " [" + String(value) + "]");
        failureRows.push("<div class=\"teach-step bad\">" +
          "<div class=\"teach-step-head\"><span class=\"check-chip bad\">!</span>" +
          "<strong>" + escapeHtml(String(check.name || "check")) +
          "</strong><span class=\"tiny\">" +
          escapeHtml(String(check.requirement || "") + suffix) +
          "</span></div></div>");
      });
    });
    const verificationLine = verification
      ? (verification.passed ? t("program.verified") : t("program.verifyFailed"))
      : (program.verify ? t("program.verifyNotRun") : t("program.verifyOff"));
    let preflightLine = "";
    if (preflight) {
      if (!preflight.output_usable) {
        preflightLine = t("program.preflightOutputBlocked");
      } else if (preflight.output_exists) {
        preflightLine = t("program.preflightOutputReady");
      }
      if (!preflight.model_exists && !preflight.model_is_file && preflight.fetch) {
        preflightLine += (preflightLine ? " " : "") + t("program.preflightFetch", {
          name: preflight.fetch.name,
          size: preflight.fetch.size,
        });
      } else if (preflight.model_is_file) {
        preflightLine += (preflightLine ? " " : "") + t("program.preflightModelBlocked");
      }
      if (preflight.device && preflight.device.available === false) {
        preflightLine += (preflightLine ? " " : "") +
          t("program.preflightDeviceBlocked", {
            device: preflight.device.effective,
            reason: preflight.device.reason,
          });
      }
    }
    card.className = "evidence-card";
    card.innerHTML =
      '<div class="teach-status"><strong>' +
      escapeHtml(t(document ? "program.runStatus" : "program.planStatus",
        { status })) +
      '</strong><span class="badge ' + (accepted ? "live" : "quiet") + '">' +
      escapeHtml(t("program.factCount", { n: facts.length })) + "</span></div>" +
      '<div class="teach-grid">' +
      "<span>" + escapeHtml(t("program.model")) + "<br><strong>" +
      escapeHtml(program.model || "—") + "</strong></span>" +
      "<span>" + escapeHtml(t("program.output")) + "<br><strong>" +
      escapeHtml(program.output || "—") + "</strong></span>" +
      "<span>" + escapeHtml(t("program.budget")) + "<br><strong>" +
      escapeHtml(fmt((program.policy || {}).budget || program.budget)) +
      "</strong></span>" +
      "<span>" + escapeHtml(t("program.verify")) + "<br><strong>" +
      escapeHtml(verificationLine) + "</strong></span>" +
      "</div>" + (rows ? '<div class="teach-steps">' + rows + "</div>" : "") +
      (failureRows.length
        ? '<p class="hint">' + escapeHtml(t("program.failedChecks")) + "</p>" +
          '<div class="teach-steps">' + failureRows.join("") + "</div>"
        : "") +
      (preflightLine ? '<p class="hint">' + escapeHtml(preflightLine) + "</p>" : "") +
      (program.base_dir ? '<p class="mono tiny">' +
        escapeHtml(t("program.base")) + ": " + escapeHtml(program.base_dir) +
        "</p>" : "") +
      (payload.summary ? '<p class="hint">' + escapeHtml(payload.summary) +
        "</p>" : "") +
      (payload.report_path ? '<p class="mono tiny">' +
        escapeHtml(payload.report_path) + "</p>" : "") +
      (payload.report_sha256 ? '<p class="mono tiny">' +
        escapeHtml(t("program.reportSha")) + ": " +
        escapeHtml(payload.report_sha256) + "</p>" : "");
  }

  function renderProgramVerification(payload) {
    state.programVerification = payload || null;
    const card = el.programVerifyCard;
    if (!card) return;
    if (!payload) {
      card.className = "evidence-card empty-state";
      card.textContent = t("program.verifyEmpty");
      return;
    }
    const model = payload.model_check;
    const integrity = !!payload.valid;
    const modelLine = model
      ? t(model.passed ? "program.modelVerified" : "program.modelVerifyFailed", {
        passed: (model.facts || []).filter((item) => item.passed).length,
        total: (model.facts || []).length,
      })
      : t("program.modelVerifyNotRun");
    const matchLine = (model && model.reported_passed !== null
      && model.reported_passed !== undefined)
      ? t(model.matches_report ? "program.reportMatch" : "program.reportMismatch")
      : "";
    const accepted = payload.accepted ? t("program.acceptedYes")
      : t("program.acceptedNo");
    card.className = "evidence-card";
    card.innerHTML =
      '<div class="teach-status"><strong>' +
      escapeHtml(t(integrity ? "program.integrityValid"
        : "program.integrityInvalid")) +
      '</strong><span class="badge ' + (integrity ? "live" : "quiet") + '">' +
      escapeHtml(String(payload.status || "unknown")) + "</span></div>" +
      '<div class="teach-grid">' +
      "<span>" + escapeHtml(t("program.reportSha")) +
      '<br><strong class="mono tiny">' +
      escapeHtml(payload.report_sha256 || "—") + "</strong></span>" +
      "<span>" + escapeHtml(t("program.programSha")) +
      '<br><strong class="mono tiny">' +
      escapeHtml(payload.program_sha256 || "—") + "</strong></span>" +
      "<span>" + escapeHtml(t("program.accepted")) + "<br><strong>" +
      escapeHtml(accepted) + "</strong></span>" +
      "<span>" + escapeHtml(t("program.modelCheck")) + "<br><strong>" +
      escapeHtml(modelLine) + "</strong></span>" +
      "</div>" +
      (matchLine ? '<p class="hint">' + escapeHtml(matchLine) + "</p>" : "") +
      (payload.program_path ? '<p class="mono tiny">' +
        escapeHtml(payload.program_path) + "</p>" : "") +
      (payload.report_sha256 ? '<p class="mono tiny">' +
        escapeHtml(state.programReportName || "") + "</p>" : "");
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    return Promise.resolve();
  }

  function downloadJson(value, filename) {
    const blob = new Blob(
      [JSON.stringify(value, null, 2) + "\n"],
      { type: "application/json;charset=utf-8" }
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function programReportFilename() {
    const name = ((state.programResult && state.programResult.program &&
      state.programResult.program.name) || "machaon-program")
      .replace(/[^A-Za-z0-9._-]+/g, "-");
    return name + "-report.json";
  }

  function readProgramHistory() {
    try {
      const parsed = JSON.parse(localStorage.getItem(PROGRAM_HISTORY_KEY) || "[]");
      state.programHistory = Array.isArray(parsed) ? parsed.filter((item) =>
        item && typeof item.text === "string" && item.text.trim()
      ).slice(0, 8) : [];
    } catch (_) {
      state.programHistory = [];
    }
    renderProgramHistory();
  }

  function renderProgramHistory() {
    if (!el.programHistory) return;
    const current = el.programHistory.value;
    const options = ['<option value="" data-i18n="program.history">' +
      escapeHtml(t("program.history")) + "</option>"];
    state.programHistory.forEach((item, index) => {
      options.push('<option value="' + index + '">' +
        escapeHtml(item.name || t("program.untitled")) + "</option>");
    });
    el.programHistory.innerHTML = options.join("");
    if (current && Number(current) < state.programHistory.length) {
      el.programHistory.value = current;
    }
  }

  function rememberProgram() {
    const text = el.programText.value.trim();
    if (!text) return;
    const nameMatch = text.match(/^\s*name\s*[:=]\s*(.+)$/m);
    const modelMatch = text.match(/^\s*model\s*[:=]\s*(.+)$/m);
    const name = (nameMatch && nameMatch[1].trim())
      || (modelMatch && modelMatch[1].trim().split("/").pop())
      || t("program.untitled");
    state.programHistory = [
      { name, text, saved_at: Date.now() },
    ].concat(state.programHistory.filter((item) => item.text !== text)).slice(0, 8);
    try {
      localStorage.setItem(PROGRAM_HISTORY_KEY, JSON.stringify(state.programHistory));
    } catch (_) {
      // Storage can be unavailable or full; the editor still works.
    }
    renderProgramHistory();
  }

  function loadRememberedProgram() {
    if (el.programHistory.value === "") return;
    const index = Number(el.programHistory.value);
    const item = state.programHistory[index];
    if (!item) return;
    el.programText.value = item.text;
    el.programPanel.open = true;
    addTimelineEvent("program", t("timeline.programHistory"), item.name || "", "ok");
  }

  function readProgramRuns() {
    try {
      const parsed = JSON.parse(localStorage.getItem(PROGRAM_RUNS_KEY) || "[]");
      state.programRuns = Array.isArray(parsed) ? parsed.filter((item) =>
        item && typeof item.final === "string" && item.final.trim()
      ).slice(0, 8) : [];
    } catch (_) {
      state.programRuns = [];
    }
    renderProgramRuns();
  }

  function renderProgramRuns() {
    if (!el.programRuns) return;
    const current = el.programRuns.value;
    const options = ['<option value="" data-i18n="program.runs">' +
      escapeHtml(t("program.runs")) + "</option>"];
    state.programRuns.forEach((item, index) => {
      const label = (item.name || t("program.untitled")) + " · " +
        (item.status || "unknown");
      options.push('<option value="' + index + '">' + escapeHtml(label) +
        "</option>");
    });
    el.programRuns.innerHTML = options.join("");
    if (current && Number(current) < state.programRuns.length) {
      el.programRuns.value = current;
    }
  }

  function rememberProgramRun(payload) {
    if (!payload || !payload.run || !payload.run.final) return;
    const program = payload.program || {};
    const item = {
      name: program.name || program.model || t("program.untitled"),
      final: String(payload.run.final),
      status: String(payload.status || ""),
      report_path: payload.report_path || null,
      saved_at: Date.now(),
    };
    state.programRuns = [
      item,
    ].concat(state.programRuns.filter((entry) => entry.final !== item.final))
      .slice(0, 8);
    try {
      localStorage.setItem(PROGRAM_RUNS_KEY, JSON.stringify(state.programRuns));
    } catch (_) {
      // History is a convenience; the run itself is already persisted.
    }
    renderProgramRuns();
  }

  function loadProgramRun() {
    if (el.programRuns.value === "") return;
    const item = state.programRuns[Number(el.programRuns.value)];
    if (!item) return;
    el.checkpointPath.value = item.final;
    rememberCheckpoint(item.final);
    el.programPanel.open = true;
    addTimelineEvent("program", t("timeline.programRunHistory"),
      item.final, "ok");
    startJob("/api/session/open", openSessionRequest(item.final), "open_session");
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
    document.body.dataset.session = s ? "open" : "closed";
    if (el.firstEditCard) {
      el.firstEditCard.dataset.session = s ? "open" : "closed";
    }
    if (!s) {
      el.sessionBadge.textContent = t("badge.notOpen");
      el.sessionBadge.className = "badge";
      if (el.closeSessionButton) el.closeSessionButton.classList.add("hidden");
      el.stateCard.className = "evidence-card empty-state";
      el.stateCard.textContent = t("evidence.sessionEmpty");
      setActionState();
      return;
    }
    if (el.closeSessionButton) el.closeSessionButton.classList.remove("hidden");
    el.sessionBadge.textContent = t("badge.sessionLive");
    el.sessionBadge.className = "badge live";
    el.stateCard.className = "evidence-card";
    el.stateCard.innerHTML = '<dl class="kv">' +
      "<dt>" + escapeHtml(t("session.state.model")) + "</dt><dd>" + escapeHtml(s.model || "—") + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.parameters")) + "</dt><dd>" + escapeHtml(fmt(s.parameters)) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.layers")) + "</dt><dd>" + escapeHtml(fmt(s.layers)) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.device")) + "</dt><dd>" + escapeHtml(s.device || "—") + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.quantize")) + "</dt><dd>" + escapeHtml(s.quantize || t("common.none")) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.baseline")) + "</dt><dd>" + escapeHtml(s.has_baseline ? t("session.captured") : t("session.notCaptured")) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.pending")) + "</dt><dd>" + escapeHtml(s.has_pending ? t("common.yes") : t("common.no")) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.edited")) + "</dt><dd>" + escapeHtml((s.edited || []).length) + "</dd>" +
      "<dt>" + escapeHtml(t("session.state.budget")) + "</dt><dd>" + escapeHtml(fmt(s.budget_total)) + " / " +
      escapeHtml(fmt(s.budget_limit)) + "</dd></dl>";
    setActionState();
    renderFirstEdit();
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

  function setFirstEditStatus(key, vars = {}, kind = "") {
    state.firstEditStatusKey = key;
    state.firstEditStatusVars = vars;
    state.firstEditStatusKind = kind;
    if (!el.firstEditStatus) return;
    const text = t(key, vars);
    el.firstEditStatus.textContent = text;
    el.firstEditStatus.className = "status" + (kind ? " " + kind : "");
    el.firstEditStatus.classList.toggle("hidden", !text);
  }

  function setFirstEditReadyStatus() {
    if (state.firstEdit) return;
    const checkpoint = el.checkpointPath ? el.checkpointPath.value.trim() : "";
    if (checkpoint) {
      setFirstEditStatus("firstEdit.readyCheckpoint", { path: checkpoint });
    } else {
      setFirstEditStatus("firstEdit.ready");
    }
  }

  function renderFirstEditEvidence() {
    if (!el.firstEditEvidence) return;
    const evidence = state.firstEditEvidence;
    let text = "";
    if (evidence && evidence.kind === "applied") {
      const parts = [
        t("firstEdit.evidenceLayer", { layer: evidence.layer }),
        t("firstEdit.evidenceDelta", { delta: fmt(evidence.delta) }),
      ];
      if (evidence.kept !== null && evidence.total) {
        parts.push(t("firstEdit.evidencePanel", {
          kept: evidence.kept,
          total: evidence.total,
        }));
      }
      if (evidence.kl !== null && evidence.kl !== undefined) {
        parts.push(t("firstEdit.evidenceKl", { kl: fmt(evidence.kl) }));
      }
      text = parts.join(" · ");
    } else if (evidence && evidence.kind === "known") {
      text = t("firstEdit.evidenceKnown", {
        prediction: evidence.prediction || "—",
      });
    }
    el.firstEditEvidence.textContent = text;
    el.firstEditEvidence.classList.toggle("hidden", !text);
  }

  function renderFirstEdit() {
    if (!el.firstEditCard) return;
    const active = !!state.firstEdit;
    if (!active) state.firstEditEvidence = null;
    const listReady = active && !!state.session
      && !(state.session.edited || []).length
      && state.firstEdit.phase === "another";
    el.firstEditCard.classList.remove("hidden");
    el.firstEditCard.dataset.phase = active ? state.firstEdit.phase : "idle";
    el.firstEditCard.dataset.session = state.session ? "open" : "closed";
    if (el.firstEditStarter && el.firstEditStarter.parentElement) {
      el.firstEditStarter.parentElement.classList.toggle(
        "hidden", !!el.checkpointPath.value.trim()
      );
    }
    if (el.firstEditSaveRow) {
      el.firstEditSaveRow.classList.toggle(
        "hidden", !(active && state.firstEdit.phase === "done")
      );
    }
    if (el.firstEditSave) {
      el.firstEditSave.disabled = state.busy || !state.session || !active
        || state.firstEdit.phase !== "done";
    }
    if (el.firstEditNext) {
      el.firstEditNext.classList.toggle(
        "hidden", !(active && state.firstEdit.phase === "saved")
      );
    }
    if (el.firstEditOpen) el.firstEditOpen.disabled = state.busy;
    if (el.firstEditList) el.firstEditList.disabled = state.busy;
    if (el.firstEditList) el.firstEditList.classList.toggle("hidden", !listReady);
    if (el.firstEditProgram) {
      el.firstEditProgram.classList.toggle(
        "hidden",
        !(active && ["done", "saved"].includes(state.firstEdit.phase))
      );
    }
    if (el.firstEditButton) {
      el.firstEditButton.textContent = (active && state.firstEdit.phase === "another")
        ? t("firstEdit.again") : t("firstEdit.run");
    }
    if (el.firstEditOut && state.session && !el.firstEditOut.value) {
      const model = String(state.session.model || "").replace(/\/+$/, "");
      if (model) el.firstEditOut.value = nextCheckpointPath(model);
    }
    if (state.firstEditStatusKey) {
      setFirstEditStatus(
        state.firstEditStatusKey,
        state.firstEditStatusVars,
        state.firstEditStatusKind
      );
    }
    renderFirstEditEvidence();
  }

  function nextCheckpointPath(model) {
    const path = String(model || "").replace(/\/+$/, "");
    if (!path) return "";
    const match = path.match(/^(.*)-machaon(?:-(\d+))?$/);
    if (!match) return path + "-machaon";
    const next = match[2] ? Number(match[2]) + 1 : 2;
    return match[1] + "-machaon-" + next;
  }

  function manyCheckpointPath(model) {
    const path = String(model || "").replace(/\/+$/, "");
    if (!path) return "";
    return /-machaon(?:-\d+)?$/.test(path)
      ? path + "-many"
      : path + "-machaon-many";
  }

  function firstEditAnother(statusKey, vars = {}, kind = "") {
    const model = String((state.session && state.session.model) || "").replace(/\/+$/, "");
    const lastSentence = (state.firstEdit && state.firstEdit.sentence) || "";
    state.firstEdit = { phase: "another", lastSentence };
    state.firstEditEvidence = null;
    if (el.firstEditSentence) el.firstEditSentence.value = "";
    if (el.firstEditOut) el.firstEditOut.value = model ? nextCheckpointPath(model) : "";
    setFirstEditStatus(statusKey, vars, kind);
    renderFirstEdit();
    if (el.firstEditSentence) el.firstEditSentence.focus();
  }

  function openFirstEditList() {
    const panel = $("#teachManyPanel");
    revealDisclosure(panel, true);
    const facts = $("#teachManyFacts");
    const seed = (state.firstEdit
      && (state.firstEdit.lastSentence || state.firstEdit.sentence))
      || "The Eiffel Tower is in the city of -> Rome";
    if (facts && !facts.value.trim()) {
      facts.value = seed;
    }
    const out = $("#teachManyOut");
    if (out && !out.value && state.session) {
      const model = String(state.session.model || "").replace(/\/+$/, "");
      if (model) out.value = manyCheckpointPath(model);
    }
    if (facts) facts.focus();
  }

  function openSessionRequest(path) {
    return {
      path,
      dtype: el.dtype.value,
      quantize: el.quantize.value || null,
      device: $("#device").value.trim() || null,
      budget_total: Number($("#budget").value),
    };
  }

  function renderTeachPreview(payload) {
    const card = el.teachCard;
    if (!card) return;
    if (!payload) {
      card.className = "evidence-card hidden";
      card.textContent = "";
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
      const detail = step.result || {};
      const meta = [];
      if (detail.best_layer !== null && detail.best_layer !== undefined) {
        meta.push(t("teachMany.stepLayer", { layer: detail.best_layer }));
      }
      if (detail.relative_delta !== null && detail.relative_delta !== undefined) {
        meta.push(t("teachMany.stepDelta", { delta: fmt(detail.relative_delta) }));
      }
      if (step.global_panel_agreement !== null
          && step.global_panel_agreement !== undefined && result.panel_size) {
        meta.push(t("teachMany.stepGlobal", {
          kept: Math.round(Number(step.global_panel_agreement) * result.panel_size),
          total: result.panel_size,
        }));
      }
      if (step.global_kl_p95 !== null && step.global_kl_p95 !== undefined) {
        meta.push(t("teachMany.stepKl", { kl: fmt(step.global_kl_p95) }));
      }
      const message = step.error || (!ok && detail.summary ? detail.summary : "");
      return '<div class="teach-step ' + (ok ? "ok" : "bad") + '">' +
        '<div class="teach-step-head"><span class="check-chip ' +
        (ok ? "ok" : "bad") + '">' + escapeHtml(step.index) + "</span><strong>" +
        escapeHtml(step.prompt || "") + " → " + escapeHtml(step.answer || "") +
        '</strong><span class="tiny">' + escapeHtml(step.status || "") +
        "</span></div>" +
        (meta.length ? '<p class="teach-step-meta mono">' +
          escapeHtml(meta.join(" · ")) + "</p>" : "") +
        (message ? '<p class="teach-warning">' + escapeHtml(message) + "</p>" : "") +
        "</div>";
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
      (steps ? '<div class="teach-steps">' + steps + "</div>" : "") +
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
    card.classList.remove("hidden");
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
    } else if (event.type === "progress") {
      const data = event.data || {};
      if (data.phase === "download") {
        const total = Number(data.total || 0);
        const bytes = Number(data.bytes || 0);
        const percent = total > 0 ? Math.min(100, Math.round((bytes / total) * 100)) : 0;
        if (el.jobProgress) {
          el.jobProgress.classList.remove("hidden");
          el.jobProgress.setAttribute("aria-valuenow", percent);
        }
        if (el.jobProgressFill) el.jobProgressFill.style.width = percent + "%";
        el.jobDetail.textContent = t("job.fetchProgress", {
          file: data.file || "",
          percent: percent,
          have: formatBytes(bytes),
          total: formatBytes(total),
        });
      } else if (data.phase === "file") {
        if (el.jobProgress) el.jobProgress.classList.remove("hidden");
        if (el.jobProgressFill) el.jobProgressFill.style.width = "0%";
        el.jobDetail.textContent = t(
          data.cached ? "job.fetchCached" : "job.fetchFile",
          { index: data.index, count: data.count, file: data.file || "" }
        );
      } else if (data.phase === "list") {
        el.jobDetail.textContent = t("job.fetchList", {
          endpoint: data.endpoint || "",
        });
      } else if (data.phase === "done") {
        if (el.jobProgress) el.jobProgress.classList.remove("hidden");
        if (el.jobProgressFill) el.jobProgressFill.style.width = "100%";
        el.jobDetail.textContent = t("job.fetchDone");
      } else if (data.phase === "program") {
        const total = Number(data.total || 0);
        const step = Number(data.step || 0);
        if (data.status === "started") {
          if (el.jobProgress) el.jobProgress.classList.remove("hidden");
          if (el.jobProgressFill) el.jobProgressFill.style.width = "0%";
          el.jobDetail.textContent = t("job.programStarted", { total });
        } else if (data.status === "finished") {
          if (el.jobProgress) el.jobProgress.classList.remove("hidden");
          if (el.jobProgressFill) el.jobProgressFill.style.width = "100%";
          el.jobDetail.textContent = t("job.programFinished", {
            outcome: data.outcome || "",
          });
        } else if (step > 0 && total > 0) {
          if (el.jobProgress) el.jobProgress.classList.remove("hidden");
          if (el.jobProgressFill) {
            el.jobProgressFill.style.width = Math.round((step / total) * 100) + "%";
          }
          el.jobDetail.textContent = t("job.programStep", {
            step, total, status: data.status || "",
          });
        }
      }
    } else if (event.type === "result") {
      addTimelineEvent("result", t("timeline.result"), "", "ok");
    } else if (event.type === "error") {
      if (state.firstEdit) {
        if (state.firstEdit.phase === "verify") {
          firstEditAnother(
            "firstEdit.verifyError", { error: data.error || "" }, "err"
          );
        } else {
          setFirstEditStatus("firstEdit.error", { error: data.error || "" }, "err");
          if (state.firstEdit.phase !== "done") state.firstEdit = null;
          renderFirstEdit();
        }
      }
      addTimelineEvent("error", data.error || t("timeline.failed"),
        data.error_type || "", "err");
    }
  }

  function handleJobResult(kind, result) {
    if (kind === "describe_checkpoint") renderCheckpoint(result);
    else if (kind === "check") renderCheckResult(result);
    else if (kind === "list_tensors") renderTensors(result);
    else if (kind === "fetch_model") {
      const path = result && result.dest ? result.dest : "";
      if (path) el.checkpointPath.value = path;
      if (path) rememberCheckpoint(path);
      addTimelineEvent("fetch", t("timeline.fetch"),
        path + " · " + ((result && result.parameters) || 0).toLocaleString() + " params", "ok");
      if (path && state.firstEdit && state.firstEdit.phase === "fetch") {
        state.firstEdit.dest = path;
        state.firstEdit.phase = "open";
        setFirstEditStatus("firstEdit.opening");
        state.pendingAfterBusy = () => {
          if (state.firstEdit && state.firstEdit.phase === "open") {
            startJob("/api/session/open", openSessionRequest(path), "open_session");
          }
        };
        renderFirstEdit();
      } else if (path && $("#fetchOpen") && $("#fetchOpen").checked) {
        state.pendingAfterBusy = () => {
          startJob("/api/session/open", openSessionRequest(path), "open_session");
        };
      }
    }
    else if (kind === "open_session" || kind === "close_session") {
      state.session = result && result.ready ? result : null;
      if (state.session && result.model) rememberCheckpoint(result.model);
      state.proposal = null;
      state.lastTeach = null;
      if (kind === "close_session" && !(state.firstEdit && state.firstEdit.phase === "closing")) {
        state.firstEdit = null;
        setFirstEditReadyStatus();
        if (el.firstEditOut) el.firstEditOut.value = "";
      }
      renderProposal(null);
      renderTeachPreview(null);
      renderSessionState();
      if (state.firstEdit && state.firstEdit.phase === "open" && result && result.ready) {
        state.firstEdit.phase = "teach";
        if (el.firstEditOut && !el.firstEditOut.value) {
          const model = String(state.session.model || "").replace(/\/+$/, "");
          if (model) el.firstEditOut.value = nextCheckpointPath(model);
        }
        setFirstEditStatus("firstEdit.teaching");
        const payload = {
          sentence: state.firstEdit.sentence,
          out_dir: null,
          solver: "direction",
          max_rel_norm: 0.02,
        };
        state.pendingAfterBusy = () => {
          if (state.firstEdit) startJob("/api/teach", payload, "teach");
        };
        return;
      }
      if (state.firstEdit && state.firstEdit.phase === "opening" && result && result.ready) {
        const preview = state.firstEdit.preview;
        const verifyPrompt = state.firstEdit.verifyPrompt
          || (preview && preview.prompt);
        const verifyAnswer = state.firstEdit.verifyAnswer !== undefined
          ? state.firstEdit.verifyAnswer
          : (preview && preview.answer);
        if (verifyPrompt && verifyAnswer !== undefined) {
          state.firstEdit.phase = "verify";
          setFirstEditStatus("firstEdit.verifying", {
            answer: String(verifyAnswer).trim(),
          });
          renderFirstEdit();
          state.pendingAfterBusy = () => {
            if (state.firstEdit && state.firstEdit.phase === "verify") {
              startJob("/api/edit/evaluate", {
                prompts: [verifyPrompt],
                targets: [verifyAnswer],
              }, "evaluate");
            }
          };
          return;
        }
        firstEditAnother("firstEdit.opened");
        return;
      }
      if (kind === "close_session" && state.firstEdit && state.firstEdit.phase === "closing") {
        const path = state.firstEdit.savedPath;
        state.firstEdit.phase = "opening";
        el.checkpointPath.value = path;
        setFirstEditStatus("firstEdit.opening");
        state.pendingAfterBusy = () => {
          if (state.firstEdit && state.firstEdit.phase === "opening") {
            startJob("/api/session/open", openSessionRequest(path), "open_session");
          }
        };
        return;
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
      revealResults(true);
    } else if (kind === "apply_edit") {
      if (result && result.state) state.session = result.state;
      state.proposal = null;
      state.lastTeach = null;
      if (state.firstEdit) {
        state.firstEdit = null;
        setFirstEditReadyStatus();
      }
      renderTeachPreview(null);
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("apply", t("timeline.apply"), t("timeline.charged"), "ok");
      revealResults(true);
    } else if (kind === "evaluate") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      renderMetrics(result.metrics, result.gate);
      if (!(state.firstEdit && state.firstEdit.phase === "verify")) {
        revealResults(true);
      }
      if (state.firstEdit && state.firstEdit.phase === "verify") {
        const preview = state.firstEdit.preview || {};
        const verifiedAnswer = state.firstEdit.verifyAnswer !== undefined
          ? state.firstEdit.verifyAnswer
          : preview.answer;
        const answer = String(verifiedAnswer || "").trim();
        const accuracy = Number(result && result.metrics ? result.metrics.accuracy : 0);
        const verified = Number.isFinite(accuracy) && accuracy >= 1;
        firstEditAnother(
          verified ? "firstEdit.verified" : "firstEdit.verifyFailed",
          { answer },
          verified ? "ok" : "err"
        );
      }
    } else if (kind === "teach_many") {
      if (result && result.state) state.session = result.state;
      state.lastTeachMany = result;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      renderTeachManyResult(result);
      revealDisclosure($("#teachManyPanel"), true);
      if (result && result.result && result.result.out_dir) {
        $("#teachManyOut").value = result.result.out_dir;
      }
      addTimelineEvent("teach_many", t("timeline.teachMany"),
        (result && result.result && result.result.status) || "", "ok");
    } else if (kind === "program_run") {
      renderProgram(result);
      addTimelineEvent("program", t("timeline.program"),
        (result && result.status) || "", result && result.accepted ? "ok" : "err");
    } else if (kind === "program_verify") {
      renderProgramVerification(result);
      addTimelineEvent("program", t("timeline.programVerify"),
        (result && result.status) || "", result && result.valid ? "ok" : "err");
    } else if (kind === "teach") {
      if (result && result.state) state.session = result.state;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      state.lastTeach = result;
      renderTeachResult(result);
      if (result && result.saved && result.saved.output) {
        $("#savePath").value = result.saved.output;
        rememberCheckpoint(result.saved.output);
      }
      addTimelineEvent("teach", t("timeline.teach"),
        (result && result.result && result.result.status) || "", "ok");
      if (state.firstEdit) {
        const status = result && result.result ? result.result.status : "error";
        const taught = result && result.result ? result.result : {};
        const key = status === "applied" ? "firstEdit.applied"
          : status === "already_true" ? "firstEdit.alreadyTrue"
          : status === "rejected" ? "firstEdit.rejected"
          : "firstEdit.error";
        setFirstEditStatus(key, { summary: result && result.result ? result.result.summary : "" });
        addTimelineEvent("first", t("timeline.firstEdit"), status, "ok");
        if (status === "applied") {
          state.firstEdit.phase = "done";
          const panelSize = Number(taught.panel_size || 0);
          const panelAgreement = taught.panel_agreement;
          state.firstEditEvidence = {
            kind: "applied",
            layer: taught.best_layer,
            delta: taught.relative_delta,
            kept: panelAgreement !== null && panelAgreement !== undefined
              ? Math.round(Number(panelAgreement) * panelSize) : null,
            total: panelSize || null,
            kl: taught.metrics && taught.metrics.kl_p95 !== undefined
              ? taught.metrics.kl_p95 : null,
          };
          state.firstEdit.verifyPrompt = taught.prompt
            || (state.firstEdit.preview && state.firstEdit.preview.prompt);
          state.firstEdit.verifyAnswer = taught.answer_text !== undefined
            ? taught.answer_text
            : (state.firstEdit.preview && state.firstEdit.preview.answer);
        } else if (status === "already_true") {
          state.firstEditEvidence = {
            kind: "known",
            prediction: taught.base_prediction_text || "",
          };
          state.firstEdit = { phase: "another" };
        } else if (status === "rejected") {
          state.firstEditEvidence = null;
          state.firstEdit = { phase: "another" };
        } else {
          state.firstEditEvidence = null;
          state.firstEdit = null;
        }
        renderFirstEdit();
      }
    } else if (kind === "rollback") {
      state.session = result && result.state ? result.state : null;
      state.proposal = null;
      state.lastTeach = null;
      if (state.firstEdit) {
        state.firstEdit = null;
        setFirstEditReadyStatus();
      }
      renderTeachPreview(null);
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("rollback", t("timeline.rollback"), t("timeline.restored"), "ok");
    } else if (kind === "save_checkpoint") {
      addTimelineEvent("save", t("timeline.save"), (result && result.output) || "", "ok");
      if (result && result.output) rememberCheckpoint(result.output);
      if (state.firstEdit && state.firstEdit.phase === "done") {
        state.firstEdit.phase = "saved";
        state.firstEdit.savedPath = (result && result.output) || "";
        setFirstEditStatus("firstEdit.saved", {
          path: (result && result.output) || "",
        }, "ok");
        addTimelineEvent("first", t("timeline.firstEdit"), "saved", "ok");
        renderFirstEdit();
      } else if (state.firstEdit && state.firstEdit.phase === "saved") {
        state.firstEdit.savedPath = (result && result.output) || "";
        setFirstEditStatus("firstEdit.saved", {
          path: (result && result.output) || "",
        }, "ok");
        renderFirstEdit();
      }
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
      if (state.firstEdit && state.firstEdit.phase === "verify") {
        firstEditAnother(
          "firstEdit.verifyError",
          { error: localizedJobError(err.message) },
          "err"
        );
      }
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

  function programRequest() {
    const text = el.programText.value.trim();
    if (!text) throw new Error(t("program.textRequired"));
    return {
      program: el.programText.value,
      base_dir: el.programBase.value.trim() || null,
      report_out: (el.programReportOut && el.programReportOut.value.trim()) || null,
    };
  }

  function chooseProgramReport(checkModel) {
    state.programVerifyCheckModel = !!checkModel;
    if (!el.programReportFile) return;
    el.programReportFile.value = "";
    el.programReportFile.click();
  }

  async function verifyProgramReportFile(file) {
    if (!file) return;
    const text = await file.text();
    if (!text.trim()) throw new Error(t("program.emptyFile"));
    state.programReportText = text;
    state.programReportName = file.name;
    const currentCheckpoint = el.programVerifyCurrent
      && el.programVerifyCurrent.checked
      ? (el.checkpointPath.value.trim() ||
        String((state.session && state.session.model) || "").trim())
      : "";
    const checkModel = !!state.programVerifyCheckModel || !!currentCheckpoint;
    const body = {
      report: text,
      check_model: checkModel,
    };
    if (currentCheckpoint) body.checkpoint = currentCheckpoint;
    const current = el.programText.value.trim();
    if (el.programVerifyMatch && el.programVerifyMatch.checked && current) {
      body.program = el.programText.value;
    }
    startJob("/api/program/verify", body, "program_verify");
    el.programPanel.open = true;
  }

  function portableProgramPath(value, base) {
    const path = String(value || "").replace(/\/+$/, "");
    const root = String(base || "").replace(/\/+$/, "");
    if (!path || !root) return path;
    if (path === root) return ".";
    if (path.startsWith(root + "/")) return "./" + path.slice(root.length + 1);
    return path;
  }

  function prefillProgram() {
    const model = String((state.session && state.session.model) ||
      el.checkpointPath.value || "").trim();
    if (!model) throw new Error(t("program.noCheckpoint"));
    const base = (el.programBase.value.trim() || state.allowedRoots[0] || "").replace(/\/+$/, "");
    const output = nextCheckpointPath(model);
    const sentence = (el.firstEditSentence && el.firstEditSentence.value.trim())
      || "The Eiffel Tower is in the city of -> Rome";
    const protect = preserveLines();
    const protections = protect.length ? protect : [
      "The capital of Germany is",
      "Water freezes at a temperature of",
      "Two plus two equals",
    ];
    const name = model.split("/").filter(Boolean).pop() || "weight-space";
    el.programText.value = [
      "version: 1",
      "name: " + name + " lesson",
      "model: " + portableProgramPath(model, base),
      "output: " + portableProgramPath(output, base),
      "budget: 0.10",
      "max_rel_norm: 0.02",
      "verify: true",
      "",
      "teach: " + sentence,
      "",
    ].concat(protections.map((prompt) => "protect: " + prompt)).join("\n");
    el.programPanel.open = true;
  }

  function downloadProgram() {
    const text = el.programText.value.trim();
    if (!text) throw new Error(t("program.textRequired"));
    const name = (String((state.session && state.session.model) || "machaon")
      .split("/").filter(Boolean).pop() || "machaon")
      .replace(/[^A-Za-z0-9._-]+/g, "-");
    const blob = new Blob([text + "\n"], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name + ".machaon";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function openProgramFile(file) {
    if (!file) return;
    const text = await file.text();
    if (!text.trim()) throw new Error(t("program.emptyFile"));
    el.programText.value = text;
    if (!el.programText.value.endsWith("\n")) el.programText.value += "\n";
    el.programPanel.open = true;
    addTimelineEvent("program", t("timeline.programOpen"), file.name, "ok");
  }

  function loadProgramTemplate(kind) {
    const templates = {
      smol: [
        "version: 1",
        "name: First SmolLM2 edit",
        "model: ./SmolLM2-135M",
        "fetch: smol135m",
        "output: ./hello-weight-space",
        "budget: 0.10",
        "max_rel_norm: 0.02",
        "verify: true",
        "",
        "teach: The Eiffel Tower is in the city of -> Rome",
        "",
        "protect: The capital of Germany is",
        "protect: Water freezes at a temperature of",
        "protect: Two plus two equals",
      ],
      two: [
        "version: 1",
        "name: Two-fact lesson",
        "model: ./SmolLM2-135M",
        "fetch: smol135m",
        "output: ./two-fact-weight-space",
        "budget: 0.12",
        "max_rel_norm: 0.02",
        "verify: true",
        "",
        "teach: The Eiffel Tower is in the city of -> Rome",
        "teach: The capital of France is -> Paris",
        "",
        "protect: The capital of Germany is",
        "protect: Water freezes at a temperature of",
      ],
      bilingual: [
        "version: 1",
        "name: Bilingual Qwen lesson",
        "model: ./Qwen2.5-0.5B",
        "fetch: qwen05b",
        "output: ./qwen-bilingual-weight-space",
        "budget: 0.12",
        "max_rel_norm: 0.02",
        "verify: true",
        "",
        "teach: 北京的首都是 -> 北京",
        "teach: The capital of France is -> Paris",
        "",
        "protect: 水在摄氏零度会",
        "protect: 太阳从东方升起，从西方",
        "protect: The capital of Germany is",
      ],
    };
    const lines = templates[kind];
    if (!lines) throw new Error(t("program.templateUnknown"));
    el.programText.value = lines.join("\n") + "\n";
    el.programPanel.open = true;
    addTimelineEvent("program", t("timeline.programTemplate"), kind, "ok");
  }

  async function buildProgramFromFacts() {
    const raw = el.programFacts.value.split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"));
    if (!raw.length) throw new Error(t("program.factsRequired"));
    const base = (el.programBase.value.trim() || state.allowedRoots[0] || "")
      .replace(/\/+$/, "");
    let model = String((state.session && state.session.model)
      || el.checkpointPath.value || "").trim();
    let fetchStarter = null;
    if (!model) {
      const selector = el.firstEditStarter || el.starterModel;
      const option = selector && selector.options[selector.selectedIndex];
      const starterName = (option && option.dataset && option.dataset.name)
        || "SmolLM2-135M";
      fetchStarter = (selector && selector.value) || "smol135m";
      model = base ? base + "/" + starterName : starterName;
    }
    const output = nextCheckpointPath(model);
    const name = model.split("/").filter(Boolean).pop() || "weight-space";
    const payload = await api("/api/program/new", {
      method: "POST",
      body: JSON.stringify({
        facts: raw,
        model: portableProgramPath(model, base),
        output: portableProgramPath(output, base),
        fetch: fetchStarter,
        name: name + " facts",
        protect: preserveLines(),
        budget: 0.10,
        max_rel_norm: 0.02,
      }),
    });
    el.programText.value = payload.text;
    el.programPanel.open = true;
    addTimelineEvent("program", t("timeline.programBuild"), raw.length + " facts", "ok");
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
          rememberCheckpoint(model.path);
          setFirstEditReadyStatus();
          renderFirstEdit();
          el.modelsList.classList.add("hidden");
        });
        el.modelsList.appendChild(button);
      });
    } catch (err) {
      addTimelineEvent("error", err.message, t("errors.modelDiscovery"), "err");
    }
  }

  el.tokenButton.addEventListener("click", () => openTokenDialog());
  if (el.firstEditStarter && el.starterModel) {
    el.firstEditStarter.addEventListener("change", () => {
      el.starterModel.value = el.firstEditStarter.value;
    });
    el.starterModel.addEventListener("change", () => {
      el.firstEditStarter.value = el.starterModel.value;
    });
  }
  if (el.firstEditDevice) {
    el.firstEditDevice.addEventListener("change", () => {
      setFirstEditDevice(el.firstEditDevice.value);
    });
  }
  if ($("#device")) {
    $("#device").addEventListener("input", syncFirstEditDeviceFromInput);
  }
  if (el.checkpointPath) {
    el.checkpointPath.addEventListener("input", () => {
      if (state.busy || state.firstEdit) return;
      setFirstEditReadyStatus();
      renderFirstEdit();
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
  async function startFirstEdit() {
    if (!el.firstEditCard || state.busy) return;
    const sentence = el.firstEditSentence.value.trim();
    if (!sentence) {
      setFirstEditStatus("firstEdit.invalid", {}, "err");
      return;
    }
    if (el.quantize.value && el.dtype.value !== "auto") {
      setFirstEditStatus("firstEdit.error", { error: t("errors.quantizeDtype") }, "err");
      return;
    }
    let preview;
    try {
      preview = await api("/api/teach/preview", {
        method: "POST",
        body: JSON.stringify({ sentence }),
      });
    } catch (err) {
      setFirstEditStatus("firstEdit.invalid", { error: err.message }, "err");
      addTimelineEvent("error", err.message, t("teach.invalid"), "err");
      return;
    }
    state.lastTeach = preview;
    renderTeachPreview(preview);
    state.firstEditEvidence = null;
    $("#prompt").value = preview.prompt;
    $("#target").value = preview.answer;
    if (state.session) {
      const starter = (state.firstEdit && state.firstEdit.starter) || null;
      state.firstEdit = { sentence, preview, phase: "teach", starter };
      setFirstEditStatus("firstEdit.teaching");
      if (el.firstEditOut && !el.firstEditOut.value) {
        const model = String(state.session.model || "").replace(/\/+$/, "");
        if (model) el.firstEditOut.value = nextCheckpointPath(model);
      }
      renderFirstEdit();
      startJob("/api/teach", {
        sentence,
        out_dir: null,
        solver: "direction",
        max_rel_norm: 0.02,
      }, "teach");
      return;
    }
    const checkpoint = el.checkpointPath.value.trim();
    if (checkpoint) rememberCheckpoint(checkpoint);
    const root = (state.allowedRoots[0] || "").replace(/\/+$/, "");
    const starterOption = el.firstEditStarter && el.firstEditStarter.selectedOptions
      ? el.firstEditStarter.selectedOptions[0] : null;
    const starter = el.firstEditStarter ? el.firstEditStarter.value : "smol135m";
    const starterName = (starterOption && starterOption.dataset.name) || "SmolLM2-135M";
    const dest = checkpoint || (root ? root + "/" + starterName : "");
    if (!dest) {
      setFirstEditStatus("firstEdit.noRoot", {}, "err");
      return;
    }
    state.firstEdit = {
      sentence,
      preview,
      dest,
      phase: checkpoint ? "open" : "fetch",
      starter: checkpoint ? null : starter,
    };
    if (!checkpoint) {
      setFirstEditStatus("firstEdit.downloading", { model: starterName });
      startJob("/api/models/fetch", { starter, dest }, "fetch_model");
    } else {
      setFirstEditStatus("firstEdit.opening");
      startJob("/api/session/open", openSessionRequest(dest), "open_session");
    }
  }

  if (el.firstEditButton) {
    el.firstEditButton.addEventListener("click", startFirstEdit);
  }
  if (el.firstEditSave) {
    el.firstEditSave.addEventListener("click", () => {
      if (!state.session || !state.firstEdit || state.firstEdit.phase !== "done") return;
      if (state.busy) return;
      let out = (el.firstEditOut.value || "").trim();
      if (!out) {
        const model = String(state.session.model || "").replace(/\/+$/, "");
        out = model ? nextCheckpointPath(model) : "";
      }
      if (!out) {
        setFirstEditStatus("firstEdit.error", { error: t("errors.outputRequired") }, "err");
        return;
      }
      setFirstEditStatus("firstEdit.saving");
      startJob("/api/session/save", { out_dir: out }, "save_checkpoint");
    });
  }
  if (el.firstEditOpen) {
    el.firstEditOpen.addEventListener("click", () => {
      if (state.busy || !state.firstEdit || state.firstEdit.phase !== "saved") return;
      state.firstEdit.phase = "closing";
      setFirstEditStatus("firstEdit.closing");
      startJob("/api/session/close", { force: true }, "close_session");
    });
  }
  if (el.firstEditList) {
    el.firstEditList.addEventListener("click", () => {
      if (state.busy) return;
      openFirstEditList();
    });
  }
  if (el.firstEditProgram) {
    el.firstEditProgram.addEventListener("click", async () => {
      if (state.busy || !state.session || !state.firstEdit) return;
      const sentence = String(
        state.firstEdit.sentence || el.firstEditSentence.value || ""
      ).trim();
      if (!sentence) {
        return addTimelineEvent("error", t("firstEdit.invalid", {
          error: t("program.factsRequired"),
        }), "", "err");
      }
      const model = String(state.session.model || "").trim();
      const output = String(el.firstEditOut.value || "").trim() || null;
      const starter = state.firstEdit.starter || null;
      const body = {
        facts: [sentence],
        model,
        output,
        fetch: starter,
        name: (model.split("/").filter(Boolean).pop() || "checkpoint") + " first edit",
        protect: preserveLines(),
        budget: Number($("#budget").value) || 0.10,
        dtype: el.dtype.value,
        quantize: el.quantize.value || null,
        device: $("#device").value.trim() || null,
      };
      try {
        const payload = await api("/api/program/new", {
          method: "POST",
          body: JSON.stringify(body),
        });
        el.programText.value = payload.text;
        el.programPanel.open = true;
        addTimelineEvent("program", t("timeline.firstProgram"), "", "ok");
        if (el.programPanel.scrollIntoView) {
          el.programPanel.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      } catch (err) {
        addTimelineEvent("error", localizedJobError(err.message), "", "err");
      }
    });
  }
  $("#teachManyButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", t("errors.sessionFirst"), "", "err");
    const facts = $("#teachManyFacts").value.split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"));
    if (!facts.length) return addTimelineEvent("error", t("teachMany.factsRequired"), "", "err");
    let out = $("#teachManyOut").value.trim();
    if (!out && state.session.model) {
      out = manyCheckpointPath(state.session.model);
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
  $("#programPlanButton").addEventListener("click", async () => {
    let body;
    try {
      body = programRequest();
    } catch (err) {
      return addTimelineEvent("error", err.message, "", "err");
    }
    showBusy(t("program.planning"), t("program.planningDetail"));
    try {
      const payload = await api("/api/program/plan", {
        method: "POST",
        body: JSON.stringify(body),
      });
      rememberProgram();
      renderProgram(payload);
      el.programPanel.open = true;
      addTimelineEvent("program", t("timeline.programPlan"),
        (payload.facts || []).length + " facts", "ok");
    } catch (err) {
      addTimelineEvent("error", localizedJobError(err.message), "", "err");
    } finally {
      stopBusy("done");
    }
  });
  $("#programPrefillButton").addEventListener("click", () => {
    try {
      prefillProgram();
      addTimelineEvent("program", t("timeline.programPrefill"), "", "ok");
    } catch (err) {
      addTimelineEvent("error", err.message, "", "err");
    }
  });
  $("#programHistoryLoad").addEventListener("click", loadRememberedProgram);
  $("#programRunHistoryLoad").addEventListener("click", loadProgramRun);
  $("#programCopyButton").addEventListener("click", async () => {
    const text = el.programText.value.trim();
    if (!text) return addTimelineEvent("error", t("program.textRequired"), "", "err");
    try {
      await copyText(text + "\n");
      addTimelineEvent("program", t("timeline.programCopy"), "", "ok");
    } catch (err) {
      addTimelineEvent("error", err.message, "", "err");
    }
  });
  $("#programReportButton").addEventListener("click", () => {
    if (!state.programResult) return;
    downloadJson(state.programResult, programReportFilename());
    addTimelineEvent("program", t("timeline.programReport"), "", "ok");
  });
  if (el.programVerifyButton) {
    el.programVerifyButton.addEventListener("click", () => chooseProgramReport(false));
  }
  if (el.programVerifyModelButton) {
    el.programVerifyModelButton.addEventListener("click", () => chooseProgramReport(true));
  }
  if (el.programVerifyLastButton) {
    el.programVerifyLastButton.addEventListener("click", () => {
      if (!state.programResult) {
        return addTimelineEvent("error", t("program.noResultToVerify"), "", "err");
      }
      const checkpoint = (el.programVerifyCurrent
        && el.programVerifyCurrent.checked)
        ? (el.checkpointPath.value.trim() ||
          String((state.session && state.session.model) || "").trim())
        : "";
      const body = {
        report: JSON.stringify(state.programResult),
        check_model: !!checkpoint,
      };
      if (checkpoint) body.checkpoint = checkpoint;
      startJob(
        "/api/program/verify",
        body,
        "program_verify",
      );
    });
  }
  if (el.programReportFile) {
    el.programReportFile.addEventListener("change", async () => {
      const file = el.programReportFile.files && el.programReportFile.files[0];
      try {
        await verifyProgramReportFile(file);
      } catch (err) {
        addTimelineEvent("error", localizedJobError(err.message), "", "err");
      } finally {
        el.programReportFile.value = "";
      }
    });
  }
  $("#programUseOutputButton").addEventListener("click", () => {
    const final = state.programResult && state.programResult.run
      && state.programResult.run.final;
    if (!final) return addTimelineEvent("error", t("program.noOutput"), "", "err");
    el.checkpointPath.value = final;
    rememberCheckpoint(final);
    el.programPanel.open = true;
    startJob("/api/session/open", openSessionRequest(final), "open_session");
  });
  $("#programDownloadButton").addEventListener("click", () => {
    try {
      downloadProgram();
      addTimelineEvent("program", t("timeline.programDownload"), "", "ok");
    } catch (err) {
      addTimelineEvent("error", err.message, "", "err");
    }
  });
  $("#programFileButton").addEventListener("click", () => el.programFile.click());
  el.programTemplateSmol.addEventListener("click", () => loadProgramTemplate("smol"));
  el.programTemplateTwo.addEventListener("click", () => loadProgramTemplate("two"));
  if (el.programTemplateBilingual) {
    el.programTemplateBilingual.addEventListener(
      "click", () => loadProgramTemplate("bilingual")
    );
  }
  el.programBuildButton.addEventListener("click", async () => {
    try {
      await buildProgramFromFacts();
    } catch (err) {
      addTimelineEvent("error", localizedJobError(err.message), "", "err");
    }
  });
  el.programFile.addEventListener("change", async () => {
    const file = el.programFile.files && el.programFile.files[0];
    try {
      await openProgramFile(file);
    } catch (err) {
      addTimelineEvent("error", err.message, "", "err");
    } finally {
      el.programFile.value = "";
    }
  });
  $("#programRunButton").addEventListener("click", () => {
    let body;
    try {
      body = programRequest();
    } catch (err) {
      return addTimelineEvent("error", err.message, "", "err");
    }
    rememberProgram();
    startJob("/api/program/run", body, "program_run");
  });
  if (el.checkButton) {
    el.checkButton.addEventListener("click", () => {
      const path = el.checkpointPath.value.trim();
      const facts = el.checkSentence.value.split(/\r?\n/)
        .map((line) => line.trim()).filter(Boolean);
      if (!path) return addTimelineEvent("error", t("errors.pathRequired"), "", "err");
      if (!facts.length) return addTimelineEvent("error", t("teach.sentenceRequired"), "", "err");
      if (el.quantize.value && el.dtype.value !== "auto") {
        return addTimelineEvent("error", t("errors.quantizeDtype"), "", "err");
      }
      renderCheckResult(null);
      const payload = {
        path,
        dtype: el.dtype.value,
        device: $("#device").value.trim() || null,
        quantize: el.quantize.value || null,
      };
      if (facts.length === 1) payload.sentence = facts[0];
      else payload.facts = facts;
      startJob("/api/check", payload, "check");
    });
  }
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
    const body = openSessionRequest(path);
    if (body.quantize && body.dtype !== "auto") {
      return addTimelineEvent("error", t("errors.quantizeDtype"), "", "err");
    }
    startJob("/api/session/open", body, "open_session");
  });
  if (el.closeSessionButton) {
    el.closeSessionButton.addEventListener("click", () => {
      if (state.busy || !state.session) return;
      const edited = (state.session.edited || []).length > 0;
      if (edited && !window.confirm(t("session.confirmClose"))) return;
      startJob("/api/session/close", { force: edited }, "close_session");
    });
  }
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
