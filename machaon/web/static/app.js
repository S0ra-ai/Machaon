/* Machaon local console: no framework, no build step, no remote requests. */
(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    token: "",
    session: null,
    proposal: null,
    lastJob: null,
    events: [],
    busy: false,
  };

  const el = {
    connection: $("#connectionPill"),
    version: $("#versionPill"),
    tokenButton: $("#tokenButton"),
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
      openTokenDialog("The token is missing or expired.");
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

  function fmt(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return String(value);
      if (Math.abs(value) >= 1000) return value.toLocaleString();
      if (Math.abs(value) > 0 && Math.abs(value) < 0.001) return value.toExponential(3);
      return Number(value.toFixed(5)).toString();
    }
    if (typeof value === "boolean") return value ? "yes" : "no";
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
    el.jobTitle.textContent = label || "Working";
    el.jobDetail.textContent = detail || "The model operation is running.";
    el.jobPill.textContent = "running";
    el.jobPill.className = "badge live";
    setActionState();
  }

  function stopBusy(status) {
    state.busy = false;
    el.jobBanner.classList.add("hidden");
    el.jobPill.textContent = status || "idle";
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

  function setConnection(kind, text) {
    el.connection.className = "pill " + kind;
    el.connection.innerHTML = "<i></i> " + escapeHtml(text);
  }

  async function boot() {
    setConnection("", "checking");
    try {
      const health = await api("/api/health");
      el.version.textContent = "v" + health.version;
      el.allowedRoots.textContent = (health.allowed_roots || []).join(" · ");
      const session = await api("/api/session/state");
      state.session = session && session.ready ? session : null;
      setConnection("live", "local · authenticated");
      renderSessionState();
      if (state.session) el.checkpointPath.value = state.session.model || el.checkpointPath.value;
    } catch (err) {
      setConnection("error", "offline");
      if (err.message !== "unauthorised") {
        el.version.textContent = "v—";
        openTokenDialog("Cannot reach the local server. Start machaon-web and open its printed URL.");
      }
    }
  }

  function renderCheckpoint(summary) {
    if (!summary) {
      el.checkpointSummary.className = "summary-grid empty-state";
      el.checkpointSummary.textContent = "Header-only metadata appears here after an inspect.";
      return;
    }
    el.checkpointSummary.className = "summary-grid";
    el.checkpointSummary.innerHTML = [
      metric("tensors", summary.tensors, true),
      metric("parameters", summary.parameters),
      metric("shards", summary.shards),
      metric("dtypes", Object.keys(summary.dtypes || {}).join(" · ")),
      metric("path", summary.path),
    ].join("");
    el.checkpointBadge.textContent = "inspected";
    el.checkpointBadge.className = "badge live";
  }

  function renderTensors(rows) {
    if (!rows || !rows.length) {
      el.tensorRows.className = "table-wrap empty-state";
      el.tensorRows.textContent = "No matching tensors.";
      return;
    }
    el.tensorRows.className = "table-wrap";
    el.tensorRows.innerHTML = "<table><thead><tr><th>tensor</th><th>shape</th><th>dtype</th></tr></thead><tbody>" +
      rows.map((row) => "<tr><td title=\"" + escapeHtml(row.name) + "\">" +
        escapeHtml(row.name) + "</td><td>" + escapeHtml(JSON.stringify(row.shape)) +
        "</td><td>" + escapeHtml(row.dtype) + "</td></tr>").join("") +
      "</tbody></table>";
  }

  function renderSessionState() {
    const s = state.session;
    if (!s) {
      el.sessionBadge.textContent = "not open";
      el.sessionBadge.className = "badge";
      el.stateCard.className = "evidence-card empty-state";
      el.stateCard.textContent = "No model session is open.";
      setActionState();
      return;
    }
    el.sessionBadge.textContent = "session live";
    el.sessionBadge.className = "badge live";
    el.stateCard.className = "evidence-card";
    el.stateCard.innerHTML = '<dl class="kv">' +
      "<dt>model</dt><dd>" + escapeHtml(s.model || "—") + "</dd>" +
      "<dt>parameters</dt><dd>" + escapeHtml(fmt(s.parameters)) + "</dd>" +
      "<dt>layers</dt><dd>" + escapeHtml(fmt(s.layers)) + "</dd>" +
      "<dt>quantize</dt><dd>" + escapeHtml(s.quantize || "none") + "</dd>" +
      "<dt>baseline</dt><dd>" + escapeHtml(s.has_baseline ? "captured" : "not captured") + "</dd>" +
      "<dt>pending</dt><dd>" + escapeHtml(s.has_pending ? "yes" : "no") + "</dd>" +
      "<dt>edited</dt><dd>" + escapeHtml((s.edited || []).length) + "</dd>" +
      "<dt>budget</dt><dd>" + escapeHtml(fmt(s.budget_total)) + " / " +
      escapeHtml(fmt(s.budget_limit)) + "</dd></dl>";
    setActionState();
  }

  function renderProposal(p) {
    if (!p) {
      el.proposalCard.className = "evidence-card empty-state";
      el.proposalCard.textContent = "No proposal. The solve does not touch weights.";
      return;
    }
    state.proposal = p;
    el.proposalCard.className = "evidence-card";
    el.proposalCard.innerHTML = '<div class="verdict pass"><strong>solved</strong><span class="tiny">' +
      escapeHtml(p.solver || "") + "</span></div><dl class=\"kv\">" +
      "<dt>module</dt><dd>" + escapeHtml(p.module) + "</dd>" +
      "<dt>layer</dt><dd>" + escapeHtml(p.layer) + "</dd>" +
      "<dt>relative delta</dt><dd>" + escapeHtml(fmt(p.relative_delta)) + "</dd>" +
      "<dt>target probability</dt><dd>" + escapeHtml(fmt(p.target_probability)) + "</dd>" +
      "<dt>steps</dt><dd>" + escapeHtml(fmt(p.steps)) + "</dd>" +
      "<dt>budget after</dt><dd>" + escapeHtml(fmt(p.budget_after)) + "</dd></dl>";
    setActionState();
  }

  function renderGate(gate) {
    if (!gate) {
      el.gateCard.className = "evidence-card empty-state";
      el.gateCard.textContent = "Capture a baseline and evaluate an applied edit to produce evidence.";
      return;
    }
    const passed = !!gate.passed;
    el.gateCard.className = "evidence-card";
    const failures = (gate.failures || []).concat(gate.missing || []);
    el.gateCard.innerHTML = '<div class="verdict ' + (passed ? "pass" : "fail") +
      '"><strong>' + (passed ? "PASS" : "FAIL") + '</strong><span class="tiny">fail-closed</span></div>' +
      '<dl class="kv"><dt>gates</dt><dd>' + escapeHtml(fmt(gate.n_gates)) + "</dd>" +
      "<dt>failures</dt><dd>" + escapeHtml(fmt(failures.length)) + "</dd></dl>" +
      (failures.length ? '<div class="failures">' + failures.map(escapeHtml).join(" · ") + "</div>" : "");
  }

  function renderMetrics(metrics, gate) {
    if (metrics) {
      const entries = Object.entries(metrics).filter(([key]) => key !== "n");
      el.gateCard.className = "evidence-card";
      el.gateCard.innerHTML = '<div class="verdict ' + (gate && gate.passed ? "pass" : "fail") +
        '"><strong>' + (gate ? (gate.passed ? "PASS" : "FAIL") : "MEASURED") +
        '</strong><span class="tiny">metrics</span></div><dl class="kv">' +
        entries.map(([key, value]) => "<dt>" + escapeHtml(key) + "</dt><dd>" +
          escapeHtml(fmt(value)) + "</dd>").join("") + "</dl>" +
        (gate && (gate.failures || []).length ? '<div class="failures">' +
          gate.failures.map(escapeHtml).join(" · ") + "</div>" : "");
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
    el.eventCount.textContent = state.events.length + " events";
  }

  function recordEvent(event) {
    state.events.push(event);
    if (state.events.length > 250) state.events.shift();
    el.rawEvents.textContent = JSON.stringify(state.events.slice(-80), null, 2);
    el.eventCount.textContent = state.events.length + " events";
    const data = event.data || {};
    if (event.type === "status") {
      addTimelineEvent("status", data.status || "status", event.kind || state.lastJob || "",
        data.status === "done" ? "ok" : data.status === "error" ? "err" : "");
    } else if (event.type === "result") {
      addTimelineEvent("result", "job result received", "", "ok");
    } else if (event.type === "error") {
      addTimelineEvent("error", data.error || "job failed", data.error_type || "", "err");
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
      addTimelineEvent("baseline", "baseline captured", JSON.stringify(result.baseline || {}), "ok");
    } else if (kind === "propose_edit") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      renderProposal(result && result.proposal);
      if (result && result.densified !== undefined) {
        addTimelineEvent("densify", "quantized window densified",
          String(result.densified) + " module(s)", "ok");
      }
    } else if (kind === "apply_edit") {
      if (result && result.state) state.session = result.state;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("apply", "edit applied", "budget charged atomically", "ok");
    } else if (kind === "evaluate") {
      if (result && result.state) state.session = result.state;
      renderSessionState();
      renderMetrics(result.metrics, result.gate);
    } else if (kind === "rollback") {
      state.session = result && result.state ? result.state : null;
      state.proposal = null;
      renderSessionState();
      renderProposal(null);
      addTimelineEvent("rollback", "edits restored", "original bytes returned", "ok");
    } else if (kind === "save_checkpoint") {
      addTimelineEvent("save", "checkpoint written", (result && result.output) || "", "ok");
    } else if (kind === "audit_checkpoint_edit" || kind === "apply_delta") {
      addTimelineEvent(kind, "checkpoint operation complete", JSON.stringify(result), "ok");
    }
  }

  function attachJob(jobId, kind) {
    state.lastJob = jobId;
    showBusy(kind.replaceAll("_", " "), "Job " + jobId + " · streaming events");
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
      addTimelineEvent("error", err.message, path, "err");
    }
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
        el.modelsList.innerHTML = '<span class="empty-state">No checkpoints found below the allowed roots. Type a path directly.</span>';
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
      addTimelineEvent("error", err.message, "model discovery", "err");
    }
  }

  el.tokenButton.addEventListener("click", () => openTokenDialog());
  $("#tokenSave").addEventListener("click", saveToken);
  $("#tokenCancel").addEventListener("click", () => el.dialog.close());
  $("#modelsButton").addEventListener("click", findModels);
  $("#describeButton").addEventListener("click", () => {
    const path = el.checkpointPath.value.trim();
    if (!path) return addTimelineEvent("error", "checkpoint path is required", "", "err");
    startJob("/api/checkpoints/describe", { path }, "describe_checkpoint");
  });
  $("#tensorsButton").addEventListener("click", () => {
    const body = tensorRequest();
    if (!body.path) return addTimelineEvent("error", "checkpoint path is required", "", "err");
    startJob("/api/checkpoints/tensors", body, "list_tensors");
  });
  $("#openButton").addEventListener("click", () => {
    const path = el.checkpointPath.value.trim();
    if (!path) return addTimelineEvent("error", "checkpoint path is required", "", "err");
    const body = {
      path,
      dtype: $("#dtype").value,
      quantize: $("#quantize").value || null,
      device: $("#device").value.trim() || null,
      budget_total: Number($("#budget").value),
    };
    if (body.quantize && body.dtype !== "auto") {
      return addTimelineEvent("error", "quantized loading uses dtype=auto", "", "err");
    }
    startJob("/api/session/open", body, "open_session");
  });
  $("#proposeButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", "open a session first", "", "err");
    startJob("/api/edit/propose", proposalRequest(), "propose_edit");
  });
  $("#applyButton").addEventListener("click", () => {
    if (!state.session) return;
    startJob("/api/edit/apply", {}, "apply_edit");
  });
  $("#baselineButton").addEventListener("click", () => {
    if (!state.session) return addTimelineEvent("error", "open a session first", "", "err");
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
    if (!out) return addTimelineEvent("error", "output path is required", "", "err");
    startJob("/api/session/save", { out_dir: out }, "save_checkpoint");
  });
  el.dtype.addEventListener("change", () => {
    if (el.quantize.value) el.dtype.value = "auto";
  });
  el.quantize.addEventListener("change", () => {
    if (el.quantize.value) el.dtype.value = "auto";
  });

  state.token = readToken();
  if (!state.token) {
    setConnection("error", "token required");
    openTokenDialog("Paste the token printed when you started machaon-web.");
  } else {
    boot();
  }
})();
