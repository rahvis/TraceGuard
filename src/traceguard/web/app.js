"use strict";

const API = {
  health: "/api/health",
  config: "/api/config",
  cases: "/api/cases",
  paperResults: "/api/paper/results",
  runs: "/api/runs",
  verifyReceipt: "/api/receipts/verify",
};

const PAPER_PROVENANCE = "paper_reported_unverified";
const LIVE_PROVENANCE = "current_live_replication";
const FIXTURE_PROVENANCE = "deterministic_fixture";
const TERMINAL_STATES = new Set(["complete", "completed", "failed", "error", "cancelled", "canceled"]);
const AGENT_NAMES = [
  "intake",
  "clinical_extraction",
  "coverage_assessment",
  "criteria_check",
  "necessity_review",
  "determination",
];

const state = {
  azureConfigured: null,
  runId: null,
  provider: null,
  condition: null,
  source: null,
  pollTimer: null,
  traceEvents: [],
  receipt: null,
  terminal: false,
  finalizing: false,
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindEvents();
  updateProviderHelp();
  updateConditionHelp();
  void Promise.allSettled([loadPaperResults(), loadRuntime(), loadCases()]);
});

function cacheElements() {
  const ids = [
    "artifact-version", "error-region", "error-message", "dismiss-error",
    "paper-results", "paper-results-note", "reload-paper-results", "health-summary",
    "health-details", "config-details", "refresh-runtime", "dataset-badge", "case-select",
    "case-help", "provider-select", "provider-help", "condition-select", "dag-condition",
    "seed-input", "run-form", "start-run", "cancel-run", "run-status", "run-status-dot",
    "run-id-label", "run-provenance", "timeline-body", "event-count", "metric-events",
    "metric-hops", "metric-duration", "metric-egress", "receipt-status-icon",
    "receipt-status", "receipt-detail", "verify-receipt",
  ];
  ids.forEach((id) => {
    elements[toCamelCase(id)] = document.getElementById(id);
  });
  elements.agentNodes = Array.from(document.querySelectorAll(".agent-node"));
  elements.copyButtons = Array.from(document.querySelectorAll(".copy-button"));
}

function bindEvents() {
  elements.dismissError.addEventListener("click", clearError);
  elements.reloadPaperResults.addEventListener("click", () => void loadPaperResults());
  elements.refreshRuntime.addEventListener("click", () => void loadRuntime());
  elements.providerSelect.addEventListener("change", updateProviderHelp);
  elements.conditionSelect.addEventListener("change", updateConditionHelp);
  elements.runForm.addEventListener("submit", startRun);
  elements.cancelRun.addEventListener("click", cancelRun);
  elements.verifyReceipt.addEventListener("click", () => void verifyReceipt());
  elements.copyButtons.forEach((button) => button.addEventListener("click", copyCommand));
  window.addEventListener("beforeunload", closeRunConnections);
}

function toCamelCase(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function safeText(value, fallback = "—", maxLength = 120) {
  if (value === undefined || value === null || value === "") return fallback;
  const text = String(value).replace(/[\u0000-\u001f\u007f]/g, " ").trim();
  if (!text) return fallback;
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function finiteNumber(value) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function humanize(value) {
  return safeText(value, "Reference result", 90)
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    credentials: "same-origin",
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    throw new Error(`${options.method || "GET"} ${path} failed with HTTP ${response.status}.`);
  }
  if (response.status === 204) return {};
  try {
    return await response.json();
  } catch {
    throw new Error(`${options.method || "GET"} ${path} returned invalid JSON.`);
  }
}

function showError(message) {
  elements.errorMessage.textContent = safeText(message, "An unexpected console error occurred.", 260);
  elements.errorRegion.hidden = false;
}

function clearError() {
  elements.errorRegion.hidden = true;
  elements.errorMessage.textContent = "";
}

async function loadPaperResults() {
  elements.paperResults.setAttribute("aria-busy", "true");
  elements.paperResults.classList.add("loading-grid");
  elements.paperResults.replaceChildren(...[0, 1, 2].map(() => {
    const card = document.createElement("div");
    card.className = "skeleton-card";
    card.setAttribute("aria-hidden", "true");
    return card;
  }));
  elements.paperResultsNote.textContent = "Loading the checked-in paper reference endpoint…";

  try {
    const payload = await apiFetch(API.paperResults);
    renderPaperResults(payload);
  } catch (error) {
    renderPaperResultsError();
    showError(error.message);
  } finally {
    elements.paperResults.setAttribute("aria-busy", "false");
    elements.paperResults.classList.remove("loading-grid");
  }
}

function renderPaperResults(payload) {
  const items = collectPaperItems(payload);
  elements.paperResults.replaceChildren();

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "result-empty";
    empty.textContent = "The endpoint returned no paper reference values. No result is inferred or substituted.";
    elements.paperResults.append(empty);
  } else {
    items.slice(0, 12).forEach((item) => {
      const card = document.createElement("article");
      card.className = "result-card";

      const label = document.createElement("p");
      label.className = "result-label";
      label.textContent = humanize(item.label);

      const value = document.createElement("div");
      value.className = "result-value";
      const main = document.createElement("span");
      main.textContent = safeText(item.value, "Not reported", 48);
      value.append(main);
      if (item.unit) {
        const unit = document.createElement("small");
        unit.textContent = safeText(item.unit, "", 24);
        value.append(unit);
      }

      const provenance = document.createElement("div");
      provenance.className = "result-provenance";
      provenance.textContent = PAPER_PROVENANCE;
      card.append(label, value, provenance);
      elements.paperResults.append(card);
    });
  }

  const source = firstDefined(payload.source, payload.paper_source, payload.citation);
  const sourceLabel = source ? ` · source: ${safeText(source, "", 90)}` : "";
  elements.paperResultsNote.textContent = `${items.length} reference value${items.length === 1 ? "" : "s"}${sourceLabel} · not reproduced in this session`;

  if (payload.reproduced === true || (payload.provenance && payload.provenance !== PAPER_PROVENANCE)) {
    showError("The paper reference endpoint reported an unexpected provenance. Values remain labeled paper_reported_unverified in this console.");
  }
}

function collectPaperItems(payload) {
  const roots = [payload.results, payload.reference_results, payload.metrics, payload.claims];
  const root = roots.find((value) => value !== undefined) ?? [];
  const items = [];
  const ignored = new Set(["provenance", "reproduced", "source", "citation", "notes", "generated_at"]);

  function visit(value, fallbackLabel = "Reference result", depth = 0) {
    if (items.length >= 16 || depth > 4 || value === null || value === undefined) return;
    if (Array.isArray(value)) {
      value.forEach((entry, index) => visit(entry, `${fallbackLabel} ${index + 1}`, depth + 1));
      return;
    }
    if (!isObject(value)) {
      if (["string", "number"].includes(typeof value)) items.push({ label: fallbackLabel, value });
      return;
    }

    const metricValue = firstDefined(
      value.paper_reported_value, value.reported_value, value.value, value.result, value.estimate,
    );
    if (["string", "number"].includes(typeof metricValue)) {
      items.push({
        label: firstDefined(value.label, value.name, value.metric, value.title, fallbackLabel),
        value: metricValue,
        unit: firstDefined(value.unit, value.measure),
      });
      return;
    }

    Object.entries(value).forEach(([key, nested]) => {
      if (!ignored.has(key)) visit(nested, key, depth + 1);
    });
  }

  visit(root);
  return items;
}

function renderPaperResultsError() {
  const empty = document.createElement("div");
  empty.className = "result-empty";
  empty.textContent = "Paper references are unavailable. No cached or hardcoded values are shown.";
  elements.paperResults.replaceChildren(empty);
  elements.paperResultsNote.textContent = "Reference endpoint unavailable · no reproduction claim";
}

async function loadRuntime() {
  elements.healthSummary.innerHTML = '<span class="status-dot" aria-hidden="true"></span><span>Checking local API…</span>';
  const [healthResult, configResult] = await Promise.allSettled([
    apiFetch(API.health),
    apiFetch(API.config),
  ]);

  if (healthResult.status === "fulfilled") renderHealth(healthResult.value);
  else {
    renderHealthUnavailable();
    showError(healthResult.reason.message);
  }

  if (configResult.status === "fulfilled") renderConfig(configResult.value);
  else {
    renderConfigUnavailable();
    showError(configResult.reason.message);
  }
}

function renderHealth(health) {
  const online = String(health.status || "").toLowerCase() === "ok" || String(health.status || "").toLowerCase() === "healthy";
  state.azureConfigured = health.azure_openai_configured === true;
  elements.healthSummary.replaceChildren(statusDot(online ? "online" : "warning"), document.createTextNode(online ? "Local API healthy" : "Local API reported a warning"));
  elements.artifactVersion.textContent = health.version ? `Runtime ${safeText(health.version, "", 32)}` : "Runtime version not reported";
  renderDefinitionList(elements.healthDetails, [
    ["Status", safeText(health.status, "Unknown", 32)],
    ["Version", safeText(health.version, "Not reported", 40)],
    ["LLM provider", state.azureConfigured
      ? "Azure OpenAI configured server-side"
      : "Not configured"],
  ]);
  updateProviderHelp();
}

function renderHealthUnavailable() {
  state.azureConfigured = null;
  elements.healthSummary.replaceChildren(statusDot("offline"), document.createTextNode("Local API unavailable"));
  renderDefinitionList(elements.healthDetails, [["Status", "Unreachable"], ["Version", "—"], ["LLM provider", "Unknown"]]);
}

function renderConfig(config) {
  const modelSource = isObject(config.models)
    ? config.models
    : { fast: config.model_fast, deep: config.model_deep, review: config.model_review, vision: config.model_vision };
  const models = ["fast", "deep", "review", "vision"].filter((key) => modelSource[key]).map((key) => `${key}: ${safeText(modelSource[key], "", 36)}`).join(" · ") || "Not reported";
  renderDefinitionList(elements.configDetails, [
    ["Models", models || "Not reported"],
    ["Max tokens / run", formatInteger(config.max_tokens_per_run)],
    ["Canonical hops", formatInteger(config.canonical_hops)],
    ["Step deadline", formatMilliseconds(config.step_deadline_ms)],
    ["Egress ceiling", formatBytes(config.step_egress_bytes)],
    ["Vision used", config.vision_used === true ? "Yes" : "No"],
  ]);
}

function renderConfigUnavailable() {
  renderDefinitionList(elements.configDetails, [["Models", "Unavailable"], ["Canonical hops", "—"], ["Step deadline", "—"], ["Egress ceiling", "—"]]);
}

function renderDefinitionList(target, rows) {
  target.replaceChildren(...rows.map(([term, description]) => {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = description;
    row.append(dt, dd);
    return row;
  }));
}

function statusDot(className) {
  const dot = document.createElement("span");
  dot.className = `status-dot ${className}`;
  dot.setAttribute("aria-hidden", "true");
  return dot;
}

async function loadCases() {
  try {
    const payload = await apiFetch(API.cases);
    const cases = Array.isArray(payload) ? payload : Array.isArray(payload.cases) ? payload.cases : [];
    const dataset = isObject(payload.dataset) ? payload.dataset : {};
    const syntheticDeclared = dataset.synthetic === true || payload.synthetic === true || cases.every((item) => isObject(item) && item.synthetic === true);
    if (!cases.length) throw new Error("GET /api/cases returned no cases.");
    if (!syntheticDeclared) throw new Error("The case endpoint did not declare a synthetic dataset; case loading was refused.");

    const options = cases.map((item, index) => {
      const option = document.createElement("option");
      const id = firstDefined(item.id, item.case_id);
      option.value = safeText(id, "", 100);
      const descriptors = [safeText(id, `synthetic-${index + 1}`, 45)];
      if (item.specialty) descriptors.push(safeText(item.specialty, "", 30));
      if (item.topic) descriptors.push(safeText(item.topic, "", 38));
      option.textContent = descriptors.join(" · ");
      return option;
    }).filter((option) => option.value);
    if (!options.length) throw new Error("Synthetic cases did not include usable identifiers.");

    elements.caseSelect.replaceChildren(...options);
    elements.caseSelect.disabled = false;
    const version = dataset.version ? ` · ${safeText(dataset.version, "", 20)}` : "";
    elements.datasetBadge.textContent = `Synthetic${version}`;
    const hash = safeText(dataset.hash, "", 18);
    elements.caseHelp.textContent = hash ? `Synthetic dataset hash: ${hash}${String(dataset.hash).length > 18 ? "…" : ""}` : "Synthetic identifiers and public labels only.";
  } catch (error) {
    elements.caseSelect.replaceChildren(new Option("Synthetic cases unavailable", ""));
    elements.caseSelect.disabled = true;
    elements.datasetBadge.textContent = "Dataset unavailable";
    showError(error.message);
  }
}

function updateProviderHelp() {
  const selected = elements.providerSelect.value;
  const configured = state.azureConfigured;
  if (selected === "fixture") {
    elements.providerHelp.textContent = "Deterministic offline plumbing check; not evidence of paper or LLM performance.";
  } else if (configured === true) {
    elements.providerHelp.textContent = "Uses the server-side environment credential. It is never sent to this page.";
  } else if (configured === false) {
    elements.providerHelp.textContent = "The server reports no Azure OpenAI credential; select fixture or configure the server environment.";
  } else {
    elements.providerHelp.textContent = "Provider availability is not yet known. Credentials remain server-side.";
  }
}

function updateConditionHelp() {
  const descriptions = {
    adaptive: "Adaptive plan: data-dependent hops and repair loops may remain observable.",
    structure_only: "Canonical structure: control flow is fixed; timing and egress size are not padded.",
    full_pad: "Full pad: fixed structure with public per-step timing and egress ceilings.",
  };
  elements.dagCondition.textContent = descriptions[elements.conditionSelect.value] || "Unknown condition";
}

async function startRun(event) {
  event.preventDefault();
  clearError();
  const caseId = elements.caseSelect.value;
  const provider = elements.providerSelect.value;
  const condition = elements.conditionSelect.value;
  const seed = Number(elements.seedInput.value);

  if (!caseId) return showError("Select a declared synthetic case before starting a run.");
  if (!["fixture", "azure"].includes(provider)) return showError("Select a supported execution provider.");
  if (!["adaptive", "structure_only", "full_pad"].includes(condition)) return showError("Select a supported trace condition.");
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 2147483647) return showError("Seed must be an integer from 0 through 2147483647.");
  if (provider === "azure" && state.azureConfigured === false) return showError("Azure OpenAI is not configured on the server. No credential is accepted by this page.");

  closeRunConnections();
  resetRunDisplay();
  setControlsRunning(true);
  setRunStatus("Submitting run…", "running");
  state.provider = provider;
  state.condition = condition;

  try {
    const response = await apiFetch(API.runs, {
      method: "POST",
      body: JSON.stringify({ case_id: caseId, provider, condition, seed, kind: "single" }),
    });
    const runId = firstDefined(response.run_id, response.id);
    if (!runId) throw new Error("POST /api/runs did not return a run identifier.");
    state.runId = safeText(runId, "", 160);
    elements.runIdLabel.textContent = `run ${state.runId}`;
    setRunProvenance(provider === "fixture" ? FIXTURE_PROVENANCE : LIVE_PROVENANCE);
    setRunStatus(safeText(response.status, "Queued", 32), "running");
    connectEventStream();
    scheduleRunPoll();
  } catch (error) {
    setRunStatus("Run could not start", "failed");
    setControlsRunning(false);
    showError(error.message);
  }
}

function resetRunDisplay() {
  state.runId = null;
  state.traceEvents = [];
  state.receipt = null;
  state.terminal = false;
  state.finalizing = false;
  elements.timelineBody.innerHTML = '<tr class="empty-row"><td colspan="6">Waiting for trace metadata…</td></tr>';
  elements.eventCount.textContent = "0 events";
  [elements.metricEvents, elements.metricHops, elements.metricDuration, elements.metricEgress].forEach((element) => { element.textContent = "—"; });
  elements.agentNodes.forEach((node) => {
    node.classList.remove("running", "done", "failed");
    node.querySelector("small").textContent = "Idle";
  });
  setReceiptStatus("Awaiting a completed run", "No receipt loaded.", "idle");
  elements.verifyReceipt.disabled = true;
}

function setControlsRunning(running) {
  [elements.caseSelect, elements.providerSelect, elements.conditionSelect, elements.seedInput, elements.startRun].forEach((element) => { element.disabled = running; });
  elements.cancelRun.hidden = !running;
  elements.cancelRun.disabled = !running;
}

function setRunStatus(label, mode = "") {
  elements.runStatus.textContent = safeText(label, "Unknown", 70);
  elements.runStatusDot.className = `status-dot ${mode}`.trim();
}

function setRunProvenance(value) {
  const known = [PAPER_PROVENANCE, LIVE_PROVENANCE, FIXTURE_PROVENANCE];
  const provenance = known.includes(value) ? value : (state.provider === "fixture" ? FIXTURE_PROVENANCE : LIVE_PROVENANCE);
  const pill = document.createElement("span");
  pill.className = `provenance-pill ${provenance === FIXTURE_PROVENANCE ? "provenance-pill-fixture" : provenance === PAPER_PROVENANCE ? "provenance-pill-paper" : "provenance-pill-live"}`;
  pill.textContent = provenance;
  elements.runProvenance.replaceChildren(pill);
}

function connectEventStream() {
  if (!window.EventSource || !state.runId) {
    showError("Live event streaming is unavailable in this browser; run status will still be polled.");
    return;
  }
  const source = new EventSource(`${API.runs}/${encodeURIComponent(state.runId)}/events`);
  state.source = source;
  source.onopen = () => {
    if (!state.terminal) setRunStatus("Running · event stream connected", "running");
  };
  source.onmessage = handleServerEvent;
  ["run_started", "step_started", "step_completed", "trace", "complete", "completed", "run_completed", "receipt"].forEach((eventName) => {
    source.addEventListener(eventName, handleServerEvent);
  });
  source.onerror = (event) => {
    if (typeof event.data === "string" && event.data) handleServerEvent(event);
    else if (!state.terminal) setRunStatus("Running · event stream reconnecting", "running");
  };
}

function handleServerEvent(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    return;
  }
  if (!isObject(payload)) return;
  const eventType = safeText(firstDefined(payload.type, payload.event_type, event.type), "message", 50).toLowerCase();
  const metadata = extractTraceMetadata(payload, eventType);
  if (metadata.agent || metadata.hop !== null || metadata.durationMs !== null || metadata.egressBytes !== null) {
    appendTraceEvent(metadata);
  }

  const reportedStatus = safeText(firstDefined(payload.status, isObject(payload.data) && payload.data.status), "", 30).toLowerCase();
  const terminalEvent = ["complete", "completed", "run_complete", "run_completed", "failed", "error", "cancelled", "canceled"].includes(eventType) && !metadata.agent;
  if (terminalEvent || TERMINAL_STATES.has(reportedStatus)) {
    const failed = eventType.includes("error") || eventType.includes("fail") || reportedStatus === "failed" || reportedStatus === "error";
    void finalizeRun(failed ? "failed" : (reportedStatus || "completed"));
  }
}

function extractTraceMetadata(payload, eventType = "trace") {
  const data = isObject(payload.data) ? payload.data : payload;
  const rawAgent = firstDefined(data.step_type, data.agent, data.step, data.node, payload.step_type, payload.step);
  const agent = normalizeAgent(rawAgent);
  const durationMs = finiteNumber(firstDefined(data.duration_ms, data.elapsed_ms, payload.duration_ms));
  const durationSeconds = finiteNumber(firstDefined(data.duration_s, payload.duration_s));
  const egressBytes = finiteNumber(firstDefined(data.egress_bytes, data.bytes, data.output_bytes, payload.egress_bytes));
  const hop = finiteNumber(firstDefined(data.hop, data.hop_index, payload.hop));
  let status = safeText(firstDefined(data.status, payload.step_status), "", 24).toLowerCase();
  if (!status) {
    if (/error|fail/.test(eventType)) status = "failed";
    else if (/complete|finish|end/.test(eventType)) status = "completed";
    else if (/start|running/.test(eventType)) status = "running";
    else status = "observed";
  }
  return {
    agent,
    agentLabel: agent ? agent[0].toUpperCase() + agent.slice(1) : "Step",
    hop: hop === null ? null : Math.max(0, Math.trunc(hop)),
    durationMs: durationMs !== null ? Math.max(0, durationMs) : durationSeconds !== null ? Math.max(0, durationSeconds * 1000) : null,
    egressBytes: egressBytes === null ? null : Math.max(0, Math.trunc(egressBytes)),
    status,
  };
}

function normalizeAgent(value) {
  if (!value) return null;
  const normalized = String(value).toLowerCase().replace(/[^a-z]/g, "");
  if (normalized.includes("intake") || normalized.includes("orchestr")) return "intake";
  if (
    normalized.includes("clinicalextraction") ||
    normalized.includes("extract") ||
    normalized.includes("clinical") ||
    normalized.includes("research") ||
    normalized.includes("retriev")
  )
    return "clinical_extraction";
  if (normalized.includes("coverage") || normalized.includes("assessment") || normalized.includes("draft"))
    return "coverage_assessment";
  if (normalized.includes("criteria") || normalized.includes("citat") || normalized.includes("cite"))
    return "criteria_check";
  if (normalized.includes("necessity") || normalized.includes("review")) return "necessity_review";
  if (normalized.includes("determination") || normalized.includes("editor") || normalized.includes("edit"))
    return "determination";
  return null;
}

function appendTraceEvent(metadata) {
  state.traceEvents.push(metadata);
  if (state.traceEvents.length === 1) elements.timelineBody.replaceChildren();
  const row = document.createElement("tr");
  const values = [
    state.traceEvents.length,
    metadata.agentLabel,
    metadata.hop === null ? "—" : metadata.hop,
    formatMilliseconds(metadata.durationMs),
    formatBytes(metadata.egressBytes),
  ];
  values.forEach((value) => {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  });
  const statusCell = document.createElement("td");
  const status = document.createElement("span");
  status.className = `event-state ${metadata.status}`;
  status.textContent = safeText(metadata.status, "observed", 20);
  statusCell.append(status);
  row.append(statusCell);
  elements.timelineBody.append(row);
  elements.eventCount.textContent = `${state.traceEvents.length} event${state.traceEvents.length === 1 ? "" : "s"}`;
  updateAgentNode(metadata);
  renderMetrics({}, state.traceEvents);
}

function updateAgentNode(metadata) {
  if (!metadata.agent || !AGENT_NAMES.includes(metadata.agent)) return;
  const node = elements.agentNodes.find((item) => item.dataset.agent === metadata.agent);
  if (!node) return;
  node.classList.remove("running", "done", "failed");
  if (["failed", "error"].includes(metadata.status)) node.classList.add("failed");
  else if (["completed", "complete", "done"].includes(metadata.status)) node.classList.add("done");
  else node.classList.add("running");
  node.querySelector("small").textContent = metadata.status === "completed" ? "Complete" : humanize(metadata.status);
}

function scheduleRunPoll() {
  clearTimeout(state.pollTimer);
  if (!state.runId || state.terminal) return;
  state.pollTimer = window.setTimeout(async () => {
    try {
      const detail = await apiFetch(`${API.runs}/${encodeURIComponent(state.runId)}`);
      const status = safeText(detail.status, "running", 30).toLowerCase();
      if (detail.provenance) setRunProvenance(detail.provenance);
      if (TERMINAL_STATES.has(status)) await finalizeRun(status);
      else {
        setRunStatus(humanize(status), "running");
        scheduleRunPoll();
      }
    } catch (error) {
      showError(error.message);
      scheduleRunPoll();
    }
  }, 1800);
}

async function finalizeRun(status) {
  if (state.finalizing || state.terminal) return;
  state.finalizing = true;
  state.terminal = true;
  closeRunConnections();
  const failed = ["failed", "error"].includes(status);
  const cancelled = ["cancelled", "canceled"].includes(status);
  setRunStatus(failed ? "Failed" : cancelled ? "Cancelled" : "Completed", failed ? "failed" : cancelled ? "warning" : "complete");
  if (!failed && !cancelled) markRunningNodesDone();
  await loadRunArtifacts();
  setControlsRunning(false);
  state.finalizing = false;
}

function closeRunConnections() {
  if (state.source) state.source.close();
  state.source = null;
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function markRunningNodesDone() {
  elements.agentNodes.forEach((node) => {
    if (node.classList.contains("running")) {
      node.classList.remove("running");
      node.classList.add("done");
      node.querySelector("small").textContent = "Complete";
    }
  });
}

async function cancelRun() {
  if (!state.runId || state.terminal) return;
  elements.cancelRun.disabled = true;
  setRunStatus("Cancellation requested…", "warning");
  try {
    await apiFetch(`${API.runs}/${encodeURIComponent(state.runId)}/cancel`, { method: "POST" });
    await finalizeRun("cancelled");
  } catch (error) {
    elements.cancelRun.disabled = false;
    showError(error.message);
  }
}

async function loadRunArtifacts() {
  if (!state.runId) return;
  const base = `${API.runs}/${encodeURIComponent(state.runId)}`;
  const [tracesResult, metricsResult, receiptsResult] = await Promise.allSettled([
    apiFetch(`${base}/traces`), apiFetch(`${base}/metrics`), apiFetch(`${base}/receipts`),
  ]);

  if (tracesResult.status === "fulfilled") {
    const traces = Array.isArray(tracesResult.value) ? tracesResult.value : tracesResult.value.traces;
    if (Array.isArray(traces) && traces.length) renderAuthoritativeTraces(traces);
  } else showError(tracesResult.reason.message);

  if (metricsResult.status === "fulfilled") {
    renderMetrics(isObject(metricsResult.value.metrics) ? metricsResult.value.metrics : metricsResult.value, state.traceEvents);
  } else renderMetrics({}, state.traceEvents);

  if (receiptsResult.status === "fulfilled") {
    const body = receiptsResult.value;
    const receipts = Array.isArray(body) ? body : Array.isArray(body.receipts) ? body.receipts : body.receipt ? [body.receipt] : [];
    state.receipt = receipts.length ? receipts[receipts.length - 1] : null;
    if (state.receipt) {
      setReceiptStatus("Receipt loaded", `Integrity metadata loaded for run ${state.runId}.`, "idle");
      elements.verifyReceipt.disabled = false;
      await verifyReceipt();
    } else {
      setReceiptStatus("No receipt emitted", "This run returned no verifiable receipt.", "idle");
    }
  } else {
    setReceiptStatus("Receipt unavailable", "The receipt endpoint could not be read.", "invalid");
    showError(receiptsResult.reason.message);
  }
}

function renderAuthoritativeTraces(traces) {
  state.traceEvents = [];
  elements.timelineBody.replaceChildren();
  elements.agentNodes.forEach((node) => {
    node.classList.remove("running", "done", "failed");
    node.querySelector("small").textContent = "Idle";
  });
  traces.forEach((trace) => {
    if (!isObject(trace)) return;
    const metadata = extractTraceMetadata(trace, safeText(firstDefined(trace.type, trace.event_type), "trace", 32).toLowerCase());
    if (metadata.agent || metadata.hop !== null || metadata.durationMs !== null || metadata.egressBytes !== null) appendTraceEvent(metadata);
  });
  if (!state.traceEvents.length) {
    elements.timelineBody.innerHTML = '<tr class="empty-row"><td colspan="6">The completed run returned no trace metadata.</td></tr>';
  }
}

function renderMetrics(metrics, traces) {
  const durationFallback = traces.reduce((sum, item) => sum + (item.durationMs || 0), 0);
  const egressFallback = traces.reduce((sum, item) => sum + (item.egressBytes || 0), 0);
  const observedHops = traces.map((item) => item.hop).filter((value) => value !== null);
  const events = finiteNumber(firstDefined(metrics.trace_events, metrics.event_count, metrics.total_events, metrics.steps)) ?? traces.length;
  const hops = finiteNumber(firstDefined(metrics.total_hops, metrics.hops, metrics.hop_count)) ?? (observedHops.length ? Math.max(...observedHops) : null);
  const duration = finiteNumber(firstDefined(metrics.total_duration_ms, metrics.duration_ms, metrics.latency_ms)) ?? (traces.some((item) => item.durationMs !== null) ? durationFallback : null);
  const egress = finiteNumber(firstDefined(metrics.total_egress_bytes, metrics.egress_bytes, metrics.bytes)) ?? (traces.some((item) => item.egressBytes !== null) ? egressFallback : null);
  elements.metricEvents.textContent = formatInteger(events);
  elements.metricHops.textContent = formatInteger(hops);
  elements.metricDuration.textContent = formatMilliseconds(duration);
  elements.metricEgress.textContent = formatBytes(egress);
}

async function verifyReceipt() {
  if (!state.receipt) return;
  elements.verifyReceipt.disabled = true;
  setReceiptStatus("Verifying receipt…", "Checking local signature and integrity fields.", "idle");
  try {
    const result = await apiFetch(API.verifyReceipt, { method: "POST", body: JSON.stringify({ receipt: state.receipt }) });
    const valid = firstDefined(result.valid, result.verified, result.ok, isObject(result.result) && result.result.valid) === true;
    const checks = isObject(result.checks) ? Object.keys(result.checks).length : Array.isArray(result.checks) ? result.checks.length : null;
    setReceiptStatus(valid ? "Receipt verified" : "Receipt verification failed", valid ? `${checks === null ? "Integrity checks passed" : `${checks} integrity checks evaluated`}; this is not hardware attestation.` : "The verifier did not accept this receipt.", valid ? "valid" : "invalid");
  } catch (error) {
    setReceiptStatus("Verification unavailable", "The local verifier endpoint returned an error.", "invalid");
    showError(error.message);
  } finally {
    elements.verifyReceipt.disabled = false;
  }
}

function setReceiptStatus(title, detail, mode) {
  elements.receiptStatus.textContent = title;
  elements.receiptDetail.textContent = detail;
  elements.receiptStatusIcon.className = `receipt-icon ${mode === "valid" ? "valid" : mode === "invalid" ? "invalid" : ""}`;
  elements.receiptStatusIcon.textContent = mode === "valid" ? "✓" : mode === "invalid" ? "!" : "—";
}

function formatInteger(value) {
  const number = finiteNumber(value);
  return number === null ? "—" : Math.max(0, Math.trunc(number)).toLocaleString();
}

function formatMilliseconds(value) {
  const number = finiteNumber(value);
  if (number === null) return "—";
  if (number < 1000) return `${Math.round(number).toLocaleString()} ms`;
  return `${(number / 1000).toFixed(number < 10000 ? 2 : 1)} s`;
}

function formatBytes(value) {
  const number = finiteNumber(value);
  if (number === null) return "—";
  if (number < 1024) return `${Math.round(number).toLocaleString()} B`;
  return `${(number / 1024).toFixed(number < 10240 ? 2 : 1)} KiB`;
}

async function copyCommand(event) {
  const button = event.currentTarget;
  const target = document.getElementById(button.dataset.copyTarget);
  if (!target || !navigator.clipboard) return showError("Clipboard access is unavailable in this browser.");
  try {
    await navigator.clipboard.writeText(target.textContent);
    const previous = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => { button.textContent = previous; }, 1400);
  } catch {
    showError("The browser denied clipboard access.");
  }
}
