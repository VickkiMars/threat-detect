/**
 * dashboard.js - Real-Time SecOps Polling & UI Interaction
 * Pure Sky Blue & Crisp White Light Design System | Zero Borders
 * Impeccable Design Quality & Ergonomics
 */

document.addEventListener("DOMContentLoaded", () => {
  // API Host Resolution (adopted from Endurance)
  const API_BASE = window.GRACE_API_BASE || `${location.protocol}//${location.host}`;

  // DOM Elements
  const elConnBadge = document.getElementById("conn-badge");
  const elConnDot = document.getElementById("conn-dot");
  const elConnText = document.getElementById("conn-text");
  const elHeaderUptime = document.getElementById("header-uptime");

  const btnSimStart = document.getElementById("btn-sim-start");
  const btnSimPause = document.getElementById("btn-sim-pause");
  const btnSimReset = document.getElementById("btn-sim-reset");

  const elTotalFlows = document.getElementById("kpi-total-flows");
  const elBenignFlows = document.getElementById("kpi-benign-flows");
  const elThreats = document.getElementById("kpi-threats");
  const elAlertsCount = document.getElementById("kpi-alerts-count");
  const elAlertBadge = document.getElementById("alert-badge-count");
  
  const tabAlertsActive = document.getElementById("tab-alerts-active");
  const tabAlertsHistory = document.getElementById("tab-alerts-history");
  const elAlertList = document.getElementById("alert-list");
  const elAlertHistoryList = document.getElementById("alert-history-list");
  const elAlertActionGroup = document.getElementById("alert-action-group");
  
  const elStreamBody = document.getElementById("stream-table-body");
  
  const elCpuVal = document.getElementById("gauge-cpu-val");
  const elCpuBar = document.getElementById("gauge-cpu-bar");
  const elMemVal = document.getElementById("gauge-mem-val");
  const elMemBar = document.getElementById("gauge-mem-bar");
  const elThroughputVal = document.getElementById("gauge-throughput-val");
  const elLatencyVal = document.getElementById("gauge-latency-val");
  const elLatencyBar = document.getElementById("gauge-latency-bar");

  // State cache to avoid redundant DOM re-renders
  let lastAlertIds = new Set();
  let lastFlowId = 0;
  let activeAlertTab = "active";
  let maxTotalFlows = 0;
  let maxBenignFlows = 0;
  let maxThreats = 0;

  // ── LocalStorage persistence for live feed ─────────────────────────────────
  const LS_KEY = "grace_feed_cache";
  const LS_MAX = 100; // max rows retained across sessions

  function lsLoad() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  }

  function lsSave(rows) {
    try {
      // Keep the most recent LS_MAX rows sorted desc by flow_id
      const sorted = [...rows].sort((a, b) => b.flow_id - a.flow_id).slice(0, LS_MAX);
      localStorage.setItem(LS_KEY, JSON.stringify(sorted));
    } catch { /* storage full – silently skip */ }
  }

  function lsClear() {
    try { localStorage.removeItem(LS_KEY); } catch { }
  }

  // Seed knownDetections map from cache on boot
  const knownDetections = new Map();
  lsLoad().forEach(d => knownDetections.set(d.flow_id, d));
  // ───────────────────────────────────────────────────────────────────────────

  // Threat Severity styling map adhering to industry SecOps red/orange standard
  const severityStyles = {
    CRITICAL: {
      card: "bg-rose-50 text-slate-900 shadow-xs hover:bg-rose-100/80",
      badge: "bg-rose-600 text-white font-extrabold shadow-xs",
      ipText: "text-rose-700 font-bold",
      metaText: "text-rose-600 font-medium",
      descText: "text-slate-800 font-medium",
      btn: "bg-rose-600 text-white hover:bg-rose-700 font-bold"
    },
    HIGH: {
      card: "bg-orange-50 text-slate-900 shadow-xs hover:bg-orange-100/80",
      badge: "bg-orange-600 text-white font-bold shadow-xs",
      ipText: "text-orange-700 font-bold",
      metaText: "text-orange-600 font-medium",
      descText: "text-slate-800 font-medium",
      btn: "bg-orange-600 text-white hover:bg-orange-700 font-bold"
    },
    MEDIUM: {
      card: "bg-amber-50 text-slate-900 shadow-xs hover:bg-amber-100/80",
      badge: "bg-amber-500 text-white font-bold shadow-xs",
      ipText: "text-amber-800 font-bold",
      metaText: "text-amber-700 font-medium",
      descText: "text-slate-700 font-medium",
      btn: "bg-amber-600 text-white hover:bg-amber-700 font-bold"
    },
    LOW: {
      card: "bg-slate-50 text-slate-800 shadow-xs hover:bg-slate-100",
      badge: "bg-slate-200 text-slate-700 font-semibold",
      ipText: "text-slate-900 font-bold",
      metaText: "text-slate-500",
      descText: "text-slate-600 font-medium",
      btn: "bg-slate-800 text-white hover:bg-slate-900 font-semibold"
    }
  };

  // Simulation Controls Handler
  function updateSimButtons(isRunning) {
    if (!btnSimStart || !btnSimPause) return;
    if (isRunning) {
      btnSimStart.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer bg-white text-sky-700 shadow-xs";
      btnSimPause.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer text-slate-600 hover:text-slate-900 hover:bg-slate-200 active:scale-95";
    } else {
      btnSimStart.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer text-slate-600 hover:text-slate-900 hover:bg-slate-200 active:scale-95";
      btnSimPause.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer bg-white text-amber-700 shadow-xs";
    }
  }

  async function handleSimStart() {
    try {
      const res = await fetch(`${API_BASE}/api/simulation/start`, { method: "POST" });
      if (res.ok) {
        updateSimButtons(true);
        fetchSummary();
        fetchRecentDetections();
      }
    } catch (err) {
      console.error("Simulation start error:", err);
    }
  }

  async function handleSimPause() {
    try {
      const res = await fetch(`${API_BASE}/api/simulation/pause`, { method: "POST" });
      if (res.ok) {
        updateSimButtons(false);
        fetchSummary();
      }
    } catch (err) {
      console.error("Simulation pause error:", err);
    }
  }

  async function handleSimReset() {
    const resetIcon = btnSimReset ? btnSimReset.querySelector("svg") : null;
    try {
      if (resetIcon) resetIcon.classList.add("animate-spin");
      if (btnSimReset) btnSimReset.classList.add("bg-sky-100", "text-sky-800");

      const res = await fetch(`${API_BASE}/api/simulation/reset`, { method: "POST" });
      if (res.ok) {
        updateSimButtons(false);
        lastFlowId = 0;
        lastAlertIds.clear();
        knownDetections.clear();
        lsClear();
        maxTotalFlows = 0;
        maxBenignFlows = 0;
        maxThreats = 0;

        // Immediately zero KPI metrics in DOM
        if (elTotalFlows) elTotalFlows.textContent = "0";
        if (elBenignFlows) elBenignFlows.textContent = "0";
        if (elThreats) elThreats.textContent = "0";
        if (elAlertsCount) elAlertsCount.textContent = "0 Alerts";
        if (elAlertBadge) elAlertBadge.textContent = "0";

        // Reset live stream table to clean placeholder
        if (elStreamBody) {
          elStreamBody.innerHTML = `
            <tr>
              <td colspan="8" class="text-center py-12 text-slate-400 font-sans text-xs">
                Flow stream replay pointer reset to Window #1. Operational logs purged. Click &ldquo;Start&rdquo; to begin live classification.
              </td>
            </tr>`;
        }

        // Reset alert lists
        if (elAlertList) {
          elAlertList.innerHTML = `
            <div class="text-center py-12 text-slate-400 text-xs bg-sky-50/60 rounded-xl p-8">
              No unacknowledged security threats. Perimeter secure.
            </div>`;
        }

        if (elAlertHistoryList) {
          elAlertHistoryList.innerHTML = `
            <div class="text-center py-12 text-slate-400 text-xs bg-sky-50/60 rounded-xl p-8">
              No historical resolved security threats recorded yet.
            </div>`;
        }

        // Reset gauges
        if (elThroughputVal) elThroughputVal.textContent = "0.0 fps";
        if (elLatencyVal) elLatencyVal.textContent = "< 0.05 ms";
        if (elLatencyBar) elLatencyBar.style.width = "2%";

        // Brief banner update
        if (elConnText) {
          const originalText = elConnText.textContent;
          elConnText.textContent = "Stream Reset";
          setTimeout(() => {
            elConnText.textContent = originalText;
          }, 1500);
        }

        await fetchSummary();
      }
    } catch (err) {
      console.error("Simulation reset error:", err);
    } finally {
      if (resetIcon) {
        setTimeout(() => {
          resetIcon.classList.remove("animate-spin");
          if (btnSimReset) btnSimReset.classList.remove("bg-sky-100", "text-sky-800");
        }, 500);
      }
    }
  }

  if (btnSimStart) btnSimStart.addEventListener("click", handleSimStart);
  if (btnSimPause) btnSimPause.addEventListener("click", handleSimPause);
  if (btnSimReset) btnSimReset.addEventListener("click", handleSimReset);

  // Alert Panel Tab Switcher
  function switchAlertTab(targetTab) {
    activeAlertTab = targetTab;
    if (targetTab === "active") {
      tabAlertsActive.className = "px-2.5 py-1 rounded-md text-xs font-bold transition-all cursor-pointer bg-sky-500 text-white shadow-xs";
      tabAlertsHistory.className = "px-2.5 py-1 rounded-md text-xs font-semibold text-slate-600 hover:text-slate-900 transition-all cursor-pointer";
      elAlertList.classList.remove("hidden");
      elAlertHistoryList.classList.add("hidden");
      if (elAlertActionGroup) elAlertActionGroup.classList.remove("hidden");
      fetchAlerts();
    } else {
      tabAlertsHistory.className = "px-2.5 py-1 rounded-md text-xs font-bold transition-all cursor-pointer bg-sky-500 text-white shadow-xs";
      tabAlertsActive.className = "px-2.5 py-1 rounded-md text-xs font-semibold text-slate-600 hover:text-slate-900 transition-all cursor-pointer";
      elAlertList.classList.add("hidden");
      elAlertHistoryList.classList.remove("hidden");
      if (elAlertActionGroup) elAlertActionGroup.classList.add("hidden");
      fetchAlertHistory();
    }
  }

  if (tabAlertsActive) tabAlertsActive.addEventListener("click", () => switchAlertTab("active"));
  if (tabAlertsHistory) tabAlertsHistory.addEventListener("click", () => switchAlertTab("history"));

  async function fetchSummary() {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (!res.ok) throw new Error("Status failed");
      const data = await res.json();
      
      // Update connection indicator
      if (elConnDot) elConnDot.className = "size-2 rounded-full bg-sky-500 animate-pulse";
      if (elConnText) elConnText.textContent = "Perimeter Online";

      console.log(`[GRACE SecOps] Status synced: flows=${data.total_flows_processed}, threats=${data.malicious_threats}, sim_running=${data.simulation?.running}, uptime=${data.uptime}`);

      // Uptime
      if (elHeaderUptime && data.uptime) {
        elHeaderUptime.textContent = data.uptime;
      }

      // Simulation status
      if (data.simulation) {
        updateSimButtons(data.simulation.running);
      }
      
      maxTotalFlows = Math.max(maxTotalFlows, Number(data.total_flows_processed || 0));
      maxBenignFlows = Math.max(maxBenignFlows, Number(data.benign_flows || 0));
      maxThreats = Math.max(maxThreats, Number(data.malicious_threats || 0));

      elTotalFlows.textContent = maxTotalFlows.toLocaleString();
      elBenignFlows.textContent = maxBenignFlows.toLocaleString();
      elThreats.textContent = maxThreats.toLocaleString();
      
      const unackCount = data.unacknowledged_alerts || 0;
      elAlertsCount.textContent = `${unackCount} Alerts`;
      if (elAlertBadge) elAlertBadge.textContent = unackCount;
      
      if (data.latest_telemetry) {
        const cpu = data.latest_telemetry.cpu_percent || 0;
        const mem = data.latest_telemetry.memory_rss_mb || 0;
        const tput = data.latest_telemetry.throughput_fps || 0;
        
        if (elCpuVal) elCpuVal.textContent = `${cpu.toFixed(1)}%`;
        if (elCpuBar) elCpuBar.style.width = `${Math.min(cpu, 100)}%`;
        
        if (elMemVal) elMemVal.textContent = `${mem.toFixed(1)} MB`;
        const memPct = Math.min((mem / 512.0) * 100, 100);
        if (elMemBar) elMemBar.style.width = `${memPct}%`;
        
        if (elThroughputVal) {
          elThroughputVal.textContent = `${tput.toFixed(1)} flows/s`;
        }
      }
    } catch (err) {
      console.warn("Telemetry fetch warning:", err);
      if (elConnDot) elConnDot.className = "size-2 rounded-full bg-red-500";
      if (elConnText) elConnText.textContent = "Reconnecting...";
    }
  }

  async function fetchAlerts() {
    try {
      const res = await fetch(`${API_BASE}/api/alerts/unacknowledged?limit=20`);
      if (!res.ok) return;
      const data = await res.json();
      const alerts = data.alerts || [];
      
      const currentIds = new Set(alerts.map(a => a.alert_id));
      if (alerts.length === 0) {
        elAlertList.innerHTML = `
          <div class="text-center py-12 text-slate-400 text-xs bg-sky-50/60 rounded-xl p-8">
            No unacknowledged security threats. Perimeter secure.
          </div>`;
        lastAlertIds.clear();
        return;
      }

      // Re-render if alert IDs changed
      const idsChanged = currentIds.size !== lastAlertIds.size || 
        [...currentIds].some(id => !lastAlertIds.has(id));
        
      if (idsChanged) {
        lastAlertIds = currentIds;
        elAlertList.innerHTML = alerts.map(a => {
          const style = severityStyles[a.severity] || severityStyles.LOW;
          return `
            <div class="p-4 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-all duration-200 hover:-translate-y-0.5 ${style.card}" id="alert-card-${a.alert_id}">
              <div class="space-y-1">
                <div class="flex flex-wrap items-center gap-2 text-xs">
                  <span class="px-2.5 py-0.5 rounded text-[10px] uppercase tracking-wider ${style.badge}">${a.severity}</span>
                  <span class="font-mono ${style.ipText}">${a.src_ip} &rarr; ${a.dst_ip}</span>
                  <span class="font-mono text-[11px] ${style.metaText}">(${a.protocol})</span>
                  <span class="text-[11px] font-medium ${style.metaText}">Confidence: ${(a.confidence * 100).toFixed(1)}%</span>
                </div>
                <p class="text-xs sm:text-sm ${style.descText}">${a.message}</p>
              </div>
              <button class="px-3.5 py-1.5 rounded-lg text-xs transition-all duration-150 shrink-0 cursor-pointer shadow-xs active:scale-95 ${style.btn}" data-id="${a.alert_id}" onclick="handleAcknowledge(${a.alert_id})">
                Acknowledge
              </button>
            </div>
          `;
        }).join("");
      }
    } catch (err) {
      console.warn("Alert fetch error:", err);
    }
  }

  async function fetchAlertHistory() {
    try {
      const res = await fetch(`${API_BASE}/api/alerts/history?limit=30`);
      if (!res.ok) return;
      const data = await res.json();
      const alerts = data.alerts || [];

      if (alerts.length === 0) {
        elAlertHistoryList.innerHTML = `
          <div class="text-center py-12 text-slate-400 text-xs bg-sky-50/60 rounded-xl p-8">
            No historical resolved security threats recorded yet.
          </div>`;
        return;
      }

      elAlertHistoryList.innerHTML = alerts.map(a => {
        const timePart = a.acknowledged_at ? a.acknowledged_at.split("T")[1]?.slice(0, 8) : "Triaged";
        const badgeColor = a.severity === 'CRITICAL' ? 'bg-rose-100 text-rose-800 font-extrabold' :
                           a.severity === 'HIGH' ? 'bg-orange-100 text-orange-800 font-bold' :
                           a.severity === 'MEDIUM' ? 'bg-amber-100 text-amber-800 font-bold' :
                           'bg-slate-200 text-slate-700 font-semibold';
        return `
          <div class="p-3.5 rounded-xl bg-slate-50 border-0 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 shadow-xs hover:bg-slate-100 transition-all duration-150">
            <div class="space-y-1">
              <div class="flex flex-wrap items-center gap-2 text-xs">
                <span class="px-2 py-0.5 rounded text-[10px] uppercase ${badgeColor}">${a.severity}</span>
                <span class="font-mono text-slate-800 font-semibold">${a.src_ip} &rarr; ${a.dst_ip}</span>
                <span class="font-mono text-[11px] text-slate-500">(${a.protocol})</span>
                <span class="text-[11px] text-slate-600 font-medium">${(a.confidence * 100).toFixed(1)}% conf</span>
              </div>
              <p class="text-xs text-slate-600">${a.message}</p>
            </div>
            <div class="flex items-center gap-1.5 shrink-0 text-right">
              <span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-emerald-100 text-emerald-800">
                <svg class="size-3 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>
                Triaged (${timePart})
              </span>
            </div>
          </div>
        `;
      }).join("");
    } catch (err) {
      console.warn("Alert history fetch error:", err);
    }
  }

  function renderDetectionRow(d) {
    const isMalicious = d.predicted_class === 1;
    const pillStyle = isMalicious
      ? "bg-rose-600 text-white font-bold"
      : "bg-slate-100 text-slate-700 font-semibold";
    const rowBg = isMalicious ? "hover:bg-rose-50/50" : "hover:bg-slate-50";
    const confText = isMalicious ? "text-rose-700 font-bold" : "text-slate-800 font-semibold";
    const pillText = isMalicious ? "MALICIOUS" : "BENIGN";
    const timeStr = (d.timestamp || "").split("T")[1]?.slice(0, 8) || d.timestamp || "";
    return `
      <tr class="${rowBg} transition-colors duration-150">
        <td class="px-2.5 py-2 font-bold text-slate-900">#${d.window_sequence_id}</td>
        <td class="px-2 py-2 text-slate-500 font-mono text-[11px]">${timeStr}</td>
        <td class="px-2.5 py-2 text-slate-700 font-mono font-medium">${d.src_ip}</td>
        <td class="px-2.5 py-2 text-slate-700 font-mono font-medium">${d.dst_ip}</td>
        <td class="px-2.5 py-2 text-slate-500 font-semibold">${d.protocol}</td>
        <td class="px-2.5 py-2">
          <span class="px-2 py-0.5 rounded-full text-[10px] tracking-wider uppercase ${pillStyle}">
            ${pillText}
          </span>
        </td>
        <td class="px-2.5 py-2 ${confText} tabular-nums">${(d.confidence * 100).toFixed(1)}%</td>
        <td class="px-2.5 py-2 text-slate-700 font-mono font-medium tabular-nums text-right">${(d.inference_latency_ms || 0).toFixed(3)} ms</td>
      </tr>`;
  }

  function renderFeedFromCache() {
    if (knownDetections.size === 0) {
      elStreamBody.innerHTML = `<tr><td colspan="8" class="text-center py-10 text-slate-400 font-sans text-xs">Awaiting streaming flow input...</td></tr>`;
      return;
    }
    // Sort descending by flow_id (newest first) and show up to 25
    const sorted = [...knownDetections.values()]
      .sort((a, b) => b.flow_id - a.flow_id)
      .slice(0, 25);

    // Update latency gauge from freshest row
    if (elLatencyVal && sorted[0].inference_latency_ms !== undefined) {
      const lat = sorted[0].inference_latency_ms || 0;
      elLatencyVal.textContent = `${lat.toFixed(3)} ms`;
      if (elLatencyBar) {
        elLatencyBar.style.width = `${Math.max(Math.min((lat / 50.0) * 100, 100), 2)}%`;
      }
    }

    elStreamBody.innerHTML = sorted.map(renderDetectionRow).join("");
  }

  async function fetchRecentDetections() {
    try {
      const res = await fetch(`${API_BASE}/api/detections/recent?limit=25`);
      if (!res.ok) {
        // API unavailable – render whatever is cached
        renderFeedFromCache();
        return;
      }
      const data = await res.json();
      const detections = data.detections || [];

      console.log(`[GRACE SecOps] /api/detections/recent returned ${detections.length} detections (cached: ${knownDetections.size}, lastFlowId: ${lastFlowId})`);
      if (detections.length === 0 && knownDetections.size === 0) {
        console.warn("[GRACE SecOps] No detections returned from API. Open /api/debug to inspect database & engine diagnostics.");
      }

      // Merge API results into knownDetections (skip legacy seq #1001)
      let hasNew = false;
      for (const d of detections) {
        if (d.window_sequence_id === 1001) continue;
        if (!knownDetections.has(d.flow_id)) {
          knownDetections.set(d.flow_id, d);
          hasNew = true;
        }
      }

      // Persist updated cache to localStorage
      if (hasNew || knownDetections.size > 0) {
        lsSave([...knownDetections.values()]);
      }

      // Update lastFlowId to track freshest known flow
      if (knownDetections.size > 0) {
        const maxId = Math.max(...knownDetections.keys());
        if (maxId !== lastFlowId) {
          lastFlowId = maxId;
          renderFeedFromCache();
        } else if (hasNew) {
          renderFeedFromCache();
        } else if (knownDetections.size > 0 && elStreamBody.querySelector("td[colspan]")) {
          // Cache exists but DOM still shows placeholder – render it
          renderFeedFromCache();
        }
      } else {
        elStreamBody.innerHTML = `<tr><td colspan="8" class="text-center py-10 text-slate-400 font-sans text-xs">Awaiting streaming flow input...</td></tr>`;
      }
    } catch (err) {
      console.warn("Detections fetch error:", err);
      // Fallback: render whatever is in cache even if the request threw
      renderFeedFromCache();
    }
  }

  // Global single acknowledge handler
  window.handleAcknowledge = async function(alertId) {
    try {
      const res = await fetch(`${API_BASE}/api/alerts/${alertId}/acknowledge`, { method: "POST" });
      if (res.ok) {
        lastAlertIds.delete(alertId);
        const card = document.getElementById(`alert-card-${alertId}`);
        if (card) {
          card.style.opacity = "0.2";
          card.style.transform = "scale(0.97)";
          setTimeout(() => {
            card.remove();
            if (elAlertList && elAlertList.querySelectorAll("[id^='alert-card-']").length === 0) {
              elAlertList.innerHTML = `
                <div class="text-center py-12 text-slate-400 text-xs bg-sky-50/60 rounded-xl p-8">
                  No unacknowledged security threats. Perimeter secure.
                </div>`;
            }
          }, 200);
        }
        fetchSummary();
      }
    } catch (err) {
      console.error("Failed to acknowledge alert:", err);
    }
  };

  // Global bulk acknowledge handler
  window.handleAcknowledgeAll = async function() {
    try {
      const res = await fetch(`${API_BASE}/api/alerts/acknowledge_all`, { method: "POST" });
      if (res.ok) {
        lastAlertIds.clear();
        if (elAlertList) {
          elAlertList.innerHTML = `
            <div class="text-center py-12 text-slate-400 text-xs bg-sky-50/60 rounded-xl p-8">
              No unacknowledged security threats. Perimeter secure.
            </div>`;
        }
        fetchSummary();
      }
    } catch (err) {
      console.error("Failed to acknowledge all alerts:", err);
    }
  };

  // Initial load – render cached rows immediately before first API response
  if (knownDetections.size > 0) renderFeedFromCache();
  fetchSummary();
  fetchAlerts();
  fetchRecentDetections();

  // Adaptive polling loop: 3.0s interval, pauses when tab is hidden (saves CPU & Vercel Fluid compute)
  let pollTimer = null;
  const POLL_INTERVAL_MS = 3000;

  function runPoll() {
    if (document.hidden) return;
    fetchSummary();
    if (activeAlertTab === "active") {
      fetchAlerts();
    } else {
      fetchAlertHistory();
    }
    fetchRecentDetections();
  }

  function startPolling() {
    if (!pollTimer) {
      pollTimer = setInterval(runPoll, POLL_INTERVAL_MS);
    }
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopPolling();
    } else {
      runPoll();
      startPolling();
    }
  });

  startPolling();
});


