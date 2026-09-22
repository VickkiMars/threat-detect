/**
 * dashboard.js - Real-Time SecOps Polling & UI Interaction
 * Pure Sky Blue & Crisp White Light Design System | Zero Borders
 * Impeccable Design Quality & Ergonomics
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const elTotalFlows = document.getElementById("kpi-total-flows");
  const elBenignFlows = document.getElementById("kpi-benign-flows");
  const elThreats = document.getElementById("kpi-threats");
  const elAlertsCount = document.getElementById("kpi-alerts-count");
  const elAlertBadge = document.getElementById("alert-badge-count");
  const elAlertList = document.getElementById("alert-list");
  const elStreamBody = document.getElementById("stream-table-body");
  
  const elCpuVal = document.getElementById("gauge-cpu-val");
  const elCpuBar = document.getElementById("gauge-cpu-bar");
  const elMemVal = document.getElementById("gauge-mem-val");
  const elMemBar = document.getElementById("gauge-mem-bar");
  const elThroughputVal = document.getElementById("gauge-throughput-val");

  // State cache to avoid redundant DOM re-renders
  let lastAlertIds = new Set();
  let lastFlowId = 0;

  // Severity styling map strictly using Sky Blue, White, and Slate (Zero Borders)
  const severityStyles = {
    CRITICAL: {
      card: "bg-sky-600 text-white shadow-sm",
      badge: "bg-white text-sky-800 font-extrabold",
      ipText: "text-white font-bold",
      metaText: "text-sky-100",
      descText: "text-white font-medium",
      btn: "bg-white text-sky-800 hover:bg-sky-50 font-bold"
    },
    HIGH: {
      card: "bg-sky-500 text-white shadow-sm",
      badge: "bg-sky-100 text-sky-800 font-bold",
      ipText: "text-white font-bold",
      metaText: "text-sky-100",
      descText: "text-white font-medium",
      btn: "bg-white text-sky-700 hover:bg-sky-50 font-bold"
    },
    MEDIUM: {
      card: "bg-sky-100 text-slate-800 shadow-xs",
      badge: "bg-sky-200 text-sky-900 font-bold",
      ipText: "text-slate-900 font-bold",
      metaText: "text-sky-700",
      descText: "text-slate-700 font-medium",
      btn: "bg-sky-600 text-white hover:bg-sky-700 font-bold"
    },
    LOW: {
      card: "bg-slate-100 text-slate-800 shadow-xs",
      badge: "bg-slate-200 text-slate-700 font-semibold",
      ipText: "text-slate-900 font-bold",
      metaText: "text-slate-500",
      descText: "text-slate-600 font-medium",
      btn: "bg-slate-800 text-white hover:bg-slate-900 font-semibold"
    }
  };

  async function fetchSummary() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      
      elTotalFlows.textContent = Number(data.total_flows_processed || 0).toLocaleString();
      elBenignFlows.textContent = Number(data.benign_flows || 0).toLocaleString();
      elThreats.textContent = Number(data.malicious_threats || 0).toLocaleString();
      
      const unackCount = data.unacknowledged_alerts || 0;
      elAlertsCount.textContent = `${unackCount} Alerts`;
      elAlertBadge.textContent = unackCount;
      
      if (data.latest_telemetry) {
        const cpu = data.latest_telemetry.cpu_percent || 0;
        const mem = data.latest_telemetry.memory_rss_mb || 0;
        const tput = data.latest_telemetry.throughput_fps || 0;
        
        elCpuVal.textContent = `${cpu.toFixed(1)}%`;
        elCpuBar.style.width = `${Math.min(cpu, 100)}%`;
        
        elMemVal.textContent = `${mem.toFixed(1)} MB / 512 MB`;
        const memPct = Math.min((mem / 512.0) * 100, 100);
        elMemBar.style.width = `${memPct}%`;
        
        if (elThroughputVal) {
          elThroughputVal.textContent = `${tput.toFixed(1)} flows/s`;
        }
      }
    } catch (err) {
      console.warn("Telemetry fetch error:", err);
    }
  }

  async function fetchAlerts() {
    try {
      const res = await fetch("/api/alerts/unacknowledged?limit=20");
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

  async function fetchRecentDetections() {
    try {
      const res = await fetch("/api/detections/recent?limit=25");
      if (!res.ok) return;
      const data = await res.json();
      const detections = data.detections || [];
      
      if (detections.length === 0) {
        elStreamBody.innerHTML = `<tr><td colspan="8" class="text-center py-10 text-slate-400 font-sans text-xs">Awaiting streaming flow input...</td></tr>`;
        return;
      }
      
      if (detections[0].flow_id === lastFlowId) {
        return; // No new detections
      }
      lastFlowId = detections[0].flow_id;
      
      elStreamBody.innerHTML = detections.map(d => {
        const isMalicious = d.predicted_class === 1;
        const pillStyle = isMalicious 
          ? "bg-sky-600 text-white font-bold"
          : "bg-sky-100 text-sky-800 font-semibold";
        const pillText = isMalicious ? "MALICIOUS" : "BENIGN";
        const timeStr = d.timestamp.split("T")[1]?.slice(0, 8) || d.timestamp;
        
        return `
          <tr class="hover:bg-sky-50 transition-colors duration-150">
            <td class="px-2.5 py-2 font-bold text-slate-900">#${d.window_sequence_id}</td>
            <td class="px-2 py-2 text-slate-500 font-mono text-[11px]">${timeStr}</td>
            <td class="px-2.5 py-2 text-slate-700 font-mono font-medium">${d.src_ip}</td>
            <td class="px-2.5 py-2 text-slate-700 font-mono font-medium">${d.dst_ip}</td>
            <td class="px-2 py-2 text-slate-500 font-semibold">${d.protocol}</td>
            <td class="px-2.5 py-2">
              <span class="px-2 py-0.5 rounded-full text-[10px] tracking-wider uppercase ${pillStyle}">
                ${pillText}
              </span>
            </td>
            <td class="px-2.5 py-2 text-slate-800 font-semibold tabular-nums">${(d.confidence * 100).toFixed(1)}%</td>
            <td class="px-2.5 py-2 text-sky-600 font-mono font-bold tabular-nums text-right">${d.inference_latency_ms.toFixed(3)} ms</td>
          </tr>
        `;
      }).join("");
    } catch (err) {
      console.warn("Detections fetch error:", err);
    }
  }

  // Global acknowledge handler
  window.handleAcknowledge = async function(alertId) {
    try {
      const res = await fetch(`/api/alerts/${alertId}/acknowledge`, { method: "POST" });
      if (res.ok) {
        const card = document.getElementById(`alert-card-${alertId}`);
        if (card) {
          card.style.opacity = "0.2";
          card.style.transform = "scale(0.97)";
          setTimeout(() => card.remove(), 200);
        }
        fetchSummary();
      }
    } catch (err) {
      console.error("Failed to acknowledge alert:", err);
    }
  };

  // Initial load
  fetchSummary();
  fetchAlerts();
  fetchRecentDetections();

  // Polling loop every 1.5 seconds
  setInterval(() => {
    fetchSummary();
    fetchAlerts();
    fetchRecentDetections();
  }, 1500);
});
