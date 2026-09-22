/**
 * dashboard.js - Real-Time SecOps Polling & UI Interaction
 * Pure Blue, Black, and White Design System | Zero Borders
 * AI-Powered Network Threat Detection System
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

  // Severity styling map strictly using Blue, Black, and White (Zero Borders)
  const severityStyles = {
    CRITICAL: {
      card: "bg-blue-600 text-white shadow-lg",
      badge: "bg-black text-white font-black",
      ipText: "text-white",
      metaText: "text-blue-100",
      descText: "text-white"
    },
    HIGH: {
      card: "bg-blue-700 text-white shadow-md",
      badge: "bg-black text-blue-200 font-bold",
      ipText: "text-white",
      metaText: "text-blue-100",
      descText: "text-white"
    },
    MEDIUM: {
      card: "bg-[#0f1d3d] text-white shadow-sm",
      badge: "bg-blue-600 text-white font-bold",
      ipText: "text-white",
      metaText: "text-blue-200",
      descText: "text-blue-100"
    },
    LOW: {
      card: "bg-[#0b1630] text-blue-100 shadow-sm",
      badge: "bg-blue-900 text-blue-200 font-semibold",
      ipText: "text-white",
      metaText: "text-blue-300",
      descText: "text-blue-200"
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
      elAlertsCount.textContent = unackCount;
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
          <div class="text-center py-12 text-blue-300/60 text-xs bg-[#040814] rounded-xl p-8">
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
                  <span class="font-mono font-semibold ${style.ipText}">${a.src_ip} &rarr; ${a.dst_ip}</span>
                  <span class="font-mono text-[11px] ${style.metaText}">(${a.protocol})</span>
                  <span class="text-[11px] font-medium ${style.metaText}">Confidence: ${(a.confidence * 100).toFixed(1)}%</span>
                </div>
                <p class="text-xs sm:text-sm font-medium ${style.descText}">${a.message}</p>
              </div>
              <button class="px-3.5 py-1.5 rounded-lg text-xs font-bold bg-white text-black hover:bg-blue-400 hover:text-black transition-all duration-150 shrink-0 cursor-pointer shadow-md active:scale-95" data-id="${a.alert_id}" onclick="handleAcknowledge(${a.alert_id})">
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
        elStreamBody.innerHTML = `<tr><td colspan="8" class="text-center py-10 text-blue-300/60 font-sans text-xs">Awaiting streaming flow input...</td></tr>`;
        return;
      }
      
      if (detections[0].flow_id === lastFlowId) {
        return; // No new detections
      }
      lastFlowId = detections[0].flow_id;
      
      elStreamBody.innerHTML = detections.map(d => {
        const isMalicious = d.predicted_class === 1;
        const pillStyle = isMalicious 
          ? "bg-blue-600 text-white font-bold"
          : "bg-[#0c162e] text-blue-200 font-semibold";
        const pillText = isMalicious ? "MALICIOUS" : "BENIGN";
        const timeStr = d.timestamp.split("T")[1]?.slice(0, 8) || d.timestamp;
        
        return `
          <tr class="hover:bg-[#0c162e] transition-colors duration-150">
            <td class="px-3 py-2 font-bold text-white">#${d.window_sequence_id}</td>
            <td class="px-3 py-2 text-blue-300 font-mono text-[11px]">${timeStr}</td>
            <td class="px-3 py-2 text-blue-100 font-mono">${d.src_ip}</td>
            <td class="px-3 py-2 text-blue-100 font-mono">${d.dst_ip}</td>
            <td class="px-3 py-2 text-blue-300 font-semibold">${d.protocol}</td>
            <td class="px-3 py-2">
              <span class="px-2.5 py-0.5 rounded-full text-[10px] tracking-wider uppercase ${pillStyle}">
                ${pillText}
              </span>
            </td>
            <td class="px-3 py-2 text-white tabular-nums">${(d.confidence * 100).toFixed(1)}%</td>
            <td class="px-3 py-2 text-blue-400 font-mono tabular-nums">${d.inference_latency_ms.toFixed(3)} ms</td>
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
