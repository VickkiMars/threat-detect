/**
 * dashboard.js - Real-Time SecOps Polling & UI Interaction
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

  // State cache to avoid unnecessary DOM rebuilds
  let lastAlertIds = new Set();
  let lastFlowId = 0;

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
        elAlertList.innerHTML = `<div class="empty-state">No unacknowledged security threats. Perimeter secure.</div>`;
        lastAlertIds.clear();
        return;
      }

      // Re-render if alert set changed
      const idsChanged = currentIds.size !== lastAlertIds.size || 
        [...currentIds].some(id => !lastAlertIds.has(id));
        
      if (idsChanged) {
        lastAlertIds = currentIds;
        elAlertList.innerHTML = alerts.map(a => `
          <div class="alert-item severity-${a.severity}" id="alert-card-${a.alert_id}">
            <div class="alert-info">
              <div class="alert-meta">
                <span class="severity-pill pill-${a.severity}">${a.severity}</span>
                <span>${a.src_ip} &rarr; ${a.dst_ip}</span>
                <span>(${a.protocol})</span>
                <span>Confidence: ${(a.confidence * 100).toFixed(1)}%</span>
              </div>
              <div class="alert-desc">${a.message}</div>
            </div>
            <button class="btn-ack" data-id="${a.alert_id}" onclick="handleAcknowledge(${a.alert_id})">
              Acknowledge
            </button>
          </div>
        `).join("");
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
        elStreamBody.innerHTML = `<tr><td colspan="7" class="empty-state">Awaiting streaming flow input...</td></tr>`;
        return;
      }
      
      if (detections[0].flow_id === lastFlowId) {
        return; // No new detections
      }
      lastFlowId = detections[0].flow_id;
      
      elStreamBody.innerHTML = detections.map(d => {
        const isMalicious = d.predicted_class === 1;
        const pillClass = isMalicious ? "class-attack" : "class-benign";
        const pillText = isMalicious ? "MALICIOUS" : "BENIGN";
        const timeStr = d.timestamp.split("T")[1]?.slice(0, 8) || d.timestamp;
        
        return `
          <tr>
            <td><strong>#${d.window_sequence_id}</strong></td>
            <td><code>${timeStr}</code></td>
            <td><code>${d.src_ip}</code></td>
            <td><code>${d.dst_ip}</code></td>
            <td>${d.protocol}</td>
            <td><span class="class-pill ${pillClass}">${pillText}</span></td>
            <td>${(d.confidence * 100).toFixed(1)}%</td>
            <td><code>${d.inference_latency_ms.toFixed(3)} ms</code></td>
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
          card.style.opacity = "0.4";
          card.style.transform = "scale(0.98)";
          setTimeout(() => card.remove(), 250);
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
