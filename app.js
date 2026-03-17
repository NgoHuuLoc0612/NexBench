/**
 * app.js — NEXBENCH Frontend Engine
 *
 * Modules:
 *  - API client (REST + SSE + Socket.IO polling)
 *  - Gauge renderer (canvas arc)
 *  - Sparkline / line chart engine (canvas 2D)
 *  - Radar chart
 *  - Bar chart (cache latency, comparison)
 *  - Volume canvas renderer (CPU-side ray-march result → ImageData)
 *  - WebGL particle storm (GPU stress visualizer)
 *  - 3-D volume marching-cubes isosurface (WebGL)
 *  - Real-time telemetry ring-buffers
 *  - Job scheduler & toast notifications
 *  - Panel router
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Constants & Config
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE    = "http://127.0.0.1:5000";
const SSE_URL     = `${API_BASE}/api/telemetry/stream`;
const POLL_INTERVAL_MS = 1000;
const RT_HISTORY  = 120; // samples to keep for realtime charts

const COLORS = {
  accent:  "#00e5ff",
  accent2: "#7b2fff",
  accent3: "#ff3860",
  accent4: "#00ff9d",
  accent5: "#ffb800",
  text2:   "#8a99bb",
  text3:   "#4a5570",
  surface: "#1c2130",
  bg:      "#080a0e",
  bg2:     "#0d1018",
};

// ─────────────────────────────────────────────────────────────────────────────
// API Client
// ─────────────────────────────────────────────────────────────────────────────

const API = {
  async get(path) {
    const r = await fetch(`${API_BASE}${path}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
  async post(path, body = {}) {
    const r = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

const STATE = {
  systemInfo:    null,
  liveMetrics:   null,
  connected:     false,
  scores:        {},
  jobHistory:    [],
  activeJobs:    {},
  rtData: {
    cpu:     { ts: [], vals: [] },
    ram:     { ts: [], vals: [] },
    temp:    { ts: [], vals: [] },
    net_up:  { ts: [], vals: [] },
    net_dn:  { ts: [], vals: [] },
    per_core:{ ts: [], vals: [] }, // array of arrays
  },
  rtPaused:      false,
  glParticles:   null,
  vol3dAnim:     null,
  volumeAnim:    null,
  vol3dData:     null,
  vol3dAngle:    { x: 0.4, y: 0 },
  mbwSweepResults: [],
};

// ─────────────────────────────────────────────────────────────────────────────
// Loader
// ─────────────────────────────────────────────────────────────────────────────

const loaderSteps = [
  "Detecting hardware…",
  "Initializing C++ engine…",
  "Calibrating telemetry…",
  "Loading WebGL context…",
  "Compiling shaders…",
  "Ready.",
];

async function runLoader() {
  const bar    = document.getElementById("loaderBar");
  const status = document.getElementById("loaderStatus");

  // Advance loader text quickly while we wait for the server to respond.
  // Steps 0-3 are cosmetic; step 4 is held until the server ping resolves.
  let stepIdx = 0;
  const advanceStep = () => {
    if (stepIdx >= loaderSteps.length - 1) return;
    status.textContent = loaderSteps[stepIdx];
    bar.style.width    = `${Math.round((stepIdx + 1) / loaderSteps.length * 100)}%`;
    stepIdx++;
  };

  advanceStep(); // "Detecting hardware…"
  await sleep(120);
  advanceStep(); // "Initializing C++ engine…"
  await sleep(120);
  advanceStep(); // "Calibrating telemetry…"
  await sleep(120);
  advanceStep(); // "Loading WebGL context…"

  // Ping the server; if it takes longer than 150 ms show "Compiling shaders…"
  const pingTimer = setTimeout(() => advanceStep(), 150); // "Compiling shaders…"
  try { await API.get("/api/system_info"); } catch { /* handled later */ }
  clearTimeout(pingTimer);

  // Final step
  status.textContent = loaderSteps[loaderSteps.length - 1]; // "Ready."
  bar.style.width    = "100%";
  await sleep(180);

  document.getElementById("loader").style.transition = "opacity 0.4s";
  document.getElementById("loader").style.opacity    = "0";
  await sleep(400);
  document.getElementById("loader").style.display = "none";
  document.getElementById("app").style.display    = "block";
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel Router
// ─────────────────────────────────────────────────────────────────────────────

function initRouter() {
  document.querySelectorAll(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      const id = tab.dataset.panel;
      document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`panel-${id}`).classList.add("active");
      if (id === "history") refreshHistory();
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas Drawing Primitives
// ─────────────────────────────────────────────────────────────────────────────

function clearCanvas(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

function drawGauge(ctx, value, max, label, colorStop) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const cx = W / 2, cy = H / 2 + 10, r = Math.min(W, H) * 0.38;
  const startAngle = Math.PI * 0.75, endAngle = Math.PI * 2.25;
  const pct = Math.max(0, Math.min(1, value / max));
  const valAngle = startAngle + pct * (endAngle - startAngle);

  clearCanvas(ctx);

  // track
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = COLORS.surface;
  ctx.lineWidth   = 12;
  ctx.lineCap     = "round";
  ctx.stroke();

  // tick marks
  for (let i = 0; i <= 10; i++) {
    const a = startAngle + (i / 10) * (endAngle - startAngle);
    const x1 = cx + (r - 16) * Math.cos(a);
    const y1 = cy + (r - 16) * Math.sin(a);
    const x2 = cx + (r - 8) * Math.cos(a);
    const y2 = cy + (r - 8) * Math.sin(a);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = COLORS.text3; ctx.lineWidth = 1; ctx.stroke();
  }

  // value arc (gradient)
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0, colorStop[0]);
  grad.addColorStop(1, colorStop[1]);
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, valAngle);
  ctx.strokeStyle = grad;
  ctx.lineWidth   = 12;
  ctx.lineCap     = "round";
  ctx.stroke();

  // glow
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, valAngle);
  ctx.strokeStyle = colorStop[1] + "44";
  ctx.lineWidth   = 20;
  ctx.stroke();
}

function drawLineChart(ctx, datasets, opts = {}) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const pad = opts.pad || { top: 16, right: 16, bottom: 28, left: 40 };
  const iW  = W - pad.left - pad.right;
  const iH  = H - pad.top  - pad.bottom;

  clearCanvas(ctx);
  ctx.fillStyle = COLORS.bg2;
  ctx.fillRect(0, 0, W, H);

  // grid
  const ySteps = 4;
  const allVals = datasets.flatMap(d => d.values);
  const minV = opts.minY !== undefined ? opts.minY : Math.min(...allVals, 0);
  const maxV = opts.maxY !== undefined ? opts.maxY : Math.max(...allVals, 1) * 1.1;

  for (let i = 0; i <= ySteps; i++) {
    const y = pad.top + iH - (i / ySteps) * iH;
    const v = minV + (i / ySteps) * (maxV - minV);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + iW, y);
    ctx.strokeStyle = COLORS.surface; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = COLORS.text3; ctx.font = "9px JetBrains Mono";
    ctx.textAlign = "right";
    ctx.fillText(v.toFixed(1), pad.left - 6, y + 3);
  }

  // datasets
  datasets.forEach(({ values, color, label, fill }) => {
    if (!values || values.length < 2) return;
    const n = values.length;
    const xScale = iW / (n - 1);
    const yScale = iH / (maxV - minV || 1);

    const px = i => pad.left + i * xScale;
    const py = v => pad.top + iH - (v - minV) * yScale;

    if (fill) {
      ctx.beginPath();
      ctx.moveTo(px(0), py(values[0]));
      for (let i = 1; i < n; i++) {
        const mx = (px(i - 1) + px(i)) / 2;
        ctx.bezierCurveTo(mx, py(values[i-1]), mx, py(values[i]), px(i), py(values[i]));
      }
      ctx.lineTo(px(n - 1), pad.top + iH);
      ctx.lineTo(px(0),     pad.top + iH);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + iH);
      grad.addColorStop(0, color + "44");
      grad.addColorStop(1, color + "00");
      ctx.fillStyle = grad; ctx.fill();
    }

    ctx.beginPath();
    ctx.moveTo(px(0), py(values[0]));
    for (let i = 1; i < n; i++) {
      const mx = (px(i - 1) + px(i)) / 2;
      ctx.bezierCurveTo(mx, py(values[i-1]), mx, py(values[i]), px(i), py(values[i]));
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();

    // latest dot
    const lx = px(n - 1), ly = py(values[n - 1]);
    ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
  });

  // x-axis label
  if (opts.xLabel) {
    ctx.fillStyle = COLORS.text3; ctx.font = "9px JetBrains Mono"; ctx.textAlign = "center";
    ctx.fillText(opts.xLabel, W / 2, H - 4);
  }
}

function drawBarChart(ctx, labels, values, colors, opts = {}) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const pad = { top: 20, right: 16, bottom: 40, left: 60 };
  const iW  = W - pad.left - pad.right;
  const iH  = H - pad.top  - pad.bottom;

  clearCanvas(ctx);
  ctx.fillStyle = COLORS.bg2; ctx.fillRect(0, 0, W, H);

  const maxV = Math.max(...values, 1);
  const barW = iW / labels.length * 0.7;
  const gap  = iW / labels.length;

  // grid lines
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + iH - (i / 4) * iH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y);
    ctx.strokeStyle = COLORS.surface; ctx.lineWidth = 1; ctx.stroke();
    const v = (maxV * i / 4).toFixed(1);
    ctx.fillStyle = COLORS.text3; ctx.font = "9px JetBrains Mono"; ctx.textAlign = "right";
    ctx.fillText(v, pad.left - 6, y + 3);
  }

  values.forEach((v, i) => {
    const x   = pad.left + i * gap + (gap - barW) / 2;
    const bH  = (v / maxV) * iH;
    const y   = pad.top + iH - bH;
    const col = Array.isArray(colors) ? (colors[i] || COLORS.accent) : colors;

    // bar gradient
    const grad = ctx.createLinearGradient(0, y, 0, y + bH);
    grad.addColorStop(0, col);
    grad.addColorStop(1, col + "44");
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, barW, bH);

    // glow
    ctx.shadowColor = col; ctx.shadowBlur = 10;
    ctx.fillRect(x, y, barW, 2);
    ctx.shadowBlur = 0;

    // label
    ctx.fillStyle = COLORS.text2; ctx.font = "9px JetBrains Mono"; ctx.textAlign = "center";
    ctx.fillText(labels[i], x + barW / 2, H - pad.bottom + 14);

    // value
    ctx.fillStyle = col; ctx.font = "bold 10px JetBrains Mono";
    ctx.fillText(v.toFixed(1), x + barW / 2, y - 4);
  });
}

function drawRadarChart(ctx, labels, datasets) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const cx = W / 2, cy = H / 2;
  const r  = Math.min(W, H) * 0.38;
  const n  = labels.length;

  clearCanvas(ctx);
  ctx.fillStyle = COLORS.bg2; ctx.fillRect(0, 0, W, H);

  const angle = i => (i / n) * Math.PI * 2 - Math.PI / 2;
  const px    = (i, frac) => cx + Math.cos(angle(i)) * r * frac;
  const py    = (i, frac) => cy + Math.sin(angle(i)) * r * frac;

  // grid rings
  for (let ring = 1; ring <= 4; ring++) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = angle(i % n);
      const x = cx + Math.cos(a) * r * (ring / 4);
      const y = cy + Math.sin(a) * r * (ring / 4);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.surface; ctx.lineWidth = 1; ctx.stroke();
  }

  // spokes
  for (let i = 0; i < n; i++) {
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px(i, 1), py(i, 1));
    ctx.strokeStyle = COLORS.border; ctx.lineWidth = 1; ctx.stroke();
    // labels
    ctx.fillStyle = COLORS.text2; ctx.font = "10px JetBrains Mono"; ctx.textAlign = "center";
    ctx.fillText(labels[i], px(i, 1.15), py(i, 1.15) + 4);
  }

  // datasets
  datasets.forEach(({ values, color }) => {
    ctx.beginPath();
    const maxV = Math.max(...values, 1);
    values.forEach((v, i) => {
      const frac = v / maxV;
      i === 0 ? ctx.moveTo(px(i, frac), py(i, frac))
              : ctx.lineTo(px(i, frac), py(i, frac));
    });
    ctx.closePath();
    ctx.fillStyle   = color + "30";
    ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();

    values.forEach((v, i) => {
      const frac = v / maxV;
      ctx.beginPath(); ctx.arc(px(i, frac), py(i, frac), 4, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Gauge Manager
// ─────────────────────────────────────────────────────────────────────────────

const GAUGES = {
  cpu:  { ctx: null, color: [COLORS.accent2, COLORS.accent],  max: 100 },
  ram:  { ctx: null, color: [COLORS.accent4, "#00ccff"],       max: 100 },
  temp: { ctx: null, color: [COLORS.accent5, COLORS.accent3],  max: 100 },
  gpu:  { ctx: null, color: [COLORS.accent2, COLORS.accent5],  max: 100 },
};

function initGauges() {
  GAUGES.cpu.ctx  = document.getElementById("gaugeCpu").getContext("2d");
  GAUGES.ram.ctx  = document.getElementById("gaugeRam").getContext("2d");
  GAUGES.temp.ctx = document.getElementById("gaugeTemp").getContext("2d");
  GAUGES.gpu.ctx  = document.getElementById("gaugeGpu").getContext("2d");
  Object.values(GAUGES).forEach(g => drawGauge(g.ctx, 0, g.max, "", g.color));
}

function updateGauges(metrics) {
  const cpu  = metrics.cpu_pct   || 0;
  const ram  = metrics.ram_used_pct || 0;
  const temp = metrics.cpu_temp_c || 0;
  const gpu  = (STATE.systemInfo?.gpus?.[0]?.util_pct) || 0;

  drawGauge(GAUGES.cpu.ctx,  cpu,  100, "CPU",  GAUGES.cpu.color);
  drawGauge(GAUGES.ram.ctx,  ram,  100, "RAM",  GAUGES.ram.color);
  drawGauge(GAUGES.temp.ctx, temp, 100, "TEMP", GAUGES.temp.color);
  drawGauge(GAUGES.gpu.ctx,  gpu,  100, "GPU",  GAUGES.gpu.color);

  document.getElementById("gCpuVal").textContent  = `${cpu.toFixed(0)}%`;
  document.getElementById("gRamVal").textContent  = `${ram.toFixed(0)}%`;
  document.getElementById("gTempVal").textContent = temp > 0 ? `${temp.toFixed(0)}°C` : "—°C";
  document.getElementById("gGpuVal").textContent  = `${gpu.toFixed(0)}%`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Realtime Charts
// ─────────────────────────────────────────────────────────────────────────────

const RT_CTXS = {};

function initRealtimeCharts() {
  RT_CTXS.cpu  = document.getElementById("rtCpuChart").getContext("2d");
  RT_CTXS.core = document.getElementById("rtCoreChart").getContext("2d");
  RT_CTXS.ram  = document.getElementById("rtRamChart").getContext("2d");
  RT_CTXS.net  = document.getElementById("rtNetChart").getContext("2d");
  RT_CTXS.temp = document.getElementById("rtTempChart").getContext("2d");
}

function pushRtData(key, val) {
  const d = STATE.rtData[key];
  if (!d) return;
  d.ts.push(Date.now() / 1000);
  d.vals.push(val);
  if (d.vals.length > RT_HISTORY) { d.vals.shift(); d.ts.shift(); }
}

function updateRealtimeCharts(metrics) {
  if (STATE.rtPaused) return;
  pushRtData("cpu",  metrics.cpu_pct      || 0);
  pushRtData("ram",  metrics.ram_used_pct || 0);
  pushRtData("temp", metrics.cpu_temp_c   || 0);
  const netUp = metrics.net_sent_mb || 0;
  const netDn = metrics.net_recv_mb || 0;
  pushRtData("net_up", netUp);
  pushRtData("net_dn", netDn);

  const cd = STATE.rtData.cpu;
  if (cd.vals.length < 2) return;

  drawLineChart(RT_CTXS.cpu, [
    { values: cd.vals, color: COLORS.accent, fill: true }
  ], { minY: 0, maxY: 100 });

  drawLineChart(RT_CTXS.ram, [
    { values: STATE.rtData.ram.vals, color: COLORS.accent4, fill: true }
  ], { minY: 0, maxY: 100 });

  drawLineChart(RT_CTXS.net, [
    { values: STATE.rtData.net_up.vals, color: COLORS.accent5 },
    { values: STATE.rtData.net_dn.vals, color: COLORS.accent2 },
  ]);

  drawLineChart(RT_CTXS.temp, [
    { values: STATE.rtData.temp.vals, color: COLORS.accent3, fill: true }
  ], { minY: 0, maxY: 100 });

  // per-core (simulated from total if not available)
  if (metrics.cpu_per_core) {
    const cores = metrics.cpu_per_core;
    const W = RT_CTXS.core.canvas.width, H = RT_CTXS.core.canvas.height;
    RT_CTXS.core.clearRect(0, 0, W, H);
    RT_CTXS.core.fillStyle = COLORS.bg2;
    RT_CTXS.core.fillRect(0, 0, W, H);
    const bw = (W - 8) / cores.length;
    cores.forEach((v, i) => {
      const bh = (v / 100) * (H - 20);
      const x  = 4 + i * bw;
      const y  = H - bh - 4;
      const hue = 180 + v * 0.8;
      RT_CTXS.core.fillStyle = `hsl(${hue}, 100%, 60%)`;
      RT_CTXS.core.fillRect(x + 1, y, bw - 2, bh);
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Telemetry SSE / polling
// ─────────────────────────────────────────────────────────────────────────────

function startTelemetry() {
  // Try SSE first
  try {
    const es = new EventSource(SSE_URL);
    es.onopen = () => setConnected(true);
    es.onmessage = e => {
      const m = JSON.parse(e.data);
      handleMetrics(m);
    };
    es.onerror = () => {
      setConnected(false);
      // fall back to polling
      es.close();
      startPolling();
    };
  } catch {
    startPolling();
  }
}

function startPolling() {
  setInterval(async () => {
    try {
      const m = await API.get("/api/live_metrics");
      handleMetrics(m);
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, POLL_INTERVAL_MS);
}

function handleMetrics(m) {
  STATE.liveMetrics = m;
  // ticker
  document.getElementById("tCpu").textContent  = (m.cpu_pct || 0).toFixed(0);
  document.getElementById("tRam").textContent  = (m.ram_used_pct || 0).toFixed(0);
  document.getElementById("tTemp").textContent = m.cpu_temp_c ? m.cpu_temp_c.toFixed(0) : "—";
  document.getElementById("tNetUp").textContent = (m.net_sent_mb || 0).toFixed(1);
  document.getElementById("tNetDn").textContent = (m.net_recv_mb || 0).toFixed(1);

  updateGauges(m);
  updateRealtimeCharts(m);
  updateMemBar(m);
}

function setConnected(ok) {
  STATE.connected = ok;
  const dot   = document.getElementById("connDot");
  const label = document.getElementById("connLabel");
  dot.className   = "status-dot" + (ok ? " online" : " error");
  label.textContent = ok ? "Connected" : "Offline";
}

// ─────────────────────────────────────────────────────────────────────────────
// System Info
// ─────────────────────────────────────────────────────────────────────────────

async function loadSystemInfo() {
  try {
    const info = await API.get("/api/system_info");
    STATE.systemInfo = info;
    renderSystemInfo(info);
  } catch (e) {
    renderSystemInfoFallback();
  }
}

function renderSystemInfo(info) {
  const cpu = info.cpu || {};
  document.getElementById("cpuName").textContent = cpu.brand || info.machine || "Unknown CPU";
  document.getElementById("cpuSub").textContent  =
    `${cpu.physical_cores || info.cpu_count_physical || "?"} cores · ` +
    `${cpu.logical_cores  || info.cpu_count_logical  || "?"} threads · ` +
    `${(info.cpu_freq_mhz || 0).toFixed(0)} MHz`;

  // feature badges
  const badges = document.getElementById("cpuFeats");
  badges.innerHTML = "";
  ["sse2","avx2","avx512f","fma","aes_ni"].forEach(f => {
    const b = document.createElement("span");
    b.className = "feat-badge" + (cpu[f] ? "" : " inactive");
    b.textContent = f.toUpperCase().replace("_"," ");
    badges.appendChild(b);
  });

  // GPU
  const gpu = (info.gpus || [])[0] || {};
  document.getElementById("gpuName").textContent = gpu.name || "No GPU";
  document.getElementById("gpuSub").textContent  =
    `${gpu.vram_mb || 0} MB VRAM · Driver ${gpu.driver || "?"} · ${gpu.temp_c || 0}°C`;

  // Memory
  const ramTot = info.ram_total_gb || 0;
  const ramAvl = info.ram_avail_gb || 0;
  document.getElementById("memTotal").textContent = `${ramTot} GB`;
  document.getElementById("memSub").textContent   = `${(ramTot - ramAvl).toFixed(1)} GB used`;

  // OS
  document.getElementById("osName").textContent = info.os || "Unknown";
  document.getElementById("osSub").textContent  =
    `${info.machine || ""} · Python ${info.python || "?"} · ` +
    (info.core_available ? "C++ core ✓" : "Simulation mode");

  // GPU device list
  const gpuList = document.getElementById("gpuDeviceList");
  gpuList.innerHTML = (info.gpus || []).map(g => `
    <div class="gpu-device-card">
      <div class="gpu-device-name">#${g.index} ${g.name}</div>
      <div class="gpu-device-detail">
        VRAM: ${g.vram_mb} MB &nbsp;·&nbsp;
        Driver: ${g.driver} &nbsp;·&nbsp;
        Temp: ${g.temp_c}°C &nbsp;·&nbsp;
        Util: ${g.util_pct}% &nbsp;·&nbsp;
        Type: ${g.type}
      </div>
    </div>
  `).join("");
}

function renderSystemInfoFallback() {
  ["cpuName","gpuName","memTotal","osName"].forEach(id => {
    document.getElementById(id).textContent = "—";
  });
}

function updateMemBar(m) {
  const bar = document.getElementById("memBar");
  const pct = m.ram_used_pct || 0;
  bar.style.width = pct + "%";
  bar.style.background = pct > 85
    ? `linear-gradient(90deg, ${COLORS.accent3}, #ff6b35)`
    : pct > 60
      ? `linear-gradient(90deg, ${COLORS.accent5}, #ffdd00)`
      : `linear-gradient(90deg, ${COLORS.accent2}, ${COLORS.accent})`;
  document.getElementById("memSub").textContent = `${(STATE.systemInfo?.ram_total_gb || 0) - (STATE.liveMetrics?.ram_used_gb || 0) > 0 ? ((STATE.systemInfo?.ram_total_gb || 0) - (STATE.liveMetrics?.ram_used_gb || 0)).toFixed(1) : "?"} GB free`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Job System & Toast Notifications
// ─────────────────────────────────────────────────────────────────────────────

const TOAST_CONTAINER = (() => {
  const d = document.createElement("div");
  d.id = "jobOverlay";
  document.body.appendChild(d);
  return d;
})();

function showToast(job) {
  const d = document.createElement("div");
  d.className = "job-toast"; d.id = `toast-${job.job_id}`;
  d.innerHTML = `
    <div class="job-toast-name">${job.name.toUpperCase()}</div>
    <div class="job-toast-bar-track"><div class="job-toast-bar" id="tbar-${job.job_id}" style="width:5%"></div></div>
    <div class="job-toast-status" id="tstat-${job.job_id}"><span class="spinner"></span> Running…</div>
  `;
  TOAST_CONTAINER.appendChild(d);

  // animate bar indeterminate
  let pct = 5;
  const iv = setInterval(() => {
    pct = Math.min(pct + 2, 90);
    const bar = document.getElementById(`tbar-${job.job_id}`);
    if (bar) bar.style.width = pct + "%";
    else clearInterval(iv);
  }, 200);
  STATE.activeJobs[job.job_id] = { el: d, interval: iv };
}

function completeToast(job_id, success) {
  const j = STATE.activeJobs[job_id];
  if (!j) return;
  clearInterval(j.interval);
  const bar  = document.getElementById(`tbar-${job_id}`);
  const stat = document.getElementById(`tstat-${job_id}`);
  if (bar)  bar.style.width = "100%";
  if (stat) stat.innerHTML  = success ? "✓ Complete" : "✗ Error";
  if (stat) stat.style.color = success ? COLORS.accent4 : COLORS.accent3;
  setTimeout(() => { j.el.remove(); delete STATE.activeJobs[job_id]; }, 2500);
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

function getParam(id) {
  const el = document.getElementById(id);
  if (!el) return undefined;
  if (el.tagName === "SELECT") return isNaN(+el.value) ? el.value : +el.value;
  return +el.value;
}

const BENCH_CONFIG = {
  matmul: {
    endpoint: "/api/bench/matmul",
    params: () => ({ N: getParam("mmN"), iters: getParam("mmIter"), threads: getParam("mmThreads") }),
    resultEl: "resMatmul",
    chartEl:  "chartMatmul",
    scoreId:  "sc-matmul",
    scoreKey: "gflops",
    unit:     "GFLOPS",
  },
  fft: {
    endpoint: "/api/bench/fft",
    params: () => ({ n: getParam("fftN"), iters: getParam("fftIter"), threads: getParam("fftThreads") }),
    resultEl: "resFft",
    chartEl:  "chartFft",
    scoreId:  "sc-fft",
    scoreKey: "gflops",
    unit:     "GFLOPS",
  },
  memory_bw: {
    endpoint: "/api/bench/memory_bw",
    params: () => ({ mb: getParam("mbwBuf"), iters: getParam("mbwIter") }),
    resultEl: "resMbw",
    chartEl:  "chartMbw",
    scoreId:  "sc-membw",
    scoreKey: "bandwidth_gbps",
    unit:     "GB/s",
  },
  cache_latency: {
    endpoint: "/api/bench/cache_latency",
    params: () => ({}),
    resultEl: "resCache",
    chartEl:  "chartCache",
  },
  integer: {
    endpoint: "/api/bench/integer",
    params: () => ({ iters: getParam("intIter") }),
    resultEl: "resInt",
    chartEl:  "chartInt",
    scoreId:  "sc-int",
    scoreKey: "gops",
    unit:     "GOPS",
  },
  branch_torture: {
    endpoint: "/api/bench/branch_torture",
    params: () => ({ iters: getParam("brIter") }),
    resultEl: "resBranch",
    chartEl:  "chartBranch",
    scoreId:  "sc-branch",
    scoreKey: "gops",
    unit:     "GOPS",
  },
  stress_suite: {
    endpoint: "/api/bench/stress_suite",
    params: () => ({ duration_s: getParam("stressDur"), threads: getParam("stressThreads") }),
    resultEl: "resStress",
    chartEl:  "chartStress",
  },
  volume_shader: {
    endpoint: "/api/bench/volume_shader",
    params: () => ({
      width:     getParam("vsW"),
      height:    getParam("vsH"),
      max_steps: getParam("vsSteps"),
      step_size: getParam("vsStep") / 1000,
      time_val:  getParam("vsTime") / 100,
      threads:   getParam("vsThreads"),
    }),
    resultEl: "shaderStats",
    chartEl:  null,
    scoreId:  "sc-vol",
    scoreKey: "gflops",
    unit:     "GFLOPS",
  },
};

const BENCH_HISTORY = {};

async function runBenchmark(name) {
  const cfg = BENCH_CONFIG[name];
  if (!cfg) return;

  const resEl = document.getElementById(cfg.resultEl);
  if (resEl) { resEl.className = "bench-result running"; resEl.textContent = "⏳ Running…"; }

  let job;
  try {
    job = await API.post(cfg.endpoint, cfg.params());
  } catch (e) {
    if (resEl) { resEl.className = "bench-result error"; resEl.textContent = `Error: ${e.message}`; }
    return;
  }
  showToast({ job_id: job.job_id, name });

  // Poll for result
  const result = await pollJob(job.job_id);
  completeToast(job.job_id, !!result);

  if (!result) {
    if (resEl) { resEl.className = "bench-result error"; resEl.textContent = "Benchmark failed"; }
    return;
  }

  // Handle result per benchmark type
  handleBenchmarkResult(name, result, cfg, resEl);
}

async function pollJob(job_id) {
  for (let i = 0; i < 600; i++) {
    await sleep(500);
    try {
      const j = await API.get(`/api/jobs/${job_id}`);
      if (j.status === "done")   return j.result;
      if (j.status === "error")  return null;
    } catch { }
  }
  return null;
}

function handleBenchmarkResult(name, result, cfg, resEl) {
  // Store history
  if (!BENCH_HISTORY[name]) BENCH_HISTORY[name] = [];
  BENCH_HISTORY[name].push({ ts: Date.now(), result });

  // Score card update
  if (cfg.scoreId && cfg.scoreKey && result[cfg.scoreKey] !== undefined) {
    const val = result[cfg.scoreKey];
    STATE.scores[name] = val;
    const cell = document.getElementById(cfg.scoreId);
    if (cell) {
      cell.querySelector(".sc-val").textContent = val.toFixed(2);
      cell.querySelector(".sc-unit").textContent = cfg.unit;
      cell.classList.add("updated");
      setTimeout(() => cell.classList.remove("updated"), 700);
    }
    updateRadarChart();
  }

  // Generic result text
  if (resEl) {
    resEl.className = "bench-result success";
    resEl.innerHTML = formatResult(name, result);
  }

  // Charts
  if (cfg.chartEl) renderBenchChart(name, result, cfg.chartEl);

  // Volume shader special
  if (name === "volume_shader") renderVolumeResult(result);
}

function formatResult(name, r) {
  if (r.simulated) return `[SIM] ${JSON.stringify(r).slice(0, 200)}`;
  switch (name) {
    case "matmul":        return `GFLOPS: ${r.gflops.toFixed(2)} · Per thread: ${r.per_thread.toFixed(2)} · Matrix: ${r.matrix_size}×${r.matrix_size}`;
    case "fft":           return `GFLOPS: ${r.gflops.toFixed(2)} · FFT size: ${r.fft_size} · Iters: ${r.iterations}`;
    case "memory_bw":     return `Bandwidth: ${r.bandwidth_gbps.toFixed(2)} GB/s · Buffer: ${r.buffer_mb} MB`;
    case "cache_latency": return Object.entries(r).filter(([k])=>!k.startsWith("s")).map(([k,v])=>`${k}: ${typeof v === "number" ? v.toFixed(2)+"ns" : v}`).join(" · ");
    case "integer":       return `GOPS: ${r.gops.toFixed(2)}`;
    case "branch_torture":return `GOPS: ${r.gops.toFixed(2)}`;
    case "stress_suite":  return `Duration: ${r.duration_s}s · Threads: ${r.threads} · Iters: ${r.total_iters}`;
    case "volume_shader": return `GFLOPS: ${r.gflops?.toFixed(2)} · Rays/s: ${(r.rays_per_second||0).toLocaleString()} · Avg steps: ${(r.avg_march_steps||0).toFixed(1)} · Quality: ${((r.convergence_quality||0)*100).toFixed(0)}%`;
    default:              return JSON.stringify(r).slice(0, 300);
  }
}

function renderBenchChart(name, result, chartId) {
  const canvas = document.getElementById(chartId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const hist = BENCH_HISTORY[name] || [];
  const vals = hist.map(h => {
    const cfg = BENCH_CONFIG[name];
    return cfg.scoreKey ? (h.result[cfg.scoreKey] || 0) : 0;
  });

  if (name === "cache_latency") {
    const keys   = Object.keys(result).filter(k => !k.startsWith("s"));
    const values = keys.map(k => result[k]);
    const colors = [COLORS.accent4, COLORS.accent, COLORS.accent5, COLORS.accent3];
    drawBarChart(ctx, keys, values, colors, {});
    return;
  }

  if (name === "stress_suite" && result.timestamps) {
    drawLineChart(ctx, [
      { values: result.cpu_load_pct || [], color: COLORS.accent, fill: true },
    ], { minY: 0, maxY: 100, xLabel: "Time (s)" });
    return;
  }

  // History line
  if (vals.length >= 1) {
    drawLineChart(ctx, [
      { values: vals, color: COLORS.accent2, fill: true },
    ]);
  }
}

function updateRadarChart() {
  const canvas = document.getElementById("radarCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const labels = ["MatMul", "FFT", "Mem BW", "Integer", "Volume", "Branch"];
  const vals = [
    STATE.scores.matmul        || 0,
    STATE.scores.fft           || 0,
    STATE.scores.memory_bw     || 0,
    STATE.scores.integer       || 0,
    STATE.scores.volume_shader || 0,
    STATE.scores.branch_torture|| 0,
  ];
  drawRadarChart(ctx, labels, [{ values: vals, color: COLORS.accent }]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Volume Shader Result Renderer
// ─────────────────────────────────────────────────────────────────────────────

function renderVolumeResult(result) {
  const canvas = document.getElementById("volumeCanvas");
  if (!canvas || !result.pixels) return;
  const W = result.width, H = result.height;
  canvas.width  = W;
  canvas.height = H;
  const ctx     = canvas.getContext("2d");
  const img     = ctx.createImageData(W, H);

  // pixels is H×(W*4) nested array
  const pixels = result.pixels;
  for (let y = 0; y < H; y++) {
    const row = pixels[y];
    if (!row) continue;
    for (let x = 0; x < W; x++) {
      const R = Math.min(255, Math.round((row[x * 4 + 0] || 0) * 255));
      const G = Math.min(255, Math.round((row[x * 4 + 1] || 0) * 255));
      const B = Math.min(255, Math.round((row[x * 4 + 2] || 0) * 255));
      const A = Math.min(255, Math.round((row[x * 4 + 3] || 0) * 255 + 30));
      const idx = (y * W + x) * 4;
      img.data[idx + 0] = R;
      img.data[idx + 1] = G;
      img.data[idx + 2] = B;
      img.data[idx + 3] = Math.max(A, 30);
    }
  }
  ctx.putImageData(img, 0, 0);

  document.getElementById("shaderStats").innerHTML =
    `GFLOPS: <b style="color:${COLORS.accent}">${(result.gflops||0).toFixed(2)}</b> &nbsp;·&nbsp; ` +
    `Rays/s: <b style="color:${COLORS.accent4}">${(result.rays_per_second||0).toLocaleString()}</b> &nbsp;·&nbsp; ` +
    `Avg steps: <b style="color:${COLORS.accent5}">${(result.avg_march_steps||0).toFixed(1)}</b> &nbsp;·&nbsp; ` +
    `Quality: <b style="color:${COLORS.accent2}">${((result.convergence_quality||0)*100).toFixed(0)}%</b>`;
}

// Volume Shader Animation
let _vsAnimId = null;
let _vsTimeState = 0;

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btnAnimateVolume")?.addEventListener("click", () => {
    if (_vsAnimId) { clearInterval(_vsAnimId); _vsAnimId = null; return; }
    _vsAnimId = setInterval(async () => {
      _vsTimeState = (_vsTimeState + 0.1) % 6.28;
      document.getElementById("vsTime").value = Math.round(_vsTimeState * 100);
      document.getElementById("vsTimeVal").textContent = _vsTimeState.toFixed(2);
      await runBenchmark("volume_shader");
    }, 1500);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// WebGL Particle Storm
// ─────────────────────────────────────────────────────────────────────────────

function initWebGL() {
  const canvas = document.getElementById("glCanvas");
  if (!canvas) return;
  const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
  if (!gl) { document.getElementById("glStats").textContent = "WebGL not available"; return; }

  const VS_SRC = `
    attribute vec2 a_pos;
    attribute float a_life;
    attribute float a_seed;
    uniform float u_time;
    varying float v_life;
    varying float v_seed;
    void main() {
      float t  = u_time + a_seed * 6.2831;
      float px = a_pos.x + sin(t * 1.7 + a_seed * 3.14) * 0.02;
      float py = a_pos.y + cos(t * 2.3 + a_seed * 1.41) * 0.02;
      v_life = a_life;
      v_seed = a_seed;
      gl_Position = vec4(px, py, 0.0, 1.0);
      gl_PointSize = 1.5 + a_life * 3.0;
    }
  `;

  const FS_SRC = `
    precision mediump float;
    varying float v_life;
    varying float v_seed;
    void main() {
      float d = length(gl_PointCoord - 0.5);
      if (d > 0.5) discard;
      float h = v_seed;
      float r = 0.5 + 0.5*sin(h*6.28 + 0.0);
      float g = 0.5 + 0.5*sin(h*6.28 + 2.09);
      float b = 0.5 + 0.5*sin(h*6.28 + 4.19);
      gl_FragColor = vec4(r, g, b, v_life * (1.0 - d*2.0));
    }
  `;

  function compileShader(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }

  const prog = gl.createProgram();
  gl.attachShader(prog, compileShader(gl.VERTEX_SHADER,   VS_SRC));
  gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS_SRC));
  gl.linkProgram(prog);
  gl.useProgram(prog);

  const N = parseInt(document.getElementById("particleCount")?.value || "50000");
  const pos   = new Float32Array(N * 2);
  const life  = new Float32Array(N);
  const seed  = new Float32Array(N);
  const vel   = new Float32Array(N * 2);

  for (let i = 0; i < N; i++) {
    pos[i*2]   = (Math.random() - 0.5) * 2;
    pos[i*2+1] = (Math.random() - 0.5) * 2;
    vel[i*2]   = (Math.random() - 0.5) * 0.004;
    vel[i*2+1] = (Math.random() - 0.5) * 0.004;
    life[i]    = Math.random();
    seed[i]    = Math.random();
  }

  const posBuf  = gl.createBuffer();
  const lifeBuf = gl.createBuffer();
  const seedBuf = gl.createBuffer();

  const aPos  = gl.getAttribLocation(prog, "a_pos");
  const aLife = gl.getAttribLocation(prog, "a_life");
  const aSeed = gl.getAttribLocation(prog, "a_seed");
  const uTime = gl.getUniformLocation(prog, "u_time");

  gl.bindBuffer(gl.ARRAY_BUFFER, seedBuf);
  gl.bufferData(gl.ARRAY_BUFFER, seed, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(aSeed);
  gl.vertexAttribPointer(aSeed, 1, gl.FLOAT, false, 0, 0);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

  let rafId = null, t0 = performance.now(), frames = 0;

  function frame(ts) {
    // Update positions on CPU (simulate)
    for (let i = 0; i < N; i++) {
      pos[i*2]   += vel[i*2];
      pos[i*2+1] += vel[i*2+1];
      life[i]    -= 0.003;
      if (life[i] <= 0 || pos[i*2] > 1 || pos[i*2] < -1 ||
          pos[i*2+1] > 1 || pos[i*2+1] < -1) {
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * 0.8;
        pos[i*2]   = Math.cos(angle) * r;
        pos[i*2+1] = Math.sin(angle) * r;
        vel[i*2]   = Math.cos(angle + Math.PI/2) * (0.001 + Math.random() * 0.003);
        vel[i*2+1] = Math.sin(angle + Math.PI/2) * (0.001 + Math.random() * 0.003);
        life[i]    = 0.3 + Math.random() * 0.7;
      }
    }

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.03, 0.04, 0.06, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, lifeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, life, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aLife);
    gl.vertexAttribPointer(aLife, 1, gl.FLOAT, false, 0, 0);

    gl.uniform1f(uTime, ts * 0.001);
    gl.drawArrays(gl.POINTS, 0, N);

    frames++;
    const elapsed = (ts - t0) / 1000;
    if (elapsed >= 1) {
      document.getElementById("glStats").textContent =
        `WebGL · ${N.toLocaleString()} particles · ${frames} FPS · Render: ${(elapsed/frames*1000).toFixed(2)}ms/frame`;
      frames = 0; t0 = ts;
    }

    rafId = requestAnimationFrame(frame);
  }

  STATE.glParticles = { stop: () => { if (rafId) cancelAnimationFrame(rafId); } };
  rafId = requestAnimationFrame(frame);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Volume Renderer (WebGL isosurface + noise field)
// ─────────────────────────────────────────────────────────────────────────────

class Vol3DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl");
    this.angle = { x: 0.4, y: 0 };
    this.animId = null;
    this.volumeData = null;
    this.prog = null;
    this._init();
  }

  _init() {
    const gl = this.gl;
    if (!gl) return;

    const VS = `
      attribute vec3 a_pos;
      attribute vec3 a_norm;
      uniform mat4 u_mvp;
      uniform mat4 u_model;
      varying vec3 v_norm;
      varying vec3 v_pos;
      void main() {
        v_norm = normalize((u_model * vec4(a_norm, 0.0)).xyz);
        v_pos  = (u_model * vec4(a_pos, 1.0)).xyz;
        gl_Position = u_mvp * vec4(a_pos, 1.0);
      }
    `;
    const FS = `
      precision mediump float;
      varying vec3 v_norm;
      varying vec3 v_pos;
      void main() {
        vec3 light = normalize(vec3(1.0, 1.5, 2.0));
        float diff  = max(dot(v_norm, light), 0.0);
        float spec  = pow(max(dot(reflect(-light, v_norm), normalize(-v_pos)), 0.0), 32.0);
        float h = (v_pos.y + 1.0) * 0.5;
        vec3 col = mix(vec3(0.0, 0.9, 1.0), vec3(0.5, 0.0, 1.0), h);
        col = col * (0.2 + 0.7 * diff) + vec3(0.6) * spec * 0.5;
        gl_FragColor = vec4(col, 0.9);
      }
    `;

    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s); return s;
    };

    this.prog = gl.createProgram();
    gl.attachShader(this.prog, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(this.prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(this.prog);

    this.posBuf  = gl.createBuffer();
    this.normBuf = gl.createBuffer();
    this.idxBuf  = gl.createBuffer();
    this.triCount = 0;

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  // Marching-cubes (simplified) isosurface extraction
  buildMesh(volumeFlat, nx, ny, nz, isoVal) {
    const sample = (x, y, z) => {
      x = Math.max(0, Math.min(nx-1, x));
      y = Math.max(0, Math.min(ny-1, y));
      z = Math.max(0, Math.min(nz-1, z));
      return volumeFlat[z * ny * nx + y * nx + x];
    };

    const verts = [], norms = [], indices = [];
    let vi = 0;

    const edgeInterp = (p1, p2, v1, v2, iso) => {
      const t = (iso - v1) / (v2 - v1 + 1e-8);
      return [p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1]), p1[2] + t*(p2[2]-p1[2])];
    };

    // Marching cubes table (edges only — simplified to surface quads per cube face)
    for (let z = 0; z < nz-1; z++)
    for (let y = 0; y < ny-1; y++)
    for (let x = 0; x < nx-1; x++) {
      const vals = [
        sample(x,y,z),   sample(x+1,y,z),   sample(x+1,y+1,z),   sample(x,y+1,z),
        sample(x,y,z+1), sample(x+1,y,z+1), sample(x+1,y+1,z+1), sample(x,y+1,z+1),
      ];
      let mask = 0;
      for (let i=0;i<8;i++) if (vals[i] > isoVal) mask |= (1<<i);
      if (mask === 0 || mask === 255) continue;

      // Very simplified — emit quads on faces that cross iso
      const fx = (x+0.5)/(nx-1)*2-1, fy = (y+0.5)/(ny-1)*2-1, fz = (z+0.5)/(nz-1)*2-1;
      const s = 1.0/(nx-1);

      // Gradient normal (central diff)
      const gx = sample(x+1,y,z) - sample(x-1,y,z);
      const gy = sample(x,y+1,z) - sample(x,y-1,z);
      const gz = sample(x,y,z+1) - sample(x,y,z-1);
      const nl = Math.sqrt(gx*gx+gy*gy+gz*gz) + 1e-8;

      const emit = (dx, dy, dz) => {
        verts.push(fx+dx*s*2, fy+dy*s*2, fz+dz*s*2);
        norms.push(-gx/nl, -gy/nl, -gz/nl);
      };

      emit(-0.5,-0.5,-0.5); emit(0.5,-0.5,-0.5); emit(0.5,0.5,-0.5); emit(-0.5,0.5,-0.5);
      indices.push(vi, vi+1, vi+2, vi, vi+2, vi+3);
      vi += 4;

      if (vi > 60000) break; // GPU limit guard
    }

    this.triCount = indices.length;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(norms), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.DYNAMIC_DRAW);
  }

  render() {
    const gl = this.gl;
    if (!gl || !this.triCount) return;
    const W = this.canvas.width, H = this.canvas.height;
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.03, 0.04, 0.06, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.prog);

    // Build MVP
    const aspect = W / H;
    const proj   = perspective(Math.PI/4, aspect, 0.1, 100);
    const view   = lookAt([0,0,4],[0,0,0],[0,1,0]);
    const model  = matMul(rotX(this.angle.x), rotY(this.angle.y));
    const mv     = matMul(view, model);
    const mvp    = matMul(proj, mv);

    const uMvp   = gl.getUniformLocation(this.prog, "u_mvp");
    const uModel = gl.getUniformLocation(this.prog, "u_model");
    gl.uniformMatrix4fv(uMvp,   false, mvp);
    gl.uniformMatrix4fv(uModel, false, model);

    const aPos  = gl.getAttribLocation(this.prog, "a_pos");
    const aNorm = gl.getAttribLocation(this.prog, "a_norm");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos,  3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.normBuf);
    gl.enableVertexAttribArray(aNorm);
    gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.drawElements(gl.TRIANGLES, Math.min(this.triCount, 65535), gl.UNSIGNED_SHORT, 0);
  }

  startAnim() {
    this.stopAnim();
    const tick = () => {
      this.angle.y += 0.008 * (getParam("v3rotspd") / 50);
      this.render();
      this.animId = requestAnimationFrame(tick);
    };
    this.animId = requestAnimationFrame(tick);
  }

  stopAnim() { if (this.animId) { cancelAnimationFrame(this.animId); this.animId = null; } }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix Math (mini WebGL math lib)
// ─────────────────────────────────────────────────────────────────────────────

function perspective(fov, asp, near, far) {
  const f = 1 / Math.tan(fov/2), d = far - near;
  return [f/asp,0,0,0, 0,f,0,0, 0,0,-(far+near)/d,-1, 0,0,-2*far*near/d,0];
}

function lookAt(eye, at, up) {
  const z = norm3(sub3(eye, at));
  const x = norm3(cross3(up, z));
  const y = cross3(z, x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
          -dot3(x,eye),-dot3(y,eye),-dot3(z,eye),1];
}

function rotX(a) { const c=Math.cos(a),s=Math.sin(a); return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]; }
function rotY(a) { const c=Math.cos(a),s=Math.sin(a); return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]; }

function matMul(a, b) {
  const r = new Array(16).fill(0);
  for (let i=0;i<4;i++) for (let j=0;j<4;j++) for (let k=0;k<4;k++) r[i*4+j] += a[i*4+k]*b[k*4+j];
  return r;
}

const sub3   = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot3   = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross3 = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm3  = a => { const l=Math.sqrt(dot3(a,a))||1; return [a[0]/l,a[1]/l,a[2]/l]; };

// ─────────────────────────────────────────────────────────────────────────────
// Volume 3D Panel
// ─────────────────────────────────────────────────────────────────────────────

let vol3dRenderer = null;

function initVol3d() {
  const canvas = document.getElementById("vol3dCanvas");
  if (!canvas) return;
  vol3dRenderer = new Vol3DRenderer(canvas);
}

async function generateVolume() {
  const params = {
    nx: getParam("v3nx"), ny: getParam("v3ny"), nz: getParam("v3nz"),
    scale: getParam("v3scale"), t_offset: getParam("v3t"),
    octaves: getParam("v3oct"),
  };

  document.getElementById("vol3dStats").textContent = "⏳ Generating noise volume…";

  let job;
  try { job = await API.post("/api/bench/noise_volume", params); }
  catch (e) { document.getElementById("vol3dStats").textContent = `Error: ${e.message}`; return; }

  const result = await pollJob(job.job_id);
  if (!result || !result.volume) {
    document.getElementById("vol3dStats").textContent = "Failed to generate volume";
    return;
  }

  STATE.vol3dData = result;
  const [nz, ny, nx] = result.shape;
  const flat = result.volume.flat(3);
  const iso  = getParam("v3iso") / 100;

  const t0 = performance.now();
  vol3dRenderer.buildMesh(flat, nx, ny, nz, iso);
  const buildMs = (performance.now() - t0).toFixed(1);
  vol3dRenderer.render();

  document.getElementById("vol3dStats").innerHTML =
    `Volume: ${nx}×${ny}×${nz} &nbsp;·&nbsp; ` +
    `Triangles: ${vol3dRenderer.triCount.toLocaleString()} &nbsp;·&nbsp; ` +
    `Mesh build: ${buildMs}ms &nbsp;·&nbsp; ` +
    `ISO: ${iso.toFixed(2)} &nbsp;·&nbsp; ` +
    (result.simulated ? "SIM mode" : "C++ FBM");
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory Bandwidth Sweep
// ─────────────────────────────────────────────────────────────────────────────

async function runMbwSweep() {
  const sizes = [16, 64, 128, 256, 512, 1024];
  const results = [];
  const resEl = document.getElementById("resMbwSweep");
  resEl.textContent = "Running sweep…";

  for (const mb of sizes) {
    const job = await API.post("/api/bench/memory_bw", { mb, iters: 3 });
    const r   = await pollJob(job.job_id);
    if (r) results.push({ mb, bw: r.bandwidth_gbps });
    resEl.textContent = `Progress: ${results.length}/${sizes.length}`;
  }

  STATE.mbwSweepResults = results;
  const canvas = document.getElementById("chartMbwSweep");
  if (!canvas) return;
  drawBarChart(canvas.getContext("2d"),
    results.map(r => `${r.mb}MB`),
    results.map(r => r.bw),
    results.map((_, i) => `hsl(${180 + i*30}, 90%, 60%)`));
  resEl.textContent = `Sweep complete. Peak: ${Math.max(...results.map(r=>r.bw)).toFixed(2)} GB/s`;
}

// ─────────────────────────────────────────────────────────────────────────────
// History
// ─────────────────────────────────────────────────────────────────────────────

async function refreshHistory() {
  let data;
  try { data = await API.get("/api/jobs"); }
  catch { return; }

  const rows = [...(data.history || []), ...(data.jobs || [])].slice(-50).reverse();
  const table = document.getElementById("historyTable");
  if (!rows.length) { table.innerHTML = "<p style='color:var(--text3)'>No history yet.</p>"; return; }

  table.innerHTML = `
    <table class="hist-table">
      <thead><tr>
        <th>Job ID</th><th>Name</th><th>Status</th><th>Duration</th><th>Key Result</th>
      </tr></thead>
      <tbody>
        ${rows.map(j => `
          <tr>
            <td style="color:var(--text3);font-size:9px">${j.job_id?.slice(0,8)}…</td>
            <td>${j.name || "—"}</td>
            <td><span class="status-badge ${j.status}">${j.status}</span></td>
            <td>${j.duration_s ? j.duration_s.toFixed(2)+"s" : "—"}</td>
            <td style="color:var(--accent);font-size:11px">${extractKeyResult(j)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  // Comparison chart
  const scores = rows.filter(j => j.status === "done" && j.result).slice(0, 20);
  if (scores.length) {
    const ctx = document.getElementById("historyComparison").getContext("2d");
    const labels = scores.map(j => j.name?.substring(0,10) || "?");
    const vals   = scores.map(j => extractNumericResult(j));
    drawBarChart(ctx, labels, vals, COLORS.accent);
  }
}

function extractKeyResult(job) {
  if (!job.result) return "—";
  const r = job.result;
  if (r.gflops !== undefined) return `${r.gflops.toFixed(2)} GFLOPS`;
  if (r.bandwidth_gbps !== undefined) return `${r.bandwidth_gbps.toFixed(2)} GB/s`;
  if (r.gops !== undefined) return `${r.gops.toFixed(2)} GOPS`;
  if (r.rays_per_second !== undefined) return `${r.rays_per_second.toLocaleString()} rays/s`;
  return "—";
}

function extractNumericResult(job) {
  if (!job.result) return 0;
  const r = job.result;
  return r.gflops || r.bandwidth_gbps || r.gops || 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Range slider labels
// ─────────────────────────────────────────────────────────────────────────────

function bindRangeLabels() {
  const pairs = [
    ["mmN","mmNVal"],["mmIter","mmIterVal"],["mmThreads","mmThreadsVal"],
    ["fftIter","fftIterVal"],["fftThreads","fftThreadsVal"],
    ["intIter","intIterVal"],["brIter","brIterVal"],
    ["stressDur","stressDurVal"],["stressThreads","stressThreadsVal"],
    ["vsW","vsWVal"],["vsH","vsHVal"],["vsSteps","vsStepsVal"],
    ["vsStep","vsStepVal"],["vsThreads","vsThreadsVal"],
    ["mbwIter","mbwIterVal"],
    ["v3nx","v3nxVal"],["v3ny","v3nyVal"],["v3nz","v3nzVal"],
    ["v3scale","v3scaleVal"],["v3oct","v3octVal"],
    ["v3t","v3tVal"],["v3opacity","v3opacityVal"],["v3iso","v3isoVal"],
    ["v3rotspd","v3rotspdVal"],
  ];

  pairs.forEach(([inputId, labelId]) => {
    const inp = document.getElementById(inputId);
    const lbl = document.getElementById(labelId);
    if (!inp || !lbl) return;
    const update = () => {
      let v = +inp.value;
      if (inputId === "vsStep")    v = (v / 1000).toFixed(3);
      else if (inputId === "vsTime") v = (v / 100).toFixed(2);
      else if (inputId === "v3iso")  v = (v / 100).toFixed(2);
      lbl.textContent = v;
    };
    inp.addEventListener("input", update);
    update();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Run-all suite
// ─────────────────────────────────────────────────────────────────────────────

async function runFullSuite() {
  const btn = document.getElementById("btnRunAll");
  btn.disabled = true;
  btn.textContent = "⏳ Running…";

  const tests = ["matmul","fft","memory_bw","cache_latency","integer","branch_torture"];
  for (const t of tests) {
    await runBenchmark(t);
    await sleep(200);
  }

  btn.disabled = false;
  btn.textContent = "▶ Run Full Suite";
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

async function main() {
  await runLoader();

  initRouter();
  initGauges();
  initRealtimeCharts();
  bindRangeLabels();
  initVol3d();

  // Bench run buttons
  document.querySelectorAll("[data-bench]").forEach(btn => {
    btn.addEventListener("click", () => runBenchmark(btn.dataset.bench));
  });

  // Special buttons
  document.getElementById("btnRunAll")?.addEventListener("click", runFullSuite);
  document.getElementById("btnStopStress")?.addEventListener("click", () => {
    API.post("/api/bench/stop_stress").catch(() => {});
  });
  document.getElementById("btnParticles")?.addEventListener("click", () => initWebGL());
  document.getElementById("btnStopParticles")?.addEventListener("click", () => {
    STATE.glParticles?.stop();
  });
  document.getElementById("btnGenVolume")?.addEventListener("click", generateVolume);
  document.getElementById("btnAnimVol3d")?.addEventListener("click", () => {
    if (vol3dRenderer?.animId) vol3dRenderer.stopAnim();
    else vol3dRenderer?.startAnim();
  });
  document.getElementById("btnMbwSweep")?.addEventListener("click", runMbwSweep);
  document.getElementById("btnRefreshHistory")?.addEventListener("click", refreshHistory);
  document.getElementById("btnClearRt")?.addEventListener("click", () => {
    Object.values(STATE.rtData).forEach(d => { d.ts = []; d.vals = []; });
  });
  document.getElementById("rtPause")?.addEventListener("change", e => {
    STATE.rtPaused = e.target.checked;
  });

  // Realtime iso rebuild on slider change for 3D vol
  ["v3iso", "v3color", "v3opacity"].forEach(id => {
    document.getElementById(id)?.addEventListener("input", () => {
      if (!STATE.vol3dData || !vol3dRenderer) return;
      const [nz, ny, nx] = STATE.vol3dData.shape;
      const flat = STATE.vol3dData.volume.flat(3);
      vol3dRenderer.buildMesh(flat, nx, ny, nz, getParam("v3iso") / 100);
      vol3dRenderer.render();
    });
  });

  // Start connectivity & telemetry
  await loadSystemInfo();
  startTelemetry();

  // Initial radar placeholder
  updateRadarChart();

  // Connectivity status (server was already pinged inside runLoader)
  try {
    await API.get("/api/live_metrics");
    setConnected(true);
  } catch {
    setConnected(false);
  }
}

main();
