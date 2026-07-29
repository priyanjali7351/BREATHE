/* BREATHE dashboard — polls /api/live and renders the instrument.
   Signature detail: the central orb's breathing period tracks live HR, and the
   particle field density/speed tracks real PM2.5. */

const $ = (id) => document.getElementById(id);
const RING_LEN = 578;                 // 2πr for r=92
const CAT_VARS = {                    // category → accent colour driving the whole theme
  "Safe":          "#34e0a1",
  "Moderate Risk": "#f2c14e",
  "High Risk":     "#f28c3b",
  "Critical":      "#f24b4b",
};

const state = {
  mode: "auto", sim: false,
  city: "Delhi", age: 30, condition: "Healthy",
  simHr: 72, simPm: 12,
};

/* ── Particle field (canvas) — density & speed scale with PM2.5 ──────────── */
const canvas = $("particles"), ctx = canvas.getContext("2d");
let particles = [], W, H, targetCount = 40, driftBoost = 1;
function resize() { W = canvas.width = innerWidth; H = canvas.height = innerHeight; }
addEventListener("resize", resize); resize();

function ensureParticles() {
  while (particles.length < targetCount) {
    particles.push({ x: Math.random()*W, y: Math.random()*H,
      r: Math.random()*1.8 + 0.4, s: Math.random()*0.4 + 0.1,
      o: Math.random()*0.5 + 0.15 });
  }
  if (particles.length > targetCount) particles.length = targetCount;
}
function drawParticles(color) {
  ctx.clearRect(0, 0, W, H);
  for (const p of particles) {
    p.y -= p.s * driftBoost; p.x += Math.sin(p.y * 0.01) * 0.2;
    if (p.y < -5) { p.y = H + 5; p.x = Math.random()*W; }
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 7);
    ctx.fillStyle = color; ctx.globalAlpha = p.o; ctx.fill();
  }
  ctx.globalAlpha = 1;
  requestAnimationFrame(() => drawParticles(color));
}
let particleColor = "#34e0a1";
drawParticles(particleColor);

/* ── ECG trace buffer ───────────────────────────────────────────────────── */
let ecg = new Array(60).fill(23);
// Null-safe DOM helpers — a missing/renamed element no longer aborts the whole
// render (which was freezing every field after the first null).
const setText = (id, v) => { const e = $(id); if (e) e.textContent = v; };
const setHTML = (id, v) => { const e = $(id); if (e) e.innerHTML = v; };
const setStyle = (id, prop, v) => { const e = $(id); if (e) e.style.setProperty(prop, v); };

function pushEcg(hr) {
  // fake a QRS-ish spike each tick scaled a little by HR
  const amp = hr ? Math.min(1, hr / 160) : 0.15;
  const beat = [23, 21, 26, 6, 40, 20, 23].map(v => 23 + (v - 23) * (0.4 + amp));
  ecg = ecg.slice(beat.length).concat(beat);
  const step = 240 / (ecg.length - 1);
  const p = $("ecgPath");
  if (p) p.setAttribute("d", ecg.map((v, i) => `${i ? "L" : "M"}${(i*step).toFixed(1)},${v.toFixed(1)}`).join(" "));
}

/* ── Apply a live payload to the UI ─────────────────────────────────────── */
function render(d) {
  const color = CAT_VARS[d.category] || "#34e0a1";
  document.documentElement.style.setProperty("--glow", color);
  document.documentElement.style.setProperty("--accent", color);
  particleColor = color;

  // Orb: fill ring by PHRS, number + category
  setText("phrsVal", Math.round(d.phrs));
  setText("phrsCat", d.category);
  const orbFill = $("orbFill");
  if (orbFill) {
    orbFill.style.stroke = color;
    orbFill.style.strokeDashoffset = RING_LEN * (1 - Math.min(d.phrs, 100) / 100);
  }

  // Breathing period tracks HR (60/HR seconds); default calm 4s when no HR
  const breath = d.hr ? Math.max(1.6, Math.min(6, 60 / d.hr)) : 4;
  setStyle("orbWrap", "--breath", breath.toFixed(2) + "s");

  // Air panel
  setText("aqiVal", Math.round(d.aqi));
  setText("cityName", d.city);
  if (d.mode === "sensor") {
    setHTML("airLabel", `Local · <b>your air</b>`);
    setText("airSub", `dust sensor · ${d.pm25} µg/m³ PM2.5`);
  } else {
    setHTML("airLabel", `Ambient · <b>${d.city}</b>`);
    setText("airSub", `live city air · PM2.5 ${d.pm25} µg/m³`);
  }
  setHTML("pollChips", Object.entries(d.pollutants || {})
    .map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join(""));

  // Body panel
  setText("hrVal", d.hr ? Math.round(d.hr) : "—");
  setText("activityName", (d.activity || "").toLowerCase());
  setText("ageChip", d.profile.age);
  setText("condChip", d.profile.condition);
  setText("hrSub", d.hr
    ? (d.hr_source === "sim" ? "simulated exertion" : "live watch → ventilation")
    : "no signal");
  pushEcg(d.hr);

  // Particles react to PM2.5
  targetCount = Math.round(30 + Math.min(d.pm25, 400) * 0.9);
  driftBoost = 1 + Math.min(d.pm25, 400) / 120;
  ensureParticles();

  // Status dots
  setDot("dotWatch", d.hardware.watch ? (d.hr_stale ? "stale" : "live") : (d.hr_source==="sim" ? "live" : "off"));
  setDot("dotSensor", d.hardware.sensor ? "live" : (d.air_source==="sim" ? "live" : "off"));
  setDot("dotApi", d.mode === "auto" && d.air_source === "city_api" ? "live" : "off");
  setText("sourceLine", `source · air:${d.air_source} · hr:${d.hr_source}`);
}
function setDot(id, cls) { const e = $(id); if (e) e.className = `dot ${cls}`; }

/* ── Polling ─────────────────────────────────────────────────────────────── */
async function tick() {
  const q = new URLSearchParams({
    city: state.city, mode: state.mode, age: state.age, condition: state.condition,
    sim: state.sim, sim_hr: state.simHr, sim_pm: state.simPm,
  });
  try {
    // no-store + cache-bust: stop the browser serving a stale cached /api/live,
    // which freezes the numbers even though the server returns fresh data.
    const r = await fetch(`/api/live?${q}&_t=${Date.now()}`, { cache: "no-store" });
    const d = await r.json();
    console.log("[live] hr:", d.hr, "src:", d.hr_source, "phrs:", d.phrs, "aqi:", d.aqi);
    render(d);
  } catch (e) {
    console.error("[render error]", e);  // surface it instead of swallowing
  }
}
setInterval(tick, 1000);

/* ── Controls wiring ────────────────────────────────────────────────────── */
$("modeswitch").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  state.mode = b.dataset.mode;
  [...$("modeswitch").children].forEach(x => x.classList.toggle("on", x === b));
  tick();
});
$("dockToggle").addEventListener("click", () => $("dock").classList.toggle("open"));
$("simChk").addEventListener("change", (e) => {
  state.sim = e.target.checked;
  $("simControls").style.display = state.sim ? "block" : "none";
  $("hwControls").style.display = state.sim ? "none" : "block";
});
$("citySel").addEventListener("change", e => { state.city = e.target.value; tick(); });
$("ageInp").addEventListener("input", e => { state.age = +e.target.value || 30; tick(); });
$("condSel").addEventListener("change", e => { state.condition = e.target.value; tick(); });
$("simHr").addEventListener("input", e => { state.simHr = +e.target.value; $("simHrVal").textContent = e.target.value; });
$("simPm").addEventListener("input", e => { state.simPm = +e.target.value; $("simPmVal").textContent = e.target.value; });
$("scanWatchBtn").addEventListener("click", async () => {
  $("scanWatchBtn").textContent = "…";
  $("connectMsg").textContent = "scanning BLE (6s)…";
  try {
    const { devices } = await (await fetch("/api/watch/scan")).json();
    $("watchSel").innerHTML = `<option value="">— not connected —</option>` +
      devices.map(d => `<option value="${d.address}">${d.name} · ${d.rssi}dBm · ${d.address.slice(0,8)}…</option>`).join("");
    // auto-select the first Galaxy Watch if present
    const gw = devices.find(d => /galaxy watch|watch|band|hr/i.test(d.name));
    if (gw) $("watchSel").value = gw.address;
    $("connectMsg").textContent = `${devices.length} devices found — pick your watch, then Go Live`;
  } catch { $("connectMsg").textContent = "scan failed"; }
  $("scanWatchBtn").textContent = "Scan";
});

$("connectBtn").addEventListener("click", async () => {
  const watchAddr = $("watchSel").value;
  const q = new URLSearchParams({
    watch: !!watchAddr, watch_addr: watchAddr,
    sensor: !!$("portSel").value, sensor_port: $("portSel").value,
  });
  $("connectBtn").textContent = "Connecting…";
  $("connectMsg").textContent = "connecting to watch — keep the broadcaster app awake…";
  try {
    const r = await (await fetch(`/api/hardware/connect?${q}`, { method: "POST" })).json();
    $("connectBtn").textContent = "Live ✓";
    $("connectMsg").textContent = `watch:${r.watch_running} · sensor:${r.sensor_running} — HR appears within a few seconds`;
  } catch { $("connectBtn").textContent = "Failed — retry"; $("connectMsg").textContent = "connect failed"; }
});

/* ── Populate selects ───────────────────────────────────────────────────── */
(async function init() {
  try {
    const { cities } = await (await fetch("/api/cities")).json();
    $("citySel").innerHTML = cities.map(c => `<option ${c===state.city?"selected":""}>${c}</option>`).join("");
  } catch {}
  try {
    const { ports } = await (await fetch("/api/ports")).json();
    $("portSel").innerHTML = `<option value="">—</option>` + ports.map(p => `<option>${p}</option>`).join("");
  } catch {}
  tick();
})();
