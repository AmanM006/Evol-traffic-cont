"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import styles from "./TrafficDashboard.module.css";

// ── Types ────────────────────────────────────────────────────────────────────

interface SimState {
  timestamp: number;
  sim_time_seconds: number;
  sim_hour: number;
  lights: { NS: "GREEN" | "YELLOW" | "RED"; EW: "GREEN" | "YELLOW" | "RED" };
  queues: { N: number; S: number; E: number; W: number };
  emergency_active: boolean;
  emergency_direction: string | null;
  controller_mode: string;
  current_timings: { ns_green: number; ew_green: number };
  stats: {
    total_wait_vs: number;
    total_cleared: number;
    total_arrived: number;
    avg_wait: number;
  };
}

interface MetricsSummary {
  avg_wait_time: number;
  avg_queue_length: number;
  total_throughput: number;
  ticks_recorded: number;
}

interface QueuePoint {
  tick: number;
  N: number;
  S: number;
  E: number;
  W: number;
  total: number;
}

interface HistoryPoint {
  generation: number;
  best_fitness: number;
  best_ns_green: number;
  best_ew_green: number;
}

// ── Mock Data (dev without backend) ──────────────────────────────────────────

const MOCK_GA: SimState = {
  timestamp: 100,
  sim_time_seconds: 100.0,
  sim_hour: 8,
  lights: { NS: "GREEN", EW: "RED" },
  queues: { N: 8, S: 5, E: 12, W: 3 },
  emergency_active: false,
  emergency_direction: null,
  controller_mode: "ga",
  current_timings: { ns_green: 38.5, ew_green: 21.5 },
  stats: { total_wait_vs: 5000, total_cleared: 450, total_arrived: 500, avg_wait: 11.1 },
};

const MOCK_FIXED: SimState = {
  ...MOCK_GA,
  controller_mode: "fixed",
  current_timings: { ns_green: 30.0, ew_green: 30.0 },
  queues: { N: 14, S: 9, E: 18, W: 7 },
  stats: { total_wait_vs: 9200, total_cleared: 380, total_arrived: 500, avg_wait: 24.2 },
};

// ── Constants ─────────────────────────────────────────────────────────────────

const GA_PORT = 5000;
const FIXED_PORT = 5001;
const USE_MOCK = false; // flip to true to dev offline

const LIGHT_COLOR: Record<string, string> = {
  GREEN: "#00ff88",
  YELLOW: "#ffd600",
  RED: "#ff3b3b",
};

const QUEUE_COLORS = { N: "#00c8ff", S: "#00ffb3", E: "#ff8c00", W: "#ff4466" };

function fmtTime(sec: number) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function fmtHour(h: number) {
  const ampm = h >= 12 ? "PM" : "AM";
  const hr = h % 12 || 12;
  return `${hr}:00 ${ampm}`;
}
function num(n: number, d = 1) {
  return n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}

// ── Mini sparkline canvas ─────────────────────────────────────────────────────

function Sparkline({
  data,
  color,
  height = 40,
}: {
  data: number[];
  color: string;
  height?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || data.length < 2) return;
    const ctx = c.getContext("2d")!;
    const w = c.width,
      h = c.height;
    ctx.clearRect(0, 0, w, h);
    const max = Math.max(...data, 1);
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - (v / max) * h * 0.85 - 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // fill under
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = color + "22";
    ctx.fill();
  }, [data, color]);
  return (
    <canvas
      ref={ref}
      width={200}
      height={height}
      style={{ width: "100%", height }}
    />
  );
}

// ── Intersection Canvas ───────────────────────────────────────────────────────

function IntersectionView({
  state,
  accentColor,
}: {
  state: SimState;
  accentColor: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width,
      H = canvas.height;
    const cx = W / 2,
      cy = H / 2;
    const roadW = 56;
    const boxHalf = roadW * 1.1;

    ctx.clearRect(0, 0, W, H);

    // ── Background ──
    ctx.fillStyle = "#0a0d12";
    ctx.fillRect(0, 0, W, H);

    // Grid dots
    ctx.fillStyle = "#ffffff08";
    for (let x = 0; x < W; x += 20)
      for (let y = 0; y < H; y += 20)
        ctx.fillRect(x, y, 1, 1);

    // ── Roads ──
    const drawRoad = (x: number, y: number, w: number, h: number) => {
      ctx.fillStyle = "#1a1f2a";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = "#ffffff10";
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x, y, w, h);
    };
    drawRoad(cx - roadW / 2, 0, roadW, H); // vertical road
    drawRoad(0, cy - roadW / 2, W, roadW); // horizontal road

    // Dashed center lines
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = "#ffffff18";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, cy - boxHalf);
    ctx.moveTo(cx, cy + boxHalf);
    ctx.lineTo(cx, H);
    ctx.moveTo(0, cy);
    ctx.lineTo(cx - boxHalf, cy);
    ctx.moveTo(cx + boxHalf, cy);
    ctx.lineTo(W, cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Intersection box ──
    ctx.fillStyle = "#1a1f2a";
    ctx.fillRect(cx - boxHalf, cy - boxHalf, boxHalf * 2, boxHalf * 2);

    // ── Traffic lights ──
    const drawLight = (
      lx: number,
      ly: number,
      color: "GREEN" | "YELLOW" | "RED"
    ) => {
      // housing
      ctx.fillStyle = "#111418";
      ctx.beginPath();
      ctx.roundRect(lx - 8, ly - 22, 16, 44, 3);
      ctx.fill();

      // Lights: R, Y, G
      const positions = [ly - 13, ly, ly + 13];
      const colors = ["RED", "YELLOW", "GREEN"];
      positions.forEach((py, i) => {
        const isOn = colors[i] === color;
        ctx.beginPath();
        ctx.arc(lx, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = isOn ? LIGHT_COLOR[colors[i]] : colors[i] === "RED" ? "#3d1010" : colors[i] === "YELLOW" ? "#3d3000" : "#0d3020";
        ctx.fill();
        if (isOn) {
          ctx.shadowColor = LIGHT_COLOR[colors[i]];
          ctx.shadowBlur = 12;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });
    };

    // N light (south-facing, sits above intersection)
    drawLight(cx + boxHalf + 12, cy - boxHalf / 2, state.lights.NS);
    // S light
    drawLight(cx - boxHalf - 12, cy + boxHalf / 2, state.lights.NS);
    // E light
    drawLight(cx + boxHalf / 2, cy + boxHalf + 12, state.lights.EW);
    // W light
    drawLight(cx - boxHalf / 2, cy - boxHalf - 12, state.lights.EW);

    // ── Vehicle queues ──
    const carW = 10,
      carH = 6,
      carGap = 3;

    const drawQueue = (
      dir: "N" | "S" | "E" | "W",
      count: number,
      isGreen: boolean
    ) => {
      const color = QUEUE_COLORS[dir];
      const maxShow = Math.min(count, 10);
      const isEmergency =
        state.emergency_active && state.emergency_direction === dir;

      for (let i = 0; i < maxShow; i++) {
        let x = 0, y = 0;
        const offset = boxHalf + 8 + i * (carH + carGap);
        if (dir === "N") { x = cx - carW / 2; y = cy - offset - carH; }
        if (dir === "S") { x = cx - carW / 2; y = cy + offset; }
        if (dir === "E") { x = cx + offset; y = cy - carW / 2; }
        if (dir === "W") { x = cx - offset - carH; y = cy - carW / 2; }

        const isHoriz = dir === "E" || dir === "W";
        const cw = isHoriz ? carH : carW;
        const ch = isHoriz ? carW : carH;

        ctx.fillStyle = isEmergency ? "#ff4466" : isGreen && i === 0 ? color : color + "88";
        ctx.shadowColor = isGreen && i === 0 ? color : "transparent";
        ctx.shadowBlur = isGreen && i === 0 ? 6 : 0;
        ctx.beginPath();
        ctx.roundRect(x, y, cw, ch, 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // overflow badge — placed with clear gap after last car
      if (count > 10) {
        const extra = count - 10;
        const badgeOffset = boxHalf + 8 + 10 * (carH + carGap) + 16;
        let tx = cx, ty = cy;
        if (dir === "N") { tx = cx; ty = cy - badgeOffset; }
        if (dir === "S") { tx = cx; ty = cy + badgeOffset; }
        if (dir === "E") { tx = cx + badgeOffset; ty = cy + 1; }
        if (dir === "W") { tx = cx - badgeOffset; ty = cy + 1; }

        // Badge background
        ctx.fillStyle = color + "33";
        const bw = 38, bh = 16;
        ctx.beginPath();
        ctx.roundRect(tx - bw / 2, ty - bh / 2, bw, bh, 4);
        ctx.fill();
        ctx.strokeStyle = color + "66";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.font = "700 9px 'JetBrains Mono', monospace";
        ctx.fillStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`+${extra}`, tx, ty);
        ctx.textBaseline = "alphabetic";
      }
    };

    const nsGreen = state.lights.NS === "GREEN";
    const ewGreen = state.lights.EW === "GREEN";
    drawQueue("N", state.queues.N, nsGreen);
    drawQueue("S", state.queues.S, nsGreen);
    drawQueue("E", state.queues.E, ewGreen);
    drawQueue("W", state.queues.W, ewGreen);

    // ── Direction labels — fixed at canvas edges, never overlap queues ──
    ctx.font = "700 11px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const edgePad = 14; // px from edge
    const labelDefs = [
      { d: "N", x: cx,        y: edgePad,      color: QUEUE_COLORS.N },
      { d: "S", x: cx,        y: H - edgePad,  color: QUEUE_COLORS.S },
      { d: "E", x: W - edgePad, y: cy,          color: QUEUE_COLORS.E },
      { d: "W", x: edgePad,   y: cy,            color: QUEUE_COLORS.W },
    ];
    labelDefs.forEach(({ d, x, y, color: lc }) => {
      // Pill background
      ctx.fillStyle = lc + "22";
      ctx.beginPath();
      ctx.roundRect(x - 11, y - 9, 22, 18, 5);
      ctx.fill();
      ctx.strokeStyle = lc + "55";
      ctx.lineWidth = 1;
      ctx.stroke();
      // Letter
      ctx.fillStyle = lc;
      ctx.fillText(d, x, y);
    });
    ctx.textBaseline = "alphabetic";

    // ── Emergency overlay ──
    if (state.emergency_active) {
      const pulse = (Math.sin(Date.now() / 150) + 1) / 2;
      ctx.fillStyle = `rgba(255, 50, 80, ${0.08 * pulse})`;
      ctx.fillRect(0, 0, W, H);

      ctx.font = "bold 20px sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = `rgba(255, 50, 80, ${0.7 + 0.3 * pulse})`;
      ctx.fillText("🚑", cx, cy + 6);
    }

    // ── Center intersection glow ──
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, boxHalf);
    grad.addColorStop(0, accentColor + "15");
    grad.addColorStop(1, "transparent");
    ctx.fillStyle = grad;
    ctx.fillRect(cx - boxHalf, cy - boxHalf, boxHalf * 2, boxHalf * 2);
  });

  return (
    <canvas
      ref={ref}
      width={320}
      height={320}
      style={{ width: "100%", maxWidth: 320, display: "block", margin: "0 auto" }}
    />
  );
}

// ── Mini Chart (queue over time) ──────────────────────────────────────────────

function QueueChart({
  data,
  accentColor,
}: {
  data: QueuePoint[];
  accentColor: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || data.length < 2) return;
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width,
      H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const maxVal = Math.max(...data.map((d) => d.total), 1);
    const pad = { t: 8, r: 8, b: 20, l: 28 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    // Grid
    ctx.strokeStyle = "#ffffff12";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75, 1].forEach((f) => {
      const y = pad.t + plotH * (1 - f);
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
      ctx.fillStyle = "#ffffffaa";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(String(Math.round(maxVal * f)), pad.l - 4, y + 3);
    });

    // Lines per direction
    const dirs: Array<keyof QueuePoint> = ["N", "S", "E", "W"];
    const colors = [QUEUE_COLORS.N, QUEUE_COLORS.S, QUEUE_COLORS.E, QUEUE_COLORS.W];

    dirs.forEach((d, i) => {
      ctx.beginPath();
      data.forEach((pt, idx) => {
        const x = pad.l + (idx / (data.length - 1)) * plotW;
        const y = pad.t + plotH * (1 - (pt[d] as number) / maxVal);
        idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = colors[i] + "cc";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // X axis label
    ctx.fillStyle = "#ffffff88";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("← TICKS", pad.l, H - 4);
  }, [data, accentColor]);

  return (
    <canvas
      ref={ref}
      width={400}
      height={120}
      style={{ width: "100%", height: 120 }}
    />
  );
}

// ── Pod (one controller side) ─────────────────────────────────────────────────

function ControllerPod({
  label,
  badge,
  accentColor,
  dimColor,
  port,
  mockState,
}: {
  label: string;
  badge: string;
  accentColor: string;
  dimColor: string;
  port: number;
  mockState: SimState;
}) {
  const [state, setState] = useState<SimState | null>(null);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [queueHistory, setQueueHistory] = useState<QueuePoint[]>([]);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [online, setOnline] = useState(false);
  const [waitHistory, setWaitHistory] = useState<number[]>([]);
  const [throughputHistory, setThroughputHistory] = useState<number[]>([]);

  const BASE = `http://127.0.0.1:${port}`;

  // State polling — 10 fps
  useEffect(() => {
    if (USE_MOCK) {
      setState(mockState);
      setOnline(true);
      return;
    }
    const id = setInterval(async () => {
      try {
        const res = await fetch(`${BASE}/state`);
        const data: SimState = await res.json();
        setState(data);
        setOnline(true);
        setWaitHistory((prev) => [...prev.slice(-120), data.stats.avg_wait]);
        setThroughputHistory((prev) => [...prev.slice(-120), data.stats.total_cleared]);
      } catch {
        setOnline(false);
      }
    }, 100);
    return () => clearInterval(id);
  }, [BASE, mockState]);

  // Metrics — every 2s
  useEffect(() => {
    if (USE_MOCK) return;
    const id = setInterval(async () => {
      try {
        const res = await fetch(`${BASE}/metrics`);
        const data = await res.json();
        setMetrics(data);
      } catch {}
    }, 2000);
    return () => clearInterval(id);
  }, [BASE]);

  // Queue history — every 3s
  useEffect(() => {
    if (USE_MOCK) return;
    const id = setInterval(async () => {
      try {
        const res = await fetch(`${BASE}/metrics/queues`);
        const data = await res.json();
        setQueueHistory(data);
      } catch {}
    }, 3000);
    return () => clearInterval(id);
  }, [BASE]);

  // GA history — every 5s
  useEffect(() => {
    if (USE_MOCK) return;
    const id = setInterval(async () => {
      try {
        const res = await fetch(`${BASE}/history`);
        const data = await res.json();
        setHistory(data);
      } catch {}
    }, 5000);
    return () => clearInterval(id);
  }, [BASE]);

  const s = state;

  return (
    <div className={styles.pod} style={{ "--accent": accentColor, "--dim": dimColor } as React.CSSProperties}>
      {/* Pod header */}
      <div className={styles.podHeader}>
        <div className={styles.podTitle}>
          <span className={styles.podBadge} style={{ background: accentColor + "22", border: `1px solid ${accentColor}44`, color: accentColor }}>
            {badge}
          </span>
          <span className={styles.podLabel}>{label}</span>
        </div>
        <div className={styles.podStatus}>
          <div className={styles.statusDot} style={{ background: online ? accentColor : "#444", boxShadow: online ? `0 0 6px ${accentColor}` : "none" }} />
          <span style={{ color: online ? accentColor : "#555" }}>{online ? "LIVE" : "OFFLINE"}</span>
        </div>
      </div>

      {!online && !USE_MOCK && (
        <div className={styles.offlineMsg}>
          <div className={styles.offlineIcon}>⬡</div>
          <div>Backend offline</div>
          <div className={styles.offlineSub}>Run: <code>python main.py --mode {label.toLowerCase().includes("genetic") ? "ga" : "fixed"} --port {port}</code></div>
        </div>
      )}

      {(s || USE_MOCK) && (() => {
        const d = s || mockState;
        return (
          <>
            {/* Intersection visual */}
            <div className={styles.intersectionWrap}>
              {d.emergency_active && (
                <div className={styles.emergencyBanner}>
                  🚑 EMERGENCY — {d.emergency_direction} APPROACH
                </div>
              )}
              <IntersectionView state={d} accentColor={accentColor} />
            </div>

            {/* Phase indicator */}
            <div className={styles.phaseRow}>
              <PhaseLight direction="NS" color={d.lights.NS} />
              <div className={styles.phaseDivider} />
              <PhaseLight direction="EW" color={d.lights.EW} />
            </div>

            {/* Timings */}
            <div className={styles.timingsRow}>
              <TimingBar label="↕ NS" value={d.current_timings.ns_green} max={60} color={accentColor} />
              <TimingBar label="↔ EW" value={d.current_timings.ew_green} max={60} color={dimColor} />
            </div>

            {/* Stats grid */}
            <div className={styles.statsGrid}>
              <StatCard label="AVG WAIT" value={`${num(d.stats.avg_wait)}s`} accent={accentColor} big />
              <StatCard label="CLEARED" value={d.stats.total_cleared.toLocaleString()} accent={accentColor} />
              <StatCard label="ARRIVED" value={d.stats.total_arrived.toLocaleString()} accent={accentColor} />
              <StatCard label="SIM TIME" value={fmtTime(d.sim_time_seconds)} accent={accentColor} mono />
            </div>

            {/* Queue counts */}
            <div className={styles.queueRow}>
              {(["N", "S", "E", "W"] as const).map((dir) => (
                <div key={dir} className={styles.queueCell}>
                  <div className={styles.queueDir} style={{ color: QUEUE_COLORS[dir] }}>{dir}</div>
                  <div className={styles.queueCount}>{d.queues[dir]}</div>
                </div>
              ))}
            </div>

            {/* Wait time sparkline */}
            {waitHistory.length > 4 && (
              <div className={styles.chartSection}>
                <div className={styles.chartLabel}>AVG WAIT TREND (s)</div>
                <Sparkline data={waitHistory} color={accentColor} height={48} />
              </div>
            )}

            {/* Throughput chart */}
            {throughputHistory.length > 4 && (
              <div className={styles.chartSection}>
                <div className={styles.chartLabel}>CUMULATIVE THROUGHPUT</div>
                <ThroughputChart data={throughputHistory} color={accentColor} />
              </div>
            )}

            {/* Queue chart */}
            {queueHistory.length > 4 && (
              <div className={styles.chartSection}>
                <div className={styles.chartLabel}>QUEUE HISTORY — PER DIRECTION</div>
                <QueueChart data={queueHistory} accentColor={accentColor} />
                <div className={styles.chartLegend}>
                  {(["N", "S", "E", "W"] as const).map((dir) => (
                    <span key={dir} style={{ color: QUEUE_COLORS[dir] }}>● {dir}</span>
                  ))}
                </div>
              </div>
            )}

            {/* GA Evolution */}
            {history.length > 1 && (
              <div className={styles.chartSection}>
                <div className={styles.chartLabel}>GA EVOLUTION — {history.length} RUNS</div>
                <FitnessChart history={history} accentColor={accentColor} />
                <div className={styles.evolutionList} style={{ marginTop: 10 }}>
                  {history.slice(-4).map((h) => (
                    <div key={h.generation} className={styles.evolutionRow}>
                      <span className={styles.evGen}>GEN {h.generation}</span>
                      <span className={styles.evBar}>
                        <span style={{ width: `${(h.best_ns_green / 60) * 100}%`, background: accentColor }} />
                      </span>
                      <span className={styles.evVal} style={{ color: accentColor }}>NS {h.best_ns_green}s</span>
                      <span className={styles.evBar}>
                        <span style={{ width: `${(h.best_ew_green / 60) * 100}%`, background: dimColor }} />
                      </span>
                      <span className={styles.evVal} style={{ color: dimColor }}>EW {h.best_ew_green}s</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        );
      })()}
    </div>
  );
}

// ── Small components ──────────────────────────────────────────────────────────

function PhaseLight({ direction, color }: { direction: string; color: "GREEN" | "YELLOW" | "RED" }) {
  // Use the ACTUAL color for the dot — never use accent color here
  const dotColor = color === "GREEN" ? "#00ff88" : color === "YELLOW" ? "#ffd600" : "#ff3b3b";
  return (
    <div className={styles.phaseLight}>
      <div
        className={styles.phaseDot}
        style={{
          background: dotColor,
          boxShadow: `0 0 8px ${dotColor}, 0 0 16px ${dotColor}55`,
        }}
      />
      <span className={styles.phaseDir}>{direction}</span>
      <span className={styles.phaseColorLabel} style={{ color: dotColor }}>{color}</span>
    </div>
  );
}

function TimingBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = (value / max) * 100;
  return (
    <div className={styles.timingBar}>
      <div className={styles.timingLabel}>{label}</div>
      <div className={styles.timingTrack}>
        <div className={styles.timingFill} style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}66` }} />
      </div>
      <div className={styles.timingVal} style={{ color }}>{value}s</div>
    </div>
  );
}

function StatCard({ label, value, accent, big, mono }: { label: string; value: string; accent: string; big?: boolean; mono?: boolean }) {
  return (
    <div className={styles.statCard}>
      <div className={styles.statLabel}>{label}</div>
      <div className={styles.statValue} style={{
        color: accent,
        fontSize: big ? "26px" : "18px",
        fontFamily: mono ? "'JetBrains Mono', monospace" : "'JetBrains Mono', monospace"
      }}>
        {value}
      </div>
    </div>
  );
}

// ── Throughput over time chart ────────────────────────────────────────────────

function ThroughputChart({ data, color }: { data: number[]; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || data.length < 2) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const pad = { t: 6, r: 8, b: 20, l: 36 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    const maxVal = Math.max(...data, 1);

    // Grid lines
    ctx.strokeStyle = "#ffffff12";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75, 1].forEach((f) => {
      const y = pad.t + plotH * (1 - f);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
      ctx.fillStyle = "#ffffffaa";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(Math.round(maxVal * f).toString(), pad.l - 4, y + 3);
    });

    // Area fill
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.l + (i / (data.length - 1)) * plotW;
      const y = pad.t + plotH * (1 - v / maxVal);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    const lastX = pad.l + plotW;
    const lastY = pad.t + plotH * (1 - data[data.length - 1] / maxVal);
    ctx.lineTo(lastX, pad.t + plotH);
    ctx.lineTo(pad.l, pad.t + plotH);
    ctx.closePath();
    ctx.fillStyle = color + "22";
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.l + (i / (data.length - 1)) * plotW;
      const y = pad.t + plotH * (1 - v / maxVal);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#ffffff99";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("← TICKS", pad.l, H - 4);
  }, [data, color]);
  return <canvas ref={ref} width={400} height={100} style={{ width: "100%", height: 100 }} />;
}

// ── GA Fitness chart ──────────────────────────────────────────────────────────

function FitnessChart({ history, accentColor }: { history: HistoryPoint[]; accentColor: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || history.length < 2) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const pad = { t: 6, r: 8, b: 20, l: 48 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    const fitVals = history.map(h => h.best_fitness);
    const nsVals = history.map(h => h.best_ns_green);
    const ewVals = history.map(h => h.best_ew_green);
    const maxFit = Math.max(...fitVals, 0.01);
    const maxTiming = 60;

    // Grid
    ctx.strokeStyle = "#ffffff12";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75, 1].forEach((f) => {
      const y = pad.t + plotH * (1 - f);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
      ctx.fillStyle = "#ffffffaa";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText((maxFit * f).toFixed(3), pad.l - 4, y + 3);
    });

    const drawLine = (vals: number[], maxV: number, color: string, dash: number[] = []) => {
      ctx.setLineDash(dash);
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = pad.l + (i / (vals.length - 1)) * plotW;
        const y = pad.t + plotH * (1 - v / maxV);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    };

    drawLine(fitVals, maxFit, accentColor);
    drawLine(nsVals, maxTiming, "#00c8ff", [4, 3]);
    drawLine(ewVals, maxTiming, "#ff8c0099", [4, 3]);

    ctx.fillStyle = "#ffffff99";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("← EVOLUTIONS", pad.l, H - 4);
  }, [history, accentColor]);

  return (
    <div>
      <canvas ref={ref} width={400} height={110} style={{ width: "100%", height: 110 }} />
      <div className={styles.chartLegend}>
        <span style={{ color: accentColor }}>● FITNESS</span>
        <span style={{ color: "#00c8ff" }}>- NS GREEN</span>
        <span style={{ color: "#ff8c00" }}>- EW GREEN</span>
      </div>
    </div>
  );
}

// ── Live wait comparison chart (center column) ────────────────────────────────

function WaitComparisonChart({
  gaHistory,
  fixedHistory,
}: {
  gaHistory: number[];
  fixedHistory: number[];
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const pad = { t: 8, r: 10, b: 22, l: 42 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    const len = Math.max(gaHistory.length, fixedHistory.length, 2);
    const allVals = [...gaHistory, ...fixedHistory];
    const maxVal = Math.max(...allVals, 1);

    // Grid
    ctx.strokeStyle = "#ffffff12";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75, 1].forEach((f) => {
      const y = pad.t + plotH * (1 - f);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
      ctx.fillStyle = "#ffffffaa";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(Math.round(maxVal * f) + "s", pad.l - 4, y + 3);
    });

    const drawLine = (vals: number[], color: string) => {
      if (vals.length < 2) return;
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = pad.l + (i / (len - 1)) * plotW;
        const y = pad.t + plotH * (1 - v / maxVal);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      // Last value dot
      const lastX = pad.l + ((vals.length - 1) / (len - 1)) * plotW;
      const lastY = pad.t + plotH * (1 - vals[vals.length - 1] / maxVal);
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    };

    drawLine(fixedHistory, "#ff8c00");
    drawLine(gaHistory, "#00ff88");

    ctx.fillStyle = "#ffffff99";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("← TIME", pad.l, H - 5);
  }, [gaHistory, fixedHistory]);

  return (
    <div>
      <canvas ref={ref} width={190} height={120} style={{ width: "100%", height: 120 }} />
      <div className={styles.chartLegend} style={{ justifyContent: "center" }}>
        <span style={{ color: "#00ff88" }}>● GA</span>
        <span style={{ color: "#ff8c00" }}>● FIXED</span>
      </div>
    </div>
  );
}

// ── Comparison bar ────────────────────────────────────────────────────────────

function ComparisonStrip({
  ga,
  fixed,
}: {
  ga: SimState | null;
  fixed: SimState | null;
}) {
  if (!ga || !fixed) return null;
  const gaWait = ga.stats.avg_wait;
  const fixedWait = fixed.stats.avg_wait;
  const improvement = fixedWait > 0 ? ((fixedWait - gaWait) / fixedWait) * 100 : 0;
  const gaAhead = gaWait < fixedWait;
  const gaTotal = Object.values(ga.queues).reduce((a, b) => a + b, 0);
  const fixedTotal = Object.values(fixed.queues).reduce((a, b) => a + b, 0);

  return (
    <div className={styles.compStrip}>
      <div className={styles.compTitle}>LIVE COMPARISON</div>
      <div className={styles.compGrid}>
        <CompRow
          label="AVG WAIT"
          gaVal={`${num(gaWait)}s`}
          fixVal={`${num(fixedWait)}s`}
          gaWins={gaAhead}
        />
        <CompRow
          label="TOTAL QUEUED"
          gaVal={`${gaTotal}`}
          fixVal={`${fixedTotal}`}
          gaWins={gaTotal <= fixedTotal}
        />
        <CompRow
          label="CLEARED"
          gaVal={ga.stats.total_cleared.toLocaleString()}
          fixVal={fixed.stats.total_cleared.toLocaleString()}
          gaWins={ga.stats.total_cleared >= fixed.stats.total_cleared}
        />
      </div>
      {Math.abs(improvement) > 1 && (
        <div className={styles.compResult} style={{ color: gaAhead ? "#00ff88" : "#ff3b3b" }}>
          {gaAhead ? `▲ GA is ${num(improvement, 0)}% better on wait time` : `▼ Fixed-time ahead by ${num(-improvement, 0)}%`}
        </div>
      )}
    </div>
  );
}

function CompRow({ label, gaVal, fixVal, gaWins }: { label: string; gaVal: string; fixVal: string; gaWins: boolean }) {
  return (
    <div className={styles.compRow}>
      <span className={styles.compLabel}>{label}</span>
      <span className={styles.compGA} style={{ color: gaWins ? "#00ff88" : "#aaa" }}>{gaVal}</span>
      <span className={styles.compVs}>vs</span>
      <span className={styles.compFixed} style={{ color: !gaWins ? "#ff8c00" : "#aaa" }}>{fixVal}</span>
    </div>
  );
}

// ── Main dashboard ────────────────────────────────────────────────────────────

export default function TrafficDashboard() {
  const [gaState, setGaState] = useState<SimState | null>(null);
  const [fixedState, setFixedState] = useState<SimState | null>(null);
  const [gaWaitHistory, setGaWaitHistory] = useState<number[]>([]);
  const [fixedWaitHistory, setFixedWaitHistory] = useState<number[]>([]);

  // Poll both for the comparison chart
  useEffect(() => {
    if (USE_MOCK) {
      setGaState(MOCK_GA);
      setFixedState(MOCK_FIXED);
      return;
    }
    const idGA = setInterval(async () => {
      try {
        const r = await fetch(`http://127.0.0.1:${GA_PORT}/state`);
        const d: SimState = await r.json();
        setGaState(d);
        setGaWaitHistory((p) => [...p.slice(-80), d.stats.avg_wait]);
      } catch {}
    }, 500);
    const idFixed = setInterval(async () => {
      try {
        const r = await fetch(`http://127.0.0.1:${FIXED_PORT}/state`);
        const d: SimState = await r.json();
        setFixedState(d);
        setFixedWaitHistory((p) => [...p.slice(-80), d.stats.avg_wait]);
      } catch {}
    }, 500);
    return () => { clearInterval(idGA); clearInterval(idFixed); };
  }, []);

  return (
    <div className={styles.root}>
      {/* Top bar */}
      <header className={styles.topBar}>
        <div className={styles.topLeft}>
          <div className={styles.logo}>◈ ATSC</div>
          <div className={styles.logoSub}>Evolutionary Traffic Signal Control</div>
        </div>
        <div className={styles.topCenter}>
          <span className={styles.topTag}>Genetic Algorithm</span>
          <span className={styles.topVs}>vs</span>
          <span className={styles.topTag}>Fixed-Time</span>
        </div>
        <div className={styles.topRight}>
          <div className={styles.topDate}>{new Date().toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" })}</div>
        </div>
      </header>

      <div className={styles.layout}>
        {/* GA Pod */}
        <ControllerPod
          label="Genetic Algorithm"
          badge="GA"
          accentColor="#00ff88"
          dimColor="#00c8ff"
          port={GA_PORT}
          mockState={MOCK_GA}
        />

        {/* Center comparison strip */}
        <div className={styles.centerCol}>
          <ComparisonStrip ga={gaState} fixed={fixedState} />
          {(gaWaitHistory.length > 3 || fixedWaitHistory.length > 3) && (
            <div className={styles.waitCompChart}>
              <div className={styles.waitCompTitle}>WAIT TIME — LIVE</div>
              <WaitComparisonChart gaHistory={gaWaitHistory} fixedHistory={fixedWaitHistory} />
            </div>
          )}
          <div className={styles.centerDecor}>
            <div className={styles.vsCircle}>VS</div>
          </div>
        </div>

        {/* Fixed-Time Pod */}
        <ControllerPod
          label="Fixed-Time"
          badge="FIXED"
          accentColor="#ff8c00"
          dimColor="#ff4466"
          port={FIXED_PORT}
          mockState={MOCK_FIXED}
        />
      </div>

      {/* Footer */}
      <footer className={styles.footer}>
        <span>PORT {GA_PORT} — GA</span>
        <span className={styles.footerDot}>◆</span>
        <span>PORT {FIXED_PORT} — FIXED</span>
        <span className={styles.footerDot}>◆</span>
        <span>Poll interval 100ms</span>
      </footer>
    </div>
  );
}