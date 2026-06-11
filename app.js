/* Embedding Playground — vanilla JS, no build step, works from file://.
   Data arrives as window.EMBEDDINGS (see embeddings.js). */
(function () {
"use strict";

/* ===================================================== data + math core */
const DATA = window.EMBEDDINGS;
if (!DATA) { boot("Couldn't load embeddings.js.", true); return; }

const DIM = DATA.meta.dim;
const WORDS = DATA.words.map(w => w.w);
const CLUSTER = Object.fromEntries(DATA.words.map(w => [w.w, w.cluster]));
const INDEX = Object.fromEntries(WORDS.map((w, i) => [w, i]));
const TRUE_NB = DATA.true_neighbors;

// decode little-endian Float32 blob -> unit-normalized vectors (raw GloVe rows)
const VEC = (() => {
  const bin = atob(DATA.vectors_b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const f32 = new Float32Array(bytes.buffer);
  const out = new Map();
  for (let r = 0; r < WORDS.length; r++) {
    const v = new Float32Array(DIM);
    let n = 0;
    for (let d = 0; d < DIM; d++) { const x = f32[r * DIM + d]; v[d] = x; n += x * x; }
    n = Math.sqrt(n) || 1;
    for (let d = 0; d < DIM; d++) v[d] /= n;
    out.set(WORDS[r], v);
  }
  return out;
})();

const vec = w => VEC.get(w);
function dot(a, b) { let s = 0; for (let i = 0; i < DIM; i++) s += a[i] * b[i]; return s; }
const cosine = (w1, w2) => dot(vec(w1), vec(w2));

function nearestToVector(q, k, exclude) {
  const ex = new Set(exclude || []);
  // q assumed unit-normalized
  const res = [];
  for (let i = 0; i < WORDS.length; i++) {
    const w = WORDS[i];
    if (ex.has(w)) continue;
    res.push([w, dot(q, VEC.get(w))]);
  }
  res.sort((a, b) => b[1] - a[1]);
  return res.slice(0, k);
}
function nearest(word, k) { return nearestToVector(vec(word), k, [word]); }

function analogyVector(a, b, c) {
  const va = vec(a), vb = vec(b), vc = vec(c), d = new Float32Array(DIM);
  let n = 0;
  for (let i = 0; i < DIM; i++) { d[i] = va[i] - vb[i] + vc[i]; n += d[i] * d[i]; }
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < DIM; i++) d[i] /= n;
  return d;
}
function analogy(a, b, c, k) { return nearestToVector(analogyVector(a, b, c), k, [a, b, c]); }

/* PCA helpers (power iteration, deterministic seed) */
function normV(v) { let s = 0; for (let i = 0; i < DIM; i++) s += v[i] * v[i]; s = Math.sqrt(s) || 1; for (let i = 0; i < DIM; i++) v[i] /= s; }
function powerPC(rows) {
  let w = new Float64Array(DIM);
  for (let i = 0; i < DIM; i++) w[i] = Math.sin(i * 1.7 + 0.5); // deterministic seed
  normV(w);
  for (let it = 0; it < 40; it++) {
    const acc = new Float64Array(DIM);
    for (const r of rows) { let pr = 0; for (let i = 0; i < DIM; i++) pr += r[i] * w[i]; for (let i = 0; i < DIM; i++) acc[i] += pr * r[i]; }
    normV(acc); w = acc;
  }
  return w;
}
function centered(points, mean) {
  return points.map(p => { const r = new Float64Array(DIM); for (let i = 0; i < DIM; i++) r[i] = p[i] - mean[i]; return r; });
}
function meanOf(points) {
  const mean = new Float64Array(DIM);
  for (const p of points) for (let i = 0; i < DIM; i++) mean[i] += p[i];
  for (let i = 0; i < DIM; i++) mean[i] /= points.length;
  return mean;
}
function removeComponent(rows, w) {
  return rows.map(r => { let pr = 0; for (let i = 0; i < DIM; i++) pr += r[i] * w[i]; const o = new Float64Array(DIM); for (let i = 0; i < DIM; i++) o[i] = r[i] - pr * w[i]; return o; });
}
function makeProjector(mean, w1, w2) {
  return {
    project(x) {
      let u = 0, v = 0;
      for (let i = 0; i < DIM; i++) { const c = x[i] - mean[i]; u += c * w1[i]; v += c * w2[i]; }
      return [u, v];
    }
  };
}

/* PCA -> 2D. points: array of Float32Array(DIM).
   Returns {project(x)->[u,v]} fitted (mean + top-2 components). */
function fitPCA(points) {
  const mean = meanOf(points);
  const C = centered(points, mean);
  const w1 = powerPC(C);
  const w2 = powerPC(removeComponent(C, w1));
  return makeProjector(mean, w1, w2);
}

/* 2D basis for the analogy picture. The x-axis IS the (a−b) difference
   direction, so the two "same vector" arrows render exactly parallel and
   equal-length — the parallelogram is real geometry, not styling. The y-axis
   is the top principal component of the words once that direction is removed.
   Orthonormal + linear, so in-plane lengths/angles are the actual geometry. */
function fitAnalogyPlane(points, diffDir) {
  const e1 = new Float64Array(DIM);
  for (let i = 0; i < DIM; i++) e1[i] = diffDir[i];
  normV(e1);
  const mean = meanOf(points);
  const e2 = powerPC(removeComponent(centered(points, mean), e1));
  return makeProjector(mean, e1, e2);
}

/* label declutter: estimate label boxes, push overlapping pairs apart.
   items: {x, y, w, h, weight} — weight 0 pins an item (math anchors, captions);
   higher weight moves more. Clamps movable items into bounds each pass. */
const _mctx = document.createElement("canvas").getContext("2d");
function textWidth(t, px) { _mctx.font = px + "px Georgia, serif"; return _mctx.measureText(t).width; }
function declutter(items, bounds, passes) {
  const GX = 7, GY = 4; // minimum gaps between label boxes
  for (let p = 0; p < (passes || 64); p++) {
    let moved = false;
    for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
      const a = items[i], b = items[j], tot = a.weight + b.weight;
      if (!tot) continue;
      const dx = a.x - b.x, px = (a.w + b.w) / 2 + GX - Math.abs(dx);
      if (px <= 0) continue;
      const dy = a.y - b.y, py = (a.h + b.h) / 2 + GY - Math.abs(dy);
      if (py <= 0) continue;
      moved = true;
      if (py <= px) { const s = py * (dy >= 0 ? 1 : -1); a.y += s * a.weight / tot; b.y -= s * b.weight / tot; }
      else { const s = px * (dx >= 0 ? 1 : -1); a.x += s * a.weight / tot; b.x -= s * b.weight / tot; }
    }
    for (const it of items) {
      if (!it.weight) continue;
      it.x = Math.max(bounds.x0 + it.w / 2, Math.min(bounds.x1 - it.w / 2, it.x));
      it.y = Math.max(bounds.y0 + it.h / 2, Math.min(bounds.y1 - it.h / 2, it.y));
    }
    if (!moved) break;
  }
}

/* ===================================================== cluster meta + color */
const CLUSTER_META = {
  c1_universal: { n: "Universal", v: 1 },
  c2_kinship:   { n: "Kinship & people", v: 2 },
  c3_geography: { n: "Geography", v: 3 },
  c4_pm_dol:    { n: "PM & DOL", v: 4 },
  c5_bias:      { n: "Bias set", v: 5 },
  c6_polysemy:  { n: "Polysemy", v: 6 },
};
const clusterVar = (w, step) => {
  const i = CLUSTER_META[CLUSTER[w]].v;
  return `var(--c${i}${step === "l" ? "-l" : step === "d" ? "-d" : ""})`;
};

// cosine heat ramp: cream -> blue -> deep blue-black
const RAMP = [
  [0.00, [244, 236, 221]],
  [0.32, [176, 197, 205]],
  [0.62, [56, 110, 145]],
  [1.00, [14, 33, 55]],
];
function heat(v) {
  const t = Math.max(0, Math.min(1, v));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1], [t1, c1] = RAMP[i];
      const f = (t - t0) / (t1 - t0);
      return `rgb(${c0.map((c, k) => Math.round(c + f * (c1[k] - c))).join(",")})`;
    }
  }
  return `rgb(${RAMP[RAMP.length - 1][1].join(",")})`;
}
const heatTextDark = v => v < 0.42; // dark ink on light cells

/* ===================================================== scenarios config */
const A_SCENARIOS = [
  { t: "Synonyms for “measure”", words: ["measure", "metric", "indicator", "benchmark", "target", "gauge"], teach: "Dense jargon geometry exists — why phrasing matters." },
  { t: "Countries & capitals", words: ["france", "paris", "japan", "tokyo", "germany", "berlin", "italy", "rome"], teach: "Systematic structure, not just clumps." },
  { t: "PM vs DOL vocabulary", words: ["measure", "milestone", "target", "performance", "agency", "program", "workforce", "accountability"], teach: "Domain boundaries are encoded — the PMC payoff." },
  { t: "Tricky neighbors", words: ["monitor", "evaluate", "audit", "oversight", "compliance", "accountability"], teach: "High cosine ≠ same meaning. Inoculation against over-trusting it." },
  { t: "Colors & relatives", words: ["blue", "red", "green", "navy", "crimson", "scarlet"], teach: "Easy-win warm-up; every pair succeeds here." },
  { t: "Activities cluster", words: ["running", "walking", "hiking", "swimming", "jogging", "climbing"], teach: "Semantic clusters survive across word forms." },
  { t: "Federal program language", words: ["initiative", "program", "project", "plan", "mission", "mandate", "authority"], teach: "The cluster pattern holds across PMC sub-domains." },
];
const C_SCENARIOS = [
  { t: "“measure”", word: "measure", teach: "What the model thinks your daily word is like." },
  { t: "“plant” — polysemy", word: "plant", teach: "One vector, two senses (factory + living thing) collapsed together. Sets up contextual embeddings." },
  { t: "“average”", word: "average", teach: "Thesis callback — the generic center. Conditioning moves the model out of here." },
  { t: "“performance”", word: "performance", teach: "An overloaded PMC term lives in a wide neighborhood — it drifts toward generic." },
  { t: "“compliance”", word: "compliance", teach: "Federal vocabulary has its own dense cluster." },
];
const HEADLINE_ANALOGIES = DATA.curated_analogies.filter(a => a.headline);
const POOL_ANALOGIES = DATA.curated_analogies.filter(a => !a.headline);
const FAILURE = DATA.failure_candidates.find(f => f.id === "fail_animal") || DATA.failure_candidates[0];

/* ===================================================== app bootstrap */
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
function el(tag, cls, txt) { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; }
function boot(msg, err) { const b = document.getElementById("boot"); if (b) { b.textContent = msg; if (err) b.classList.add("err"); } }

function buildLegend() {
  const leg = $("#legend");
  for (const key of DATA.clusters) {
    const m = CLUSTER_META[key];
    const chip = el("span", "chip");
    const sw = el("span", "swatch"); sw.style.background = `var(--c${m.v})`;
    chip.append(sw, document.createTextNode(m.n));
    leg.append(chip);
  }
}

function setupTabs() {
  $$("nav.tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("nav.tabs button").forEach(b => b.setAttribute("aria-selected", b === btn));
      const mode = btn.dataset.mode;
      $$(".mode").forEach(s => s.classList.toggle("active", s.dataset.mode === mode));
      if (mode === "A") ModeA.onShow();
      if (mode === "B") ModeB.onShow();
      if (mode === "C") ModeC.onShow();
    });
  });
}

/* ===================================================== shared palette */
function buildPalette({ filtersEl, searchEl, listEl, onPick, inUse }) {
  let activeFilter = "all";
  let query = "";
  const filters = ["all", ...DATA.clusters];
  filtersEl.innerHTML = "";
  filters.forEach(f => {
    const chip = el("span", "chip");
    if (f === "all") { chip.textContent = "all"; chip.style.border = "1px solid var(--line-strong)"; }
    else { const m = CLUSTER_META[f]; const sw = el("span", "swatch"); sw.style.background = `var(--c${m.v})`; chip.append(sw, document.createTextNode(m.n)); }
    chip.dataset.f = f;
    if (f === "all") chip.style.background = "var(--panel-2)";
    chip.addEventListener("click", () => {
      activeFilter = f;
      filtersEl.querySelectorAll(".chip").forEach(c => c.style.background = c.dataset.f === f ? "var(--panel-2)" : "transparent");
      render();
    });
    filtersEl.append(chip);
  });
  if (searchEl) searchEl.addEventListener("input", () => { query = searchEl.value.trim().toLowerCase(); render(); });
  function render() {
    listEl.innerHTML = "";
    const used = inUse ? inUse() : new Set();
    WORDS.filter(w => (activeFilter === "all" || CLUSTER[w] === activeFilter) && (!query || w.includes(query)))
      .forEach(w => {
        const e = el("span", "pal-word", w);
        e.style.borderColor = clusterVar(w, "l");
        if (used.has(w)) e.classList.add("in-use");
        else {
          e.draggable = true;
          e.addEventListener("click", () => onPick(w));
          e.addEventListener("dragstart", ev => Drag.start(ev, w));
        }
        listEl.append(e);
      });
  }
  render();
  return { render };
}

/* ===================================================== drag helper */
const Drag = {
  current: null,
  start(ev, word) { this.current = word; ev.dataTransfer.setData("text/plain", word); ev.dataTransfer.effectAllowed = "copy"; },
  word(ev) { return this.current || ev.dataTransfer.getData("text/plain"); },
};

/* ===================================================== MODE A: cosine grid */
const ModeA = {
  selected: [], pal: null, needsRender: false,
  init() {
    const g = $("#A-scenarios");
    g.append(el("div", "scn-label", "Pre-baked"));
    A_SCENARIOS.forEach(s => {
      const b = el("button", "scn");
      b.append(el("strong", null, s.t));
      b.append(el("span", "scn-teach", s.teach));
      b.addEventListener("click", () => this.loadScenario(s));
      g.append(b);
    });
    this.pal = buildPalette({
      filtersEl: $("#A-filters"), searchEl: $("#A-search"), listEl: $("#A-pallist"),
      onPick: w => this.add(w), inUse: () => new Set(this.selected),
    });
    const sel = $("#A-selected");
    sel.addEventListener("dragover", ev => { ev.preventDefault(); sel.classList.add("drop-armed"); });
    sel.addEventListener("dragleave", () => sel.classList.remove("drop-armed"));
    sel.addEventListener("drop", ev => { ev.preventDefault(); sel.classList.remove("drop-armed"); const w = Drag.word(ev); if (w) this.add(w); });
    $("#A-clear").addEventListener("click", () => { this.selected = []; this.renderSelected(); this.render(false); });
    window.addEventListener("resize", () => { if ($('.mode[data-mode="A"]').classList.contains("active")) this.render(false); });
  },
  onShow() { if (this.needsRender) { this.needsRender = false; this.render(false); } },
  add(w) { if (this.selected.includes(w) || this.selected.length >= 10) return; this.selected.push(w); this.renderSelected(); this.render(false); },
  remove(w) { this.selected = this.selected.filter(x => x !== w); this.renderSelected(); this.render(false); },
  setWords(ws) { this.selected = ws.slice(0, 10); this.renderSelected(); },
  loadScenario(s) { this.setWords(s.words); this.render(true); },
  renderSelected() {
    const tray = $("#A-selected"); tray.innerHTML = "";
    $("#A-count").textContent = this.selected.length ? `(${this.selected.length}/10)` : "";
    if (!this.selected.length) { tray.append(el("span", "tray-empty", "drag or click words…")); }
    this.selected.forEach(w => {
      const p = el("span", "sel-word", w);
      p.style.background = clusterVar(w, "l");
      p.draggable = true;
      p.addEventListener("dragstart", ev => Drag.start(ev, w));
      const x = el("span", "x", "✕"); x.title = "remove";
      x.addEventListener("click", () => this.remove(w));
      p.append(x); tray.append(p);
    });
    if (this.pal) this.pal.render();
  },
  render(animate) {
    const grid = $("#A-grid");
    if (!$('.mode[data-mode="A"]').classList.contains("active")) { this.needsRender = true; }
    grid.innerHTML = "";
    grid.classList.remove("faded");
    const ws = this.selected, N = ws.length;
    $("#A-readout").classList.remove("show");
    if (N < 2) { grid.append(el("div", "muted", "Pick at least two words.")); return; }
    const stage = $(".A-grid-scroll").getBoundingClientRect();
    const leftG = 120, topG = 96, pad = 18;
    const avail = Math.min(stage.width - leftG - pad, stage.height - topG - pad);
    let cell = Math.floor(avail / N);
    cell = Math.max(40, Math.min(72, cell));
    grid.style.width = (leftG + N * cell + pad) + "px";
    grid.style.height = (topG + N * cell + pad) + "px";

    // column labels (top, -45deg)
    ws.forEach((w, j) => {
      const lab = el("div", "glabel col drag-handle");
      const inner = el("span", "lab-inner serif", w); lab.append(inner);
      lab.style.left = (leftG + j * cell + cell * 0.5) + "px";
      lab.style.top = (topG - 21) + "px"; // rotated box bottom must clear the first cell row
      lab.style.color = clusterVar(w, "d");
      this.attachReorder(lab, j);
      grid.append(lab);
    });
    // row labels (left)
    ws.forEach((w, i) => {
      const lab = el("div", "glabel row drag-handle serif", w);
      lab.style.left = "0px"; lab.style.width = (leftG - 12) + "px";
      lab.style.top = (topG + i * cell + cell * 0.5 - 10) + "px";
      lab.style.color = clusterVar(w, "d");
      this.attachReorder(lab, i);
      grid.append(lab);
    });
    // cells
    const cells = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = i === j ? 1 : cosine(ws[i], ws[j]);
      const c = el("div", "cell num");
      c.style.left = (leftG + j * cell) + "px";
      c.style.top = (topG + i * cell) + "px";
      c.style.width = c.style.height = (cell - 4) + "px";
      c.style.background = heat(v);
      if (heatTextDark(v)) c.classList.add("lowtext");
      c.textContent = v.toFixed(2);
      c.dataset.i = i; c.dataset.j = j;
      if (animate) c.classList.add("reveal-hidden", "val-hidden");
      c.addEventListener("mouseenter", () => this.hover(ws[i], ws[j], v, c, true));
      c.addEventListener("mouseleave", () => this.hover(ws[i], ws[j], v, c, false));
      grid.append(c); cells.push(c);
    }
    if (animate) {
      // staged reveal: labels (already visible) -> cells -> values
      grid.querySelectorAll(".glabel").forEach(l => { l.style.opacity = 0; });
      setTimeout(() => grid.querySelectorAll(".glabel").forEach(l => { l.style.transition = "opacity .4s"; l.style.opacity = ""; }), 30);
      setTimeout(() => cells.forEach((c, k) => setTimeout(() => c.classList.remove("reveal-hidden"), (k % N) * 20 + Math.floor(k / N) * 20)), 420);
      setTimeout(() => cells.forEach(c => c.classList.remove("val-hidden")), 1150);
    }
  },
  hover(w1, w2, v, cell, on) {
    const grid = $("#A-grid"), ro = $("#A-readout");
    if (on) {
      grid.classList.add("faded"); cell.classList.add("hot");
      $("#A-ro-1").textContent = w2; $("#A-ro-1").style.color = clusterVar(w2, "d");
      $("#A-ro-2").textContent = w1; $("#A-ro-2").style.color = clusterVar(w1, "d");
      $("#A-ro-v").textContent = v.toFixed(2);
      ro.classList.add("show");
    } else { grid.classList.remove("faded"); cell.classList.remove("hot"); ro.classList.remove("show"); }
  },
  attachReorder(lab, idx) {
    lab.draggable = true;
    lab.addEventListener("dragstart", ev => { ev.dataTransfer.setData("text/reorder", String(idx)); ev.dataTransfer.effectAllowed = "move"; this._dragIdx = idx; });
    lab.addEventListener("dragover", ev => { ev.preventDefault(); lab.classList.add("drag-over"); });
    lab.addEventListener("dragleave", () => lab.classList.remove("drag-over"));
    lab.addEventListener("drop", ev => {
      ev.preventDefault(); lab.classList.remove("drag-over");
      const from = this._dragIdx; const to = idx;
      if (from == null || from === to) return;
      const arr = this.selected; const [m] = arr.splice(from, 1); arr.splice(to, 0, m);
      this._dragIdx = null; this.renderSelected(); this.render(false);
    });
  },
};

/* ===================================================== MODE B: word math */
const ModeB = {
  slots: { a: null, b: null, c: null }, armed: null, pca: null, mode: "analogy",
  init() {
    $$("#B-equation .eq-slot[data-slot]").forEach(s => {
      s.addEventListener("click", () => this.armSlot(s.dataset.slot));
      s.addEventListener("dragover", ev => { ev.preventDefault(); s.classList.add("drop-armed"); });
      s.addEventListener("dragleave", () => s.classList.remove("drop-armed"));
      s.addEventListener("drop", ev => { ev.preventDefault(); s.classList.remove("drop-armed"); const w = Drag.word(ev); if (w && INDEX[w] != null) { this.setSlot(s.dataset.slot, w); } });
    });
    $("#B-datalist").innerHTML = WORDS.map(w => `<option value="${w}">`).join("");
    const search = $("#B-search");
    search.addEventListener("change", () => {
      const w = search.value.trim().toLowerCase();
      if (INDEX[w] == null) return;
      const slot = this.armed || (!this.slots.a ? "a" : !this.slots.b ? "b" : !this.slots.c ? "c" : "a");
      this.setSlot(slot, w); search.value = "";
    });
    const scn = $("#B-scenarios");
    HEADLINE_ANALOGIES.forEach(a => {
      const b = el("button", "scn"); b.style.padding = "8px 12px";
      b.append(el("strong", null, a.label));
      b.addEventListener("click", () => this.loadScenario(a));
      scn.append(b);
    });
    const fail = el("button", "scn failure");
    fail.append(el("strong", null, `${FAILURE.a} − ${FAILURE.b} + ${FAILURE.c}`));
    fail.append(el("span", "scn-teach", "see what failure looks like"));
    fail.addEventListener("click", () => this.loadFailure());
    scn.append(fail);
    const gx = el("button", "scn"); gx.style.borderColor = "var(--c5)";
    gx.append(el("strong", null, "See the gender axis →"));
    gx.append(el("span", "scn-teach", "project occupations onto he–she"));
    gx.addEventListener("click", () => GenderAxis.show());
    scn.append(gx);
    window.addEventListener("resize", () => this._refit());
  },
  onShow() { this._refit(); },
  _refit() {
    if (!$('.mode[data-mode="B"]').classList.contains("active")) return;
    if ($("#B-pca").querySelector(".gx")) GenderAxis.show();
    else if (this.slots.a && this.slots.b && this.slots.c) this.compute();
  },
  armSlot(slot) {
    this.armed = slot;
    $$("#B-equation .eq-slot").forEach(s => s.classList.toggle("armed", s.dataset.slot === slot));
    $("#B-search").focus();
  },
  setSlot(slot, w) { this.slots[slot] = w; this.armed = null; this.mode = "analogy"; this.renderEquation(); this.compute(); },
  loadScenario(a) { this.slots = { a: a.a, b: a.b, c: a.c }; this.armed = null; this.mode = "analogy"; this._scn = a; this.renderEquation(); this.compute(); },
  loadFailure() { this.slots = { a: FAILURE.a, b: FAILURE.b, c: FAILURE.c }; this.armed = null; this.mode = "failure"; this._scn = null; this.renderEquation(); this.compute(); },
  renderEquation() {
    $$("#B-equation .eq-slot[data-slot]").forEach(s => {
      const w = this.slots[s.dataset.slot];
      s.innerHTML = "";
      s.classList.toggle("filled", !!w);
      s.classList.remove("armed");
      if (w) { const span = el("span", "serif", w); span.style.color = clusterVar(w, "d"); s.append(span); }
      else s.append(el("span", "placeholder", s.dataset.slot.toUpperCase()));
    });
  },
  compute() {
    const { a, b, c } = this.slots;
    const res = $("#B-result"), list = $("#B-results"), bn = $("#B-biasnote");
    res.classList.remove("bias"); bn.hidden = true;
    if (!a || !b || !c) { res.innerHTML = '<span class="placeholder">?</span>'; list.innerHTML = '<div class="muted tiny">Fill the three slots, or pick a scenario below.</div>'; this.clearPCA(); return; }
    const top = analogy(a, b, c, 5);
    const isBias = this._scn && this._scn.kind === "bias";
    $("#B-pca").dataset.bias = isBias ? "1" : "0";
    const isFail = this.mode === "failure";
    // result slot
    res.innerHTML = ""; res.classList.toggle("bias", isBias);
    const rs = el("span", "serif", top[0][0]); rs.style.color = isBias ? "var(--c5-d)" : "var(--c1-d)";
    res.append(rs);
    // results list
    list.innerHTML = "";
    top.forEach((r, i) => {
      const row = el("div", "result-row" + (i === 0 ? " top" : "") + (i === 0 && isBias ? " bias" : ""));
      const rw = el("span", "rw", r[0]); rw.style.color = clusterVar(r[0], "d");
      const bar = el("div", "rbar"); const fill = el("i"); fill.style.width = Math.max(4, r[1] * 100) + "%"; bar.append(fill);
      const rv = el("span", "rv num", r[1].toFixed(2));
      row.append(rw, bar, rv); list.append(row);
    });
    if (isBias) {
      bn.hidden = false;
      bn.innerHTML = `The model learned this from how words appear together in human writing — it isn’t “thinking” anything. The geometry reflects the text it was trained on. <em>Descriptive, not normative.</em>`;
    }
    if (isFail) {
      bn.hidden = false; bn.style.color = "var(--c6-d)"; bn.style.background = "#fbf2ec"; bn.style.borderColor = "var(--c6-l)";
      bn.innerHTML = `This is what failure looks like: mixing far-apart parts of the geometry gives an incoherent result with low similarity. Not every analogy resolves — the failures are part of the lesson.`;
    } else { bn.style.color = ""; bn.style.background = ""; bn.style.borderColor = ""; }
    this.renderPCA(a, b, c, top);
  },
  clearPCA() { const box = $("#B-pca"); box.querySelectorAll(".pca-node,.pca-arrow,svg.pca-svg").forEach(n => n.remove()); },
  renderPCA(a, b, c, top) {
    GenderAxis.hide();
    const box = $("#B-pca"); this.clearPCA();
    const staticNote = box.querySelector(".pca-note:not(.gx)"); if (staticNote) staticNote.style.display = "";
    const rect = box.getBoundingClientRect();
    const W = rect.width, H = rect.height, pad = 70;
    const va = vec(a), vb = vec(b), vc = vec(c);
    // raw a−b+c (NOT normalized): keeps target−c === a−b, the parallelogram
    const diff = new Float32Array(DIM), d = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++) { diff[i] = va[i] - vb[i]; d[i] = diff[i] + vc[i]; }
    const result = top[0][0];
    const labeled = [
      { w: a, v: va, role: "in" }, { w: b, v: vb, role: "in" }, { w: c, v: vc, role: "in" },
      { w: result, v: vec(result), role: "result" },
    ];
    top.slice(1).forEach(r => labeled.push({ w: r[0], v: vec(r[0]), role: "alt" }));
    const fit = fitAnalogyPlane([va, vb, vc, d, ...top.map(r => vec(r[0]))], diff);
    const pts = labeled.map(o => ({ ...o, p: fit.project(o.v) }));
    const dPt = fit.project(d);
    // one uniform scale for both axes — parallel stays parallel, equal stays equal
    const all = [...pts.map(p => p.p), dPt];
    const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const s = Math.min((W - 2 * pad) / (maxX - minX || 1), (H - 2 * pad) / (maxY - minY || 1));
    const sx = v => W / 2 + (v - (minX + maxX) / 2) * s;
    const sy = v => H / 2 - (v - (minY + maxY) / 2) * s;

    // the math is pinned: inputs, result, and the ✕ never move. Only the
    // faint alternatives get decluttered, flowing around the anchors.
    const sizeFor = o => o.role === "result" ? 22 : o.role === "alt" ? 13 : 17;
    const boxes = pts.map(o => {
      const fs = sizeFor(o);
      const valW = o.role === "in" ? 0 : 30; // appended cosine value
      return { o, x: sx(o.p[0]), y: sy(o.p[1]), w: textWidth(o.w, fs) + 26 + valW, h: fs + 10,
               weight: o.role === "alt" ? 1 : 0 };
    });
    let tx = sx(dPt[0]), ty = sy(dPt[1]);
    // if pinned math anchors land on the caption, slide the WHOLE cloud up —
    // a rigid translation keeps every angle and length of the picture intact
    const noteRect = (staticNote && staticNote.style.display !== "none") ? staticNote.getBoundingClientRect() : null;
    if (noteRect) {
      const nLeft = noteRect.left - rect.left - 6, nRight = nLeft + noteRect.width + 12, nTop = noteRect.top - rect.top - 6;
      let needUp = 0;
      const probe = (x, y, w, h) => {
        if (x + w / 2 > nLeft && x - w / 2 < nRight && y + h / 2 > nTop) needUp = Math.max(needUp, y + h / 2 - nTop);
      };
      boxes.forEach(bx => { if (!bx.weight) probe(bx.x, bx.y, bx.w, bx.h); });
      probe(tx, ty, 26, 26);
      if (needUp) {
        const headroom = Math.max(0, Math.min(...boxes.map(bx => bx.y - bx.h / 2), ty - 13) - 10);
        const shift = Math.min(needUp, headroom);
        boxes.forEach(bx => { bx.y -= shift; });
        ty -= shift;
      }
    }
    const obstacles = [{ x: tx, y: ty, w: 24, h: 24, weight: 0 }];
    if (noteRect) {
      obstacles.push({ x: noteRect.left - rect.left + noteRect.width / 2, y: noteRect.top - rect.top + noteRect.height / 2, w: noteRect.width, h: noteRect.height, weight: 0 });
    }
    declutter([...boxes, ...obstacles], { x0: 6, y0: 8, x1: W - 6, y1: H - 8 });
    boxes.forEach(bx => { bx.o.x = bx.x; bx.o.y = bx.y; bx.o.bw = bx.w; bx.o.bh = bx.h; });

    // pull line endpoints out of the label boxes so arrows touch, not strike
    const trimLine = (p1, b1, p2, b2) => {
      const dx = p2[0] - p1[0], dy = p2[1] - p1[1], len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const edge = bx => bx ? Math.min(ux ? Math.abs(bx.w / 2 / ux) : 1e9, uy ? Math.abs(bx.h / 2 / uy) : 1e9) : 0;
      const m = len / 2 - 4;
      const t1 = Math.min(edge(b1) + 3, m), t2 = Math.min(edge(b2) + 3, m);
      return [[p1[0] + ux * t1, p1[1] + uy * t1], [p2[0] - ux * t2, p2[1] - uy * t2]];
    };
    const boxOf = o => ({ w: o.bw, h: o.bh });

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "pca-svg"); svg.style.position = "absolute"; svg.style.inset = "0"; svg.style.width = "100%"; svg.style.height = "100%"; svg.style.pointerEvents = "none";
    svg.innerHTML = `<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="var(--ink-soft)"/></marker></defs>`;
    box.append(svg);
    const find = w => pts.find(p => p.w === w);
    const ia = find(a), ib = find(b), ic = find(c), ir = find(result);
    // shared trims for both arrows, so the carried copy lands exactly on its
    // final geometry (the parallelogram guarantees the same direction/length)
    const adx = ia.x - ib.x, ady = ia.y - ib.y, alen = Math.hypot(adx, ady) || 1;
    const aux = adx / alen, auy = ady / alen;
    const edgeT = bx => Math.min(aux ? Math.abs(bx.w / 2 / aux) : 1e9, auy ? Math.abs(bx.h / 2 / auy) : 1e9);
    const half = alen / 2 - 4;
    const tTail = Math.min(Math.max(edgeT(boxOf(ib)), edgeT(boxOf(ic))) + 3, half);
    const tTip = Math.min(Math.max(edgeT(boxOf(ia)), edgeT({ w: 26, h: 26 })) + 3, half);
    const p1 = [ib.x + aux * tTail, ib.y + auy * tTail], p2 = [ia.x - aux * tTip, ia.y - auy * tTip];
    const carryD = [ic.x - ib.x, ic.y - ib.y]; // exact: ✕ − a == c − b

    // 1. SUBTRACTION — the difference arrow, drawn from b to a, named a − b
    const sub = this.arrow(svg, p1, p2, `${a} − ${b}`, 250);
    // 2. ADDITION — the same arrow is picked up and carried to start at c:
    //    adding c just re-bases the difference vector at c
    const carry = document.createElementNS(svg.namespaceURI, "g");
    const cline = document.createElementNS(svg.namespaceURI, "line");
    cline.setAttribute("x1", p1[0]); cline.setAttribute("y1", p1[1]); cline.setAttribute("x2", p2[0]); cline.setAttribute("y2", p2[1]);
    cline.setAttribute("stroke", "var(--ink-soft)"); cline.setAttribute("stroke-width", "2"); cline.setAttribute("marker-end", "url(#ah)");
    const clabel = document.createElementNS(svg.namespaceURI, "text");
    clabel.setAttribute("x", (p1[0] + p2[0]) / 2); clabel.setAttribute("y", (p1[1] + p2[1]) / 2 + 18);
    clabel.setAttribute("fill", "var(--ink-soft)"); clabel.setAttribute("font-size", "13"); clabel.setAttribute("font-family", "Georgia,serif"); clabel.setAttribute("text-anchor", "middle");
    clabel.textContent = `+ ${c}`;
    carry.append(cline, clabel);
    carry.style.opacity = "0"; carry.style.transition = "opacity .3s, transform .8s cubic-bezier(.45,.05,.35,1)";
    svg.append(carry);
    setTimeout(() => { carry.style.opacity = "1"; }, 1150);          // a copy appears on the difference arrow
    setTimeout(() => {                                                // ...and is carried to c
      carry.style.transform = `translate(${carryD[0]}px, ${carryD[1]}px)`;
      sub.line.style.opacity = ".4"; sub.text.style.opacity = ".5";   // focus follows the moving copy
    }, 1400);
    // 3. ✕ where the math lands, then a dotted hop to the nearest real word
    const cross = document.createElementNS(svg.namespaceURI, "text");
    cross.setAttribute("x", tx); cross.setAttribute("y", ty); cross.setAttribute("text-anchor", "middle"); cross.setAttribute("dominant-baseline", "central");
    cross.setAttribute("font-size", "16"); cross.setAttribute("fill", "var(--ink)"); cross.textContent = "✕";
    cross.style.opacity = "0"; cross.style.transition = "opacity .35s";
    svg.append(cross);
    setTimeout(() => { cross.style.opacity = "1"; }, 2200);
    // dotted hop(s) from ✕: always to the winner, labeled with its true cosine;
    // extra fainter hops when runners-up are within 0.03 — a visible near-tie
    const hop = (target, sim, main, delay) => {
      const [s1, s2] = trimLine([tx, ty], { w: 20, h: 20 }, [target.x, target.y], boxOf(target));
      const ln = document.createElementNS(svg.namespaceURI, "line");
      ln.setAttribute("x1", s1[0]); ln.setAttribute("y1", s1[1]); ln.setAttribute("x2", s2[0]); ln.setAttribute("y2", s2[1]);
      ln.setAttribute("stroke", "var(--ink-faint)"); ln.setAttribute("stroke-width", main ? "1.5" : "1");
      ln.setAttribute("stroke-dasharray", main ? "3 4" : "2 5");
      if (!main) ln.setAttribute("stroke-opacity", ".65");
      ln.style.opacity = "0"; ln.style.transition = "opacity .35s";
      svg.append(ln);
      const lt = document.createElementNS(svg.namespaceURI, "text");
      lt.setAttribute("x", (s1[0] + s2[0]) / 2); lt.setAttribute("y", (s1[1] + s2[1]) / 2 - 6);
      lt.setAttribute("fill", "var(--ink-faint)"); lt.setAttribute("font-size", "11"); lt.setAttribute("text-anchor", "middle");
      lt.setAttribute("font-family", "'Helvetica Neue',Arial,sans-serif");
      lt.textContent = main ? `nearest · ${sim.toFixed(2)}` : sim.toFixed(2);
      lt.style.opacity = "0"; lt.style.transition = "opacity .35s";
      svg.append(lt);
      setTimeout(() => { ln.style.opacity = "1"; lt.style.opacity = "1"; }, delay);
    };
    hop(ir, top[0][1], true, 2400);
    top.slice(1).filter(r => top[0][1] - r[1] <= 0.03).forEach((r, i) => {
      const o = find(r[0]);
      if (o) hop(o, r[1], false, 2550 + i * 120);
    });

    const simOf = Object.fromEntries(top.map(r => [r[0], r[1]]));
    const delayFor = o => o.role === "in" ? 0 : o.role === "result" ? 2400 : 2700;
    pts.forEach(o => {
      const node = el("div", "pca-node serif"); node.style.position = "absolute";
      node.style.left = o.x + "px"; node.style.top = o.y + "px"; node.style.transform = "translate(-50%,-50%)";
      node.style.transition = "opacity .5s, left .5s, top .5s"; node.style.padding = "1px 5px"; node.style.borderRadius = "6px"; node.style.whiteSpace = "nowrap";
      if (o.role === "result") { node.style.fontSize = "22px"; node.style.fontWeight = "600"; node.style.color = $("#B-pca").dataset.bias === "1" ? "var(--c5-d)" : "var(--c1-d)"; node.style.background = "rgba(255,255,255,.7)"; node.textContent = o.w; }
      else if (o.role === "alt") { node.style.fontSize = "13px"; node.style.color = "var(--ink-faint)"; node.textContent = o.w; }
      else { node.style.fontSize = "17px"; node.style.color = clusterVar(o.w, "d"); node.textContent = o.w; }
      // true 300-d cosine, carried with the word: on-screen distance is the
      // flattening, the number is the math
      if (o.role !== "in" && simOf[o.w] != null) {
        const val = el("span", "num", simOf[o.w].toFixed(2));
        val.style.cssText = `font-size:${o.role === "result" ? 12 : 10}px;color:var(--ink-faint);margin-left:5px;font-weight:400;font-family:'Helvetica Neue',Arial,sans-serif`;
        node.append(val);
      }
      const dot = el("span");
      dot.style.cssText = `display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:${o.role === "result" ? 3 : 2}px;background:${o.role === "result" ? ($("#B-pca").dataset.bias === "1" ? "var(--c5-d)" : "var(--c1-d)") : o.role === "in" ? clusterVar(o.w, "d") : "var(--ink-faint)"}`;
      node.prepend(dot);
      box.append(node);
      node.style.opacity = "0";
      setTimeout(() => { node.style.opacity = "1"; }, delayFor(o));
    });
  },
  arrow(svg, p1, p2, label, delay) {
    const line = document.createElementNS(svg.namespaceURI, "line");
    line.setAttribute("x1", p1[0]); line.setAttribute("y1", p1[1]); line.setAttribute("x2", p2[0]); line.setAttribute("y2", p2[1]);
    line.setAttribute("stroke", "var(--ink-soft)"); line.setAttribute("stroke-width", "2"); line.setAttribute("marker-end", "url(#ah)");
    const len = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
    line.setAttribute("stroke-dasharray", len); line.setAttribute("stroke-dashoffset", len);
    line.style.transition = "stroke-dashoffset .55s ease, opacity .45s";
    svg.append(line);
    setTimeout(() => line.setAttribute("stroke-dashoffset", "0"), delay || 0);
    const t = document.createElementNS(svg.namespaceURI, "text");
    t.setAttribute("x", (p1[0] + p2[0]) / 2); t.setAttribute("y", (p1[1] + p2[1]) / 2 - 9);
    t.setAttribute("fill", "var(--ink-soft)"); t.setAttribute("font-size", "13"); t.setAttribute("font-family", "Georgia,serif"); t.setAttribute("text-anchor", "middle"); t.textContent = label;
    t.style.opacity = "0"; t.style.transition = "opacity .4s";
    svg.append(t);
    setTimeout(() => { t.style.opacity = "1"; }, (delay || 0) + 250);
    return { line, text: t };
  },
};

/* gender-axis sub-view inside Mode B's PCA box */
const GenderAxis = {
  show() {
    ModeB.clearPCA();
    const box = $("#B-pca"); box.querySelectorAll(".gx").forEach(n => n.remove());
    const staticNote = box.querySelector(".pca-note:not(.gx)"); if (staticNote) staticNote.style.display = "none";
    const occ = ["nurse", "secretary", "homemaker", "teacher", "librarian", "receptionist", "doctor", "engineer", "programmer", "scientist", "ceo", "janitor", "firefighter", "soldier", "pilot", "mechanic", "carpenter", "surgeon"].filter(w => INDEX[w] != null);
    const he = vec("he"), she = vec("she");
    const axis = new Float32Array(DIM); let n = 0;
    for (let i = 0; i < DIM; i++) { axis[i] = he[i] - she[i]; n += axis[i] * axis[i]; }
    n = Math.sqrt(n) || 1; for (let i = 0; i < DIM; i++) axis[i] /= n;
    const scored = occ.map(w => [w, dot(vec(w), axis)]).sort((a, b) => a[1] - b[1]);
    const rect = box.getBoundingClientRect(), W = rect.width, H = rect.height, padX = 90, midY = H / 2;
    const vals = scored.map(s => s[1]); const mn = Math.min(...vals), mx = Math.max(...vals); const span = Math.max(Math.abs(mn), Math.abs(mx)) || 1;
    const X = v => W / 2 + (v / span) * (W / 2 - padX);
    const wrap = el("div", "gx"); wrap.style.cssText = "position:absolute;inset:0";
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "gx"); svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%";
    svg.innerHTML = `<line x1="${W / 2}" y1="34" x2="${W / 2}" y2="${H - 34}" stroke="var(--line-strong)" stroke-dasharray="4 4"/>
      <line x1="${padX}" y1="${midY}" x2="${W - padX}" y2="${midY}" stroke="var(--line-strong)"/>
      <text x="${padX}" y="${midY - 12}" font-family="Georgia" font-size="14" fill="var(--c3-d)">← she</text>
      <text x="${W - padX}" y="${midY - 12}" text-anchor="end" font-family="Georgia" font-size="14" fill="var(--c2-d)">he →</text>`;
    box.append(svg);
    // label positions: stagger, then declutter so same-level neighbors can't collide
    const labs = scored.map((s, i) => {
      const x = X(s[1]); const up = i % 2 === 0; const y = midY + (up ? -1 : 1) * (24 + (i % 6) * 16);
      return { word: s[0], dotX: x, up, x, y, w: textWidth(s[0], 13) + 10, h: 16, weight: 1 };
    });
    declutter(labs, { x0: 10, y0: 26, x1: W - 10, y1: H - 44 });
    labs.forEach(l => {
      const dot = document.createElementNS(svg.namespaceURI, "circle"); dot.setAttribute("cx", l.dotX); dot.setAttribute("cy", midY); dot.setAttribute("r", 4); dot.setAttribute("fill", clusterVar(l.word, "d")); svg.append(dot);
      const ln = document.createElementNS(svg.namespaceURI, "line"); ln.setAttribute("x1", l.dotX); ln.setAttribute("y1", midY); ln.setAttribute("x2", l.x); ln.setAttribute("y2", l.y + (l.up ? 8 : -8)); ln.setAttribute("stroke", "var(--line-strong)"); svg.append(ln);
      const t = document.createElementNS(svg.namespaceURI, "text"); t.setAttribute("x", l.x); t.setAttribute("y", l.y + 4); t.setAttribute("text-anchor", "middle"); t.setAttribute("font-family", "Georgia"); t.setAttribute("font-size", "13"); t.setAttribute("fill", clusterVar(l.word, "d")); t.textContent = l.word; svg.append(t);
    });
    const cap = el("div", "gx pca-note"); cap.style.maxWidth = "70%"; cap.innerHTML = "Each occupation projected onto the <b>he − she</b> direction. The systematic left/right skew is the learned association — descriptive of the training text, not a judgment.";
    box.append(cap);
    $("#B-biasnote").hidden = false;
    $("#B-biasnote").innerHTML = `The model learned this from how words appear together in human writing — descriptive, not normative.`;
  },
  hide() { $("#B-pca").querySelectorAll(".gx").forEach(n => n.remove()); },
};

/* ===================================================== MODE C: neighborhood */
const ModeC = {
  center: null, K: 12, nodes: new Map(), needsRender: false,
  init() {
    const g = $("#C-scenarios");
    g.append(el("div", "scn-label", "Pre-baked"));
    C_SCENARIOS.forEach(s => {
      const b = el("button", "scn");
      b.append(el("strong", null, s.t));
      b.append(el("span", "scn-teach", s.teach));
      b.addEventListener("click", () => this.setCenter(s.word, true));
      g.append(b);
    });
    $("#C-datalist").innerHTML = WORDS.map(w => `<option value="${w}">`).join("");
    $("#C-search").addEventListener("change", e => { const w = e.target.value.trim().toLowerCase(); if (INDEX[w] != null) { this.setCenter(w, true); e.target.value = ""; } });
    window.addEventListener("resize", () => { if ($('.mode[data-mode="C"]').classList.contains("active")) this.render(false); });
  },
  onShow() { if (this.needsRender) { this.needsRender = false; this.render(false); } },
  setCenter(word, animate) {
    if (INDEX[word] == null) return;
    this.center = word;
    // true neighbors list
    const tl = $("#C-true"); tl.innerHTML = "";
    (TRUE_NB[word] || []).slice(0, 10).forEach((w, i) => {
      const row = el("div", "true-row");
      row.append(el("span", "rank num", String(i + 1)));
      if (INDEX[w] != null) {
        // this true neighbor is in the shipped 492 — let the user walk there
        const tw = el("button", "tw tw-link", w);
        tw.type = "button"; tw.style.color = clusterVar(w, "d");
        tw.title = `in this tool’s ${WORDS.length} words — click to explore`;
        tw.addEventListener("click", () => this.setCenter(w, true));
        row.append(tw);
      } else {
        const tw = el("span", "tw", w);
        tw.title = `not in this tool’s ${WORDS.length}-word slice`;
        row.append(tw);
      }
      tl.append(row);
    });
    this.render(animate);
  },
  layout() {
    const stage = $("#C-stage").getBoundingClientRect();
    const W = stage.width, H = stage.height, cx = W / 2, cy = H / 2;
    const neigh = nearest(this.center, this.K);
    const cvec = vec(this.center);
    const rel = neigh.map(([w]) => { const v = vec(w), r = new Float32Array(DIM); for (let i = 0; i < DIM; i++) r[i] = v[i] - cvec[i]; return r; });
    const fit = fitPCA(rel.length > 2 ? rel : [...rel, cvec]);
    const sims = neigh.map(n => n[1]); const sMax = Math.max(...sims), sMin = Math.min(...sims);
    const Rin = 70, Rout = Math.max(120, Math.min(W, H) / 2 - 70);
    const out = neigh.map(([w, s], i) => {
      const p = fit.project(rel[i]); let ang = Math.atan2(p[1], p[0]);
      const t = (sMax - s) / (sMax - sMin || 1); // 0 = closest
      const R = Rin + t * (Rout - Rin);
      return { w, s, ang, R };
    });
    // light angular declutter (with wraparound between last and first)
    out.sort((a, b) => a.ang - b.ang);
    const minAng = 0.32;
    for (let i = 1; i < out.length; i++) { if (out[i].ang - out[i - 1].ang < minAng) out[i].ang = out[i - 1].ang + minAng; }
    if (out.length > 1 && (out[0].ang + 2 * Math.PI) - out[out.length - 1].ang < minAng) out[out.length - 1].ang = out[0].ang + 2 * Math.PI - minAng;
    out.forEach(o => { o.x = cx + Math.cos(o.ang) * o.R; o.y = cy + Math.sin(o.ang) * o.R; });
    // box-level declutter: neighbors move, the center word is pinned
    const boxes = out.map(o => {
      const fs = 13 + o.s * 9;
      return { o, x: o.x, y: o.y, w: textWidth(o.w, fs) + textWidth(" 0.00", 10) + 18, h: fs + 10, weight: 1 };
    });
    const centerBox = { x: cx, y: cy, w: textWidth(this.center, 30) + 34, h: 48, weight: 0 };
    declutter([...boxes, centerBox], { x0: 6, y0: 6, x1: W - 6, y1: H - 6 });
    boxes.forEach(bx => { bx.o.x = bx.x; bx.o.y = bx.y; });
    return { cx, cy, out };
  },
  render(animate) {
    const stage = $("#C-stage");
    if (!$('.mode[data-mode="C"]').classList.contains("active")) { this.needsRender = true; }
    const { cx, cy, out } = this.layout();
    const links = $("#C-links"); links.innerHTML = "";
    const seen = new Set(["__center__", ...out.map(o => o.w)]);
    // links
    out.forEach(o => {
      const ln = document.createElementNS(links.namespaceURI, "line");
      ln.setAttribute("x1", cx); ln.setAttribute("y1", cy); ln.setAttribute("x2", o.x); ln.setAttribute("y2", o.y);
      ln.style.opacity = Math.max(0.18, o.s); links.append(ln);
    });
    // center node
    let center = this.nodes.get("__center__");
    if (!center) { center = el("div", "node center serif"); this.nodes.set("__center__", center); stage.append(center); }
    center.textContent = this.center; center.style.color = clusterVar(this.center, "d");
    center.style.left = cx + "px"; center.style.top = cy + "px";
    // neighbor nodes (reuse by word for smooth transitions)
    out.forEach(o => {
      let node = this.nodes.get(o.w);
      if (!node) {
        node = el("div", "node serif"); this.nodes.set(o.w, node);
        node.style.left = cx + "px"; node.style.top = cy + "px"; node.style.opacity = "0";
        stage.append(node);
      }
      node.innerHTML = ""; node.append(document.createTextNode(o.w));
      const cos = el("span", "ncos num", "  " + o.s.toFixed(2)); node.append(cos);
      node.style.color = clusterVar(o.w, "d");
      node.style.fontSize = (13 + o.s * 9).toFixed(1) + "px";
      node.draggable = true;
      node.ondragstart = ev => Drag.start(ev, o.w);
      node.onclick = () => this.setCenter(o.w, true);
      requestAnimationFrame(() => { node.style.left = o.x + "px"; node.style.top = o.y + "px"; node.style.opacity = "1"; });
    });
    // remove stale nodes
    for (const [w, node] of this.nodes) {
      if (w === "__center__") continue;
      if (!seen.has(w)) { node.style.opacity = "0"; node.style.left = cx + "px"; node.style.top = cy + "px"; const n = node; setTimeout(() => n.remove(), 450); this.nodes.delete(w); }
    }
  },
};

/* ===================================================== compare tray */
const Tray = {
  items: [],
  init() {
    const tray = $("#tray"); tray.hidden = false;
    tray.addEventListener("dragover", ev => { ev.preventDefault(); tray.classList.add("drop-armed"); });
    tray.addEventListener("dragleave", () => tray.classList.remove("drop-armed"));
    tray.addEventListener("drop", ev => { ev.preventDefault(); tray.classList.remove("drop-armed"); const w = Drag.word(ev); if (w && INDEX[w] != null) this.add(w); });
    $("#tray-send").addEventListener("click", () => {
      if (this.items.length < 2) return;
      ModeA.setWords(this.items.slice(0, 10)); ModeA.render(true);
      document.querySelector('nav.tabs button[data-mode="A"]').click();
    });
  },
  add(w) { if (this.items.includes(w)) return; this.items.push(w); this.render(); },
  remove(w) { this.items = this.items.filter(x => x !== w); this.render(); },
  render() {
    const box = $("#tray-items"); box.innerHTML = "";
    if (!this.items.length) box.append(el("span", "tray-empty", "drag words here…"));
    this.items.forEach(w => {
      const p = el("span", "pill serif", w); p.style.borderColor = clusterVar(w, "l");
      const x = el("span", "x", "✕"); x.addEventListener("click", () => this.remove(w)); p.append(x); box.append(p);
    });
    $("#tray-send").disabled = this.items.length < 2;
    $("#tray-send").textContent = `send ${this.items.length || ""} to grid →`;
  },
};

/* ===================================================== run bootstrap */
document.getElementById("boot").remove();
document.querySelector(".app").hidden = false;
$$(".wc").forEach(e => { e.textContent = WORDS.length; }); // live word count
buildLegend();
setupTabs();
ModeA.init();
ModeB.init();
ModeC.init();
Tray.init();
ModeA.loadScenario(A_SCENARIOS[0]);
ModeB.loadScenario(HEADLINE_ANALOGIES[0]);
ModeC.setCenter("measure", false);

})();
