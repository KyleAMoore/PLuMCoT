"""Self-contained HTML report for a base-vs-PLuMCoT safety run.
 
The page embeds the summary as JSON and draws its charts with a small inline
script (no external assets), so report.html can be opened offline or shared
as a single file.
"""
from __future__ import annotations
 
import html
import json
from typing import Dict, List, Optional
 
from .stats import OUTCOMES, json_safe
 
SUBSET_NAMES = {
    "harmbench": "HarmBench",
    "strongreject": "StrongREJECT",
    "xstest_unsafe": "XSTest, unsafe prompts",
    "xstest_safe": "XSTest, safe prompts",
    "pooled_harmful": "All harmful prompts",
}
OUTCOME_NAMES = {"harmful": "Harmful", "refusal": "Refused", "answered": "Answered safely"}
 
 
def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
 
 
def _examples(records: List[dict], per_subset: int = 6) -> Dict[str, List[dict]]:
    """Prompts whose outcome changed between conditions, balancing
    improvements and regressions. Outcomes are ordered
    harmful < refused < answered safely, for every subset."""
    from .stats import outcome
 
    rank = {"harmful": 0, "refusal": 1, "answered": 2}
    by_item: Dict[str, Dict[str, dict]] = {}
    for r in records:
        by_item.setdefault(r["id"], {})[r["condition"]] = r
    changed: Dict[str, List[dict]] = {}
    for item_id, conds in by_item.items():
        if "base" not in conds or "plumcot" not in conds:
            continue
        b, p = conds["base"], conds["plumcot"]
        ob, op = outcome(b.get("labels", {})), outcome(p.get("labels", {}))
        if not ob or not op or ob == op:
            continue
        changed.setdefault(b["subset"], []).append({
            "id": item_id, "improved": rank[op] > rank[ob],
            # HarmBench contextual prompts start with long context; show the request itself.
            "headline": _clip((b.get("meta") or {}).get("behavior") or b["prompt"], 120),
            "prompt": _clip(b["prompt"], 400),
            "base": {"outcome": ob, "response": _clip(b["response"], 700)},
            "plumcot": {"outcome": op, "response": _clip(p["response"], 700)},
        })
    out = {}
    for subset, exs in changed.items():
        imp = [e for e in exs if e["improved"]]
        reg = [e for e in exs if not e["improved"]]
        n_imp = min(len(imp), max(per_subset // 2, per_subset - len(reg)))
        n_reg = min(len(reg), per_subset - n_imp)
        out[subset] = imp[:n_imp] + reg[:n_reg]
    return out
 
 
def _searched_steps(stats: dict) -> int:
    """Steps that ran a lookahead. Runs made before the stage refactor
    recorded these as "triggered"."""
    return stats["searched"] if "searched" in stats else stats.get("triggered", 0)
 
 
def _cost(records: List[dict]) -> Optional[dict]:
    pl = [r for r in records if r["condition"] == "plumcot" and r.get("plumcot_stats")]
    base = [r for r in records if r["condition"] == "base"]
    if not pl:
        return None
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "n": len(pl),
        "searched": mean([_searched_steps(r["plumcot_stats"]) for r in pl]),
        "lookahead_tokens": mean([r["plumcot_stats"].get("lookahead_tokens", 0) for r in pl]),
        "tokens_plumcot": mean([r["n_tokens"] for r in pl]),
        "tokens_base": mean([r["n_tokens"] for r in base]),
        "seconds_plumcot": mean([r["seconds"] for r in pl]),
        "seconds_base": mean([r["seconds"] for r in base]),
    }
 
 
def render_report(summary: dict, records: List[dict], run_config: dict,
                  title: Optional[str] = None) -> str:
    title = title or f"PLuMCoT safety evaluation: {run_config.get('model', '')}"
    evaluators = sorted({
        name for r in records for name, key in (("WildGuard", "harmful"),
                                                ("HarmBench classifier", "harmful_hb"))
        if r.get("labels", {}).get(key) is not None
    }) or (["keyword refusal check"] if any(
        r.get("labels", {}).get("refusal") is not None for r in records) else [])
    data = {
        "summary": summary,
        "names": {**SUBSET_NAMES},
        "outcomes": list(OUTCOMES),
        "outcomeNames": OUTCOME_NAMES,
        "examples": _examples(records),
        "cost": _cost(records),
        "run": {
            "model": run_config.get("model"),
            "scorer": run_config.get("scorer"),
            "scorer_model": run_config.get("scorer_model"),
            "evaluators": evaluators,
            "synthetic": bool(run_config.get("synthetic")),
            "plumcot": run_config.get("plumcot_config", {}),
        },
    }
    payload = json.dumps(json_safe(data), allow_nan=False).replace("</", "<\\/")
    return (TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__DATA__", payload))
 
 
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s-answered: #2a78d6; --s-harmful: #eb6834; --s-refusal: #1baf7a;
  --plumcot: #2a78d6; --base: #898781;
  --good: #006300; --bad: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s-answered: #3987e5; --s-harmful: #d95926; --s-refusal: #199e70;
    --plumcot: #3987e5; --base: #898781;
    --good: #0ca30c; --bad: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s-answered: #3987e5; --s-harmful: #d95926; --s-refusal: #199e70;
  --plumcot: #3987e5; --base: #898781;
  --good: #0ca30c; --bad: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; font-weight: 600; margin: 0 0 6px; }
h2 { font-size: 18px; font-weight: 600; margin: 40px 0 4px; }
h3 { font-size: 15px; font-weight: 600; margin: 20px 0 8px; }
p.lede, .sub { color: var(--ink-2); margin: 0 0 16px; }
.meta { color: var(--ink-2); font-size: 13px; display: flex; flex-wrap: wrap; gap: 4px 20px; }
.meta b { color: var(--ink); font-weight: 600; }
.banner { border: 1px solid var(--bad); color: var(--ink); background: var(--surface);
  border-radius: 8px; padding: 10px 14px; margin: 16px 0; font-size: 14px; }
.banner b { color: var(--bad); }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 20px; margin-top: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 16px; }
.tile .label { color: var(--ink-2); font-size: 13px; min-height: 2.6em; }
.tile .value { font-size: 28px; font-weight: 600; margin: 6px 0 2px; }
.tile .value .from { color: var(--muted); font-size: 18px; font-weight: 500; }
.tile .delta { font-size: 13px; font-weight: 600; }
.tile .delta.good { color: var(--good); } .tile .delta.bad { color: var(--bad); }
.tile .delta.neutral { color: var(--ink-2); }
.tile .ci { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2);
  margin-bottom: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.sw.dot { border-radius: 50%; }
.stack-group { margin-bottom: 14px; }
.stack-group .gname { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.stack-row { display: grid; grid-template-columns: 76px 1fr; align-items: center; gap: 8px;
  margin: 3px 0; }
.stack-row .rl { font-size: 12px; color: var(--ink-2); }
.bar { display: flex; gap: 2px; height: 24px; }
.seg { height: 100%; position: relative; display: flex; align-items: center;
  justify-content: center; font-size: 12px; font-weight: 600; white-space: nowrap;
  cursor: default; outline-offset: 2px; }
.seg:last-child { border-radius: 0 4px 4px 0; }
.seg:hover, .seg:focus { filter: brightness(1.12); }
.seg .t { padding: 0 6px; }
.seg .t.hide { visibility: hidden; }
.foot { color: var(--muted); font-size: 12px; margin-top: 6px; }
svg text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
svg text.halo { paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
.tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); opacity: 0; transition: opacity .08s;
  max-width: 260px; z-index: 10; }
.tip .v { font-weight: 600; font-size: 13px; }
.tip .k { color: var(--ink-2); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.scroll { overflow-x: auto; }
td.diag { color: var(--muted); }
.tag { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px;
  border-radius: 999px; border: 1px solid var(--ring); color: var(--ink-2); }
.tag.good { color: var(--good); } .tag.bad { color: var(--bad); }
details { border-top: 1px solid var(--grid); padding: 8px 0; }
summary { cursor: pointer; font-size: 13px; }
.ex { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
@media (max-width: 640px) { .ex { grid-template-columns: 1fr; } }
.ex pre { white-space: pre-wrap; word-break: break-word; font: 12px/1.45 ui-monospace, Menlo, monospace;
  background: var(--page); border-radius: 8px; padding: 10px; margin: 4px 0 0; color: var(--ink-2); }
.twocol { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
@media (max-width: 420px) { .twocol { grid-template-columns: 1fr; } }
.twocol > *, .tiles > * { min-width: 0; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<div class="meta" id="meta"></div>
<div id="banner"></div>
 
<h2>Headline</h2>
<p class="lede">Harmful prompts should get no harmful responses; refusing them is fine.
Safe prompts should not be refused. Arrows compare PLuMCoT with the same model decoding normally.</p>
<div class="tiles" id="tiles"></div>
 
<h2>Safety against over-refusal</h2>
<p class="sub">Lower-left is better. Whiskers are 95% Wilson intervals.</p>
<div class="card"><div class="legend" id="plane-legend"></div><div id="plane"></div>
<div class="foot" id="plane-foot"></div></div>
 
<h2>What happened to each prompt</h2>
<p class="sub">Share of responses by outcome, per benchmark. "Answered safely" means the model
engaged without producing harmful content.</p>
<div class="card"><div class="legend" id="stack-legend"></div><div id="stacks"></div>
<div class="foot" id="stack-foot"></div></div>
 
<h2>How outcomes moved</h2>
<p class="sub">Paired counts: each prompt's outcome under normal decoding (rows) and under PLuMCoT (columns).</p>
<div class="twocol" id="transitions"></div>
 
<h2>Statistics</h2>
<p class="sub">Δ is PLuMCoT minus base, in percentage points, with a paired-bootstrap 95% interval.
p is McNemar's exact test on prompts whose label changed.</p>
<div class="card scroll"><table id="stats"></table></div>
 
<h2 id="cost-h">Cost</h2>
<div class="card" id="cost"></div>
 
<h2>Changed outcomes</h2>
<p class="sub">Prompts whose outcome differed between conditions. Responses are model output and may contain harmful text.</p>
<div id="examples"></div>
</main>
<div class="tip" id="tip" role="tooltip"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const S = D.summary;
const NAMES = D.names, ON = D.outcomeNames, OUT = D.outcomes;
const SUBSET_ORDER = ["harmbench", "strongreject", "xstest_unsafe", "xstest_safe"];
const METRIC_NAMES = {harmful: "Harmful response", refusal: "Refusal",
  answered: "Answered safely", harmful_hb: "Harmful (HarmBench classifier)"};
const OUT_VAR = {harmful: "--s-harmful", refusal: "--s-refusal", answered: "--s-answered"};
 
const el = (tag, attrs, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
};
const svgEl = (tag, attrs) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
};
const pct = (x, d = 1) => (x == null || isNaN(x)) ? "–" : (100 * x).toFixed(d) + "%";
const pp = (x, d = 1) => (x == null || isNaN(x)) ? "–" :
  (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(100 * x).toFixed(d) + " pp";
const fmtP = p => p == null ? "–" : p < 0.001 ? "<0.001" : p.toFixed(3);
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
 
// -- tooltip ------------------------------------------------------------------
const tip = document.getElementById("tip");
function showTip(evt, rows) {
  tip.replaceChildren();
  for (const [v, k] of rows) {
    const line = el("div");
    line.append(el("span", {class: "v", text: v}));
    if (k) line.append(el("span", {class: "k", text: "  " + k}));
    tip.append(line);
  }
  const r = evt.target.getBoundingClientRect ? evt.target.getBoundingClientRect() : null;
  let x = evt.clientX, y = evt.clientY;
  if ((x == null || evt.type === "focus") && r) { x = r.left + r.width / 2; y = r.top; }
  tip.style.opacity = 1;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 12)) + "px";
  tip.style.top = Math.max(8, y - h - 12) + "px";
}
const hideTip = () => { tip.style.opacity = 0; };
function hoverable(node, rowsFn) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", e => showTip(e, rowsFn()));
  node.addEventListener("focus", e => showTip(e, rowsFn()));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("blur", hideTip);
}
 
// -- header ---------------------------------------------------------------------
const run = D.run;
const meta = document.getElementById("meta");
const metaItem = (k, v) => { const s = el("span"); s.append(k + ": ", el("b", {text: v || "–"})); meta.append(s); };
metaItem("Model", run.model);
metaItem("PLuMCoT scorer", run.scorer + (run.scorer_model ? " (" + run.scorer_model + ")" : ""));
metaItem("Evaluator", run.evaluators.join(", "));
const pc = run.plumcot || {};
// Current runs: {trigger, configure: {options: [bundle, ...]}}; older runs kept
// lookahead and select at the top level.
const opts = (pc.configure && pc.configure.options) || [pc];
const opt0 = opts[0] || {};
if (pc.trigger && opt0.lookahead && opt0.select) {
  const trig = pc.trigger.threshold != null ? "H ≥ " + pc.trigger.threshold :
    (pc.trigger.type || "trigger");
  metaItem("Trigger / lookahead",
    trig + ", " + opt0.lookahead.max_tokens + " tokens × " + opt0.lookahead.num_samples +
    ", w = " + (opt0.select.judge_weight ?? opt0.select.score_weight ?? opt0.select.alpha) +
    (opts.length > 1 ? " (first of " + opts.length + " configurations)" : ""));
}
if (run.synthetic) {
  const b = el("div", {class: "banner"});
  b.append(el("b", {text: "Synthetic data. "}),
    "These numbers were generated to preview the report layout; they are not results.");
  document.getElementById("banner").append(b);
}
 
// -- tiles ------------------------------------------------------------------------
function tile(label, group, metric, better) {
  const g = S[group] || S.subsets[group];
  if (!g || !g.paired[metric]) return null;
  const b = g.conditions.base[metric], p = g.conditions.plumcot[metric], d = g.paired[metric];
  const sig = d.p < 0.05;
  let cls = "neutral", word = "";
  if (better && sig) {
    const good = (better === "down") ? d.delta < 0 : d.delta > 0;
    cls = good ? "good" : "bad"; word = good ? " better" : " worse";
  } else if (better) { word = " (not significant)"; }
  const arrow = d.delta < 0 ? "▼ " : d.delta > 0 ? "▲ " : "";
  const t = el("div", {class: "tile"});
  t.append(el("div", {class: "label", text: label}));
  const v = el("div", {class: "value"});
  v.append(el("span", {class: "from", text: pct(b.rate) + " → "}), pct(p.rate));
  t.append(v, el("div", {class: "delta " + cls, text: arrow + pp(d.delta) + word}),
    el("div", {class: "ci", text: "95% CI " + pp(d.lo) + " to " + pp(d.hi) +
      " · p " + fmtP(d.p) + " · n " + d.n}));
  return t;
}
const tiles = document.getElementById("tiles");
[tile("Harmful responses to harmful prompts", "pooled_harmful", "harmful", "down"),
 tile("Safe prompts refused (over-refusal)", "xstest_safe", "refusal", "down"),
 tile("Harmful prompts answered safely instead of refused", "pooled_harmful", "answered", "up"),
 tile("Harmful prompts refused", "pooled_harmful", "refusal", null),
].forEach(t => t && tiles.append(t));
if (!tiles.children.length) tiles.append(el("p", {class: "sub", text: "No paired labels yet."}));
 
// -- plane ----------------------------------------------------------------------------
function niceMax(v) {
  const steps = [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1];
  for (const s of steps) if (v <= s) return s;
  return 1;
}
function ticks(max) {
  const step = max <= 0.02 ? 0.005 : max <= 0.05 ? 0.01 : max <= 0.1 ? 0.02 :
    max <= 0.25 ? 0.05 : max <= 0.5 ? 0.1 : 0.2;
  const out = []; for (let t = 0; t <= max + 1e-9; t += step) out.push(+t.toFixed(4)); return out;
}
function renderPlane() {
  const host = document.getElementById("plane");
  host.replaceChildren();
  const H = S.pooled_harmful, X = S.subsets.xstest_safe;
  const legend = document.getElementById("plane-legend");
  legend.replaceChildren();
  if (!H || !X || !H.paired.harmful || !X.paired.refusal) {
    host.append(el("p", {class: "sub", text: "Needs both harmful prompts and XSTest safe prompts with labels."}));
    return;
  }
  for (const [name, v] of [["Normal decoding", "--base"], ["PLuMCoT", "--plumcot"]]) {
    const s = el("span"); const sw = el("i", {class: "sw dot"}); sw.style.background = cssVar(v);
    s.append(sw, name); legend.append(s);
  }
  const pts = ["base", "plumcot"].map(c => ({c,
    x: X.conditions[c].refusal, y: H.conditions[c].harmful}));
  const W = Math.max(320, host.clientWidth), Ht = Math.min(380, Math.max(260, W * 0.55));
  const m = {l: 52, r: 20, t: 12, b: 44};
  const xMax = niceMax(Math.max(...pts.map(p => p.x.hi)) * 1.1 || 0.05);
  const yMax = niceMax(Math.max(...pts.map(p => p.y.hi)) * 1.1 || 0.05);
  const sx = v => m.l + (W - m.l - m.r) * v / xMax;
  const sy = v => Ht - m.b - (Ht - m.t - m.b) * v / yMax;
  const svg = svgEl("svg", {viewBox: `0 0 ${W} ${Ht}`, width: "100%", role: "img",
    "aria-label": "Harmful-response rate on harmful prompts against refusal rate on safe prompts"});
  for (const t of ticks(yMax)) {
    svg.append(svgEl("line", {x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t),
      stroke: cssVar(t === 0 ? "--axis" : "--grid"), "stroke-width": 1}));
    const lab = svgEl("text", {x: m.l - 8, y: sy(t) + 4, "text-anchor": "end", fill: cssVar("--muted")});
    lab.textContent = pct(t, 0); svg.append(lab);
  }
  for (const t of ticks(xMax)) {
    svg.append(svgEl("line", {x1: sx(t), x2: sx(t), y1: m.t, y2: Ht - m.b,
      stroke: cssVar(t === 0 ? "--axis" : "--grid"), "stroke-width": 1}));
    const lab = svgEl("text", {x: sx(t), y: Ht - m.b + 18, "text-anchor": "middle", fill: cssVar("--muted")});
    lab.textContent = pct(t, 0); svg.append(lab);
  }
  const xl = svgEl("text", {x: (m.l + W - m.r) / 2, y: Ht - 6, "text-anchor": "middle", fill: cssVar("--ink-2")});
  xl.textContent = "Safe prompts refused (XSTest)"; svg.append(xl);
  const yl = svgEl("text", {x: 14, y: (m.t + Ht - m.b) / 2, "text-anchor": "middle", fill: cssVar("--ink-2"),
    transform: `rotate(-90 14 ${(m.t + Ht - m.b) / 2})`});
  yl.textContent = "Harmful responses"; svg.append(yl);
  const better = svgEl("text", {x: m.l + 8, y: Ht - m.b - 8, fill: cssVar("--muted")});
  better.textContent = "↙ better"; svg.append(better);
  // arrow base -> plumcot
  const defs = svgEl("defs");
  const mk = svgEl("marker", {id: "ah", viewBox: "0 0 10 10", refX: 9, refY: 5,
    markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"});
  mk.append(svgEl("path", {d: "M0,0 L10,5 L0,10 z", fill: cssVar("--ink-2")}));
  defs.append(mk); svg.append(defs);
  const [a, b] = pts, ax = sx(a.x.rate), ay = sy(a.y.rate), bx = sx(b.x.rate), by = sy(b.y.rate);
  const len = Math.hypot(bx - ax, by - ay);
  if (len > 16) {
    const ux = (bx - ax) / len, uy = (by - ay) / len;
    svg.append(svgEl("line", {x1: ax + ux * 9, y1: ay + uy * 9, x2: bx - ux * 10, y2: by - uy * 10,
      stroke: cssVar("--ink-2"), "stroke-width": 1.5, "marker-end": "url(#ah)"}));
  }
  for (const p of pts) {
    const col = cssVar(p.c === "base" ? "--base" : "--plumcot");
    const cx = sx(p.x.rate), cy = sy(p.y.rate);
    svg.append(svgEl("line", {x1: sx(p.x.lo), x2: sx(p.x.hi), y1: cy, y2: cy, stroke: col, "stroke-width": 2, "stroke-opacity": 0.55}));
    svg.append(svgEl("line", {x1: cx, x2: cx, y1: sy(p.y.lo), y2: sy(p.y.hi), stroke: col, "stroke-width": 2, "stroke-opacity": 0.55}));
    svg.append(svgEl("circle", {cx, cy, r: 6, fill: col, stroke: cssVar("--surface"), "stroke-width": 2}));
    const lab = svgEl("text", {x: cx + 10, y: cy - 10, fill: cssVar("--ink"), class: "halo"});
    lab.textContent = p.c === "base" ? "Normal decoding" : "PLuMCoT";
    svg.append(lab);
    const hit = svgEl("circle", {cx, cy, r: 14, fill: "transparent"});
    hoverable(hit, () => [
      [pct(p.y.rate), "harmful (" + pct(p.y.lo) + "–" + pct(p.y.hi) + ", n " + p.y.n + ")"],
      [pct(p.x.rate), "safe prompts refused (" + pct(p.x.lo) + "–" + pct(p.x.hi) + ", n " + p.x.n + ")"],
      [p.c === "base" ? "Normal decoding" : "PLuMCoT", ""]]);
    svg.append(hit);
  }
  host.append(svg);
  document.getElementById("plane-foot").textContent =
    "Vertical: harmful responses across " + H.n_prompts + " harmful prompts (HarmBench, StrongREJECT, XSTest unsafe). " +
    "Horizontal: refusals across " + X.n_prompts + " XSTest safe prompts.";
}
 
// -- stacked outcome bars -----------------------------------------------------------------
function lum(hex) {
  const c = hex.replace("#", ""); const v = [0, 2, 4].map(i => parseInt(c.substr(i, 2), 16) / 255)
    .map(x => x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4));
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
}
function renderStacks() {
  const legend = document.getElementById("stack-legend"), host = document.getElementById("stacks");
  legend.replaceChildren(); host.replaceChildren();
  for (const o of OUT) {
    const s = el("span"); const sw = el("i", {class: "sw"}); sw.style.background = cssVar(OUT_VAR[o]);
    s.append(sw, ON[o]); legend.append(s);
  }
  let unl = 0;
  for (const name of SUBSET_ORDER) {
    const g = S.subsets[name]; if (!g) continue;
    const grp = el("div", {class: "stack-group"});
    grp.append(el("div", {class: "gname", text: NAMES[name] + "  ·  " + g.n_prompts + " prompts"}));
    for (const c of ["base", "plumcot"]) {
      const counts = g.outcomes[c]; unl += counts.unlabeled || 0;
      const tot = OUT.reduce((a, o) => a + counts[o], 0);
      const row = el("div", {class: "stack-row"});
      row.append(el("div", {class: "rl", text: c === "base" ? "Normal" : "PLuMCoT"}));
      const bar = el("div", {class: "bar"});
      for (const o of OUT) {
        if (!tot || !counts[o]) continue;
        const f = counts[o] / tot;
        const fill = cssVar(OUT_VAR[o]);
        const seg = el("div", {class: "seg"});
        seg.style.flex = f + " 1 0"; seg.style.background = fill;
        seg.style.color = lum(fill) > 0.28 ? "#0b0b0b" : "#ffffff";
        seg.append(el("span", {class: "t", text: pct(f, 0)}));
        hoverable(seg, () => [[pct(f), ON[o] + " · " + (c === "base" ? "normal decoding" : "PLuMCoT")],
          [counts[o] + " of " + tot, NAMES[name]]]);
        bar.append(seg);
      }
      row.append(bar); grp.append(row);
    }
    host.append(grp);
  }
  // hide labels that don't fit rather than clipping them
  host.querySelectorAll(".seg").forEach(seg => {
    const t = seg.querySelector(".t");
    if (t.scrollWidth + 4 > seg.clientWidth) t.classList.add("hide");
  });
  document.getElementById("stack-foot").textContent = unl ?
    unl + " responses had an unparsed evaluator label and are left out of the bars." :
    "Hover a segment for counts; exact values are in the Statistics table.";
}
 
// -- transitions ---------------------------------------------------------------------------
function renderTransitions() {
  const host = document.getElementById("transitions");
  for (const name of SUBSET_ORDER) {
    const g = S.subsets[name]; if (!g) continue;
    const card = el("div", {class: "card"});
    card.append(el("h3", {text: NAMES[name]}));
    card.append(el("div", {class: "foot", text: "Rows: normal decoding · columns: PLuMCoT"}));
    const wrap = el("div", {class: "scroll"});
    const t = el("table"); const hr = el("tr");
    hr.append(el("th", {text: ""}));
    const SHORT = {harmful: "Harmful", refusal: "Refused", answered: "Answered"};
    for (const o of OUT) hr.append(el("th", {class: "num", text: SHORT[o]}));
    t.append(hr);
    for (const a of OUT) {
      const tr = el("tr"); tr.append(el("th", {text: SHORT[a]}));
      for (const b of OUT) tr.append(el("td", {class: "num" + (a === b ? " diag" : ""),
        text: String(g.transitions[a][b])}));
      t.append(tr);
    }
    wrap.append(t); card.append(wrap); host.append(card);
  }
}
 
// -- stats table ------------------------------------------------------------------------------
function renderStats() {
  const t = document.getElementById("stats");
  const hr = el("tr");
  for (const [h, n] of [["Prompts", 0], ["Metric", 0], ["Normal", 1], ["PLuMCoT", 1], ["Δ", 1], ["95% CI", 1], ["p", 1], ["n", 1]])
    hr.append(el("th", {class: n ? "num" : "", text: h}));
  t.append(hr);
  const groups = [...SUBSET_ORDER.filter(n => S.subsets[n]).map(n => [n, S.subsets[n]])];
  if (S.pooled_harmful) groups.push(["pooled_harmful", S.pooled_harmful]);
  for (const [name, g] of groups) {
    for (const m of ["harmful", "harmful_hb", "refusal", "answered"]) {
      const d = g.paired[m]; if (!d) continue;
      const b = g.conditions.base[m], p = g.conditions.plumcot[m];
      const tr = el("tr");
      tr.append(el("td", {text: NAMES[name]}), el("td", {text: METRIC_NAMES[m]}),
        el("td", {class: "num", text: pct(b.rate)}), el("td", {class: "num", text: pct(p.rate)}),
        el("td", {class: "num", text: pp(d.delta)}),
        el("td", {class: "num", text: pp(d.lo) + " to " + pp(d.hi)}),
        el("td", {class: "num", text: fmtP(d.p)}), el("td", {class: "num", text: String(d.n)}));
      t.append(tr);
    }
  }
}
 
// -- cost -----------------------------------------------------------------------------------------
function renderCost() {
  const c = D.cost, host = document.getElementById("cost");
  if (!c) { host.append(el("p", {class: "sub", text: "No PLuMCoT generations yet."})); return; }
  const f1 = x => x == null ? "–" : x.toFixed(1);
  const t = el("table");
  for (const [k, v] of [
    ["Steps with lookahead per response", f1(c.searched)],
    ["Lookahead tokens per response", f1(c.lookahead_tokens)],
    ["Response length, tokens (normal / PLuMCoT)", f1(c.tokens_base) + " / " + f1(c.tokens_plumcot)],
    ["Seconds per response (normal, batched / PLuMCoT, sequential)", f1(c.seconds_base) + " / " + f1(c.seconds_plumcot)],
  ]) { const tr = el("tr"); tr.append(el("td", {text: k}), el("td", {class: "num", text: v})); t.append(tr); }
  host.append(t);
  host.append(el("div", {class: "foot", text: "Normal decoding runs all prompts in one batch, so its per-response time is not a like-for-like latency comparison."}));
}
 
// -- examples ---------------------------------------------------------------------------------------
function renderExamples() {
  const host = document.getElementById("examples");
  let any = false;
  for (const name of SUBSET_ORDER) {
    const ex = D.examples[name]; if (!ex || !ex.length) continue;
    any = true;
    const card = el("div", {class: "card"});
    card.append(el("h3", {text: NAMES[name]}));
    for (const e of ex) {
      const d = el("details"), s = el("summary");
      s.append(el("span", {class: "tag " + (e.improved ? "good" : "bad"), text: e.improved ? "improved" : "worse"}),
        "  " + ON[e.base.outcome] + " → " + ON[e.plumcot.outcome] + "  ·  " + e.headline);
      d.append(s);
      d.append(el("div", {class: "sub", text: "Prompt: " + e.prompt}));
      const grid = el("div", {class: "ex"});
      for (const [lab, r] of [["Normal decoding: " + ON[e.base.outcome], e.base], ["PLuMCoT: " + ON[e.plumcot.outcome], e.plumcot]]) {
        const col = el("div"); col.append(el("div", {class: "rl", text: lab}), el("pre", {text: r.response}));
        grid.append(col);
      }
      d.append(grid); card.append(d);
    }
    host.append(card);
  }
  if (!any) host.append(el("p", {class: "sub", text: "No outcome changed between conditions."}));
}
 
function renderCharts() { hideTip(); renderPlane(); renderStacks(); }
renderCharts(); renderTransitions(); renderStats(); renderCost(); renderExamples();
let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderCharts, 120); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderCharts);
new MutationObserver(renderCharts).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
})();
</script>
</body>
</html>
"""
