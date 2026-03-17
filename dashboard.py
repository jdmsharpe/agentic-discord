"""dashboard.py — Cost monitoring dashboard for agentic-discord.

Run:  python dashboard.py [--port 8080] [--host 127.0.0.1]
Then open http://localhost:8080 in a browser.

Reads agent:{name}:cost:{YYYY-MM-DD} hashes from Redis and renders
a live Chart.js dashboard (auto-refreshes every 30 s).
"""

import argparse
import os
from datetime import date, timedelta

from aiohttp import web
import redis.asyncio as aioredis

AGENTS = ["chatgpt", "claude", "gemini", "grok"]
AGENT_COLORS = {
    "chatgpt": "#10a37f",
    "claude": "#d4a27f",
    "gemini": "#4285f4",
    "grok": "#e7212e",
}

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

async def get_cost_data(r: aioredis.Redis, days: int = 30) -> dict:
    today = date.today()
    dates = [
        (today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)
    ]
    data: dict[str, dict] = {}
    for d in dates:
        data[d] = {}
        for agent in AGENTS:
            key = f"agent:{agent}:cost:{d}"
            raw = await r.hgetall(key)
            if raw:
                float_fields = {"total_cost", "ai_cost", "image_cost"}
                data[d][agent] = {
                    k.decode(): (
                        float(v) if k.decode() in float_fields else int(v)
                    )
                    for k, v in raw.items()
                }
    return {"dates": dates, "agents": AGENTS, "colors": AGENT_COLORS, "data": data}


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

async def handle_api(request: web.Request) -> web.Response:
    r: aioredis.Redis = request.app["redis"]
    days = min(int(request.rel_url.query.get("days", 30)), 90)
    payload = await get_cost_data(r, days=days)
    return web.json_response(payload)


async def handle_index(_: web.Request) -> web.Response:
    return web.Response(text=_DASHBOARD_HTML, content_type="text/html")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    app["redis"] = aioredis.from_url(redis_url, decode_responses=False)


async def on_cleanup(app: web.Application) -> None:
    await app["redis"].aclose()


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/costs", handle_api)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


# ---------------------------------------------------------------------------
# Dashboard HTML (self-contained, Chart.js from CDN)
# Note: all dynamic content is inserted via textContent / DOM methods —
# no user-supplied strings are placed into innerHTML.
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentic-discord · Cost Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px; }
h1 { font-size: 1.35rem; font-weight: 600; }
.subtitle { color: #64748b; font-size: 0.82rem; margin: 4px 0 20px; }
.controls { display: flex; align-items: center; gap: 8px; margin-bottom: 20px; }
.controls label { font-size: 0.8rem; color: #64748b; }
select { background: #1e2130; border: 1px solid #2d3748; color: #e2e8f0;
         padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 12px; margin-bottom: 24px; }
.card { background: #1e2130; border-radius: 10px; padding: 16px; }
.card-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
              letter-spacing: .06em; margin-bottom: 6px; }
.card-value { font-size: 1.55rem; font-weight: 700; }
.card-sub { font-size: 0.72rem; color: #64748b; margin-top: 3px; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart-box { background: #1e2130; border-radius: 10px; padding: 16px 20px; }
.chart-box h2 { font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px; }
canvas { max-height: 240px; }
@media (max-width: 680px) { .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>agentic-discord &middot; Cost Dashboard</h1>
<p class="subtitle" id="subtitle">Loading&hellip;</p>

<div class="controls">
  <label for="days">Show last:</label>
  <select id="days" onchange="loadData()">
    <option value="7">7 days</option>
    <option value="14">14 days</option>
    <option value="30" selected>30 days</option>
    <option value="90">90 days</option>
  </select>
</div>

<div class="cards" id="cards"></div>

<div class="charts">
  <div class="chart-box">
    <h2>Daily cost by agent ($)</h2>
    <canvas id="chart-daily"></canvas>
  </div>
  <div class="chart-box">
    <h2>Cumulative cost ($)</h2>
    <canvas id="chart-cumul"></canvas>
  </div>
  <div class="chart-box">
    <h2>Input tokens per day</h2>
    <canvas id="chart-input"></canvas>
  </div>
  <div class="chart-box">
    <h2>Output tokens per day</h2>
    <canvas id="chart-output"></canvas>
  </div>
  <div class="chart-box">
    <h2>Reasoning tokens per day</h2>
    <canvas id="chart-reasoning"></canvas>
  </div>
  <div class="chart-box">
    <h2>AI calls &amp; image generations per day</h2>
    <canvas id="chart-calls"></canvas>
  </div>
  <div class="chart-box">
    <h2>AI cost vs image cost per day ($)</h2>
    <canvas id="chart-cost-split"></canvas>
  </div>
  <div class="chart-box">
    <h2>Avg cost per call ($)</h2>
    <canvas id="chart-efficiency"></canvas>
  </div>
</div>

<script>
const COLORS = { chatgpt:"#10a37f", claude:"#d4a27f", gemini:"#4285f4", grok:"#e7212e" };
const _charts = {};

function makeChart(id, type, labels, datasets, yFmt) {
  if (_charts[id]) _charts[id].destroy();
  _charts[id] = new Chart(document.getElementById(id), {
    type,
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { labels: { color: "#94a3b8", boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 10, font: { size: 10 } },
             grid: { color: "#1a2030" } },
        y: { ticks: { color: "#64748b", font: { size: 10 },
                      callback: yFmt || (v => v) },
             grid: { color: "#1a2030" } },
      },
    },
  });
}

function barDataset(label, data, color, stack) {
  return { label, data, backgroundColor: color + "cc", stack: stack || "s" };
}
function lineDataset(agent, data) {
  return { label: agent, data, borderColor: COLORS[agent],
           backgroundColor: "transparent", tension: 0.3, pointRadius: 2, borderWidth: 1.5 };
}

function buildCard(labelText, valueText, subText, labelColor) {
  const card = document.createElement("div");
  card.className = "card";
  const lbl = document.createElement("div");
  lbl.className = "card-label";
  lbl.textContent = labelText;
  if (labelColor) lbl.style.color = labelColor;
  const val = document.createElement("div");
  val.className = "card-value";
  val.textContent = valueText;
  const sub = document.createElement("div");
  sub.className = "card-sub";
  sub.textContent = subText;
  card.appendChild(lbl);
  card.appendChild(val);
  card.appendChild(sub);
  return card;
}

function fmtK(v) { return v >= 1000 ? (+v/1000).toFixed(1) + "k" : v; }
const dollar = v => "$" + (+v).toFixed(4);
const millidollar = v => "$" + (+v).toFixed(5);

async function loadData() {
  const days = document.getElementById("days").value;
  const { dates, agents, data } = await fetch("/api/costs?days=" + days).then(r => r.json());

  // Build per-agent time series
  const cost = {}, aiCost = {}, imgCost = {}, inputTok = {}, outputTok = {},
        reasoningTok = {}, calls = {}, imgCalls = {};
  for (const ag of agents) {
    cost[ag] = []; aiCost[ag] = []; imgCost[ag] = [];
    inputTok[ag] = []; outputTok[ag] = []; reasoningTok[ag] = [];
    calls[ag] = []; imgCalls[ag] = [];
    for (const d of dates) {
      const r = data[d]?.[ag] || {};
      cost[ag].push(+(r.total_cost || 0));
      aiCost[ag].push(+(r.ai_cost || 0));
      imgCost[ag].push(+(r.image_cost || 0));
      inputTok[ag].push(r.input_tokens || 0);
      outputTok[ag].push(r.output_tokens || 0);
      reasoningTok[ag].push(r.reasoning_tokens || 0);
      calls[ag].push(r.ai_calls || 0);
      imgCalls[ag].push(r.image_calls || 0);
    }
  }

  const labels = dates.map(d => d.slice(5)); // MM-DD

  // Summary cards
  const totals     = Object.fromEntries(agents.map(ag => [ag, cost[ag].reduce((a,b) => a+b, 0)]));
  const totalCallsN = Object.values(calls).flat().reduce((a,b) => a+b, 0);
  const totalImgN   = Object.values(imgCalls).flat().reduce((a,b) => a+b, 0);
  const totalReason = Object.values(reasoningTok).flat().reduce((a,b) => a+b, 0);
  const grand       = Object.values(totals).reduce((a,b) => a+b, 0);
  const today       = agents.reduce((s, ag) => s + (cost[ag].at(-1) || 0), 0);
  const avgPerCall  = totalCallsN > 0 ? grand / totalCallsN : 0;

  const cardsEl = document.getElementById("cards");
  cardsEl.replaceChildren(
    buildCard("Period total",   "$" + grand.toFixed(3),        "last " + days + " days"),
    buildCard("Today",          "$" + today.toFixed(4),        dates.at(-1)),
    buildCard("Avg $/call",     "$" + avgPerCall.toFixed(4),   totalCallsN + " calls \u00b7 " + totalImgN + " images"),
    buildCard("Reasoning tok",  fmtK(totalReason),             "Grok + Gemini thinking"),
    ...agents.map(ag => {
      const agCalls = calls[ag].reduce((a,b) => a+b, 0);
      const agAvg   = agCalls > 0 ? totals[ag] / agCalls : 0;
      return buildCard(ag.toUpperCase(), "$" + totals[ag].toFixed(3),
                       agCalls + " calls \u00b7 $" + agAvg.toFixed(4) + "/call", COLORS[ag]);
    }),
  );

  // Cost charts
  makeChart("chart-daily", "bar", labels, agents.map(ag => barDataset(ag, cost[ag], COLORS[ag])), dollar);
  makeChart("chart-cumul", "line", labels, agents.map(ag => {
    let acc = 0;
    return lineDataset(ag, cost[ag].map(v => +(acc += v).toFixed(5)));
  }), dollar);

  // AI cost vs image cost — two stacks side-by-side per agent using agent-scoped stacks
  makeChart("chart-cost-split", "bar", labels, agents.flatMap(ag => [
    barDataset(ag + " AI",    aiCost[ag],  COLORS[ag], ag + "-ai"),
    barDataset(ag + " image", imgCost[ag], COLORS[ag] + "55", ag + "-img"),
  ]), dollar);

  // Token charts
  makeChart("chart-input",     "bar", labels, agents.map(ag => barDataset(ag, inputTok[ag],     COLORS[ag])), fmtK);
  makeChart("chart-output",    "bar", labels, agents.map(ag => barDataset(ag, outputTok[ag],    COLORS[ag])), fmtK);
  makeChart("chart-reasoning", "bar", labels, agents.map(ag => barDataset(ag, reasoningTok[ag], COLORS[ag])), fmtK);

  // Calls: AI calls solid, image calls faded, separate stacks so both show
  makeChart("chart-calls", "bar", labels, agents.flatMap(ag => [
    barDataset(ag,           calls[ag],    COLORS[ag],       ag + "-ai"),
    barDataset(ag + " img",  imgCalls[ag], COLORS[ag] + "55", ag + "-img"),
  ]));

  // Avg cost per call per agent (line — only where calls > 0)
  makeChart("chart-efficiency", "line", labels, agents.map(ag => {
    const data = cost[ag].map((c, i) => calls[ag][i] > 0 ? +(c / calls[ag][i]).toFixed(5) : null);
    return lineDataset(ag, data);
  }), millidollar);

  document.getElementById("subtitle").textContent =
    "Updated " + new Date().toLocaleTimeString() + " \u00b7 auto-refreshes every 30 s";
}

loadData();
setInterval(loadData, 30_000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="agentic-discord cost dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    print(f"Dashboard → http://{args.host}:{args.port}")
    web.run_app(make_app(), host=args.host, port=args.port, print=None)
