(() => {
  "use strict";

  const API_BASE = "/api";
  const REFRESH_MS = 5000;
  const MAX_MODELS = 5;

  const el = {
    market: document.getElementById("market-grid"),
    models: document.getElementById("models-grid"),
    decisions: document.getElementById("decisions-grid"),
    refresh: document.getElementById("last-refresh"),
    status: document.getElementById("status-chip"),
  };

  const state = {
    market: null,
    models: [],
    lastFetchMs: 0,
    error: null,
  };

  const providerMeta = {
    gemini: { icon: "/logos/gemini.svg", label: "Gemini" },
    openai: { icon: "/logos/openai.svg", label: "OpenAI" },
    claude: { icon: "/logos/cluade.png", label: "Claude" },
    qwen: { icon: "/logos/qwen.svg", label: "Qwen" },
    deepseek: { icon: "/logos/deepseek.svg", label: "DeepSeek" },
  };

  function toNum(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function parseTs(value) {
    if (!value) return 0;
    const ms = Date.parse(value);
    return Number.isFinite(ms) ? ms : 0;
  }

  function fmtMoney(value) {
    const n = toNum(value);
    if (n === null) return "--";
    return `$${n.toFixed(2)}`;
  }

  function fmtSigned(value) {
    const n = toNum(value);
    if (n === null) return "--";
    return `${n >= 0 ? "+" : "-"}$${Math.abs(n).toFixed(2)}`;
  }

  function fmtShortTime(tsMs) {
    if (!tsMs) return "--";
    return new Date(tsMs).toLocaleString("en-US", {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  }

  function fmtAge(tsMs) {
    if (!tsMs) return "No recent activity";
    const diff = Math.max(0, Date.now() - tsMs);
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  }

  function escapeHtml(text) {
    return String(text ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function setStatus(text, isError = false) {
    if (!el.status) return;
    el.status.textContent = text;
    el.status.classList.toggle("down", isError);
    el.status.classList.toggle("up", !isError);
  }

  async function fetchJson(path) {
    const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`${path} -> HTTP ${res.status}`);
    }
    return res.json();
  }

  async function fetchJsonSafe(path, fallbackValue) {
    try {
      return await fetchJson(path);
    } catch (err) {
      console.warn(`optional api failed: ${path}`, err);
      return fallbackValue;
    }
  }

  function modelMeta(modelId) {
    const raw = (modelId || "").toLowerCase();
    const provider = raw.split("-")[0] || raw;
    const meta = providerMeta[provider];
    if (!meta) {
      return { icon: "🧠", label: modelId || "Unknown" };
    }
    const modelName = raw.slice(provider.length + 1);
    return {
      icon: meta.icon,
      label: modelName ? `${meta.label} (${modelName})` : meta.label,
    };
  }

  function uniqueModels(models) {
    const seen = new Set();
    return (models || []).filter((m) => {
      if (!m || m === "default") return false;
      if (seen.has(m)) return false;
      seen.add(m);
      return true;
    });
  }

  function bestMark(sideQuote) {
    const bid = toNum(sideQuote?.bid);
    const ask = toNum(sideQuote?.ask);
    if (bid !== null && ask !== null) return (bid + ask) / 2;
    if (bid !== null) return bid;
    if (ask !== null) return ask;
    return null;
  }

  function calcCurrentMarketPositions(trades, market) {
    const upId = market?.up_token_id;
    const downId = market?.down_token_id;
    const tracked = {};
    if (upId) tracked[upId] = { shares: 0, avgCost: 0, direction: "UP" };
    if (downId) tracked[downId] = { shares: 0, avgCost: 0, direction: "DOWN" };

    if (!upId && !downId) {
      return { rows: [], positionValue: 0, unrealizedPnl: 0 };
    }

    const ordered = [...(trades || [])].sort((a, b) => {
      const ida = Number(a?.id) || 0;
      const idb = Number(b?.id) || 0;
      if (ida !== idb) return ida - idb;
      return parseTs(a?.ts) - parseTs(b?.ts);
    });

    for (const trade of ordered) {
      const tokenId = trade?.token_id;
      if (!tracked[tokenId]) continue;

      const qty = toNum(trade?.filled_shares) || 0;
      const px = toNum(trade?.avg_price) || 0;
      if (qty <= 0) continue;

      const p = tracked[tokenId];
      const side = String(trade?.side || "").toUpperCase();
      if (side.startsWith("BUY")) {
        const oldShares = p.shares;
        const newShares = oldShares + qty;
        if (newShares > 0) {
          p.avgCost = (p.avgCost * oldShares + px * qty) / newShares;
          p.shares = newShares;
        }
      } else if (side.startsWith("SELL")) {
        p.shares = Math.max(0, p.shares - qty);
        if (p.shares === 0) p.avgCost = 0;
      }
    }

    const upMark = bestMark(market?.up);
    const downMark = bestMark(market?.down);

    const rows = [];
    let positionValue = 0;
    let unrealizedPnl = 0;

    for (const tokenId of Object.keys(tracked)) {
      const p = tracked[tokenId];
      if (p.shares <= 0) continue;
      const mark = p.direction === "UP" ? upMark : downMark;
      const value = p.shares * (mark ?? p.avgCost);
      const upnl = mark === null ? null : p.shares * (mark - p.avgCost);
      positionValue += value;
      if (upnl !== null) unrealizedPnl += upnl;
      rows.push({
        direction: p.direction,
        shares: p.shares,
        avgCost: p.avgCost,
        mark,
        value,
        unrealizedPnl: upnl,
      });
    }

    rows.sort((a, b) => a.direction.localeCompare(b.direction));
    return { rows, positionValue, unrealizedPnl };
  }

  function extractLatestDecision(decisions) {
    const events = Array.isArray(decisions?.events) ? decisions.events : [];
    const decisionEvent = events.find((ev) => ev?.type === "ai_decision" || ev?.type === "decision");
    const runEndEvent = events.find((ev) => ev?.type === "run_end");

    const payload = decisionEvent?.payload || decisions?.last_round_ai_thinking || null;
    let actions = [];

    if (Array.isArray(payload?.decision?.actions)) {
      actions = payload.decision.actions;
    } else if (Array.isArray(payload?.actions)) {
      actions = payload.actions;
    } else if (Array.isArray(runEndEvent?.payload?.results)) {
      actions = runEndEvent.payload.results
        .map((r) => r?.action)
        .filter((a) => a && typeof a === "object");
    }

    const reasoning =
      payload?.reasoning ||
      runEndEvent?.payload?.results?.[0]?.action?.rationale ||
      "";

    const ts = decisionEvent?.ts || runEndEvent?.ts || null;
    return { actions, reasoning, ts };
  }

  function maxTsFromArray(items, key = "ts") {
    let max = 0;
    for (const item of items || []) {
      const ts = parseTs(item?.[key]);
      if (ts > max) max = ts;
    }
    return max;
  }

  async function loadModel(modelId, market) {
    const [stats, decisions, trades] = await Promise.all([
      fetchJson(`/stats?model_id=${encodeURIComponent(modelId)}`),
      fetchJson(`/decisions?model_id=${encodeURIComponent(modelId)}`),
      fetchJson(`/trades?model_id=${encodeURIComponent(modelId)}`),
    ]);

    const latestEquity = stats?.latest_equity || null;
    const profit = stats?.profit || {};

    const positionCalc = calcCurrentMarketPositions(trades?.trades || [], market);
    const latestDecision = extractLatestDecision(decisions);

    const realized =
      toNum(latestEquity?.realized_pnl) ??
      toNum(profit?.realized_pnl) ??
      0;

    const unrealized =
      toNum(latestEquity?.unrealized_pnl) ??
      toNum(positionCalc.unrealizedPnl) ??
      0;

    const positionValue =
      toNum(latestEquity?.position_value) ??
      toNum(positionCalc.positionValue) ??
      0;

    const totalEquity = toNum(latestEquity?.total_equity);
    const cashBalance = toNum(latestEquity?.cash_balance);

    const lastActivityTs = Math.max(
      parseTs(latestEquity?.ts),
      maxTsFromArray(decisions?.events || []),
      maxTsFromArray(trades?.trades || [])
    );

    return {
      modelId,
      latestEquity,
      profit,
      positions: positionCalc.rows,
      totalEquity,
      cashBalance,
      realized,
      unrealized,
      positionValue,
      tradeCount: Number(profit?.trade_count) || 0,
      latestDecision,
      lastActivityTs,
    };
  }

  function renderMarket() {
    if (!el.market) return;

    const m = state.market;
    if (!m) {
      el.market.innerHTML = `<div class="loading">Loading market snapshot...</div>`;
      return;
    }

    const current = toNum(m.current_btc_price);
    const target = toNum(m.price_to_beat);
    const delta = current !== null && target !== null ? current - target : null;

    const upBid = toNum(m?.up?.bid);
    const upAsk = toNum(m?.up?.ask);
    const downBid = toNum(m?.down?.bid);
    const downAsk = toNum(m?.down?.ask);

    const cells = [
      {
        label: "Market",
        value: escapeHtml(m.market_slug || "--"),
        cls: "small",
      },
      {
        label: "Ends In",
        value: `<span id="countdown">--:--</span>`,
      },
      {
        label: "Price To Beat",
        value: target === null ? "--" : `$${target.toFixed(2)}`,
      },
      {
        label: "BTC Price",
        value:
          current === null
            ? "--"
            : `$${current.toFixed(2)}${
                delta === null
                  ? ""
                  : ` <span class="${delta >= 0 ? "up" : "down"}">(${delta >= 0 ? "+" : ""}${delta.toFixed(2)})</span>`
              }`,
      },
      {
        label: "UP Bid / Ask",
        value: `${upBid === null ? "--" : upBid.toFixed(3)} / ${upAsk === null ? "--" : upAsk.toFixed(3)}`,
      },
      {
        label: "DOWN Bid / Ask",
        value: `${downBid === null ? "--" : downBid.toFixed(3)} / ${downAsk === null ? "--" : downAsk.toFixed(3)}`,
      },
    ];

    el.market.innerHTML = cells
      .map(
        (c) => `
        <div class="market-item">
          <div class="label">${c.label}</div>
          <div class="value ${c.cls || ""}">${c.value}</div>
        </div>
      `
      )
      .join("");

    updateCountdown();
  }

  function renderModels() {
    if (!el.models) return;

    if (state.error) {
      el.models.innerHTML = `<div class="error">${escapeHtml(state.error)}</div>`;
      return;
    }

    if (!state.models.length) {
      el.models.innerHTML = `<div class="loading">No model data yet.</div>`;
      return;
    }

    el.models.innerHTML = state.models
      .map((m) => {
        const meta = modelMeta(m.modelId);
        const posHtml = m.positions.length
          ? m.positions
              .map((p) => {
                const cls = p.direction === "UP" ? "up" : "down";
                const upnl = toNum(p.unrealizedPnl);
                return `
                <div class="pos-row">
                  <div class="pos-head">
                    <span class="${cls}">${p.direction}</span>
                    <span>${p.shares.toFixed(3)} shares</span>
                  </div>
                  <div class="pos-sub">Avg ${p.avgCost.toFixed(3)} | Mark ${p.mark === null ? "--" : p.mark.toFixed(3)}</div>
                  <div class="pos-sub">Value ${fmtMoney(p.value)} | U-PnL <span class="${(upnl || 0) >= 0 ? "up" : "down"}">${fmtSigned(upnl)}</span></div>
                </div>
              `;
              })
              .join("")
          : `<div class="empty-note">No current market position.</div>`;

        const iconHtml = meta.icon.startsWith("/")
          ? `<img src="${meta.icon}" alt="${escapeHtml(meta.label)}" />`
          : `<span class="emoji">${meta.icon}</span>`;

        const eq = m.totalEquity !== null ? fmtMoney(m.totalEquity) : "--";
        const cash = m.cashBalance !== null ? fmtMoney(m.cashBalance) : "--";

        return `
          <article class="model-card">
            <div class="model-header">
              <div class="model-name">${iconHtml}<span>${escapeHtml(meta.label)}</span></div>
              <div class="model-time">${escapeHtml(fmtAge(m.lastActivityTs))}<br/>${escapeHtml(fmtShortTime(m.lastActivityTs))}</div>
            </div>

            <div class="metrics">
              <div class="metric-line"><span class="k">Total Equity</span><span class="v">${eq}</span></div>
              <div class="metric-line"><span class="k">Cash</span><span class="v">${cash}</span></div>
              <div class="metric-line"><span class="k">Position Value</span><span class="v">${fmtMoney(m.positionValue)}</span></div>
              <div class="metric-line"><span class="k">Unrealized P&L</span><span class="v ${(m.unrealized || 0) >= 0 ? "up" : "down"}">${fmtSigned(m.unrealized)}</span></div>
              <div class="metric-line"><span class="k">Realized P&L</span><span class="v ${(m.realized || 0) >= 0 ? "up" : "down"}">${fmtSigned(m.realized)}</span></div>
              <div class="metric-line"><span class="k">Trades</span><span class="v">${m.tradeCount}</span></div>
            </div>

            <div class="positions">${posHtml}</div>
          </article>
        `;
      })
      .join("");
  }

  function actionClass(type) {
    const t = String(type || "").toLowerCase();
    if (t === "open") return "action-open";
    if (t === "close") return "action-close";
    if (t === "hold") return "action-hold";
    return "action-wait";
  }

  function renderDecisions() {
    if (!el.decisions) return;

    if (!state.models.length) {
      el.decisions.innerHTML = `<div class="loading">No decisions yet.</div>`;
      return;
    }

    el.decisions.innerHTML = state.models
      .map((m) => {
        const meta = modelMeta(m.modelId);
        const iconHtml = meta.icon.startsWith("/")
          ? `<img src="${meta.icon}" alt="${escapeHtml(meta.label)}" />`
          : `<span class="emoji">${meta.icon}</span>`;

        const actions = Array.isArray(m.latestDecision.actions) ? m.latestDecision.actions : [];
        const actionHtml = actions.length
          ? actions
              .slice(0, 4)
              .map((a) => {
                const type = String(a?.type || "wait").toLowerCase();
                const side = String(a?.side || "").replace(/_/g, " ");
                const size = toNum(a?.size);
                const price = toNum(a?.price);
                const parts = [type.toUpperCase()];
                if (side) parts.push(side);
                if (size !== null && size > 0) parts.push(`${size.toFixed(3)}sh`);
                if (price !== null && price > 0) parts.push(`@${price.toFixed(3)}`);
                return `<span class="action-chip ${actionClass(type)}">${escapeHtml(parts.join(" "))}</span>`;
              })
              .join("")
          : `<span class="empty-note">No decision action.</span>`;

        const rationale =
          actions.find((a) => a?.rationale)?.rationale ||
          m.latestDecision.reasoning ||
          "No rationale yet.";

        const reasoning = m.latestDecision.reasoning || "";

        return `
          <article class="decision-card">
            <div class="model-header">
              <div class="model-name">${iconHtml}<span>${escapeHtml(meta.label)}</span></div>
              <div class="model-time">${escapeHtml(fmtAge(parseTs(m.latestDecision.ts) || m.lastActivityTs))}</div>
            </div>

            <div class="action-list">${actionHtml}</div>
            <div class="rationale">${escapeHtml(rationale)}</div>

            ${
              reasoning
                ? `<details class="details"><summary>Show full reasoning</summary><pre>${escapeHtml(reasoning)}</pre></details>`
                : ""
            }
          </article>
        `;
      })
      .join("");
  }

  function render() {
    renderMarket();
    renderModels();
    renderDecisions();

    if (el.refresh) {
      el.refresh.textContent = state.lastFetchMs
        ? `Last refresh: ${new Date(state.lastFetchMs).toLocaleTimeString("en-US", { hour12: false })}`
        : "Loading...";
    }
  }

  function updateCountdown() {
    const node = document.getElementById("countdown");
    if (!node) return;

    const endTs = parseTs(state.market?.market_end_time);
    if (!endTs) {
      node.textContent = "--:--";
      node.className = "warn";
      return;
    }

    const left = Math.max(0, endTs - Date.now());
    const mins = Math.floor(left / 60000);
    const secs = Math.floor((left % 60000) / 1000);
    node.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    node.className = left <= 5 * 60000 ? "down" : "warn";
  }

  async function loadDashboard() {
    setStatus("Syncing", false);
    try {
      const [market, modelResp, runtimeResp] = await Promise.all([
        fetchJson("/market_prices"),
        fetchJson("/models"),
        fetchJsonSafe("/runtime_models", { models: [] }),
      ]);

      const runtimeIds = uniqueModels(runtimeResp?.models);
      const ids = (runtimeIds.length ? runtimeIds : uniqueModels(modelResp?.models)).slice(
        0,
        MAX_MODELS
      );
      const snapshots = await Promise.all(
        ids.map(async (id) => {
          try {
            return await loadModel(id, market);
          } catch (e) {
            console.error("failed model load", id, e);
            return null;
          }
        })
      );

      const rows = snapshots.filter(Boolean);
      rows.sort((a, b) => {
        if (b.lastActivityTs !== a.lastActivityTs) return b.lastActivityTs - a.lastActivityTs;
        return a.modelId.localeCompare(b.modelId);
      });

      const selected = rows.slice(0, MAX_MODELS);

      state.market = market;
      state.models = selected;
      state.lastFetchMs = Date.now();
      state.error = null;

      const source = runtimeIds.length ? "runtime" : "history";
      setStatus(`Live ${selected.length} model(s) · ${source}`, false);
      render();
    } catch (err) {
      console.error(err);
      state.error = err instanceof Error ? err.message : String(err);
      state.lastFetchMs = Date.now();
      setStatus("API error", true);
      render();
    }
  }

  render();
  loadDashboard();
  setInterval(loadDashboard, REFRESH_MS);
  setInterval(updateCountdown, 1000);
})();
