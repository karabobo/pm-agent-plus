(() => {
  "use strict";

  const API_BASE = "/api";
  const REFRESH_MS = 5000;
  const MAX_MODELS = 8; // expand to 8 models

  const el = {
    market: document.getElementById("market-grid"),
    models: document.getElementById("models-grid"),
    chat: document.getElementById("live-chat-grid"),
    filter: document.getElementById("model-filter"),
    refresh: document.getElementById("last-refresh"),
    status: document.getElementById("status-chip"),
  };

  const state = {
    market: null,
    models: [],
    events: [], // grouped by cycle_id across all models
    selectedModelId: "ALL",
    lastFetchMs: 0,
    error: null,
  };

  // Setup Event Listeners
  if (el.filter) {
    el.filter.addEventListener("change", (e) => {
      state.selectedModelId = e.target.value;
      renderChat();
    });
  }

  const providerMeta = {
    gemini: { icon: "/logos/gemini.svg", label: "Gemini" },
    openai: { icon: "/logos/openai.svg", label: "OpenAI" },
    claude: { icon: "/logos/cluade.png", label: "Claude" },
    qwen: { icon: "/logos/qwen.svg", label: "Qwen" },
    deepseek: { icon: "/logos/deepseek.svg", label: "DeepSeek" },
    grok: { icon: "✖", label: "Grok" },
    glm: { icon: "智", label: "GLM" },
    custom: { icon: "⚙", label: "Custom" },
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
      second: "2-digit",
      hour12: false,
    });
  }

  function fmtAge(tsMs) {
    if (!tsMs) return "No recent activity";
    const diff = Math.max(0, Date.now() - tsMs);
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return Math.floor(diff / 1000) + "s ago";
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
      .replace(/"/g, "&quot;")
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
    if (!meta) return { icon: "🧠", label: modelId || "Unknown" };
    const modelName = raw.slice(provider.length + 1);
    return {
      icon: meta.icon,
      label: modelName ? `${meta.label} (${modelName})` : meta.label,
    };
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
    if (!upId && !downId) return { rows: [], positionValue: 0, unrealizedPnl: 0 };

    const ordered = [...(trades || [])].sort((a, b) => parseTs(a?.ts) - parseTs(b?.ts));

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

  async function loadModel(modelId, market) {
    const [stats, decisions, trades] = await Promise.all([
      fetchJson(`/stats?model_id=${encodeURIComponent(modelId)}`),
      fetchJson(`/decisions?model_id=${encodeURIComponent(modelId)}`),
      fetchJson(`/trades?model_id=${encodeURIComponent(modelId)}`),
    ]);

    const latestEquity = stats?.latest_equity || null;
    const profit = stats?.profit || {};
    const positionCalc = calcCurrentMarketPositions(trades?.trades || [], market);

    const realized = toNum(latestEquity?.realized_pnl) ?? toNum(profit?.realized_pnl) ?? 0;
    const unrealized = toNum(latestEquity?.unrealized_pnl) ?? toNum(positionCalc.unrealizedPnl) ?? 0;
    const positionValue = toNum(latestEquity?.position_value) ?? toNum(positionCalc.positionValue) ?? 0;
    const totalEquity = toNum(latestEquity?.total_equity);
    const cashBalance = toNum(latestEquity?.cash_balance);

    const lastActivityTs = Math.max(
      parseTs(latestEquity?.ts),
      ...((decisions?.events || []).map(e => parseTs(e.ts))),
      ...((trades?.trades || []).map(t => parseTs(t.ts)))
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
      lastActivityTs,
      rawEvents: decisions?.events || []
    };
  }

  function groupEvents(allModelsData) {
    const cycles = {};
    for (const m of allModelsData) {
      if (!m || !m.rawEvents) continue;
      for (const ev of m.rawEvents) {
        let cycleId = null;
        if (ev.payload?.cycle_id) cycleId = ev.payload.cycle_id;
        else if (ev.payload?.system_state?.cycle_id) cycleId = ev.payload.system_state.cycle_id;

        if (!cycleId) continue; // fallback: ignore events without cycle

        if (!cycles[cycleId]) {
          cycles[cycleId] = {
            cycleId,
            modelId: m.modelId,
            ts: 0,
            prompt: "",
            rawResp: "",
            reasoning: "",
            actions: [],
            results: []
          };
        }

        const t = parseTs(ev.ts);
        if (t > cycles[cycleId].ts) cycles[cycleId].ts = t;

        if (ev.type === "ai_raw") {
          if (ev.payload?.prompt) cycles[cycleId].prompt = ev.payload.prompt;
          if (ev.payload?.raw) cycles[cycleId].rawResp = ev.payload.raw;
        } else if (ev.type === "ai_decision" || ev.type === "decision") {
          if (ev.payload?.decision?.actions) cycles[cycleId].actions = ev.payload.decision.actions;
          else if (ev.payload?.actions) cycles[cycleId].actions = ev.payload.actions;

          if (ev.payload?.reasoning) cycles[cycleId].reasoning = ev.payload.reasoning;
        } else if (ev.type === "run_end") {
          if (Array.isArray(ev.payload?.results)) {
            cycles[cycleId].results = ev.payload.results;
            if (cycles[cycleId].actions.length === 0) {
              // extract actions from results if ai_decision skipped storing them directly
              cycles[cycleId].actions = ev.payload.results.map(r => r.action).filter(Boolean);
            }
          }
        }
      }
    }

    const arr = Object.values(cycles);
    arr.sort((a, b) => b.ts - a.ts); // Descending (newest first)
    return arr;
  }

  function actionClass(type) {
    const t = String(type || "").toLowerCase();
    if (t === "open") return "action-open";
    if (t === "close") return "action-close";
    if (t === "hold") return "action-hold";
    return "action-wait";
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
      { label: "Market", value: escapeHtml(m.market_slug || "--"), cls: "small" },
      { label: "Ends In", value: `<span id="countdown">--:--</span>` },
      { label: "Price To Beat", value: target === null ? "--" : `$${target.toFixed(2)}` },
      {
        label: "BTC Price",
        value: current === null ? "--" : `$${current.toFixed(2)}${delta === null ? "" : ` <span class="${delta >= 0 ? "up" : "down"}">(${delta >= 0 ? "+" : ""}${delta.toFixed(2)})</span>`
          }`
      },
      { label: "UP B/A", value: `${upBid === null ? "--" : upBid.toFixed(3)} / ${upAsk === null ? "--" : upAsk.toFixed(3)}` },
      { label: "DOWN B/A", value: `${downBid === null ? "--" : downBid.toFixed(3)} / ${downAsk === null ? "--" : downAsk.toFixed(3)}` }
    ];

    el.market.innerHTML = cells.map(c => `
      <div class="market-item">
        <div class="label">${c.label}</div>
        <div class="value ${c.cls || ""}">${c.value}</div>
      </div>
    `).join("");

    updateCountdown();
  }

  function renderLeaderboard() {
    if (!el.models) return;
    if (state.error) {
      el.models.innerHTML = `<div class="error">${escapeHtml(state.error)}</div>`;
      return;
    }
    if (!state.models.length) {
      el.models.innerHTML = `<div class="loading">No model data yet.</div>`;
      return;
    }

    el.models.innerHTML = state.models.map(m => {
      const meta = modelMeta(m.modelId);
      const posHtml = m.positions.length ? m.positions.map(p => {
        const cls = p.direction === "UP" ? "up" : "down";
        return `<span class="${cls}">${p.direction}</span> ${p.shares.toFixed(3)} sh ($${(p.mark ?? p.avgCost).toFixed(3)})`;
      }).join(" | ") : "No Position";

      const iconHtml = meta.icon.startsWith("/")
        ? `<img src="${meta.icon}" alt="${escapeHtml(meta.label)}" />`
        : `<span class="emoji">${meta.icon}</span>`;

      return `
        <article class="model-card">
          <div class="model-header">
            <div class="model-name">${iconHtml}<span>${escapeHtml(meta.label)}</span></div>
            <div class="model-time">${escapeHtml(fmtAge(m.lastActivityTs))}</div>
          </div>
          <div class="metrics">
            <div class="metric-line"><span class="k">Eq:</span><span class="v">${fmtMoney(m.totalEquity)}</span></div>
            <div class="metric-line"><span class="k">P&L:</span><span class="v ${(m.realized + m.unrealized) >= 0 ? "up" : "down"}">${fmtSigned(m.realized + m.unrealized)}</span></div>
            <div class="metric-line"><span class="k">Trades:</span><span class="v">${m.tradeCount}</span></div>
          </div>
          <div class="positions-line"><b>POS:</b> ${posHtml}</div>
        </article>
      `;
    }).join("");
  }

  function updateFilterOptions() {
    if (!el.filter) return;
    const currentVal = el.filter.value;
    const modelIds = state.models.map(m => m.modelId);

    let html = `<option value="ALL">ALL MODELS</option>`;
    for (const id of modelIds) {
      const m = modelMeta(id);
      html += `<option value="${escapeHtml(id)}">${escapeHtml(m.label)}</option>`;
    }
    el.filter.innerHTML = html;

    // restore selection if still valid
    if (currentVal === "ALL" || modelIds.includes(currentVal)) {
      el.filter.value = currentVal;
    } else {
      el.filter.value = "ALL";
      state.selectedModelId = "ALL";
    }
  }

  function renderChat() {
    if (!el.chat) return;
    if (!state.events.length) {
      el.chat.innerHTML = `<div class="loading">Waiting for cycle data...</div>`;
      return;
    }

    const filtered = state.selectedModelId === "ALL"
      ? state.events
      : state.events.filter(e => e.modelId === state.selectedModelId);

    if (!filtered.length) {
      el.chat.innerHTML = `<div class="empty-note" style="padding:10px;">No events matching the filter.</div>`;
      return;
    }

    el.chat.innerHTML = filtered.map(ev => {
      const meta = modelMeta(ev.modelId);
      const iconHtml = meta.icon.startsWith("/")
        ? `<img src="${meta.icon}" style="width:16px;" alt="${escapeHtml(meta.label)}" />`
        : `<span class="emoji">${meta.icon}</span>`;

      let actionHtml = "";
      if (ev.actions && ev.actions.length > 0) {
        actionHtml = ev.actions.map(a => {
          const type = String(a.type || "wait").toLowerCase();
          const side = String(a.side || "").replace(/_/g, " ");
          const parts = [type.toUpperCase()];
          if (side) parts.push(side);
          if (a.size > 0) parts.push(`${a.size}sh`);
          return `<span class="action-chip ${actionClass(type)}">${escapeHtml(parts.join(" "))}</span>`;
        }).join("");

        const firstRationale = ev.actions.find(a => a.rationale)?.rationale;
        if (firstRationale && !ev.reasoning) ev.reasoning = firstRationale;
      } else {
        actionHtml = `<span class="action-chip action-wait">WAIT</span>`;
      }

      // We extract thinking from reasoning, or attempt to extract from XML tags in rawResp if reasoning is empty
      let thinking = ev.reasoning || "";
      if (!thinking && ev.rawResp) {
        const m = ev.rawResp.match(/<reasoning>([\s\S]*?)<\/reasoning>/i);
        if (m) thinking = m[1].trim();
      }

      return `
        <article class="chat-card">
          <div class="chat-header">
            <div style="display:flex;align-items:center;gap:6px;font-weight:700;font-size:13px;">
              ${iconHtml} ${escapeHtml(meta.label)}
            </div>
            <div style="font-size:11px;color:var(--muted)">
              ${escapeHtml(fmtShortTime(ev.ts))}
            </div>
          </div>
          <div class="chat-body">
            
            ${ev.prompt ? `
            <details class="chat-details">
              <summary>↳ Expand User Prompt Context</summary>
              <pre class="chat-prompt">${escapeHtml(ev.prompt)}</pre>
            </details>
            ` : ""}

            ${thinking ? `
            <details class="chat-details" open>
              <summary>↳ Thinking Process (Chain of Thought)</summary>
              <pre class="chat-reasoning">${escapeHtml(thinking)}</pre>
            </details>
            ` : ""}

            <div class="chat-block">
              <div class="chat-label">Executing Action</div>
              <div class="chat-action-container">${actionHtml}</div>
            </div>
            
          </div>
        </article>
      `;
    }).join("");
  }

  function render() {
    renderMarket();
    renderLeaderboard();
    updateFilterOptions();
    renderChat();

    if (el.refresh) {
      el.refresh.textContent = state.lastFetchMs
        ? `Refreshed: ${new Date(state.lastFetchMs).toLocaleTimeString("en-US", { hour12: false })}`
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

      const runtimeIds = [...new Set(runtimeResp?.models || [])].filter(m => m && m !== "default");
      const ids = (runtimeIds.length ? runtimeIds : [...new Set(modelResp?.models || [])].filter(m => m && m !== "default")).slice(0, MAX_MODELS);

      const snapshots = await Promise.all(
        ids.map(async (id) => {
          try { return await loadModel(id, market); }
          catch (e) { console.error("failed", id, e); return null; }
        })
      );

      const validModels = snapshots.filter(Boolean);

      // Sort leaderboard initially by Total Equity descending
      validModels.sort((a, b) => {
        const valA = a.totalEquity ?? -99999;
        const valB = b.totalEquity ?? -99999;
        return valB - valA;
      });

      state.market = market;
      state.models = validModels;
      state.events = groupEvents(validModels);
      state.lastFetchMs = Date.now();
      state.error = null;

      setStatus(`Live ${validModels.length} model(s)`, false);
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
