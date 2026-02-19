const state = {
  setups: [],
  watchlist: [],
  volatilePairs: [],
  stats: {},
  events: [],
  wsConnected: false,
  autoRefresh: true,
  refreshTimer: null,
  ws: null,
  wsPingTimer: null,
  selectedSetupId: null,
  tvCandidates: [],
  tvCandidateIndex: 0,
  tvInterval: "5",
  currentBacktestJobId: null,
  backtestPollTimer: null,
  latestBacktest: null,
};

const els = {
  metricTotal: document.getElementById("metricTotal"),
  metricActive: document.getElementById("metricActive"),
  metricWinRate: document.getElementById("metricWinRate"),
  metricWatchlist: document.getElementById("metricWatchlist"),
  metricVolatile: document.getElementById("metricVolatile"),
  setupsBody: document.getElementById("setupsBody"),
  setupsCount: document.getElementById("setupsCount"),
  volatileList: document.getElementById("volatileList"),
  volatileCount: document.getElementById("volatileCount"),
  watchlistList: document.getElementById("watchlistList"),
  watchlistCount: document.getElementById("watchlistCount"),
  eventsLog: document.getElementById("eventsLog"),
  wsBadge: document.getElementById("wsBadge"),
  refreshBtn: document.getElementById("refreshBtn"),
  autoRefreshToggle: document.getElementById("autoRefreshToggle"),
  searchInput: document.getElementById("searchInput"),
  statusFilter: document.getElementById("statusFilter"),
  directionFilter: document.getElementById("directionFilter"),
  confidenceFilter: document.getElementById("confidenceFilter"),
  tvFrame: document.getElementById("tvFrame"),
  tvSourceBadge: document.getElementById("tvSourceBadge"),
  tvSwitchSource: document.getElementById("tvSwitchSource"),
  tvOpenLink: document.getElementById("tvOpenLink"),
  selectedSetupCard: document.getElementById("selectedSetupCard"),
  backtestForm: document.getElementById("backtestForm"),
  backtestMode: document.getElementById("backtestMode"),
  backtestSymbol: document.getElementById("backtestSymbol"),
  backtestLookbackDays: document.getElementById("backtestLookbackDays"),
  backtestProfile: document.getElementById("backtestProfile"),
  backtestLtf: document.getElementById("backtestLtf"),
  backtestHtf: document.getElementById("backtestHtf"),
  backtestStatusBadge: document.getElementById("backtestStatusBadge"),
  backtestJobLabel: document.getElementById("backtestJobLabel"),
  backtestProgressText: document.getElementById("backtestProgressText"),
  backtestProgressBar: document.getElementById("backtestProgressBar"),
  btTrades: document.getElementById("btTrades"),
  btWinRate: document.getElementById("btWinRate"),
  btExpectancy: document.getElementById("btExpectancy"),
  btProfitFactor: document.getElementById("btProfitFactor"),
  btMaxDD: document.getElementById("btMaxDD"),
  backtestResultsBody: document.getElementById("backtestResultsBody"),
};

function setWsBadge(connected) {
  state.wsConnected = connected;
  els.wsBadge.classList.remove("badge-online", "badge-offline");
  if (connected) {
    els.wsBadge.classList.add("badge-online");
    els.wsBadge.textContent = "WS Online";
  } else {
    els.wsBadge.classList.add("badge-offline");
    els.wsBadge.textContent = "WS Offline";
  }
}

function setBacktestBadge(statusText, kind) {
  els.backtestStatusBadge.textContent = statusText;
  els.backtestStatusBadge.classList.remove("badge-online", "badge-offline");
  if (kind === "running" || kind === "completed") {
    els.backtestStatusBadge.classList.add("badge-online");
  } else {
    els.backtestStatusBadge.classList.add("badge-offline");
  }
}

function safeNum(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return "0";
  }
  return n.toFixed(digits);
}

function formatTime(value) {
  if (!value) {
    return "-";
  }
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    return "-";
  }
  return d.toLocaleString();
}

function statusClass(status) {
  if (status === "ACTIVE") {
    return "status status-active";
  }
  if (String(status).startsWith("TP")) {
    return "status status-win";
  }
  if (status === "SL_HIT") {
    return "status status-loss";
  }
  return "status status-other";
}

function directionClass(direction) {
  return direction === "LONG" ? "direction direction-long" : "direction direction-short";
}

function confidenceClass(confidence) {
  if (confidence === "HIGH") {
    return "confidence confidence-high";
  }
  if (confidence === "MEDIUM") {
    return "confidence confidence-medium";
  }
  return "confidence confidence-low";
}

function addEvent(text) {
  const item = {
    id: crypto.randomUUID(),
    text,
    time: new Date().toISOString(),
  };
  state.events.unshift(item);
  state.events = state.events.slice(0, 50);
  renderEvents();
}

function renderMetrics() {
  const total = Number(state.stats.total_setups || 0);
  const winRate = Number(state.stats.win_rate || 0);
  const active = state.setups.filter((s) => s.status === "ACTIVE").length;

  els.metricTotal.textContent = total.toString();
  els.metricActive.textContent = active.toString();
  els.metricWinRate.textContent = `${safeNum(winRate, 1)}%`;
  els.metricWatchlist.textContent = state.watchlist.length.toString();
  els.metricVolatile.textContent = state.volatilePairs.length.toString();
}

function getFilters() {
  return {
    search: els.searchInput.value.trim().toUpperCase(),
    status: els.statusFilter.value,
    direction: els.directionFilter.value,
    confidence: els.confidenceFilter.value,
  };
}

function filteredSetups() {
  const f = getFilters();
  return state.setups.filter((row) => {
    if (f.search && !String(row.symbol || "").toUpperCase().includes(f.search)) {
      return false;
    }
    if (f.status && row.status !== f.status) {
      return false;
    }
    if (f.direction && row.direction !== f.direction) {
      return false;
    }
    if (f.confidence && row.confidence !== f.confidence) {
      return false;
    }
    return true;
  });
}

function renderSetups() {
  const rows = filteredSetups();
  els.setupsCount.textContent = rows.length.toString();
  if (!rows.length) {
    els.setupsBody.innerHTML = `<tr><td colspan="8" class="empty">No setups for current filters</td></tr>`;
    return;
  }

  els.setupsBody.innerHTML = rows
    .map((s) => {
      const selectedClass = state.selectedSetupId === s.id ? "selected" : "";
      return `
      <tr class="setup-row ${selectedClass}" data-setup-id="${s.id}">
        <td class="mono">${formatTime(s.timestamp)}</td>
        <td class="mono">${s.symbol || "-"}</td>
        <td><span class="${directionClass(s.direction)}">${s.direction || "-"}</span></td>
        <td><span class="${statusClass(s.status)}">${s.status || "-"}</span></td>
        <td><span class="${confidenceClass(s.confidence)}">${s.confidence || "-"}</span></td>
        <td class="mono">${safeNum(s.entry, 4)}</td>
        <td class="mono">${safeNum(s.stop_loss, 4)}</td>
        <td class="mono">${safeNum(s.risk_reward, 2)}</td>
      </tr>
    `;
    })
    .join("");
}

function renderWatchlist() {
  els.watchlistCount.textContent = state.watchlist.length.toString();
  if (!state.watchlist.length) {
    els.watchlistList.innerHTML = `<p class="empty">Watchlist is empty</p>`;
    return;
  }

  els.watchlistList.innerHTML = state.watchlist
    .map((row) => {
      const zone = Array.isArray(row.poi_zone) ? row.poi_zone : [];
      const zoneLow = zone.length > 0 ? safeNum(zone[0], 4) : "-";
      const zoneHigh = zone.length > 1 ? safeNum(zone[1], 4) : "-";
      return `
        <article class="watch-item">
          <div class="watch-item-top">
            <span class="watch-symbol mono">${row.symbol || "-"}</span>
            <span class="direction ${row.bias === "BULLISH" ? "direction-long" : "direction-short"}">${row.bias || "-"}</span>
          </div>
          <div class="watch-zone">POI: ${zoneLow} - ${zoneHigh}</div>
          <div class="event-time">checked: ${formatTime(row.last_checked)}</div>
        </article>
      `;
    })
    .join("");
}

function renderVolatilePairs() {
  els.volatileCount.textContent = state.volatilePairs.length.toString();
  if (!state.volatilePairs.length) {
    els.volatileList.innerHTML = `<p class="empty">No volatility leaders yet</p>`;
    return;
  }

  els.volatileList.innerHTML = state.volatilePairs
    .map((row, idx) => {
      return `
        <article class="volatile-item">
          <span class="volatile-rank">#${idx + 1}</span>
          <span class="volatile-symbol mono">${row.symbol || "-"}</span>
          <span class="volatile-score">${safeNum(row.volatility, 4)}%</span>
        </article>
      `;
    })
    .join("");
}

function renderEvents() {
  if (!state.events.length) {
    els.eventsLog.innerHTML = `<p class="empty">No events yet</p>`;
    return;
  }

  els.eventsLog.innerHTML = state.events
    .map((evt) => {
      return `
        <article class="event-item">
          <div>${evt.text}</div>
          <div class="event-time">${formatTime(evt.time)}</div>
        </article>
      `;
    })
    .join("");
}

function renderSelectedSetupCard(setup) {
  if (!setup) {
    els.selectedSetupCard.innerHTML = `<p class="empty">Select a setup row to load chart</p>`;
    return;
  }
  const confluences = (setup.confluences || []).map((x) => `<li>${x}</li>`).join("");
  els.selectedSetupCard.innerHTML = `
    <div class="selected-setup-grid">
      <div class="selected-setup-item"><span>Symbol</span><strong>${setup.symbol}</strong></div>
      <div class="selected-setup-item"><span>Direction</span><strong>${setup.direction}</strong></div>
      <div class="selected-setup-item"><span>Entry</span><strong>${safeNum(setup.entry, 4)}</strong></div>
      <div class="selected-setup-item"><span>SL</span><strong>${safeNum(setup.stop_loss, 4)}</strong></div>
      <div class="selected-setup-item"><span>TP1</span><strong>${safeNum(setup.take_profits?.[0], 4)}</strong></div>
      <div class="selected-setup-item"><span>RR</span><strong>${safeNum(setup.risk_reward, 2)}</strong></div>
    </div>
    <div class="event-time" style="margin-top:8px;">${formatTime(setup.timestamp)}</div>
    <ul style="margin:8px 0 0 16px;padding:0;">${confluences}</ul>
  `;
}

function currentTvSymbol() {
  if (!state.tvCandidates.length) {
    return null;
  }
  return state.tvCandidates[state.tvCandidateIndex] || state.tvCandidates[0];
}

function tvEmbedUrl(symbol, interval) {
  const params = new URLSearchParams({
    symbol,
    interval,
    theme: "light",
    style: "1",
    withdateranges: "1",
    studies: "[]",
    timezone: "Etc/UTC",
    hidetoptoolbar: "1",
    hidesidetoolbar: "1",
  });
  return `https://s.tradingview.com/widgetembed/?${params.toString()}`;
}

function tvExternalUrl(symbol, interval) {
  const params = new URLSearchParams({ symbol, interval });
  return `https://www.tradingview.com/chart/?${params.toString()}`;
}

function renderTradingView() {
  const symbol = currentTvSymbol();
  if (!symbol) {
    els.tvSourceBadge.textContent = "-";
    els.tvFrame.removeAttribute("src");
    els.tvOpenLink.setAttribute("href", "#");
    return;
  }
  els.tvSourceBadge.textContent = `${symbol} (${state.tvCandidateIndex + 1}/${state.tvCandidates.length})`;
  els.tvFrame.src = tvEmbedUrl(symbol, state.tvInterval);
  els.tvOpenLink.href = tvExternalUrl(symbol, state.tvInterval);
}

async function resolveTradingViewSymbol(symbol) {
  try {
    const res = await fetch(`/api/tradingview/symbol/${encodeURIComponent(symbol)}`);
    if (!res.ok) {
      throw new Error("resolver failed");
    }
    const payload = await res.json();
    state.tvCandidates = payload.candidates || [];
    state.tvCandidateIndex = 0;
    state.tvInterval = payload.default_interval || "5";
    renderTradingView();
  } catch (_) {
    const base = String(symbol || "").replace(/[^A-Za-z0-9]/g, "").toUpperCase();
    state.tvCandidates = [`BINGX:${base}.P`, `BINGX:${base}`, `BINANCE:${base}.P`];
    state.tvCandidateIndex = 0;
    state.tvInterval = "5";
    renderTradingView();
  }
}

async function selectSetup(setupId) {
  const setup = state.setups.find((x) => x.id === setupId);
  if (!setup) {
    return;
  }
  state.selectedSetupId = setup.id;
  renderSetups();
  renderSelectedSetupCard(setup);
  await resolveTradingViewSymbol(setup.symbol);
}

function renderBacktestSummary(summary) {
  if (!summary) {
    els.btTrades.textContent = "0";
    els.btWinRate.textContent = "0%";
    els.btExpectancy.textContent = "0";
    els.btProfitFactor.textContent = "0";
    els.btMaxDD.textContent = "0";
    els.backtestResultsBody.innerHTML = `<tr><td colspan="7" class="empty">No backtest results yet</td></tr>`;
    return;
  }
  els.btTrades.textContent = String(summary.trades_count || 0);
  els.btWinRate.textContent = `${safeNum(summary.win_rate, 2)}%`;
  els.btExpectancy.textContent = safeNum(summary.expectancy, 4);
  els.btProfitFactor.textContent = safeNum(summary.profit_factor, 4);
  els.btMaxDD.textContent = safeNum(summary.max_drawdown, 4);

  const rows = summary.symbol_results || [];
  if (!rows.length) {
    els.backtestResultsBody.innerHTML = `<tr><td colspan="7" class="empty">No symbol-level results</td></tr>`;
    return;
  }
  els.backtestResultsBody.innerHTML = rows
    .map((row) => {
      return `
        <tr>
          <td class="mono">${row.symbol}</td>
          <td class="mono">${row.trades_count}</td>
          <td class="mono">${safeNum(row.win_rate, 2)}%</td>
          <td class="mono">${safeNum(row.expectancy, 4)}</td>
          <td class="mono">${safeNum(row.profit_factor, 4)}</td>
          <td class="mono">${safeNum(row.max_drawdown, 4)}</td>
          <td class="mono">${safeNum(row.total_pnl_percent, 4)}%</td>
        </tr>
      `;
    })
    .join("");
}

function setBacktestProgress(jobId, progress) {
  const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
  els.backtestJobLabel.textContent = jobId ? `Job: ${jobId}` : "No active backtest";
  els.backtestProgressText.textContent = `${pct}%`;
  els.backtestProgressBar.style.width = `${pct}%`;
}

function stopBacktestPolling() {
  if (state.backtestPollTimer) {
    clearInterval(state.backtestPollTimer);
    state.backtestPollTimer = null;
  }
}

async function fetchBacktestResult(jobId) {
  const res = await fetch(`/api/backtest/jobs/${encodeURIComponent(jobId)}/result`);
  if (!res.ok) {
    throw new Error("Failed to load backtest result");
  }
  const summary = await res.json();
  state.latestBacktest = summary;
  renderBacktestSummary(summary);
}

async function pollBacktestJob() {
  if (!state.currentBacktestJobId) {
    return;
  }
  try {
    const res = await fetch(`/api/backtest/jobs/${encodeURIComponent(state.currentBacktestJobId)}`);
    if (!res.ok) {
      throw new Error("Job status fetch failed");
    }
    const status = await res.json();
    setBacktestProgress(status.job_id, Number(status.progress || 0));
    if (status.status === "running" || status.status === "queued") {
      setBacktestBadge(status.status.toUpperCase(), "running");
      return;
    }
    if (status.status === "failed") {
      setBacktestBadge("FAILED", "failed");
      addEvent(`Backtest failed: ${status.error || "unknown error"}`);
      stopBacktestPolling();
      return;
    }
    if (status.status === "completed") {
      setBacktestBadge("COMPLETED", "completed");
      await fetchBacktestResult(status.job_id);
      stopBacktestPolling();
      addEvent(`Backtest completed: ${status.job_id}`);
    }
  } catch (error) {
    addEvent(`Backtest poll error: ${error.message}`);
  }
}

function startBacktestPolling() {
  stopBacktestPolling();
  state.backtestPollTimer = setInterval(() => {
    void pollBacktestJob();
  }, 2000);
}

async function runBacktest(event) {
  event.preventDefault();
  const mode = els.backtestMode.value;
  const symbol = els.backtestSymbol.value.trim().toUpperCase();
  const payload = {
    mode,
    symbol: mode === "single" ? symbol : undefined,
    lookback_days: Number(els.backtestLookbackDays.value || 14),
    profile: els.backtestProfile.value,
    ltf_timeframe: els.backtestLtf.value,
    htf_timeframe: els.backtestHtf.value,
  };
  if (mode === "single" && !symbol) {
    addEvent("Backtest validation: symbol required for single mode");
    return;
  }

  try {
    const res = await fetch("/api/backtest/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(errorText || "Backtest run failed");
    }
    const status = await res.json();
    state.currentBacktestJobId = status.job_id;
    setBacktestBadge("RUNNING", "running");
    setBacktestProgress(status.job_id, 0);
    addEvent(`Backtest started: ${status.job_id}`);
    startBacktestPolling();
  } catch (error) {
    setBacktestBadge("FAILED", "failed");
    addEvent(`Backtest start error: ${error.message}`);
  }
}

async function loadLatestBacktest() {
  try {
    const res = await fetch("/api/backtest/latest");
    if (res.status === 404) {
      renderBacktestSummary(null);
      return;
    }
    if (!res.ok) {
      throw new Error("latest backtest unavailable");
    }
    const summary = await res.json();
    state.latestBacktest = summary;
    renderBacktestSummary(summary);
    setBacktestBadge("LATEST LOADED", "completed");
  } catch (_) {
    renderBacktestSummary(null);
  }
}

function renderAll() {
  renderMetrics();
  renderSetups();
  renderVolatilePairs();
  renderWatchlist();
}

async function loadData() {
  try {
    const [setupsRes, watchlistRes, volatileRes, statsRes] = await Promise.all([
      fetch("/api/setups?limit=250"),
      fetch("/api/watchlist"),
      fetch("/api/scanner/volatile-pairs"),
      fetch("/api/stats"),
    ]);

    if (!setupsRes.ok || !watchlistRes.ok || !volatileRes.ok || !statsRes.ok) {
      throw new Error("Failed to load dashboard data");
    }

    state.setups = await setupsRes.json();
    state.watchlist = await watchlistRes.json();
    state.volatilePairs = await volatileRes.json();
    state.stats = await statsRes.json();
    renderAll();
    if (state.selectedSetupId) {
      const stillExists = state.setups.some((x) => x.id === state.selectedSetupId);
      if (!stillExists) {
        state.selectedSetupId = null;
        state.tvCandidates = [];
        renderSelectedSetupCard(null);
        renderTradingView();
      } else {
        renderSetups();
      }
    }
  } catch (error) {
    addEvent(`Data load error: ${error.message}`);
  }
}

function setupAutoRefresh() {
  if (state.refreshTimer) {
    clearInterval(state.refreshTimer);
    state.refreshTimer = null;
  }
  if (state.autoRefresh) {
    state.refreshTimer = setInterval(() => {
      void loadData();
    }, 10000);
  }
}

function connectWs() {
  if (state.ws) {
    try {
      state.ws.close();
    } catch (_) {
      // noop
    }
  }

  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => {
    setWsBadge(true);
    addEvent("WebSocket connected");
    if (state.wsPingTimer) {
      clearInterval(state.wsPingTimer);
    }
    state.wsPingTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("ping");
      }
    }, 15000);
  };

  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      const eventName = payload.event || "event";
      const symbol = payload?.data?.symbol ? ` ${payload.data.symbol}` : "";
      addEvent(`${eventName}${symbol}`);
      if (payload.event === "backtest_job_update" && payload.data?.job_id === state.currentBacktestJobId) {
        void pollBacktestJob();
      }
    } catch (_) {
      addEvent("WebSocket message received");
    }
    void loadData();
  };

  ws.onerror = () => {
    setWsBadge(false);
  };

  ws.onclose = () => {
    setWsBadge(false);
    if (state.wsPingTimer) {
      clearInterval(state.wsPingTimer);
      state.wsPingTimer = null;
    }
    setTimeout(connectWs, 2000);
  };
}

function bindUI() {
  const onFilterChange = () => renderSetups();
  els.searchInput.addEventListener("input", onFilterChange);
  els.statusFilter.addEventListener("change", onFilterChange);
  els.directionFilter.addEventListener("change", onFilterChange);
  els.confidenceFilter.addEventListener("change", onFilterChange);

  els.refreshBtn.addEventListener("click", () => {
    addEvent("Manual refresh");
    void loadData();
  });

  els.autoRefreshToggle.addEventListener("change", () => {
    state.autoRefresh = els.autoRefreshToggle.checked;
    setupAutoRefresh();
    addEvent(state.autoRefresh ? "Auto refresh enabled" : "Auto refresh disabled");
  });

  els.setupsBody.addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-setup-id]");
    if (!row) {
      return;
    }
    const setupId = row.getAttribute("data-setup-id");
    if (!setupId) {
      return;
    }
    void selectSetup(setupId);
  });

  els.tvSwitchSource.addEventListener("click", () => {
    if (!state.tvCandidates.length) {
      return;
    }
    state.tvCandidateIndex = (state.tvCandidateIndex + 1) % state.tvCandidates.length;
    renderTradingView();
  });

  els.backtestMode.addEventListener("change", () => {
    const single = els.backtestMode.value === "single";
    els.backtestSymbol.disabled = !single;
    if (!single) {
      els.backtestSymbol.value = "";
    }
  });

  els.backtestForm.addEventListener("submit", (event) => {
    void runBacktest(event);
  });
}

async function init() {
  bindUI();
  els.backtestMode.dispatchEvent(new Event("change"));
  setupAutoRefresh();
  connectWs();
  renderSelectedSetupCard(null);
  renderTradingView();
  await Promise.all([loadData(), loadLatestBacktest()]);
}

void init();
