function formatPct(value) {
  return `${Number(value ?? 0).toFixed(2)}%`;
}

function updateRiskSummary(data) {
  const stats = data.stats ?? {};
  const risk = data.risk_config ?? {};

  document.getElementById('daily-loss-used').textContent = formatPct(stats.daily_loss_used_pct);
  document.getElementById('daily-loss-limit').textContent = formatPct(risk.daily_loss_limit_pct);
  document.getElementById('consecutive-losses').textContent = String(stats.consecutive_losses ?? 0);
  document.getElementById('cooldown-until').textContent = stats.cooldown_until ?? '—';
}

async function fetchState() {
  const response = await fetch('/api/state');
  const data = await response.json();
  document.getElementById('state-view').textContent = JSON.stringify(data, null, 2);
  updateRiskSummary(data);
}

function formToJson(form) {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = Number.isNaN(Number(value)) ? value : Number(value);
  }
  return payload;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${url} failed: ${text}`);
  }
}

document.getElementById('risk-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await postJson('/api/risk-config', formToJson(event.target));
    await fetchState();
  } catch (error) {
    alert(error.message);
  }
});

document.getElementById('strategy-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await postJson('/api/strategy-config', formToJson(event.target));
    await fetchState();
  } catch (error) {
    alert(error.message);
  }
});

document.getElementById('tick-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = formToJson(event.target);
  payload.symbol = 'BTC-USDT';
  try {
    await postJson('/api/tick', payload);
    await fetchState();
  } catch (error) {
    alert(error.message);
  }
});

fetchState();
