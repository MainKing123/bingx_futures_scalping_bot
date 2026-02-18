async function fetchState() {
  const response = await fetch('/api/state');
  const data = await response.json();
  document.getElementById('state-view').textContent = JSON.stringify(data, null, 2);
}

function formToJson(form) {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = Number.isNaN(Number(value)) ? value : Number(value);
  }
  return payload;
}

document.getElementById('risk-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = formToJson(event.target);
  await fetch('/api/risk-config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  await fetchState();
});

document.getElementById('tick-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = formToJson(event.target);
  payload.symbol = 'BTC-USDT';
  await fetch('/api/tick', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  await fetchState();
});

fetchState();
