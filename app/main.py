from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models import BotState, MarketTick, RiskConfig, StrategySignal
from app.risk import RiskManager
from app.strategy import ICTSMCStrategy

app = FastAPI(title="BingX Futures Scalping Bot MVP")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

strategy = ICTSMCStrategy()
risk_manager = RiskManager()

state = BotState(
    risk_config=RiskConfig(),
    latest_signal=StrategySignal(has_signal=False, reason="No ticks processed yet"),
)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/state", response_model=BotState)
async def get_state() -> BotState:
    return state


@app.post("/api/risk-config", response_model=BotState)
async def update_risk_config(config: RiskConfig) -> BotState:
    state.risk_config = config
    return state


@app.post("/api/tick", response_model=BotState)
async def process_tick(tick: MarketTick) -> BotState:
    state.latest_signal = strategy.evaluate(tick)

    if state.latest_signal.has_signal and state.active_position is None:
        state.active_position = risk_manager.build_position(tick.symbol, state.latest_signal, state.risk_config)

    if state.active_position is not None:
        pos = state.active_position

        hit_stop = tick.price <= pos.stop_price if pos.side.value == "long" else tick.price >= pos.stop_price
        hit_tp = tick.price >= pos.take_profit if pos.side.value == "long" else tick.price <= pos.take_profit
        if hit_stop or hit_tp:
            pnl = risk_manager.close_pnl(pos, tick.price)
            state.closed_pnl_usdt += pnl
            state.active_position = None

    return state


@app.post("/api/close-position", response_model=BotState)
async def close_position(close_price: float) -> BotState:
    if state.active_position is None:
        raise HTTPException(status_code=400, detail="No active position")
    state.closed_pnl_usdt += risk_manager.close_pnl(state.active_position, close_price)
    state.active_position = None
    return state
