from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models import BotState, MarketTick, RiskConfig, StrategyConfig, StrategySignal
from app.risk import RiskManager
from app.strategy import ICTSMCStrategy

app = FastAPI(title="BingX Futures Scalping Bot MVP")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

strategy = ICTSMCStrategy()
risk_manager = RiskManager()

state = BotState(
    risk_config=RiskConfig(),
    strategy_config=StrategyConfig(),
    latest_signal=StrategySignal(has_signal=False, reason="No ticks processed yet"),
)


def _is_blocked_for_new_trades() -> tuple[bool, str]:
    now = risk_manager.utcnow()
    stats = state.stats
    risk = state.risk_config
    stats.daily_loss_used_pct = risk_manager.update_daily_loss_used_pct(
        stats.closed_pnl_usdt,
        risk,
        realized_pnl_usdt=0.0,
        now=now,
    )

    if stats.cooldown_until is not None and now < stats.cooldown_until:
        return True, f"Cooldown active until {stats.cooldown_until.isoformat()}"

    if stats.daily_loss_used_pct >= risk.daily_loss_limit_pct:
        return True, "Daily loss limit reached"

    if stats.consecutive_losses >= risk.max_consecutive_losses:
        return True, "Max consecutive losses reached"

    return False, ""


def _register_close(pnl: float) -> None:
    stats = state.stats
    stats.closed_pnl_usdt += pnl
    stats.trades_closed += 1
    stats.daily_loss_used_pct = risk_manager.update_daily_loss_used_pct(
        stats.closed_pnl_usdt,
        state.risk_config,
        realized_pnl_usdt=pnl,
    )

    if pnl >= 0:
        stats.wins += 1
        stats.consecutive_losses = 0
        return

    stats.losses += 1
    stats.consecutive_losses += 1
    stats.cooldown_until = risk_manager.should_enter_cooldown(risk_manager.utcnow(), state.risk_config)


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


@app.post("/api/strategy-config", response_model=BotState)
async def update_strategy_config(config: StrategyConfig) -> BotState:
    state.strategy_config = config
    return state


@app.post("/api/tick", response_model=BotState)
async def process_tick(tick: MarketTick) -> BotState:
    if state.active_position is not None:
        pos = state.active_position
        pos, partial_pnl = risk_manager.apply_partial_take_profit(pos, tick.price)
        if partial_pnl != 0:
            _register_close(partial_pnl)

        pos = risk_manager.apply_trailing_stop(pos, tick.price)
        hit_stop = tick.price <= pos.stop_price if pos.side.value == "long" else tick.price >= pos.stop_price
        hit_tp = tick.price >= pos.take_profit if pos.side.value == "long" else tick.price <= pos.take_profit

        if hit_stop or hit_tp:
            pnl = risk_manager.close_pnl(pos, tick.price)
            _register_close(pnl)
            state.active_position = None
            state.latest_signal = StrategySignal(has_signal=False, reason="Position closed")
            return state

        state.active_position = pos
        state.latest_signal = StrategySignal(has_signal=False, reason="Managing active position")
        return state

    state.latest_signal = strategy.evaluate(tick, state.strategy_config)
    blocked, reason = _is_blocked_for_new_trades()
    if blocked:
        state.latest_signal = StrategySignal(has_signal=False, reason=reason)
        return state

    if state.latest_signal.has_signal:
        state.active_position = risk_manager.build_position(tick.symbol, state.latest_signal, state.risk_config)

    return state


@app.post("/api/close-position", response_model=BotState)
async def close_position(close_price: float) -> BotState:
    if state.active_position is None:
        raise HTTPException(status_code=400, detail="No active position")

    pnl = risk_manager.close_pnl(state.active_position, close_price)
    _register_close(pnl)
    state.active_position = None
    return state
