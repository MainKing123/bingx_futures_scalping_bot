import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
from app.runtime_v6 import RuntimeV6Settings
from app.risk.risk_manager import RiskManager
from app.scanner.scanner import SignalScanner
from app.strategy import volium_v6
from app.api.router import status
from test_volium_v6 import frames,NOW

def test_scanner_calls_actual_v6_with_the_registered_prefix():
    async def run():
        data=frames();settings=RuntimeV6Settings(_env_file=None,volium_session_enabled=False,pair_selection='fixed')
        async def candles(symbol,tf,limit):return data[tf]
        client=SimpleNamespace(get_klines=AsyncMock(side_effect=candles))
        scanner=SignalScanner(client,settings,RiskManager(settings),SimpleNamespace(),None)
        scanner._publish_setup=AsyncMock(return_value=True)
        await scanner._scan_symbol('BTC_USDT','intraday',NOW.to_pydatetime())
        scanner._publish_setup.assert_awaited_once()
        idea=scanner._publish_setup.await_args.args[0]
        assert idea.timestamp==NOW and idea.htf_bias=='BULLISH'
        assert idea.confluences[0]=='Experimental v6; current liquidity-to-liquidity context'
        requests={call.args[1]:call.kwargs['limit'] for call in client.get_klines.await_args_list}
        assert requests=={'5m':991,'1d':110,'1h':110}
        assert scanner.last_errors=={}
    asyncio.run(run())

def test_v6_status_is_experimental_paper_and_exposes_current_profile():
    async def run():
        settings=RuntimeV6Settings(_env_file=None);risk=RiskManager(settings)
        scanner=SimpleNamespace(active_symbols=['BTC_USDT'],selection_error=None,last_scan_at=None,last_errors={},last_rejections={},selected_pairs=[])
        request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings,risk_manager=risk,
            scanner=scanner,tracker=SimpleNamespace(marked_equity=1000))))
        result=await status(request)
        assert result['strategy_profile']=='v6_current_leg'
        assert result['strategy_is_experimental'] and result['mode']=='paper'
        assert result['leverage_cap']==50 and result['risk_per_trade_percent']==.5
    asyncio.run(run())
