"""Source contracts and counterexamples, not historical profitability tests."""
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from app.runtime_settings import RuntimeSettings
from app.runtime_v6 import RuntimeV6Settings
from app.strategy import volium as v1
from app.strategy.volium_v6 import V6Parameters, current_liquidity_leg, diagnose_volium_v6_from_df, manipulation_origin

NOW = pd.Timestamp('2026-10-05T07:15Z')
ROWS = [
    (110,111,108,109),(109,113,107,112),(112,120,109,118),
    (118,119,103,105),(105,108,98,102),(102,109,100,107),
    (107,112,101,110),(110,116,106,114),(114,115,106,108),
    (108,112,104,106),(106,110,101,104),(104,105,94,96),
    (96,111,95,110),(110,113,106,112)]

def frame(rows, end, frequency):
    return pd.DataFrame(rows, columns=['open','high','low','close'], dtype=float,
                        index=pd.date_range(end=end, periods=len(rows), freq=frequency, tz='UTC'))

def mirror(df):
    return pd.DataFrame({'open':250-df.open, 'high':250-df.low,
                         'low':250-df.high, 'close':250-df.close}, index=df.index)

def frames(short=False):
    daily = list(ROWS)
    daily[2]=(112,150,109,145); daily[3]=(145,149,103,105)
    daily[7]=(110,140,106,137); daily[8]=(137,138,106,108)
    hourly=list(ROWS)
    hourly[2]=(112,130,109,128); hourly[3]=(128,129,103,105)
    hourly[7]=(110,125,106,124); hourly[8]=(124,124,106,108)
    hourly[12]=(96,121,95,120); hourly[13]=(120,124,110,117)
    hourly.extend([(117,123,115,121),(121,124,114,117),(117,120,110,115),
                   (115,125,114,123),(123,130,116,128),(128,129,117,124.5),(124.5,125,118,123.5)])
    warm=frame([(117.8,118.4,117.4,118)]*16, '2026-10-05 06:50','5min')
    warm.loc[pd.Timestamp('2026-10-05T06:30Z'),'high']=120.5
    signal=frame([(117,119,116,118),(118,119,113,114),(114,115,109.5,112),(112,117.2,111.8,117)],
                 '2026-10-05 07:10','5min')
    result={'1d':frame(daily,'2026-10-04','D'), '1h':frame(hourly,'2026-10-05 06:00','h'),
            '5m':pd.concat([warm,signal])}
    return {key:mirror(df) for key,df in result.items()} if short else result

def diagnose(data=None, **kwargs):
    return diagnose_volium_v6_from_df(symbol='BTC_USDT',frames=frames() if data is None else data,
                                      now=NOW,enforce_session_filter=False,**kwargs)

@pytest.mark.parametrize('short',[False,True])
def test_a_reaction_is_usable_before_right_hand_fractal_confirmation(short):
    daily=frame(ROWS[:13],'2026-10-04','D')
    daily=mirror(daily) if short else daily
    assert not any(p.index==11 for p in v1._pivots(daily,2))
    leg=current_liquidity_leg(daily,'1d')
    assert leg is not None and leg.direction==('SHORT' if short else 'LONG')
    assert leg.extreme==pytest.approx(156 if short else 94)
    assert leg.target==pytest.approx(134 if short else 116)
    assert pd.Timestamp(leg.reaction_close_utc)<=daily.index[-1]+pd.Timedelta(days=1)

@pytest.mark.parametrize('short',[False,True])
def test_full_entry_has_independent_manipulation_origin_and_gross_2r(short):
    result=diagnose(frames(short))
    assert result.setup is not None, result.as_dict()
    assert result.features['manipulation_origin']==pytest.approx(129.5 if short else 120.5)
    assert result.features['target']==pytest.approx(120 if short else 130)
    assert result.features['manipulation_origin']!=result.features['target']
    assert result.features['recovery_fraction']==pytest.approx(7.5/11)
    s=result.setup
    assert abs(s.take_profits[0]-s.entry)/abs(s.entry-s.stop_loss)==pytest.approx(2)
    assert s.timestamp==NOW.to_pydatetime()

@pytest.mark.parametrize('short',[False,True])
def test_h1_flow_can_continue_beyond_local_targets_toward_daily_b(short):
    data=frames(short)
    leg=current_liquidity_leg(data['1h'],'1h')
    assert leg is not None and leg.direction==('SHORT' if short else 'LONG')
    assert leg.target is None
    assert diagnose(data).setup is not None

@pytest.mark.parametrize('short',[False,True])
def test_wick_break_of_protected_daily_a_invalidates_current_context(short):
    data=frames(short)
    col='high' if short else 'low'
    data['1h'].iloc[-1,data['1h'].columns.get_loc(col)]=157 if short else 93
    result=diagnose(data)
    assert result.setup is None and result.reason=='current_context_extreme_broken'

@pytest.mark.parametrize('short',[False,True])
def test_partial_daily_target_visit_invalidates_before_daily_close(short):
    data=frames(short)
    col='low' if short else 'high'
    data['1h'].iloc[-1,data['1h'].columns.get_loc(col)]=109 if short else 141
    result=diagnose(data)
    assert result.setup is None and result.reason=='current_context_target_reached'

@pytest.mark.parametrize('short',[False,True])
def test_future_candles_and_their_extremes_cannot_change_a_known_signal(short):
    data=frames(short); before=diagnose(data).as_dict()
    assert before['accepted']
    for tf,step in [('1d','D'),('1h','h'),('5m','5min')]:
        extra=frame([(120,240,1,230)]*4,data[tf].index[-1]+4*pd.Timedelta(seconds=v1._DURATIONS[tf]),step)
        data[tf]=pd.concat([data[tf],extra])
    assert diagnose(data).as_dict()==before

def test_sweep_without_reaction_cannot_supply_an_a():
    data=frame(ROWS[:12],'2026-10-04','D')
    assert current_liquidity_leg(data,'1d') is None

def test_target_reached_does_not_resurrect_an_older_confirmed_leg():
    daily=frame(ROWS,'2026-10-04','D')
    assert current_liquidity_leg(daily,'1d') is not None
    extra=frame([(112,117,109,114)],'2026-10-05','D')
    assert current_liquidity_leg(pd.concat([daily,extra]),'1d') is None

def test_ambiguous_outside_bar_is_not_a_directional_origin():
    rows=list(ROWS[:12]); rows[11]=(104,121,94,105)
    assert current_liquidity_leg(frame(rows,'2026-10-04','D'),'1d') is None

def test_missing_higher_timeframe_bar_rejects_context():
    daily=frame(ROWS,'2026-10-04','D').drop(pd.Timestamp('2026-09-30T00:00Z'))
    assert current_liquidity_leg(daily,'1d') is None

def test_forming_m5_peak_cannot_be_used_as_manipulation_origin():
    data=frames()['5m'].iloc[:-2]
    p=manipulation_origin(data,'LONG')
    assert p.price==120.5
    assert data.index[p.known_index]+pd.Timedelta(minutes=5)<=data.index[-1]+pd.Timedelta(minutes=5)

@pytest.mark.parametrize('mode',['scalp','swing'])
def test_unvalidated_mode_is_rejected(mode):
    with pytest.raises(ValueError): diagnose(mode=mode)

@pytest.mark.parametrize('changes',[{'liquidity_mode':'strict'},{'recovery_fraction':.3},
                                    {'max_wait_bars':10},{'max_wait_bars':12.0}])
def test_v6_has_one_fixed_parameter_set(changes):
    with pytest.raises(ValueError): V6Parameters(**changes)

def test_parameters_cannot_mutate_after_registration():
    with pytest.raises(FrozenInstanceError): V6Parameters().max_wait_bars=10

@pytest.mark.parametrize('values',[{'auto_execution':True},{'volium_mode':'scalp'},{'volium_mode':'swing'}])
def test_runtime_v6_cannot_enable_live_or_other_modes(values):
    with pytest.raises(ValueError): RuntimeV6Settings(_env_file=None,**values)

def test_financial_policy_is_unchanged_and_historical_profile_remains_available():
    old=RuntimeSettings(_env_file=None).model_dump()
    new=RuntimeV6Settings(_env_file=None).model_dump()
    assert new.pop('volium_strategy_profile')=='v6_current_leg'
    old.pop('volium_strategy_profile')
    assert new==old
    assert RuntimeV6Settings(_env_file=None,volium_strategy_profile='v5_equal').volium_strategy_profile=='v5_equal'
