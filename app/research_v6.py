"""Fixed 28-case V6/control research. Register all inputs before any new PnL."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import pandas as pd
from app import research_v5 as old
from app import research_suite as common
from app.strategy.volium_v5 import diagnose_volium_v5_from_df,V5Parameters
from app.strategy.volium_v6 import diagnose_volium_v6_from_df
from app.schemas.setup import TradeSetup

MODELS=('v6-current-leg','v5-equal-control')
BLOCKS=('mexc_five','binance_known','binance_2021')
CODE_FILES=tuple(dict.fromkeys((*old.CODE_FILES,'research_v6.py','strategy/volium_v6.py','runtime_v6.py',
    'external_data.py','external_data_v6.py')))
BOUNDARIES={
    'mexc_five':('2026-04-10T15:05Z','2026-07-09T15:05Z','2026-08-23T15:05Z','2026-10-04T15:20Z'),
    'binance_known':('2024-01-01T00:00Z','2025-01-01T00:00Z','2025-07-01T00:00Z','2026-01-01T00:00Z'),
    'binance_2021':('2021-01-01T00:00Z','2021-05-01T00:00Z','2021-09-01T00:00Z','2022-01-01T00:00Z')}

def bound_file(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':common.file_hash(path)}

def register(root,out_dir):
    root=Path(root).resolve();out_dir=Path(out_dir).resolve();target=out_dir/'research_plan.json'
    if target.exists():raise ValueError('Never overwrite registered V6')
    repo=Path(__file__).parent.parent;artifacts=root/'outputs'
    frozen=json.loads((artifacts/'research_v5_source_manifest.json').read_text(encoding='utf-8'))
    import hashlib,zipfile
    capsule=artifacts/'research_v5_source.zip'
    if common.file_hash(capsule)!=frozen['sha256']:raise ValueError('V5 capsule changed')
    with zipfile.ZipFile(capsule) as archive:
        for relative in old.CODE_FILES:
            expected=hashlib.sha256(archive.read('mexc-volium/app/'+relative)).hexdigest()
            if common.file_hash(repo/'app'/relative)!=expected:raise ValueError('Frozen V5 changed: '+relative)
    prior=[];count=255
    for venue in ('mexc','binance'):
        for version in ('','_v3','_v4','_v5'):
            directory=artifacts/f'research_{venue}{version}'
            prior.append(bound_file(directory/'research_plan.json'))
            for phase in ('train','validation','final','robustness'):
                path=directory/f'{phase}_results.json'
                if path.exists():
                    payload=json.loads(path.read_text(encoding='utf-8'));assert payload.get('completed_at_utc')
                    count+=len(payload['results']);prior.append(bound_file(path))
    supplement=artifacts/'research_v5_five_pair_power/supplement_results.json'
    payload=json.loads(supplement.read_text(encoding='utf-8'));assert payload.get('completed_at_utc')
    count+=len(payload['results']);prior += [bound_file(supplement),bound_file(artifacts/'research_v5_five_pair_power/supplement_plan.json')]
    if count!=698:raise ValueError(f'Prior ledger differs: {count}')
    specs={}
    for block in BLOCKS:
        venue='mexc' if block=='mexc_five' else 'binance_usdm';resolution='5m' if venue=='mexc' else '1m'
        symbols=old.ROBUST_FIVE if venue=='mexc' else old.CORE
        cache=repo/'.cache'/({'mexc_five':'suite','binance_known':'binance_core','binance_2021':'binance_v6_2021'}[block])
        folds={phase:{'start_utc':a,'end_utc':b} for phase,a,b in zip(('train','validation','test'),BOUNDARIES[block],BOUNDARIES[block][1:])}
        spec=old.data_spec(cache,symbols,['intraday'],resolution,folds)
        path=artifacts/('research_v6_contracts.json' if venue=='mexc' else 'research_v6_binance_contracts.json')
        metadata=json.loads(path.read_text(encoding='utf-8'))
        contracts={(r['symbol'][:-4]+'_USDT' if venue!='mexc' else r['symbol']):r for r in metadata['records']}
        assert set(symbols)<=set(contracts)
        specs[block]={**spec,'symbols':symbols,'contracts':contracts,'contract_snapshot':bound_file(path),
            'known_before_v6':block!='binance_2021','external_whole_window':{'start_utc':BOUNDARIES[block][0],'end_utc':BOUNDARIES[block][-1]} if block=='binance_2021' else None}
    plan={'version':6,'runner':'app.research_v6','registered_at_utc':old.now_utc(),'blocks':specs,'models':MODELS,
        'budget':28,'prior_stage_trials':698,'cumulative_stage_trials':726,'old_shared_wallet_runs_separate':2,
        'policy':{**old.POLICY,'finite_joint_budget':{'known_folds':24,'external':4,'total':28},'prior_stage_trials':698,
            'external_min_baseline_closed':30,'no_profit_parameter_changes':True},
        'code_sha256':{name:common.file_hash(repo/'app'/name) for name in CODE_FILES},
        'inputs':[bound_file(repo/'docs/strategy_v6.md'),bound_file(repo/'docs/research_v6_protocol.md'),
            bound_file(artifacts/'strategy_v6_source_review.json'),bound_file(artifacts/'strategy_v6_visual_review.json'),
            bound_file(capsule),bound_file(artifacts/'research_v5_source_manifest.json')],
        'prior_completed_files':prior,'runtime_settings':old.runtime_settings('intraday').model_dump(mode='json'),
        'history_bars':old.history_requirements('intraday')}
    plan['plan_sha256']=common.digest(plan);common.atomic_json(target,plan)
    (out_dir/'registered_protocol.md').write_bytes((repo/'docs/research_v6_protocol.md').read_bytes())
    print('Registered all 28 fixed cases: '+plan['plan_sha256'],flush=True)
    return plan

def verify(plan):
    common.verify_plan(plan)
    for record in [*plan['inputs'],*plan['prior_completed_files'],*(spec['contract_snapshot'] for spec in plan['blocks'].values())]:
        if common.file_hash(record['path'])!=record['sha256']:raise ValueError('Frozen V6/prior input changed: '+record['path'])
    for spec in plan['blocks'].values():
        cache=Path(spec['cache_path'])
        if common.file_hash(cache/'snapshot.json')!=spec['data_manifest_sha256']:raise ValueError('Manifest changed')
        for name,expected in spec['data_sha256'].items():
            if common.file_hash(cache/name)!=expected:raise ValueError('CSV changed: '+name)

def precompute(frames,symbols,settings,window,history):
    tables={model:{} for model in MODELS};reasons={model:Counter() for model in MODELS}
    begin,end=common.utc(window['start_utc']),common.utc(window['end_utc'])
    for symbol in symbols:
        indices={tf:frames[symbol][tf].index+pd.Timedelta(seconds=old.SECONDS[tf]) for tf in ('1d','1h','5m')}
        times=indices['5m'];times=times[(times>begin)&(times<=end)]
        for index,now in enumerate(times,1):
            if index%20000==0:print(f'V6 source {symbol} {index}/{len(times)}',flush=True)
            if not old.in_volium_session(now,settings):continue
            known={tf:frames[symbol][tf].iloc[:indices[tf].searchsorted(now,side='right')].tail(history[tf]) for tf in indices}
            kwargs={'symbol':symbol,'frames':known,'settings':settings,'mode':'intraday','now':now.to_pydatetime()}
            for model,fn in zip(MODELS,(diagnose_volium_v6_from_df,diagnose_volium_v5_from_df)):
                result=fn(**kwargs) if model==MODELS[0] else fn(**kwargs,parameters=V5Parameters(liquidity_mode='equal_clusters'))
                reasons[model][result.reason]+=1
                if result.setup:
                    if result.setup.symbol!=symbol or common.utc(result.setup.timestamp)!=now:raise ValueError('Noncausal provider timestamp')
                    tables[model][(symbol,now.value)]=result.setup.model_dump(mode='json')
    return tables,{model:dict(value) for model,value in reasons.items()}

def run_group(out_dir,block,phase):
    out_dir=Path(out_dir);plan=json.loads((out_dir/'research_plan.json').read_text(encoding='utf-8'));verify(plan)
    spec=plan['blocks'][block];window=spec['external_whole_window'] if phase=='external' else spec['folds']['test' if phase=='final' else phase]
    frames,funding=common.read_dataset(spec,end_at=window['end_utc'])
    settings=old.RuntimeSettings(_env_file=None,**plan['runtime_settings'])
    tables,funnel=precompute(frames,spec['symbols'],settings,window,plan['history_bars'])
    target=out_dir/f'{block}_{phase}_results.json'
    payload={'plan_sha256':plan['plan_sha256'],'block':block,'phase':phase,'source_venue':spec['source_venue'],
        'started_at_utc':old.now_utc(),'window':window,'signal_funnel':funnel,'results':[]}
    for model in MODELS:
        signals=tables[model]
        def provider(**kwargs):
            value=signals.get((kwargs['symbol'],common.utc(kwargs['now']).value))
            return TradeSetup.model_validate(value) if value is not None else None
        schedule={symbol:[pd.Timestamp(nanos,tz='UTC') for key,nanos in signals if key==symbol] for symbol in spec['symbols']}
        for stress in (False,True):
            started=time.monotonic()
            admission=old.mexc_admission(spec['contracts']) if spec['source_venue']=='mexc' else old.binance_proxy_admission(spec['contracts'])
            result=old.portfolio_replay_v5(frames,old.runtime_settings('intraday',stress=stress),symbols=spec['symbols'],
                start_at=window['start_utc'],end_at=window['end_utc'],funding_rates=funding,signal_provider=provider,
                execution_timeframe=spec['execution_resolution'],signal_schedule=schedule,admission=admission,history_bars=plan['history_bars'])
            result.update(candidate={'id':model,'category':'fixed_source_model','engine':'v5'},phase=phase,block=block,
                market='core_plus_top3' if block=='mexc_five' else 'crypto_core',source_venue=spec['source_venue'],
                cost_variant='stress' if stress else 'baseline',requested_window=window,elapsed_seconds=round(time.monotonic()-started,3))
            payload['results'].append(common.annotate(result));common.atomic_json(target,payload)
            print(f'{block} {phase} {model} {result["cost_variant"]}: n={result["trades_count"]}, TP={result["tp_hits"]}, net={result["net_pnl_usdt"]:.4f}',flush=True)
    verify(plan);payload['completed_at_utc']=old.now_utc();common.atomic_json(target,payload);return payload

def identities(phase):
    return {(block,phase,model,cost) for block in (('binance_2021',) if phase=='external' else BLOCKS[:2]) for model in MODELS for cost in ('baseline','stress')}

def collect(directory,phase,plan):
    rows=[]
    for block in (('binance_2021',) if phase=='external' else BLOCKS[:2]):
        value=json.loads((Path(directory)/f'{block}_{phase}_results.json').read_text(encoding='utf-8'))
        if value['plan_sha256']!=plan['plan_sha256'] or not value.get('completed_at_utc'):raise ValueError('Incomplete group')
        rows+=value['results']
    actual=[(r['block'],r['phase'],r['candidate']['id'],r['cost_variant']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=identities(phase):raise ValueError('Fixed budget mismatch')
    return rows

def run_phase(directory,phase,workers=2):
    directory=Path(directory);plan=json.loads((directory/'research_plan.json').read_text(encoding='utf-8'));verify(plan)
    prerequisites=() if phase=='train' else ('train',) if phase=='validation' else ('train','validation') if phase=='final' else ('train','validation','final')
    previous=[]
    for earlier in prerequisites:
        collect(directory,earlier,plan)
        previous += [bound_file(directory/f'{block}_{earlier}_results.json') for block in BLOCKS[:2]]
    if phase in ('final','external'):
        marker=directory/f'{phase}_opened.json'
        if marker.exists():
            recorded=json.loads(marker.read_text(encoding='utf-8'))
            if recorded['plan_sha256']!=plan['plan_sha256'] or recorded['earlier_results']!=previous:raise ValueError('Opening marker changed')
        else:common.atomic_json(marker,{'plan_sha256':plan['plan_sha256'],'opened_at_utc':old.now_utc(),'earlier_results':previous})
    jobs=[]
    for block in (('binance_2021',) if phase=='external' else BLOCKS[:2]):
        path=directory/f'{block}_{phase}_results.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('completed_at_utc'):continue
        jobs.append(block)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_group,str(directory),block,phase) for block in jobs]
            for future in futures:future.result()
    collect(directory,phase,plan)
    print(f'Completed V6 {phase}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','train','validation','final','external'))
    parser.add_argument('--root',required=True);parser.add_argument('--out-dir',required=True);args=parser.parse_args()
    if args.phase=='register':register(args.root,args.out_dir)
    else:run_phase(args.out_dir,args.phase)

if __name__=='__main__':main()
