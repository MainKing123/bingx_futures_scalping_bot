"""One calendar-selected 2021 USD-M test; public archives and checksums only."""
import argparse
import asyncio
from pathlib import Path

import httpx
import pandas as pd
from app import external_data as old

START = pd.Timestamp('2021-01-01', tz='UTC')
END = pd.Timestamp('2022-01-01', tz='UTC')
MINUTE_WARMUP = pd.Timestamp('2020-12-01', tz='UTC')
DAILY_WARMUP = pd.Timestamp('2020-01-01', tz='UTC')

def validate(manifest, cache):
    if manifest.get('status') != 'complete' or len(manifest['archives']) != 72:
        raise old.ExternalDataError('Incomplete declared 72-archive snapshot')
    for symbol in old.SYMBOLS:
        for tf in ('1m','5m','1h','1d'):
            record = manifest['frames'][symbol][tf]
            begin = DAILY_WARMUP if tf == '1d' else MINUTE_WARMUP
            if record['bars'] != int((END-begin).total_seconds()/old.SECONDS[tf]) or record['missing_intervals']:
                raise old.ExternalDataError('Incomplete calendar coverage')
        if manifest['funding'][symbol]['records'] != 1095:
            raise old.ExternalDataError('Incomplete 2021 funding')
        for record in [*manifest['frames'][symbol].values(),manifest['funding'][symbol]]:
            if old.sha256((cache/record['filename']).read_bytes()) != record['sha256']:
                raise old.ExternalDataError('Frozen data changed')
    for record in manifest['archives'].values():
        path=cache/'archives'/record['filename']
        old.verify_checksum(path.read_bytes(),path.with_suffix('.zip.CHECKSUM').read_text(encoding='utf-8'),path.name)

async def fetch(cache, concurrency=4):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    target=cache/'snapshot.json'
    if target.exists():
        existing=__import__('json').loads(target.read_text(encoding='utf-8'))
        if existing.get('status')=='complete':
            validate(existing,cache);return existing
    manifest={'source':'Binance public USD-M monthly archives','source_url':old.SOURCE,
        'venue':'binance_usdm','status':'incomplete','symbols':list(old.SYMBOLS),
        'server_snapshot_utc':END.isoformat(),'snapshot_ms':int(END.timestamp()*1000),
        'outcomes_start_utc':START.isoformat(),'minute_warmup_start_utc':MINUTE_WARMUP.isoformat(),
        'warmup_start_utc':DAILY_WARMUP.isoformat(),'execution_resolution':'1m',
        'never_described_as_mexc_execution':True,'downloaded_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'frames':{},'funding':{},'archives':{}}
    old.atomic_json(target,manifest)
    semaphore=asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=90,follow_redirects=True) as client:
        for local,symbol in old.SYMBOLS.items():
            tasks=[old.download_archive(client,semaphore,cache,'klines',symbol,m,'1m') for m in old.months(MINUTE_WARMUP,END)]
            tasks += [old.download_archive(client,semaphore,cache,'klines',symbol,m,'1d') for m in old.months(DAILY_WARMUP,MINUTE_WARMUP)]
            tasks += [old.download_archive(client,semaphore,cache,'fundingRate',symbol,m) for m in old.months(START,END)]
            parts={'1m':[],'1d':[],'funding':[]}
            for kind,tf,month,csv,record in await asyncio.gather(*tasks):
                begin=pd.Timestamp(month+'-01',tz='UTC');end=begin+pd.offsets.MonthBegin(1)
                manifest['archives'][record['filename']]=record
                parts['funding' if kind=='fundingRate' else tf].append(old.parse_funding(csv,begin,end) if kind=='fundingRate' else old.parse_klines(csv,tf,begin,end))
            minutes=pd.concat(parts['1m']).sort_index()
            manifest['frames'][local]={}
            for tf in ('1m','5m','1h','1d'):
                frame=old.aggregate_minutes(minutes,tf)
                if tf=='1d':frame=pd.concat([*parts['1d'],frame]).sort_index()
                if frame.index.has_duplicates:raise old.ExternalDataError('Duplicate warmup')
                path=cache/f'{local}_{tf}.csv';old.atomic_csv(path,frame)
                manifest['frames'][local][tf]={**old.metadata(frame,tf),'filename':path.name,'sha256':old.sha256(path.read_bytes()),
                    'aggregation':'observed M1' if tf=='1m' else 'complete M1 groups; D1 native warmup'}
            funding=pd.concat(parts['funding']).sort_index();path=cache/f'{local}_funding.csv'
            old.atomic_csv(path,funding.to_frame(),date_format='%Y-%m-%dT%H:%M:%S.%f%z')
            manifest['funding'][local]={'filename':path.name,'sha256':old.sha256(path.read_bytes()),'records':len(funding),
                'cadence_hours':8,'actual_funding_rates':True,'exact_settlement_milliseconds_preserved':True}
            old.atomic_json(target,manifest)
            print(f'{local}: {len(minutes)} observed minutes; {len(funding)} actual settlements',flush=True)
    manifest['status']='complete';validate(manifest,cache);old.atomic_json(target,manifest)
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',default='.cache/binance_v6_2021')
    asyncio.run(fetch(parser.parse_args().cache))
