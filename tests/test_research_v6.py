import json
from pathlib import Path

import pytest
from app import research_v6 as study

def test_exact_budget_has_no_grid_or_profit_selection():
    all_cases=set.union(*(study.identities(p) for p in ('train','validation','final','external')))
    assert len(all_cases)==28
    assert {r[2] for r in all_cases}=={'v6-current-leg','v5-equal-control'}
    assert study.BOUNDARIES['binance_2021'][0]=='2021-01-01T00:00Z'
    assert study.BOUNDARIES['binance_2021'][-1]=='2022-01-01T00:00Z'

def write_group(tmp,block,phase,rows):
    study.common.atomic_json(tmp/f'{block}_{phase}_results.json',{'plan_sha256':'frozen','completed_at_utc':'2026-10-05','results':rows})

def row(identity):
    block,phase,model,cost=identity
    return {'block':block,'phase':phase,'candidate':{'id':model},'cost_variant':cost}

def test_complete_collection_rejects_missing_and_duplicate_trials(tmp_path):
    plan={'plan_sha256':'frozen'}
    expected=study.identities('final')
    for block in study.BLOCKS[:2]:write_group(tmp_path,block,'final',[row(k) for k in sorted(expected) if k[0]==block])
    assert len(study.collect(tmp_path,'final',plan))==8
    block='mexc_five';path=tmp_path/f'{block}_final_results.json';payload=json.loads(path.read_text(encoding='utf-8'))
    payload['results'].pop();study.common.atomic_json(path,payload)
    with pytest.raises(ValueError,match='budget'):study.collect(tmp_path,'final',plan)
    payload['results'].append(payload['results'][0]);study.common.atomic_json(path,payload)
    with pytest.raises(ValueError,match='budget'):study.collect(tmp_path,'final',plan)

def test_mutated_registered_code_is_rejected(tmp_path):
    plan={'code_sha256':{'strategy/volium_v6.py':'0'*64},'inputs':[],'prior_completed_files':[],'blocks':{}}
    plan['plan_sha256']=study.common.digest(plan)
    with pytest.raises(ValueError,match='source changed'):study.verify(plan)

def test_prerequisite_failure_cannot_open_final(tmp_path,monkeypatch):
    plan={'plan_sha256':'frozen'};study.common.atomic_json(tmp_path/'research_plan.json',plan)
    monkeypatch.setattr(study,'verify',lambda p:None)
    with pytest.raises(FileNotFoundError):study.run_phase(tmp_path,'final')
    assert not (tmp_path/'final_opened.json').exists()

def test_finished_group_cannot_be_rerun_as_new_outcome(tmp_path,monkeypatch):
    plan={'plan_sha256':'frozen'};study.common.atomic_json(tmp_path/'research_plan.json',plan)
    monkeypatch.setattr(study,'verify',lambda p:None)
    for block in study.BLOCKS[:2]:write_group(tmp_path,block,'train',[row(k) for k in study.identities('train') if k[0]==block])
    def unexpected(*args):raise AssertionError('Completed source group rerun')
    monkeypatch.setattr(study,'run_group',unexpected)
    study.run_phase(tmp_path,'train')

def test_registered_unicode_contract_names_round_trip_in_actual_runner(tmp_path):
    plan={'public_contract_label':'BTC 永续','code_sha256':{},'inputs':[],'prior_completed_files':[],'blocks':{}}
    plan['plan_sha256']=study.common.digest(plan)
    study.common.atomic_json(tmp_path/'research_plan.json',plan)
    for block in study.BLOCKS[:2]:
        payload={'plan_sha256':plan['plan_sha256'],'completed_at_utc':'2026-10-05',
                 'results':[row(k) for k in study.identities('train') if k[0]==block]}
        study.common.atomic_json(tmp_path/f'{block}_train_results.json',payload)
    study.run_phase(tmp_path,'train')
