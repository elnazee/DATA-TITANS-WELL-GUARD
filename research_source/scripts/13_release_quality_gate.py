"""Strict private research release checklist. Does NOT claim absence of reporting bias.

Fail closed when model and source audits have not been run on the authorized computer.
Only aggregate checks; never publishes source well data. Saves outputs/private_release_gate.json.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

REQUIRED_SOURCES=('daily_production.parquet','production_failures.parquet','well_jobs.parquet')
MODEL_FILES=('private_test_alerts.csv','private_historical_model.joblib','aggregate_model_metrics.json')


def read_json(p):
    if not p.is_file():return None
    try:return json.loads(p.read_text(encoding='utf-8'))
    except (ValueError,OSError):return None


def gate(data:Path,out:Path,model_script:Path):
    checks={}
    for name in REQUIRED_SOURCES:
        checks[f'source_{name}']=(data/name).is_file()
    for name in MODEL_FILES:
        checks[f'output_{name}']=(out/name).is_file()
    quality=read_json(out/'source_quality_gate.json')
    audit=read_json(out/'private_temporal_audit.json')
    metrics=read_json(out/'aggregate_model_metrics.json')
    checks['source_quality_report_present']=bool(quality)
    checks['temporal_record_replay_PASS']=bool(audit and audit.get('status')=='PASS' and audit.get('event_label_replay'))
    checks['model_test_metrics_present']=bool(metrics and 'model_ap' in metrics and 'model_roc_auc' in metrics)
    checks['known_training_script_present']=model_script.is_file()
    checks['test_split_has_two_classes']=bool(metrics and metrics.get('test_positive_well_weeks',0)>0 and
                                               metrics.get('test_well_weeks',0)>metrics.get('test_positive_well_weeks',-1))
    checks['denominators_consistent']=bool(metrics and all(
        (metrics.get(f'model_top{k}_event_coverage') or {}).get('eligible_events',0)==
        (metrics.get(f'recency_top{k}_event_coverage') or {}).get('eligible_events',-1)>0
        for k in (25,50,100)))
    status='PASS_TECHNICAL_PRECHECK' if all(checks.values()) else 'BLOCKED'
    return {'status':status, 'checks':checks,
            'training_script_sha256':hashlib.sha256(model_script.read_bytes()).hexdigest() if model_script.is_file() else None,
            'scope':'Files and temporal replay only. Does not prove absence of reporting/proxy leakage, calibration, causal loss reduction, or untouched test selection.',
            'policy':'Any metrics modified by source cleaning must be reconciled across slide deck, video, dashboard, and report.',
            'remaining_manual_signoffs':['model outcome coverage and scoring date availability',
                 'reporting regime change pre-2025',
                 'source variable units and ambiguous duplicate policy',
                 'reference test reuse and proposed policy tuning',
                 'Strathcona data confidentiality and contributor authorization',
                 'business benefits are hypothetical, not realized']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    p.add_argument('--model-script',type=Path,default=Path('scripts/02_failure_ranking.py'))
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    result=gate(a.data_dir,a.out_dir,a.model_script)
    target=a.out_dir/'private_release_gate.json'
    target.write_text(json.dumps(result,indent=2))
    print('WellGuard research release precheck:',result['status'])
    print('Checks:',len(result['checks']),'PASS:',sum(result['checks'].values()))
    print('The manual signoffs in',target,'remain required.')
    if result['status']!='PASS_TECHNICAL_PRECHECK':raise SystemExit(2)

if __name__=='__main__':main()
