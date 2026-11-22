"""WellGuard V8.7: read-only private, record-level temporal integrity checker.

Run on the authorized Windows machine after model scoring:
    python scripts/12_audit_data_leakage.py --data-dir data --out-dir outputs

Only writes an aggregate status file to outputs/. No private identifiers are printed.
Does not retrain the model or claim that label/reporting bias is absent.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_END=pd.Timestamp('2026-03-01')
VAL_START=pd.Timestamp('2026-03-22')
VAL_END=pd.Timestamp('2026-05-17')
TEST_START=pd.Timestamp('2026-06-07')
TEST_END=pd.Timestamp('2026-10-04')
HORIZON_DAYS=14


def temporal_split_checks():
    assert (VAL_START - TRAIN_END).days > HORIZON_DAYS, 'Training labels overlap validation'
    assert (TEST_START - VAL_END).days > HORIZON_DAYS, 'Validation labels overlap testing'
    return {'training_max_snapshot':'2026-03-01','validation_min_snapshot':'2026-03-22',
            'validation_max_snapshot':'2026-05-17','testing_min_snapshot':'2026-06-07',
            'testing_max_snapshot':'2026-10-04','label_horizon_days':HORIZON_DAYS,
            'development_gap_days':(VAL_START-TRAIN_END).days,
            'holdout_gap_days':(TEST_START-VAL_END).days}


def check_labels_and_past_history(test:pd.DataFrame, failures:pd.DataFrame):
    t=test.copy()
    for col in ['uwi','week_end','target_14d','failure_in_previous_90d','log_days_since_failure']:
        if col not in t: raise ValueError(f'Missing {col} from test alerts')
    t['week_end']=pd.to_datetime(t.week_end,errors='raise')
    t['uwi']=t.uwi.astype(str)
    if t[['uwi','week_end']].duplicated().any():raise ValueError('Duplicate well-weeks in test alerts')
    if not t.week_end.between(TEST_START,TEST_END).all():raise ValueError('Unexpected test week outside locked time split')
    if (t.week_end.dt.dayofweek!=6).any():raise ValueError('Snapshot not on Sunday')
    f=failures[['uwi','failure_start_date']].dropna().drop_duplicates().copy()
    f['uwi']=f.uwi.astype(str)
    f['failure_start_date']=pd.to_datetime(f.failure_start_date,errors='raise')
    f=f.sort_values('failure_start_date')
    ordered=t[['uwi','week_end']].sort_values('week_end')
    nxt=pd.merge_asof(ordered,f,left_on='week_end',right_on='failure_start_date',by='uwi',direction='forward',allow_exact_matches=False)
    prv=pd.merge_asof(ordered,f,left_on='week_end',right_on='failure_start_date',by='uwi',direction='backward',allow_exact_matches=True)
    k=ordered.assign(target_expected=(nxt.failure_start_date-ordered.week_end).dt.days.between(1,HORIZON_DAYS).astype(int).to_numpy(),
                     past_days=(ordered.week_end-prv.failure_start_date).dt.days.to_numpy())
    m=t.merge(k,on=['uwi','week_end'],how='left',validate='one_to_one')
    ybad=int((m.target_14d.astype(int)!=m.target_expected.astype(int)).sum())
    prev90=m.past_days.between(0,90).astype(int)
    prevbad=int((m.failure_in_previous_90d.astype(int)!=prev90).sum())
    days=np.log1p(m.past_days.clip(lower=0,upper=365)).fillna(np.log1p(365))
    prevlogbad=int((~np.isclose(m.log_days_since_failure.astype(float), days.astype(float),atol=1e-6,rtol=1e-6)).sum())
    if any([ybad,prevbad,prevlogbad]):
        raise AssertionError(f'Chronology mismatch: target={ybad}, last90={prevbad}, recency_log={prevlogbad}')
    return {'test_well_weeks':len(t),'test_positive_well_weeks':int(t.target_14d.sum()),
            'target_disagreements':ybad,'past_failure_indicator_disagreements':prevbad,
            'past_failure_recency_disagreements':prevlogbad}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    a=p.parse_args()
    checks={'scope':'WellGuard temporal leakage and event labeling; NOT an inspection of reporting-completeness or causal effects',
            'time_splits':temporal_split_checks(),
            'model_leakage_status':'PENDING PRIVATE TEST-ALERT REPLAY'}
    source=a.out_dir/'private_test_alerts.csv'
    failures=a.data_dir/'production_failures.parquet'
    if not source.is_file() or not failures.is_file():
        checks['reason']='Source Parquet and model-generated private_test_alerts.csv are needed for record-level replay.'
        status='INCOMPLETE'
    else:
        test=pd.read_csv(source)
        f=pd.read_parquet(failures,columns=['uwi','failure_start_date'])
        checks['event_label_replay']=check_labels_and_past_history(test,f)
        status='PASS'
        checks['model_leakage_status']='AS-OF LABEL AND FAILURE-HISTORY REPLAY PASSED; does not establish no reporting/proxy leakage'
    checks['status']=status
    a.out_dir.mkdir(parents=True,exist_ok=True)
    target=a.out_dir/'private_temporal_audit.json'
    target.write_text(json.dumps(checks,indent=2))
    print('Saved PRIVATE audit:',target,'status:',status)
    if status!='PASS': raise SystemExit(2)

if __name__=='__main__':main()
