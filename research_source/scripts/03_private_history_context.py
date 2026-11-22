"""Add *past-only* maintenance / failure context to a private alert replay.

This joins ALL 3 competition datasets in the actual application workflow.
The context is for operator review only; maintenance jobs are NOT used as
prediction features. Never commit the output CSV to GitHub.

python scripts/03_private_history_context.py --data-dir data --out-dir outputs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    p.add_argument('--jobs-pickle',type=Path,help='Internal validation only')
    p.add_argument('--failures-pickle',type=Path,help='Internal validation only')
    args=p.parse_args()
    base=args.out_dir/'private_test_alerts.csv'
    if not base.exists():raise SystemExit(f'Run 02_failure_ranking.py first; missing {base}')
    a=pd.read_csv(base,parse_dates=['week_end'])
    j=pd.read_pickle(args.jobs_pickle) if args.jobs_pickle else pd.read_parquet(args.data_dir/'well_jobs.parquet',columns=['uwi','start_date','job_type','field_est_total'])
    f=pd.read_pickle(args.failures_pickle) if args.failures_pickle else pd.read_parquet(args.data_dir/'production_failures.parquet',columns=['uwi','failure_start_date','failure_type'])
    j=j[['uwi','start_date','job_type','field_est_total']].copy();j['start_date']=pd.to_datetime(j.start_date)
    j=j[j.start_date.notna()].sort_values(['start_date','uwi','job_type']).drop_duplicates(['uwi','start_date'],keep='last')
    f=f[['uwi','failure_start_date','failure_type']].copy();f['failure_start_date']=pd.to_datetime(f.failure_start_date)
    f=f[f.failure_start_date.notna()].sort_values(['failure_start_date','uwi','failure_type']).drop_duplicates(['uwi','failure_start_date'],keep='last')
    a['_original_order']=np.arange(len(a));a=a.sort_values('week_end').reset_index(drop=True)
    a=pd.merge_asof(a,j.sort_values('start_date'),by='uwi',left_on='week_end',right_on='start_date',direction='backward')
    a=pd.merge_asof(a,f.sort_values('failure_start_date'),by='uwi',left_on='week_end',right_on='failure_start_date',direction='backward')
    a['days_since_last_job']=(a.week_end-a.start_date).dt.days
    a['days_since_last_recorded_failure']=(a.week_end-a.failure_start_date).dt.days
    a['prior_jobs_90d']=0
    jby={well:g.start_date.to_numpy(dtype='datetime64[ns]') for well,g in j.groupby('uwi')}
    for well,indices in a.groupby('uwi').groups.items():
        js=jby.get(well)
        if js is None:continue
        end=a.loc[indices,'week_end'].to_numpy(dtype='datetime64[ns]')
        start=end-np.timedelta64(90,'D')
        ct=np.searchsorted(js,end,side='right')-np.searchsorted(js,start,side='left')
        a.loc[indices,'prior_jobs_90d']=ct
    a=a.sort_values('_original_order').drop(columns=['_original_order'])
    a=a.rename(columns={'start_date':'last_job_date','job_type':'last_job_type',
                         'field_est_total':'last_job_field_estimated_cost_cad',
                         'failure_start_date':'last_recorded_failure_date',
                         'failure_type':'last_recorded_failure_type'})
    out=args.out_dir/'private_test_with_context.csv';a.to_csv(out,index=False)
    print('PRIVATE context written:',out, 'rows',len(a))
    print('Check: no future jobs:',int((a.last_job_date>a.week_end).sum()),
          'no future failures:',int((a.last_recorded_failure_date>a.week_end).sum()))
    assert not (a.last_job_date>a.week_end).any()
    assert not (a.last_recorded_failure_date>a.week_end).any()

if __name__=='__main__':main()
