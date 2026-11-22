"""Recompute fair continuous comparator, cohort stability, and scheduling sensitivity.

Use only on an authorized computer with private source outputs. Creates AGGREGATE JSON.
The policy comparisons are explicitly POST-HOC on the original test period.
Example: python scripts/05_advanced_validation.py --out-dir outputs --data-dir data
"""
from __future__ import annotations
import sys,argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.scheduling import schedule_inspections,event_recall


def validate(d:pd.DataFrame, f:pd.DataFrame):
    d=d.copy()
    d['week_end']=pd.to_datetime(d['week_end'])
    d['recency_score']=-pd.to_numeric(d['log_days_since_failure'])
    d['historical_binary_score']=pd.to_numeric(d['failure_in_previous_90d'])
    y=d.target_14d.astype(int)
    base=float(average_precision_score(y,d.recency_score))
    auc=float(roc_auc_score(y,d.recency_score))
    model_ap=float(average_precision_score(y,d.relative_risk_score))
    eligible=event_recall(d,d,f)
    results={'validation_type':'Retrospective, test-explored policy, NOT independent policy validation',
             'holdout_observations':len(d),'test_weeks':int(d.week_end.nunique()),
             'test_positive_rate':float(y.mean()),'model_ap':model_ap,
             'recency_baseline_ap':base,'recency_baseline_roc_auc':auc,
             'event_denominator':eligible['eligible_events'],
             'policies':{},'month_stability':{}}
    for cap in (25,50,100):
        for label,score,cool in [('model_standard','relative_risk_score',0),('continuous_recency','recency_score',0),
                                 ('model_1w_cooldown','relative_risk_score',1),('model_2w_cooldown','relative_risk_score',2)]:
            queue=schedule_inspections(d,capacity=cap,cooldown_weeks=cool,score_col=score)
            r=event_recall(d,queue,f)
            results['policies'][f'{label}_top{cap}']={**r,'slots':len(queue),'unique_wells':int(queue.uwi.nunique()),
                        'positive_well_week_precision':float(queue.target_14d.mean())}
    for mo,g in d.groupby(d.week_end.dt.to_period('M')):
        if g.target_14d.nunique()==2:
            results['month_stability'][str(mo)]={'observations':len(g),'positive_rate':float(g.target_14d.mean()),
                       'model_ap':float(average_precision_score(g.target_14d,g.relative_risk_score)),
                       'baseline_ap':float(average_precision_score(g.target_14d,g.recency_score))}
    # Cluster resampling of weeks (18 is small): descriptive sensitivity, not prospective certainty.
    rng=np.random.default_rng(2026)
    weeks=sorted(d.week_end.unique())
    weekparts=[d[d.week_end==wk] for wk in weeks]
    diffs=[]
    for _ in range(750):
        ixs=rng.integers(0,len(weekparts),len(weekparts))
        b=pd.concat([weekparts[i] for i in ixs],ignore_index=True)
        if b.target_14d.nunique()<2:continue
        diffs.append(float(average_precision_score(b.target_14d,b.relative_risk_score)-average_precision_score(b.target_14d,b.recency_score)))
    results['week_cluster_ap_difference_sensitivity']={'point_difference':model_ap-base,
                'replicates':len(diffs),'percentile_95_interval':[float(x) for x in np.quantile(diffs,[.025,.975])],
                'warning':'Reuses 18 test weeks; not independent wells or prospective confidence interval.'}
    # Keep each distinct event together and resample adjacent event weeks
    # as a block. Results remain descriptive for these recurring wells.
    events = f[['uwi','failure_start_date']].drop_duplicates()
    events['failure_start_date'] = pd.to_datetime(events['failure_start_date'])
    def matched(rows):
        joined = rows[['uwi','week_end']].merge(events,on='uwi')
        lead = (joined.failure_start_date - joined.week_end).dt.days
        return joined.loc[lead.between(1,14), ['uwi','failure_start_date']].drop_duplicates()
    positions = matched(d)
    for name, score in [('model','relative_risk_score'),('recency','recency_score')]:
        queue = schedule_inspections(d,capacity=25,score_col=score)
        hits = matched(queue).assign(**{name:1})
        positions = positions.merge(hits,how='left',on=['uwi','failure_start_date'])
    positions['difference'] = positions.model.fillna(0) - positions.recency.fillna(0)
    day_offset = (positions.failure_start_date - positions.failure_start_date.min()).dt.days
    blocks = positions.groupby(day_offset // 14).difference.agg(['sum','size'])
    if len(blocks):
        draws = rng.integers(0,len(blocks),size=(2000,len(blocks)))
        delta = blocks['sum'].to_numpy()[draws].sum(1) / blocks['size'].to_numpy()[draws].sum(1)
        results['top25_paired_event_recall_sensitivity'] = {
            'point_difference':float(positions.difference.mean()),
            'interval_95':[float(x) for x in np.quantile(delta,[.025,.975])],
            'event_date_blocks':len(blocks),
            'interpretation':'Descriptive paired 14-day event-date block resampling. Repeated wells and a single historical period limit inference.'}
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--failures-pickle',type=Path,default=None,help='Local internal verification only')
    a=p.parse_args()
    source=a.out_dir/'private_test_alerts.csv'
    if not source.exists():raise SystemExit('Missing private_test_alerts.csv: run 02_failure_ranking.py first')
    d=pd.read_csv(source,parse_dates=['week_end'])
    f=pd.read_pickle(a.failures_pickle) if a.failures_pickle else pd.read_parquet(a.data_dir/'production_failures.parquet',columns=['uwi','failure_start_date'])
    f['failure_start_date']=pd.to_datetime(f['failure_start_date'])
    # Prevent inclusion of unrelated future events in reported denominator.
    f=f[f.failure_start_date.between('2026-06-08','2026-10-18')]
    res=validate(d,f)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/'advanced_validation.json').write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))
    print('All policy comparisons are POST-HOC. Do not claim independently confirmed uplift.')

if __name__=='__main__':main()
