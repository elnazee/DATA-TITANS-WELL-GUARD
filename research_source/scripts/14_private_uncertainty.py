"""Private week-block bootstrap of average-precision improvement over recency rule.

Exploratory sensitivity for the already-used June–October 2026 test only.
It does not restore an untouched holdout, guarantee independent weeks, or establish
prospective value. Writes aggregate-only JSON into private outputs/.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def week_block_bootstrap(frame:pd.DataFrame,iterations:int=500,seed:int=2026):
    if iterations<20:raise ValueError('At least 20 iterations required')
    required={'week_end','target_14d','relative_risk_score','recency_baseline_score'}
    missing=required-set(frame.columns)
    if missing:raise ValueError(f'Missing private test inputs: {sorted(missing)}')
    x=frame.copy()
    x['week_end']=pd.to_datetime(x.week_end)
    weeks=sorted(x.week_end.unique())
    if len(weeks)<3:raise ValueError('At least three distinct test weeks required')
    y=x.target_14d.astype(int).to_numpy()
    if set(y)!={0,1}:raise ValueError('Both target classes required')
    model=x.relative_risk_score.to_numpy(dtype=float)
    base=x.recency_baseline_score.to_numpy(dtype=float)
    if not np.isfinite(model).all() or not np.isfinite(base).all():raise ValueError('Missing or invalid model scores')
    groups=[np.flatnonzero(x.week_end.to_numpy()==w) for w in weeks]
    rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(iterations):
        draws=rng.integers(0,len(groups),len(groups))
        selected=np.concatenate([groups[i] for i in draws])
        sy=y[selected]
        if sy.min()==sy.max():continue
        diffs.append(float(average_precision_score(sy,model[selected])-
                           average_precision_score(sy,base[selected])))
    if len(diffs)<max(10,iterations//2):raise ValueError('Insufficient class-balanced bootstrap draws')
    actual=float(average_precision_score(y,model)-average_precision_score(y,base))
    lo,hi=np.quantile(diffs,[.025,.975])
    return {'method':'resample 18 test Sunday snapshots with replacement (within-week observations grouped)',
            'reference_test_only':True,'weeks':len(weeks),'iterations_requested':iterations,'valid_draws':len(diffs),
            'ap_difference_model_minus_recency':actual,
            'descriptive_percentile_95_interval':[float(lo),float(hi)],
            'share_resampled_differences_positive':float((np.array(diffs)>0).mean()),
            'inference_limits':'Different weeks can include the same wells/events, and the test was explored before. This interval is descriptive, not a prospective or independent validation.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    p.add_argument('--iterations',type=int,default=500)
    args=p.parse_args()
    path=args.out_dir/'private_test_alerts.csv'
    if not path.is_file():
        print('NO private test replay. Train on the source first; no synthetic results will be published.')
        return 2
    result=week_block_bootstrap(pd.read_csv(path),iterations=args.iterations)
    output=args.out_dir/'private_uncertainty_aggregate.json'
    output.write_text(json.dumps(result,indent=2))
    print('Saved aggregate-only, exploratory weekly-block AP interval to:',output)
    print('Model-minus-recency AP:',round(result['ap_difference_model_minus_recency'],4),
          '95% descriptive interval:',[round(v,4) for v in result['descriptive_percentile_95_interval']])
    return 0

if __name__=='__main__':sys.exit(main())
