"""Pure, testable V6 operations analytics. Read-only research tools.

All changes are descriptive, and should not be interpreted as a diagnosed equipment
fault or causal treatment effect. Missingness is explicit rather than imputed silently.
"""
from __future__ import annotations
from typing import Any
import math
import numpy as np
import pandas as pd

SENSOR_METRICS = (
    ('oil_daily_avg_2w', 'Oil rate', 'm³/day'),
    ('rpm_avg_2w', 'Pump speed', 'RPM'),
    ('eff_avg_2w', 'Pump efficiency (source unit)', ''),
    ('torque_avg_2w', 'Rod torque', 'ft·lbf'),
    ('current_avg_2w', 'Motor current', 'A'),
    ('down_fraction_2w', 'Downtime fraction', ''),
)

def _number(value: Any) -> float | None:
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):
        return None

def health_snapshot(history: pd.DataFrame, as_of, recent_baseline_weeks: int=4) -> dict:
    """Past-only deviations versus previous weekly snapshots, no failure diagnosis.

    Snapshot as-of is latest observed week <= cutoff. Baseline excludes selected week;
    no hypothetical unobserved weekly values are inferred. Incomplete baselines are
    shown as insufficient rather than promoted to 'healthy'.
    """
    if 'week_end' not in history.columns: raise ValueError('Missing week_end')
    d=history.copy(); d['week_end']=pd.to_datetime(d['week_end'],errors='coerce')
    d=d[d['week_end'].le(pd.Timestamp(as_of))].sort_values('week_end')
    if d.empty: return {'has_snapshot':False, 'metrics':[], 'data_quality':'No record on or before as-of date'}
    row=d.iloc[-1]; prior=d.iloc[:-1].tail(recent_baseline_weeks)
    metrics=[]
    for key,label,unit in SENSOR_METRICS:
        current=_number(row.get(key))
        vals=pd.to_numeric(prior.get(key,pd.Series(dtype=float)),errors='coerce').dropna()
        base=float(vals.median()) if len(vals)>=3 else None
        change=(current-base)/abs(base)*100 if current is not None and base is not None and abs(base)>1e-8 else None
        metrics.append({'field':key,'name':label,'unit':unit,'current':current,'baseline':base,
                        'change_pct':change,'observed_baseline_weeks':int(len(vals)),
                        'flag': 'NOTABLE DEVIATION' if change is not None and abs(change)>=20 else
                                'BASELINE INSUFFICIENT' if base is None else
                                'MISSING' if current is None else 'WITHIN 20% BAND'})
    observed=sum(_number(row.get(k)) is not None for k in ('eff_avg_2w','rpm_avg_2w','torque_avg_2w','current_avg_2w'))
    lag=int((pd.Timestamp(as_of)-pd.Timestamp(row['week_end'])).days)
    return {'has_snapshot':True,'well':str(row.get('uwi','')), 'as_of':str(pd.Timestamp(as_of).date()),
            'observed_week':str(row['week_end'].date()),'age_days':lag,'pump_channels_available':observed,
            'pump_channels_possible':4,'metrics':metrics,
            'data_quality':'STALE' if lag>7 else 'LOW SENSOR COVERAGE' if observed<2 else 'AVAILABLE WITH CAVEATS'}


