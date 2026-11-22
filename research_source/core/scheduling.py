"""Transparent, deterministic, as-of inspection queues.

The selected wells are *review candidates*. Cooldowns are operational hypotheses;
never describe a retrospectively observed uplift as independently validated.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd


def rank_week(frame: pd.DataFrame, *, score_col: str='relative_risk_score', capacity: int=25,
              excluded_wells: set[str] | None=None) -> pd.DataFrame:
    if capacity < 1:
        raise ValueError('capacity must be positive')
    if any(c not in frame.columns for c in ('uwi', score_col)):
        raise KeyError('missing score or well ID')
    d = frame.copy()
    d['uwi'] = d['uwi'].astype(str)
    if excluded_wells:
        d = d[~d['uwi'].isin(set(excluded_wells))]
    # Stable, auditable tie breaking, not source file row order.
    return d.sort_values([score_col, 'uwi'], ascending=[False, True], kind='mergesort').head(capacity).copy()


def schedule_inspections(frame: pd.DataFrame, *, capacity: int=25, cooldown_weeks: int=0,
                         score_col: str='relative_risk_score', through_date=None) -> pd.DataFrame:
    if cooldown_weeks < 0:
        raise ValueError('cooldown weeks must be nonnegative')
    if frame['week_end'].isna().any():
        raise ValueError('week_end contains missing dates')
    d=frame.copy()
    d['week_end']=pd.to_datetime(d['week_end'])
    if through_date is not None:
        d=d[d['week_end']<=pd.Timestamp(through_date)]
    d=d.sort_values('week_end',kind='mergesort')
    last_selected: dict[str,int]={}
    selected=[]
    for week_idx,(week,g) in enumerate(d.groupby('week_end',sort=True)):
        banned = {well for well,last in last_selected.items() if week_idx-last<=cooldown_weeks}
        current=rank_week(g,score_col=score_col,capacity=capacity,excluded_wells=banned)
        current['review_rank']=range(1,len(current)+1)
        current['cooldown_weeks']=cooldown_weeks
        for well in current['uwi']:
            last_selected[well]=week_idx
        selected.append(current)
    return pd.concat(selected,ignore_index=True) if selected else d.iloc[:0].copy()


def event_recall(eligible: pd.DataFrame, selected: pd.DataFrame, failures: pd.DataFrame,
                 *, horizon_days: int=14) -> dict:
    """Event detection among distinct (well,start date) events eligible for advance observation."""
    f=failures[['uwi','failure_start_date']].dropna().drop_duplicates().copy()
    f['uwi']=f['uwi'].astype(str)
    f['failure_start_date']=pd.to_datetime(f['failure_start_date'])
    def match(rows):
        if rows.empty: return set(),{}
        r=rows[['uwi','week_end']].copy()
        r['uwi']=r['uwi'].astype(str)
        r['week_end']=pd.to_datetime(r['week_end'])
        m=r.merge(f,on='uwi',how='inner')
        m['lead_days']=(m['failure_start_date']-m['week_end']).dt.days
        m=m[m['lead_days'].between(1,horizon_days)]
        keys=set(zip(m['uwi'],m['failure_start_date']))
        leads={(u,d):int(v) for (u,d),v in m.groupby(['uwi','failure_start_date'])['lead_days'].max().items()}
        return keys,leads
    all_keys,_=match(eligible)
    caught_keys,lead=match(selected)
    hit=caught_keys & all_keys
    return dict(eligible_events=len(all_keys),caught_events=len(hit),
                event_recall=(len(hit)/len(all_keys) if all_keys else None),
                median_earliest_lead_days=(float(np.median([lead[k] for k in hit])) if hit else None))


def break_even_success_rate(*, flagged_events_per_week: float, fraction_addressable: float,
                            volume_per_event_m3: float, value_cad_per_m3: float,
                            inspections_per_week: int, cost_cad_per_inspection:float):
    """Required intervention success fraction for zero hypothetical net value; None if impossible denominator."""
    vals=(flagged_events_per_week,fraction_addressable,volume_per_event_m3,value_cad_per_m3,inspections_per_week,cost_cad_per_inspection)
    if any(v<0 for v in vals): raise ValueError('all inputs must be nonnegative')
    denom=flagged_events_per_week*fraction_addressable*volume_per_event_m3*value_cad_per_m3
    return inspections_per_week*cost_cad_per_inspection/denom if denom>0 else None
