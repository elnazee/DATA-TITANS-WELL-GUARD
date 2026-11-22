"""Additional past-only model features and inspection-policy experiments.

These are candidate *research* features. Never replace the locked historical
benchmark or present them as proven future operational improvements.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from core.scheduling import schedule_inspections,event_recall

HORIZON_DAYS=14
# Only measurements available at or before a weekly as-of snapshot. The
# benchmark's source features and label construction are deliberately unchanged.
TREND_INPUTS=(
    'oil_daily_avg','down_fraction','rpm_avg','torque_avg',
    'current_avg','eff_avg','lost_per_week',
)
ENGINEERED_FEATURES=tuple(
    f'{name}_{suffix}' for name in TREND_INPUTS
    for suffix in ('change_1w','history_std_4w','past_baseline_delta_8w')
)
MAINTENANCE_FEATURES=(
    'job_starts_30d','job_starts_90d','days_since_last_job_start',
)
SPLIT_DATES={
    'train_start':'2025-06-01','train_end':'2026-03-01',
    'validation_start':'2026-03-22','validation_end':'2026-05-17',
    'test_start':'2026-06-07','test_end':'2026-10-04',
}


def add_past_only_trends(weekly:pd.DataFrame)->pd.DataFrame:
    """Consecutive weekly segments prevent crossing unobserved-week gaps.

    Historical baseline excludes current week. Each feature at t depends only
    on t and dates <t for the same well; the previous-week delta is missing
    when the last recorded week is not exactly seven days earlier.
    """
    required={'uwi','week_end',*TREND_INPUTS}
    missing=required-set(weekly.columns)
    if missing: raise ValueError(f'Missing weekly columns for feature engineering: {sorted(missing)}')
    d=weekly.copy()
    d['week_end']=pd.to_datetime(d.week_end,errors='raise')
    if d[['uwi','week_end']].duplicated().any():
        raise ValueError('Duplicate well-week; cannot calculate reliable temporal features.')
    # Preserve caller row order while calculating chronologically.
    d['_wg_order']=np.arange(len(d))
    d=d.sort_values(['uwi','week_end'],kind='mergesort').copy()
    gaps=d.groupby('uwi',sort=False).week_end.diff().dt.days
    d['_wg_segment']=gaps.ne(7).groupby(d.uwi,sort=False).cumsum()
    group_keys=[d['uwi'],d['_wg_segment']]
    for col in TREND_INPUTS:
        s=pd.to_numeric(d[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
        # pandas groupby transform must align to original index.
        previous=s.groupby(group_keys,sort=False).shift(1)
        d[f'{col}_change_1w']=s-previous
        d[f'{col}_history_std_4w']=s.groupby(group_keys,sort=False).transform(
            lambda x:x.shift(1).rolling(4,min_periods=3).std(ddof=0))
        history=s.groupby(group_keys,sort=False).transform(
            lambda x:x.shift(1).rolling(8,min_periods=3).median())
        d[f'{col}_past_baseline_delta_8w']=s-history
    d=d.sort_values('_wg_order',kind='mergesort').drop(columns=['_wg_order','_wg_segment'])
    for feature in ENGINEERED_FEATURES:
        d[feature]=pd.to_numeric(d[feature],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return d


def add_prior_maintenance(weekly:pd.DataFrame,jobs:pd.DataFrame)->pd.DataFrame:
    """Count prior job *start dates* only (strictly before Sunday snapshot).

    This says nothing about work completion or when operators learned of the
    job. The dataset has no point-in-time publication timestamp, so retrospective
    entry is a residual availability risk. Missing job history yields count=0
    for the recorded log, not an assertion that no maintenance happened.
    """
    d=weekly.copy()
    d['week_end']=pd.to_datetime(d.week_end,errors='raise')
    if d[['uwi','week_end']].duplicated().any():
        raise ValueError('Duplicate well-week in maintenance join.')
    d['uwi']=d.uwi.astype(str)
    j=jobs[['uwi','start_date']].copy()
    j['uwi']=j.uwi.astype(str)
    j['start_date']=pd.to_datetime(j.start_date,errors='coerce')
    j=j[j.start_date.notna()].drop_duplicates(['uwi','start_date']).sort_values(['uwi','start_date'])
    by_well={u:g.start_date.to_numpy(dtype='datetime64[ns]') for u,g in j.groupby('uwi',sort=False)}
    d['job_starts_30d']=0
    d['job_starts_90d']=0
    d['days_since_last_job_start']=np.nan
    for u,ix in d.groupby('uwi',sort=False).groups.items():
        dates=by_well.get(u)
        if dates is None or len(dates)==0:continue
        asof=d.loc[ix,'week_end'].to_numpy(dtype='datetime64[ns]')
        upper=np.searchsorted(dates,asof,side='left')  # strict < as-of, never same-day
        lower30=np.searchsorted(dates,asof-np.timedelta64(30,'D'),side='left')
        lower90=np.searchsorted(dates,asof-np.timedelta64(90,'D'),side='left')
        d.loc[ix,'job_starts_30d']=upper-lower30
        d.loc[ix,'job_starts_90d']=upper-lower90
        valid=upper>0
        if valid.any():
            back=(asof[valid]-dates[upper[valid]-1])/np.timedelta64(1,'D')
            d.loc[np.asarray(list(ix))[valid],'days_since_last_job_start']=np.minimum(back.astype(float),365)
    for c in MAINTENANCE_FEATURES:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    return d


def check_reference_boundaries(d:pd.DataFrame, failures:pd.DataFrame, *, final_day='2026-10-18'):
    """Enforce 14-day label coverage, distinct rows, and reference chronology."""
    if d[['uwi','week_end']].duplicated().any(): raise ValueError('Duplicate candidate well-weeks')
    if d['week_end'].isna().any():raise ValueError('Missing prediction dates')
    f=pd.to_datetime(failures['failure_start_date'],errors='raise')
    # The original dataset declares future failure coverage to October 18;
    # failure log max alone cannot establish absence of failures after that.
    cutoff=pd.Timestamp(final_day)
    if pd.Timestamp(SPLIT_DATES['test_end'])+pd.Timedelta(days=14)>cutoff:
        raise ValueError('Test outcomes extend beyond the declared failure observation horizon')
    if (pd.Timestamp(SPLIT_DATES['validation_start'])-pd.Timestamp(SPLIT_DATES['train_end'])).days<=14:
        raise ValueError('Train/validation label overlap')
    if (pd.Timestamp(SPLIT_DATES['test_start'])-pd.Timestamp(SPLIT_DATES['validation_end'])).days<=14:
        raise ValueError('Validation/test label overlap')
    return True


def split_masks(d:pd.DataFrame)->dict:
    w=pd.to_datetime(d.week_end)
    return {
      'train':w.between(SPLIT_DATES['train_start'],SPLIT_DATES['train_end']),
      'validation':w.between(SPLIT_DATES['validation_start'],SPLIT_DATES['validation_end']),
      'test':w.between(SPLIT_DATES['test_start'],SPLIT_DATES['test_end']),
    }


def run_policy_evaluation(scored:pd.DataFrame,failures:pd.DataFrame,*,capacity:int=25,cooldown_weeks:int=0,score_col:str='risk_candidate') ->dict:
    if scored.empty:raise ValueError('No eligible rows for policy evaluation')
    selected=schedule_inspections(scored,capacity=capacity,cooldown_weeks=cooldown_weeks,score_col=score_col)
    recall=event_recall(scored,selected,failures,horizon_days=14)
    positive_rate=float(selected['target_14d'].mean()) if len(selected) else None
    return {**recall,'well_week_precision':positive_rate,'review_slots_used':int(len(selected)),
            'weeks':int(scored.week_end.nunique()),'capacity':capacity,
            'cooldown_weeks':cooldown_weeks}


def classification_metrics(y,score)->dict:
    y=np.asarray(y,dtype=int)
    score=np.asarray(score,dtype=float)
    if not set(np.unique(y)).issubset({0,1}) or np.isnan(score).any():raise ValueError('Invalid labels or missing scores')
    return {'ap':float(average_precision_score(y,score)) if y.sum() else None,
            'roc_auc':float(roc_auc_score(y,score)) if len(set(y))==2 else None,
            'observations':int(len(y)),'positives':int(y.sum())}
