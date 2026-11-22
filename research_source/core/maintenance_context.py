"""Enrich private historical risk replays with past-only recorded job-start counts.

Only source-recorded dates are counted. No imputation, invented maintenance
activity, model retraining, or data upload. A zero means no *recorded distinct
job start date* for that well in the prior inclusive 90-day lookback; it does
not establish that the well had no maintenance activity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_prior_job_counts(score_rows: pd.DataFrame, job_rows: pd.DataFrame,
                         *, window_days: int = 90) -> pd.DataFrame:
    """Count distinct recorded job-start dates from [as_of - 90d, as_of].

    Matches the original private context builder's UWI/start-date dedup policy.
    All dates are treated as calendar dates, because model weeks are date-based.
    Missing/invalid score week remains unknown (pd.NA), not zero.
    """
    required_score = {'uwi', 'week_end'}
    required_jobs = {'uwi', 'start_date'}
    if not required_score.issubset(score_rows):
        raise ValueError('Score records lack uwi or week_end; cannot calculate maintenance history.')
    if not required_jobs.issubset(job_rows):
        raise ValueError('Job records lack uwi or start_date; cannot calculate maintenance history.')
    if window_days <= 0:
        raise ValueError('window_days must be positive')
    result = score_rows.copy()
    result['prior_jobs_90d'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
    if result.empty:
        return result

    a = result[['uwi', 'week_end']].copy()
    a['uwi'] = a.uwi.astype('string')
    a['week_end'] = pd.to_datetime(a.week_end, errors='coerce').dt.normalize()
    a = a.dropna(subset=['uwi', 'week_end'])
    if a.empty:
        return result

    j = job_rows[['uwi', 'start_date']].copy()
    j['uwi'] = j.uwi.astype('string')
    j['start_date'] = pd.to_datetime(j.start_date, errors='coerce').dt.normalize()
    j = j.dropna(subset=['uwi', 'start_date']).drop_duplicates(['uwi', 'start_date'])
    dates_by_well = {str(well): np.sort(g.start_date.to_numpy(dtype='datetime64[ns]'))
                     for well, g in j.groupby('uwi', sort=False)}
    for well, rows in a.groupby('uwi', sort=False):
        end = rows.week_end.to_numpy(dtype='datetime64[ns]')
        dates = dates_by_well.get(str(well))
        if dates is None:
            counts = np.zeros(len(end), dtype=np.int64)
        else:
            start = end - np.timedelta64(window_days, 'D')
            counts = (np.searchsorted(dates, end, side='right') -
                      np.searchsorted(dates, start, side='left'))
        result.loc[rows.index, 'prior_jobs_90d'] = counts
    return result
