"""Pure, verifiable analytics for WellGuard Failure Intelligence (private source data).

No generated/fictional operational figures. These functions never diagnose causes.
Only weekly ranks dated 1--14 days before a failure qualify as early warnings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PERIODS = ('2021–2024', '2021–2026', '2025–2026 (Recommended)')
PERIOD_START = {
    PERIODS[0]: pd.Timestamp('2021-01-01'),
    PERIODS[1]: pd.Timestamp('2021-01-01'),
    PERIODS[2]: pd.Timestamp('2025-01-01'),
}
PERIOD_END = {PERIODS[0]: pd.Timestamp('2024-12-31 23:59:59')}


def scoped_failures(events: pd.DataFrame, *, cutoff, period: str,
                    permitted_wells=None) -> pd.DataFrame:
    if period not in PERIODS:
        raise ValueError('Unsupported reporting period')
    required = {'uwi', 'failure_start_date'}
    if not required.issubset(events.columns):
        raise ValueError(f'Failure records missing {sorted(required - set(events.columns))}')
    d = events.copy()
    d['uwi'] = d.uwi.astype(str)
    d['failure_start_date'] = pd.to_datetime(d.failure_start_date, errors='coerce')
    if '_event_key' not in d.columns:
        d['_event_key'] = [str(i) for i in range(len(d))]
    end = min(pd.Timestamp(cutoff).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
              PERIOD_END.get(period, pd.Timestamp.max))
    d = d[d.failure_start_date.between(PERIOD_START[period], end)].copy()
    if permitted_wells is not None:
        d = d[d.uwi.isin(set(map(str, permitted_wells)))].copy()
    d['year'] = d.failure_start_date.dt.year
    return d.sort_values(['failure_start_date', '_event_key'], ascending=[False, True], kind='mergesort')


def analysis_statistics(events: pd.DataFrame) -> dict:
    """Repeated failure is conservatively defined as >=2 different dates per UWI."""
    n = int(len(events))
    if n == 0:
        return {'events': 0, 'wells': 0, 'rod_events': 0, 'rod_share': None,
                'repeat_wells': 0, 'repeat_well_share': None,
                'additional_event_dates': 0, 'median_repeat_gap_days': None}
    types = events['failure_type'] if 'failure_type' in events else pd.Series('', index=events.index)
    rod = int(types.fillna('').astype(str).str.contains(r'\brod', case=False, regex=True).sum())
    uniq = events[['uwi', 'failure_start_date']].dropna().copy()
    uniq['failure_date'] = uniq.failure_start_date.dt.normalize()
    uniq = uniq.drop_duplicates(['uwi', 'failure_date'])
    bywell = uniq.groupby('uwi').failure_date.agg(list)
    repeat = bywell[bywell.apply(len) >= 2]
    gaps = []
    for dates in repeat:
        ordered = sorted(dates)
        gaps.extend([(b-a).days for a,b in zip(ordered,ordered[1:])])
    wells = int(events.uwi.nunique())
    return {'events': n, 'wells': wells, 'rod_events': rod,
            'rod_share': rod/n if n else None,
            'repeat_wells': int(len(repeat)),
            'repeat_well_share': len(repeat)/wells if wells else None,
            'additional_event_dates': int(sum(max(len(dates)-1,0) for dates in bywell)),
            'median_repeat_gap_days': float(np.median(gaps)) if gaps else None}


def annual_counts(events: pd.DataFrame, period: str, cutoff) -> pd.DataFrame:
    start = PERIOD_START[period].year
    end = min(pd.Timestamp(cutoff).year, 2024 if period == PERIODS[0] else 2026)
    if end < start:
        return pd.DataFrame(columns=['year', 'events'])
    years = pd.DataFrame({'year': list(range(start, end + 1))})
    if events.empty:
        years['events'] = 0
        return years
    counted = events.groupby(events.failure_start_date.dt.year).size().rename('events')
    years['events'] = years.year.map(counted).fillna(0).astype(int)
    return years


def pareto_types(events: pd.DataFrame, max_types: int = 5) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=['failure_type', 'count', 'cumulative_pct'])
    raw = (events.get('failure_type', pd.Series(index=events.index, dtype='object'))
           .fillna('Not recorded').replace('', 'Not recorded').astype(str).value_counts())
    top = raw.iloc[:max_types].copy()
    tail = int(raw.iloc[max_types:].sum())
    if tail:
        top.loc['Other types'] = tail
    ans = top.rename_axis('failure_type').reset_index(name='count')
    ans['cumulative_pct'] = ans['count'].cumsum() / len(events) * 100
    return ans


def recorded_causes(events: pd.DataFrame, max_causes: int = 5) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=['cause', 'count', 'share_pct'])
    raw = (events.get('cause', pd.Series(index=events.index, dtype='object'))
           .fillna('Not recorded').replace('', 'Not recorded').astype(str).value_counts())
    first = raw.iloc[:max_causes].copy()
    if int(raw.iloc[max_causes:].sum()): first.loc['Other recorded causes'] = int(raw.iloc[max_causes:].sum())
    ans = first.rename_axis('cause').reset_index(name='count')
    ans['share_pct'] = ans['count'] / len(events) * 100
    return ans


def event_history(events: pd.DataFrame) -> pd.DataFrame:
    """Add days since previous *distinct-date* failure as retrospective context."""
    if events.empty:
        x = events.copy(); x['days_since_previous_failure'] = pd.Series(dtype='Int64'); return x
    d = events.sort_values(['uwi', 'failure_start_date', '_event_key']).copy()
    # Same-date, multiple recorded rows are not separate time-separated failures.
    days = d.failure_start_date.dt.normalize()
    preceding = pd.Series(index=d.index, dtype='datetime64[ns]')
    for _, group in d.assign(_day=days).groupby('uwi', sort=False):
        unique_days = sorted(group['_day'].dropna().unique())
        prev_map = {day: (unique_days[i-1] if i > 0 else pd.NaT)
                    for i, day in enumerate(unique_days)}
        preceding.loc[group.index] = group['_day'].map(prev_map)
    d['days_since_previous_failure'] = (days - preceding).dt.days.astype('Int64')
    return d.sort_values(['failure_start_date', '_event_key'], ascending=[False, True])


def pre_failure_observations(weekly: pd.DataFrame, well: str, event_date, lookback_days=30) -> pd.DataFrame:
    """Only historical weekly snapshots ending BEFORE the recorded failure date."""
    if weekly.empty:
        return weekly.iloc[0:0].copy()
    start = pd.Timestamp(event_date).normalize() - pd.Timedelta(days=lookback_days)
    end = pd.Timestamp(event_date).normalize()
    dates = pd.to_datetime(weekly['week_end'], errors='coerce')
    d = weekly[weekly.uwi.astype(str).eq(str(well)) & dates.ge(start) & dates.lt(end)].copy()
    return d.sort_values('week_end')


def early_warning_check(scores: pd.DataFrame, well: str, event_date, *, capacity=25,
                        horizon_days=14) -> dict:
    """Historical ranking for weeks 1..14 days BEFORE one recorded event.

    Ranking is across the full eligible score cohort for each week. No post-event
    labels are used to choose the historical ranked wells, and rankings are not
    calibrated probabilities. Not all score weeks are independently held out.
    """
    if capacity not in (25,50,100): raise ValueError('Unsupported inspection capacity')
    empty = {'status': 'NO_SCORE', 'rank': None, 'eligible_count': None,
             'scored_week': None, 'first_alert_week': None, 'lead_days': None,
             'selected_for_review': False, 'score_origin': None, 'relative_risk_score': None,
             'observed_weeks': 0}
    required = {'uwi', 'week_end', 'relative_risk_score'}
    if scores is None or scores.empty or not required.issubset(scores.columns):
        return empty.copy()
    event = pd.Timestamp(event_date).normalize()
    df = scores.copy()
    df['week_end'] = pd.to_datetime(df['week_end'], errors='coerce').dt.normalize()
    df['uwi'] = df.uwi.astype(str)
    df['relative_risk_score'] = pd.to_numeric(df['relative_risk_score'], errors='coerce')
    df = df[df.week_end.notna() & df.relative_risk_score.notna()].copy()
    df['lead_days'] = (event - df.week_end).dt.days
    df = df[df.lead_days.between(1,horizon_days)]
    if df.empty: return empty.copy()
    snapshots=[]
    for week, group in df.groupby('week_end',sort=True):
        eligible = (group.sort_values(['relative_risk_score','uwi'],ascending=[False,True],kind='mergesort')
                    .drop_duplicates('uwi',keep='first').reset_index(drop=True))
        matching = eligible[eligible.uwi.eq(str(well))]
        if matching.empty:continue
        row=matching.iloc[0]
        rank=int(matching.index[0])+1
        snapshots.append({'rank':rank,'eligible_count':len(eligible),
                          'scored_week':pd.Timestamp(week),
                          'lead_days':int((event-week).days),
                          'selected_for_review':rank<=capacity,
                          'score_origin':str(row.get('score_origin','UNSPECIFIED')),
                          'relative_risk_score':float(row['relative_risk_score'])})
    if not snapshots:
        return empty.copy()
    selected=[x for x in snapshots if x['selected_for_review']]
    # Earliest qualifying selection in the prediction horizon, never >14 days.
    if selected:
        chosen=min(selected,key=lambda x:x['scored_week'])
        status='FLAGGED'
    else:
        # Most recent available rank explains a missed/low-rank event.
        chosen=max(snapshots,key=lambda x:x['scored_week'])
        status='NOT_SELECTED'
    return {**chosen, 'status':status, 'first_alert_week': chosen['scored_week'] if selected else None,
            'lead_days':chosen['lead_days'] if selected else None,
            'observed_weeks':len(snapshots)}
