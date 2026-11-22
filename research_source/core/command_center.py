"""Pure, source-grounded calculations for the WellGuard executive page.

The design reference is visual only: no screenshot metric or sample well is used.
All quantities are derived from the recorded as-of-week or archived retrospective report.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def _numeric_total(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame.columns:
        return None
    series = pd.to_numeric(frame[col], errors='coerce')
    return float(series.sum(min_count=1)) if series.notna().any() else None


def _reason(row: pd.Series) -> str:
    """Recorded evidence only; no invented equipment cause or diagnosis."""
    parts = []
    loss = pd.to_numeric(row.get('estimated_lost_oil_m3'), errors='coerce')
    down = pd.to_numeric(row.get('hours_down'), errors='coerce')
    if pd.notna(down) and down > 0:
        parts.append(f'{down:,.0f} h recorded downtime')
    if pd.notna(loss) and loss > 0:
        parts.append(f'{loss:,.1f} m³ estimated lost oil')
    if not parts:
        parts.append('Review operating trends and recorded sensor coverage')
    return '; '.join(parts)


def build_command_center(weekly: pd.DataFrame, view: pd.DataFrame,
                         geo: pd.DataFrame, model_week: pd.DataFrame,
                         cutoff, *, model_report: dict | None = None, capacity: int = 25) -> dict:
    """Prepare exact-as-of KPIs and real rankings for the selected region.

    `view` may include old last-observed rows for map and historical drilldown,
    but ONLY rows matching cutoff contribute to current-week production/downtime.
    Model ranks exist only where score file has records on the cutoff date.
    """
    if capacity not in (25,50,100):raise ValueError('Select a validated inspection capacity of 25, 50 or 100')
    asof = pd.Timestamp(cutoff)
    observed = view[view.week_end.eq(asof)].copy()
    model_week = model_week.copy()
    if not model_week.empty:
        model_week['uwi'] = model_week['uwi'].astype(str)
        model_week = model_week[model_week.week_end.eq(asof)]
        model_week = model_week.drop_duplicates('uwi')
        model_week = model_week[model_week.uwi.isin(view.uwi.astype(str))]
        model_week = model_week.dropna(subset=['relative_risk_score'])
    has_scores = not model_week.empty
    if has_scores:
        choices = model_week[['uwi', 'relative_risk_score']].merge(
            view.drop(columns=['relative_risk_score'], errors='ignore'),
            on='uwi', how='left', validate='one_to_one')
        choices = choices.sort_values(['relative_risk_score', 'uwi'], ascending=[False, True])
        queue = choices.head(capacity).copy()
        queue['review_rank'] = range(1, len(queue)+1)
        queue['priority_tier'] = np.where(queue.review_rank.le(max(1,capacity//5)), 'High review',
                                  np.where(queue.review_rank.le(max(2,capacity*3//5)), 'Review', 'Watch'))
        queue['next_step'] = 'Review evidence'
        queue['reason'] = queue.apply(_reason, axis=1)
        eligible_label = 'Model-scored eligible wells'
        eligible_count = len(model_week)
        rank_label = 'Inspection alerts by operating area'
        rank_detail = f'Model-ranked top {capacity} at selected cutoff — an uncalibrated relative score'
    else:
        # Honest non-predictive mode: available evidence, not fabricated ML scores.
        choices = observed.dropna(subset=['estimated_lost_oil_m3']).copy()
        choices = choices.sort_values(['estimated_lost_oil_m3','uwi'],ascending=[False,True])
        queue = choices.head(capacity).copy()
        queue['review_rank'] = range(1,len(queue)+1)
        queue['priority_tier'] = 'Loss-only'
        queue['next_step'] = 'Inspect evidence'
        queue['reason'] = queue.apply(_reason, axis=1)
        eligible_label = 'Wells observed this week'
        eligible_count = observed.uwi.nunique()
        rank_label = 'Estimated lost oil by operating area'
        rank_detail = 'Descriptive observation only — model ranking not available for this week'
    if queue.empty:
        area_rank = pd.DataFrame(columns=['area','value'])
    elif has_scores:
        area_rank = queue.groupby('area', dropna=False).size().reset_index(name='value')
    else:
        area_rank = observed.groupby('area', dropna=False)['estimated_lost_oil_m3'].sum(min_count=1).reset_index(name='value')
    area_rank['area'] = area_rank['area'].fillna('Unknown') if len(area_rank) else area_rank.get('area', pd.Series(dtype='str'))
    area_rank = area_rank.dropna(subset=['value']).sort_values('value',ascending=False).head(8)

    # Filtering is based on static geographic metadata, explicitly not historical feature evidence.
    visible_wells = set(view.uwi.astype(str))
    trend = weekly[weekly.uwi.astype(str).isin(visible_wells) &
                   weekly.week_end.le(asof)].copy()
    # Unlike an oil-rate estimate, a weekly total drops naturally when the final
    # reporting week has <7 source dates. Exclude partial weeks from the comparable
    # trend; the selected-week KPI remains explicitly labeled as partial.
    latest_source_days = int(observed.days.max()) if not observed.empty and 'days' in observed else None
    selected_partial = bool(latest_source_days is not None and latest_source_days < 7)
    if 'days' in trend.columns:
        full_weeks=trend.groupby('week_end')['days'].max()
        full_weeks=full_weeks[full_weeks>=7].index
        trend=trend[trend.week_end.isin(full_weeks)]
    weeks = sorted(trend.week_end.unique())[-10:]
    trend = trend[trend.week_end.isin(weeks)]
    if not trend.empty:
        trend = trend.groupby('week_end', as_index=False).agg(
            oil=('oil_total_m3',lambda s: pd.to_numeric(s,errors='coerce').sum(min_count=1)),
            lost=('estimated_lost_oil_m3',lambda s: pd.to_numeric(s,errors='coerce').sum(min_count=1)),
            wells=('uwi','nunique'))
    else:
        trend = pd.DataFrame(columns=['week_end','oil','lost','wells'])
    age = pd.to_numeric(view.age_days, errors='coerce')
    freshest = int((age.le(7)).sum())
    result = {
        'cutoff': asof, 'observed': observed, 'has_scores': has_scores,
        'selected_partial': selected_partial, 'source_days_in_selected_week': latest_source_days,
        'eligible_label': eligible_label, 'eligible_count': int(eligible_count),
        'selected_reviews': int(len(queue)) if has_scores else None,
        'review_capacity': capacity,
        'weekly_oil': _numeric_total(observed, 'oil_total_m3'),
        'weekly_lost': _numeric_total(observed,'estimated_lost_oil_m3'),
        'weekly_downtime': _numeric_total(observed, 'hours_down'),
        'current_week_coverage': freshest,
        'total_visible_wells': int(view.uwi.nunique()),
        'median_freshness_days': float(age.median()) if age.notna().any() else None,
        'area_rank': area_rank, 'rank_label': rank_label, 'rank_detail': rank_detail,
        'trend': trend, 'queue': queue,
        'benchmark': model_report or {},
    }
    return result
