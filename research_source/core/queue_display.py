"""Presentation-only formatting for the real risk-ranked inspection queue.

Canonical numerical columns are preserved in private CSV exports; the UI labels
reflect the actual feature-construction semantics in 02_failure_ranking.py.
"""
from __future__ import annotations
import pandas as pd

LABELS = {
    'review_rank': 'Rank',
    'week_end': 'Week ending',
    'uwi': 'Well (UWI)',
    'relative_risk_score': 'Model score (relative)',
    'oil_daily_avg_2w': 'Avg oil rate (2w, m³/day)',
    'lost_per_week_2w': 'Est. loss per week (2w avg, m³)',
    'down_fraction_2w': 'Days with downtime (2w, %)',
    'prior_jobs_90d': 'Recorded job-start dates (90d)',
}


def friendly_queue_table(frame: pd.DataFrame) -> pd.DataFrame:
    """A display copy only; never change source feature values or scored ranks."""
    out = frame.copy()
    if 'week_end' in out:
        dt = pd.to_datetime(out['week_end'], errors='coerce')
        out['week_end'] = dt.dt.strftime('%Y-%m-%d').where(dt.notna(), '')
    if 'down_fraction_2w' in out:
        out['down_fraction_2w'] = (pd.to_numeric(out['down_fraction_2w'], errors='coerce') * 100).round(1)
    for col in ('relative_risk_score', 'oil_daily_avg_2w', 'lost_per_week_2w'):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors='coerce').round(4 if col == 'relative_risk_score' else 2)
    if 'prior_jobs_90d' in out:
        out['prior_jobs_90d'] = pd.to_numeric(out['prior_jobs_90d'], errors='coerce').astype('Int64')
    renamed = out.rename(columns={col: label for col,label in LABELS.items() if col in out})
    assert renamed.columns.is_unique
    return renamed
