"""Evidence-bounded question-answer assistant for WellGuard.

Deterministic (works offline) and transparent. No training, cloud calls,
mechanical root-cause attribution, or use of future-dated records.
Optional local language-model narration is a separate UI capability.
"""
from __future__ import annotations
from datetime import date
from typing import Any
import math
import pandas as pd

QUESTIONS = (
    'Why is this well on the watchlist?',
    'What changed over the last few weeks?',
    'Which sensors are missing?',
    'What should the operator verify?',
    'Does maintenance history explain the score?',
    'Can we estimate production savings?',
)
FIELDS = {
    'relative_risk_score': ('Relative rank score', '', 3),
    'oil_daily_avg_2w': ('Observed mean oil rate', ' m³/day', 1),
    'lost_per_week_2w': ('Historical estimated lost oil', ' m³/week', 1),
    'down_fraction_2w': ('Two-week downtime fraction', '', 2),
    'rpm_avg_2w': ('Pump RPM', ' rpm', 1),
    'eff_avg_2w': ('Reported pump efficiency', '', 2),
    'torque_avg_2w': ('Rod torque', '', 1),
    'current_avg_2w': ('Motor current', '', 1),
}
SENSOR_FIELDS = ('eff_avg_2w', 'rpm_avg_2w', 'torque_avg_2w', 'current_avg_2w')


def printable(value: Any, unit: str = '', precision: int = 1) -> str:
    try:
        val = float(value)
        return f'{val:,.{precision}f}{unit}' if math.isfinite(val) else 'not available'
    except (ValueError, TypeError):
        return 'not available'


def _history_as_of(history: pd.DataFrame, as_of) -> pd.DataFrame:
    """Remove future snapshots even when callers provide a mixed history."""
    if history is None or history.empty or 'week_end' not in history:
        return pd.DataFrame()
    result = history.copy()
    result['week_end'] = pd.to_datetime(result['week_end'], errors='coerce')
    return result.loc[result['week_end'].le(pd.Timestamp(as_of))].sort_values('week_end')


def explain(question: str, row: dict, history: pd.DataFrame | None = None,
            as_of=None, rank: int | None = None) -> dict:
    """Return traceable facts and cautious prose for a selected well snapshot."""
    prompt = (question or '').lower().strip()
    if as_of is None:
        as_of = row.get('week_end', date.today())
    hist = _history_as_of(history, as_of)
    label = str(row.get('uwi', 'selected well'))
    when = str(pd.Timestamp(as_of).date())
    facts = [f'Well: {label} | As-of: {when} | Source: selected historical well-week']
    facts += [f'{nice}: {printable(row.get(field), unit, digits)} ({field})'
              for field, (nice, unit, digits) in FIELDS.items()]
    unknown = [field for field in SENSOR_FIELDS if printable(row.get(field)) == 'not available']
    notes = []

    if any(key in prompt for key in ('missing', 'sensor', 'quality', 'available', 'instrument')):
        heading = 'Telemetry coverage and data limitations'
        notes = [f'{len(SENSOR_FIELDS) - len(unknown)} of {len(SENSOR_FIELDS)} pump-related channels have a numeric value in this snapshot.',
                 'Missing channels: ' + (', '.join(unknown) if unknown else 'none among these four channels') + '.',
                 'A missing sensor may reflect well type, instrumentation coverage, or missing reporting; it is not evidence of a fault.']
    elif any(key in prompt for key in ('change', 'trend', 'recent', 'week', 'production', 'decline')):
        heading = 'Changes visible in observed well history'
        notes = []
        if len(hist) > 1:
            recent = hist.iloc[-1]
            prev = hist.iloc[-2]
            for key, (nice, unit, digits) in FIELDS.items():
                try:
                    current = float(recent[key]); previous = float(prev[key])
                    if math.isfinite(current) and math.isfinite(previous):
                        notes.append(f'{nice}: {previous:.{digits}f} → {current:.{digits}f}{unit}; change {current - previous:+.{digits}f}{unit} between {prev.week_end.date()} and {recent.week_end.date()}.')
                except (KeyError, TypeError, ValueError):
                    continue
        if not notes:
            notes = ['At least two valid historical snapshots are required to quantify changes; only currently available values can be reported.']
        notes.append('These are associations and trend descriptions, not evidence of a confirmed mechanical cause.')
    elif any(key in prompt for key in ('mainten', 'work order', 'service', 'repair', 'job')):
        heading = 'Maintenance context available at this date'
        jobs = row.get('prior_jobs_90d')
        notes = ([f'Prior recorded maintenance jobs over 90 days: {printable(jobs, precision=0)}.'] if jobs is not None
                 else ['Historical maintenance-job context is not included for this example.'])
        if row.get('last_job_type') is not None and pd.notna(row.get('last_job_type')):
            notes.append(f'Last recorded job type: {row["last_job_type"]}.')
        notes.append('A contemporaneous or later job must not be used to explain a prediction made before that job was recorded.')
    elif any(key in prompt for key in ('saving', 'money', 'cost', 'value', 'impact', 'recover', 'roi', 'profit')):
        heading = 'What the records do—and do not—say about value'
        notes = [f'Estimated lost oil recorded in this two-week summary: {printable(row.get("lost_per_week_2w"), " m³/week")}.',
                 'This is a historical exposure estimate, NOT recoverable volume or demonstrated avoided losses.',
                 'A net-value estimate requires verified interventions, actual inspection costs, counterfactual performance and operating assumptions. The simulator is hypothetical.']
    elif any(key in prompt for key in ('check', 'action', 'inspect', 'next', 'recommend')):
        heading = 'Evidence-based operator review checklist'
        notes = ['Confirm the well is currently producing and that each relevant measurement is recent and reliable.',
                 'Review field operating envelopes, observed downtime codes, maintenance history and reporting timestamps.',
                 'Ask an authorized field specialist whether any additional investigation is warranted.',
                 'Record the inspection decision and subsequent outcome to support prospective evaluation.']
    else:
        heading = 'Why this well may merit attention'
        score = printable(row.get('relative_risk_score'), precision=3)
        notes = [f'This historical snapshot has relative model score {score}.' +
                 (f' The well ranked {rank} within that week.' if rank is not None else ''),
                 f'Observed mean oil rate: {printable(row.get("oil_daily_avg_2w"), " m³/day")}; downtime fraction: {printable(row.get("down_fraction_2w"), precision=2)}.',
                 f'The tool reports {len(unknown)} unavailable pump-related sensor values among the four checked channels.',
                 'Inspect the measured history before making a field decision. A risk score is not a calibrated failure probability.']

    return {
        'title': heading, 'well': label, 'as_of': when, 'evidence': facts,
        'answer': notes, 'limitations': [
            'Retrospective research prototype; no independently verified failure prevention.',
            'No mechanical root-cause diagnosis or autonomous control action.',
            'No future-dated records may be used as evidence for this snapshot.',
        ],
        'engine': 'Deterministic evidence assistant (not a generative LLM)'
    }
