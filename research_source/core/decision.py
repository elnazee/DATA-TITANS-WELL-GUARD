"""Deterministic, auditable operator support. NOT autonomous maintenance advice.

Alerts are uncalibrated model relative ranks. This module neither diagnoses
mechanical failure nor estimates real prevented failures.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

FEATURE_LABELS = {
    'down_fraction_2w': 'recent downtime fraction',
    'downtime_hours_2w': 'observed downtime hours',
    'oil_daily_avg_2w': 'daily oil production',
    'lost_per_week_2w': 'historical estimated production loss',
    'watercut_2w': 'produced-water fraction',
    'eff_avg_2w': 'observed pump efficiency',
    'rpm_avg_2w': 'pump RPM',
    'torque_avg_2w': 'rod torque',
    'current_avg_2w': 'electric current',
    'production_days_2w': 'observed production days',
    'failure_in_previous_90d': 'previous recorded failure',
    'log_days_since_failure': 'time since previous recorded failure',
    'month': 'calendar month'
}


def available(value):
    try:
        return value is not None and isfinite(float(value))
    except (ValueError, TypeError):
        return False


def format_measure(value, unit='', precision=1):
    return f'{float(value):,.{precision}f}{unit}' if available(value) else 'not recorded'


def well_evidence(row:dict, rank:int|None=None)->list[tuple[str,str]]:
    """Human-readable facts from a *single as-of snapshot*, with no causal claims."""
    evidence=[]
    if rank is not None:
        evidence.append(('Rank',f'{rank} among observed eligible wells on the selected snapshot'))
    evidence += [
        ('Oil rate',f'{format_measure(row.get("oil_daily_avg_2w"), " m³/day")} (two-week summary)'),
        ('Downtime',f'{format_measure(row.get("down_fraction_2w"), precision=2)} of observed days; {format_measure(row.get("downtime_hours_2w"), " h")} mean weekly hours'),
        ('Pump RPM',f'{format_measure(row.get("rpm_avg_2w"), " rpm")} (two-week summary)'),
        ('Estimated historic lost oil',f'{format_measure(row.get("lost_per_week_2w"), " m³/week")} (not recoverable savings)'),
        ('Pump efficiency',f'{format_measure(row.get("eff_avg_2w"), precision=2)} (source unit is fractional when valid)'),
    ]
    driver_names=[str(row.get(f'driver_{i}','')) for i in (1,2,3)]
    driver_names=[FEATURE_LABELS.get(k,k) for k in driver_names if k and k!='nan']
    if driver_names:
        evidence.append(('Model drivers',', '.join(driver_names)+' (local model contributions, not confirmed causes)'))
    return evidence


def investigator_report(row:dict, rank:int|None=None)->str:
    """Safe fallback report from data; never generates a fabricated root cause."""
    label=str(row.get('uwi','Selected well'))
    facts=well_evidence(row,rank)
    lines=[f'### Evidence brief — {label}',
           'This is a **retrospective inspection priority**, not a failure diagnosis or calibrated probability.',
           '#### Observations as of the selected week']
    lines += [f'- **{k}:** {v}' for k,v in facts]
    lines += ['', '#### Suggested review sequence',
        '1. Verify the latest sensor readings and whether the well is producing; compare against field operating envelopes.',
        '2. Review recent downtime codes, operational changes and the historical maintenance log in the approved system.',
        '3. Ask an authorized field specialist to assess possible root causes before deciding whether inspection is appropriate.',
        '', '**Uncertainty:** Missing telemetry, maintenance-reporting linkage and sparse pre-2025 failure labels may affect the ranking.',
        '**No intervention has been proven to prevent a failure.**']
    return '\n'.join(lines)


def scenario_weekly_gross_exposure(reported_events:int, mean_lost_m3:float, fraction_addressable:float,
                                   inspection_effectiveness:float, realized_value_cad_per_m3:float,
                                   inspections:int, cost_per_inspection_cad:float)->dict:
    """Illustrative gross exposure only; no causal guarantee.

    Every input is explicitly user-assumed. Avoid inferring mean lost volume from
    historic total lost_oil_m3 because it is not a causal loss per equipment failure.
    """
    if not (0<=fraction_addressable<=1 and 0<=inspection_effectiveness<=1):
        raise ValueError('Fractions must be between 0 and 1')
    if min(reported_events,mean_lost_m3,realized_value_cad_per_m3,inspections,cost_per_inspection_cad)<0:
        raise ValueError('Volumes, counts and prices must be nonnegative')
    addressable=reported_events*fraction_addressable*inspection_effectiveness*mean_lost_m3
    gross=addressable*realized_value_cad_per_m3
    inspection_cost=inspections*cost_per_inspection_cad
    return {'hypothetical_addressed_events':reported_events*fraction_addressable*inspection_effectiveness,
            'hypothetical_oil_m3':addressable,'hypothetical_gross_value_cad':gross,
            'hypothetical_inspection_cost_cad':inspection_cost,
            'hypothetical_net_cad':gross-inspection_cost}
