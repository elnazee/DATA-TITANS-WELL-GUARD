"""Entirely fictional showcase data for public WellGuard Streamlit UI.

These rows are synthetic examples generated from a fixed random seed. They are NOT
anonymized Strathcona data, model outputs or reconstructed source observations.
The separate benchmark.json contains saved aggregate research results.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ZONES = ("North Sector", "Central Sector", "South Sector", "West Sector")
JOB_TYPES = ("Pump service", "Rod inspection", "Tubing maintenance", "Electrical service")


def make_demo(n_wells: int = 480, n_weeks: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20261122)
    ids = np.array([f"DEMO-{i:04d}" for i in range(1, n_wells + 1)])
    zone_index = rng.integers(0, 4, n_wells)
    zone = np.array(ZONES, dtype=object)[zone_index]
    # Coordinates are on an arbitrary illustrative canvas; no actual latitude/longitude.
    gx = rng.uniform(0, 17, n_wells) + np.array([5, 30, 32, 9])[zone_index]
    gy = rng.uniform(0, 15, n_wells) + np.array([33, 21, 4, 8])[zone_index]
    equipment = np.array(["PCP", "Other artificial lift"], dtype=object)[rng.choice([0, 1], n_wells, p=[.74, .26])]
    baseline_oil = rng.uniform(13, 95, n_wells)
    baseline_rpm = rng.uniform(110, 330, n_wells)
    baseline_risk = rng.normal(0, 1, n_wells)
    fleet = pd.DataFrame({"well_id": ids, "zone": zone, "equipment": equipment, "x": gx, "y": gy})
    weeks = pd.date_range("2026-05-17", periods=n_weeks, freq="W-SUN")
    records = []
    jobs = []
    for t, week in enumerate(weeks):
        down_hours = np.clip(rng.exponential(7.0, n_wells) + np.maximum(baseline_risk, 0) * 4 + 4 * np.sin(t / 3 + zone_index), 0, 72)
        health_drift = 0.6 * np.sin(t / 3.1 + np.arange(n_wells) / 12) + rng.normal(0, .23, n_wells)
        oil_rate = np.clip(baseline_oil * (1 - down_hours / 168) * (1 + .065 * health_drift) + rng.normal(0, 3, n_wells), 0, None)
        rpm = np.clip(baseline_rpm * (1 + .11 * health_drift) + rng.normal(0, 12, n_wells), 25, 475)
        efficiency = np.clip(.7 - .09 * baseline_risk - .08 * np.abs(health_drift) + rng.normal(0, .07, n_wells), .18, .98)
        pressure = np.clip(1350 + 420 * baseline_risk + rng.normal(0, 170, n_wells) + 90 * np.sin(t / 4), 70, 3500)
        torque = np.clip(370 + 75 * baseline_risk + rng.normal(0, 30, n_wells), 60, 900)
        risk = 0.64 * baseline_risk + .65 * (down_hours / 28) + .57 * (0.72 - efficiency) * 5 + .28 * health_drift + rng.normal(0, .22, n_wells)
        risk = 1 / (1 + np.exp(-np.clip(risk, -10, 10)))
        lost_oil = np.maximum(0, oil_rate * down_hours / 24 * rng.uniform(.38, .85, n_wells))
        failure = rng.random(n_wells) < np.clip(.015 + .043 * risk, .01, .09)
        maintain = rng.random(n_wells) < .034
        for i, yes in enumerate(maintain):
            if yes:
                jobs.append({"well_id": ids[i], "zone": zone[i], "week_end": week,
                             "job_type": JOB_TYPES[int(rng.integers(0, len(JOB_TYPES)))],
                             "field_estimated_cost_cad": round(float(rng.uniform(4000, 38000)), 2),
                             "before_oil_m3d": round(float(oil_rate[i] * rng.uniform(.85, 1.03)), 2),
                             "after_oil_m3d": round(float(oil_rate[i] * rng.uniform(.93, 1.15)), 2)})
        records.append(pd.DataFrame({"well_id": ids, "zone": zone, "week_end": week,
                         "oil_rate_m3d": oil_rate.round(2), "oil_volume_m3": (oil_rate * 7).round(2),
                         "downtime_hours": down_hours.round(2), "estimated_lost_oil_m3": lost_oil.round(2),
                         "pump_rpm": rpm.round(1), "pump_efficiency": efficiency.round(3),
                         "tubing_pressure_kpa": pressure.round(1), "rod_torque_ftlb": torque.round(1),
                         "relative_risk_score": risk.round(4), "demo_failure_record": failure,
                         "demo_job_record": maintain}))
    history = pd.concat(records, ignore_index=True)
    # Deliberate example missing channels for demonstrating data reliability controls.
    missing = np.random.default_rng(603)
    for col, rate in [("pump_rpm", .13), ("pump_efficiency", .22), ("rod_torque_ftlb", .28), ("tubing_pressure_kpa", .12)]:
        history.loc[missing.random(len(history)) < rate, col] = np.nan
    return fleet, history, pd.DataFrame(jobs)


def schedule_demo(history: pd.DataFrame, as_of, capacity: int, zone: str = "All sectors", cooldown: int = 0) -> pd.DataFrame:
    """Risk-sort synthetic latest rows and optionally exclude recently selected demo wells.

    Cooldown starts with the earliest eligible week and is applied to the
    *entire synthetic fleet*, before any zone filter, for stable scenarios.
    """
    data = history[history.week_end <= pd.Timestamp(as_of)]
    dates = sorted(data.week_end.unique())
    last_selected: dict[str, int] = {}
    selected = pd.DataFrame()
    for idx, week in enumerate(dates):
        rows = data[data.week_end.eq(week)].sort_values(
            ["relative_risk_score", "well_id"], ascending=[False, True]
        )
        if cooldown:
            rows = rows[~rows.well_id.map(lambda x: idx - last_selected.get(x, -1000) <= cooldown)]
        selected = rows.head(capacity).copy()
        for w in selected.well_id:
            last_selected[w] = idx
    if zone != "All sectors":
        selected = selected[selected.zone.eq(zone)]
    selected = selected.sort_values("relative_risk_score", ascending=False).reset_index(drop=True)
    selected.insert(0, "review_rank", np.arange(1, len(selected) + 1))
    return selected
