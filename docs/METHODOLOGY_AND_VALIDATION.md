# DATA TITANS — Methods, model validation and known limits

## Research target

The operational question is whether historical production and pump measurements can prioritize **active/eligible wells associated with a recorded failure beginning in the following 1–14 days**, within a fixed weekly review capacity. The label is a *recorded event*, not all physical failures, and model scores are relative ranks rather than calibrated probabilities.

## Analytical workflow

- Primary inputs: daily operating/production records (well-day grain), maintenance jobs (event grain), and recorded equipment failures (event grain). Align events to each well and timestamp; avoid multiplying rows by joining raw events directly to every production day.
- Data quality: verify keys and dates; inspect duplicates, missing production/sensor measurements, invalid efficiency values and placeholders. Do not invent missing measurements.
- Weekly snapshots: construct one observation per well and Sunday-ending week with only historical and current measurements. Feature families include trailing downtime, production and lost-oil summaries, sensor means, recent failure history, and seasonal month.
- Feature availability: future failures define labels *only*, while future jobs, outcomes and production measurements must not enter model predictors. Recorded job start dates are imperfect proxies for knowledge available to field staff.
- Baseline: reference model compared with a separate as-of failure-recency ranking and with limited inspection capacity (25, 50, or 100 candidate reviews/week).

## Chronological holdout design (saved reference)

| Segment | Dates | Saved eligible well-weeks |
|---|---|---:|
| Training | 2025-06-01 to 2026-03-01 | 48,865 |
| Validation | 2026-03-22 to 2026-05-17 | 11,149 |
| Test | 2026-06-07 to 2026-10-04 | 22,580 |

The gaps between periods exceed the forward 14-day label horizon; window-based features must be computed as-of, not using future outcomes. This is a retrospective evaluation, and time-aware holdout alone does not rule out every possible leakage or reporting-bias mechanism.

## Reference evaluation

The previously saved `public_data/benchmark.json` reports model ROC-AUC **0.8055**, AP **0.1637**, compared with a failure-recency baseline AP **0.1030**. The 25-review/week policy retrospectively flagged **116 of 643** eligible distinct test-horizon recorded failure events, against **75 of 643** for the recency baseline. Those eligible events are a specific evaluation subset; they must not be confused with all 2,873 original failure records. Higher review capacities increase the number of events captured at additional workload.

`public_data/monthly_summary.csv` gives the saved reference's month-by-month summary. The released data do not contain per-well predictions or the exact sample-level ROC/PR arrays. No uncertainty intervals or model curves should be inferred from aggregate data alone.

## Exploratory experiment

A later candidate experiment explored additional trend, prior-maintenance and inspection-policy features. The research period used for evaluation had also informed earlier development, so this candidate is treated as exploratory rather than a new independent holdout. Its preliminary scorecard is not part of the public reference benchmark. The candidate-study source code is in `research_source/scripts/16_experimental_model_upgrade.py`.

## Limitations that affect the business case

1. Historical failure logging is very uneven across years, so apparent risk and event distributions partly reflect recording practices.
2. Missing operating measurements vary by well and equipment; some coverage patterns can be confounded with technology and operational state.
3. Recorded failures are not all failures. Only eligible events in test snapshots enter event-recall denominators.
4. Maintenance jobs are historical observations, not proof that an inspection prevents a failure. Actual job costs and detailed repair outcomes are largely unavailable.
5. Economic outputs require declared assumptions about inspection cost, addressable events, success rates, and recoverable volume; no realized savings are established.
6. The public preview cannot independently regenerate private features, predictions, or performance curves without permission to access the restricted input data.
7. The optional Databricks notebook has not been run and has not been verified to have feature parity with the local pandas implementation.

This project demonstrates a *decision-support research methodology* and evidence about historical prioritization, not certified or prospective field performance.
