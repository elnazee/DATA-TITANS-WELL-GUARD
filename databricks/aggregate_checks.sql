-- WellGuard / DATA TITANS. Prepared aggregate-only SQL reference; NOT an executed query log.
-- Execute inside the authorized Databricks notebook after it creates the temporary views.
-- Never publish UWI-level rows or proprietary source tables.

-- Weekly source coverage (source view created by notebook)
SELECT YEAR(week_end) AS yr, COUNT(*) AS well_weeks, COUNT(DISTINCT uwi) AS wells
FROM wg_weekly_source GROUP BY YEAR(week_end) ORDER BY yr;

-- Chronological split review (source view created by notebook)
SELECT period, COUNT(*) AS well_weeks, SUM(target_14d) AS positive_well_weeks,
       ROUND(AVG(CAST(target_14d AS DOUBLE)), 5) AS positive_rate
FROM wg_modeling_split_audit
GROUP BY period ORDER BY period;

-- No source records, UWIs or private model scores are printed by these queries.
