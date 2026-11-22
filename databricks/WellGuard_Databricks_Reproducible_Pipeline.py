# Databricks notebook source
# DATA TITANS - Prepared workflow, NOT an executed historical artifact

# MAGIC %md
# MAGIC # WellGuard AI — Databricks Data Engineering & Audit Notebook
# MAGIC 
# MAGIC **DATA TITANS | DSMLC Datathon 2026 | Prepared reproducibility workflow — NOT previously executed in Databricks**
# MAGIC 
# MAGIC Research question: *How can predictive maintenance and production optimization analytics help Strathcona Resources identify high-priority wells, prioritize inspections, and support data-driven decisions to reduce operational losses?*
# MAGIC 
# MAGIC This importable Databricks notebook analyzes the **three original restricted Parquet files** and demonstrates their preparation for the local WellGuard predictive-maintenance workflow. It provides:
# MAGIC 
# MAGIC 1. Source ingestion, schema and row-count checks.
# MAGIC 2. Missingness, impossible readings and conservative duplicate handling.
# MAGIC 3. Weekly well-level aggregation and historical operating features.
# MAGIC 4. Job features using **only job start dates strictly before** the scoring snapshot.
# MAGIC 5. Recorded-failure labels for **days 1–14 after** a weekly snapshot (evaluation only).
# MAGIC 6. Chronological training/validation/test dates, eligibility checks, and aggregate counts.
# MAGIC 7. Verification, read-only aggregate queries and methodology notes.
# MAGIC 
# MAGIC **Confidentiality:** Upload the Parquet files only to a workspace/storage location explicitly approved by the event organizers and data owner. Do not upload source rows, private UWIs, detailed well outputs, credentials, or screenshots of restricted records to public GitHub. This notebook itself contains no source data or embedded secrets.
# MAGIC 
# MAGIC **Scope:** The production WellGuard model is trained separately with the Python scripts in `research_source/`. This notebook does **not** claim Databricks trained the LightGBM models, and its Spark outputs are not a claimed bit-for-bit reproduction of the pandas feature pipeline. The final metrics (e.g. AP/recall) must come from the evaluated model artifacts, not from this unexecuted notebook.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
from functools import reduce
from datetime import date

# Supply an approved restricted workspace path, e.g. /Volumes/<catalog>/<schema>/<volume>.
dbutils.widgets.text('private_data_dir', '', 'Approved private dataset directory')
DATA_DIR = dbutils.widgets.get('private_data_dir').strip().rstrip('/')
if not DATA_DIR or '<' in DATA_DIR or DATA_DIR.startswith('dbfs:/FileStore'):
    raise ValueError('Set private_data_dir to an APPROVED restricted path. Do not use public FileStore.')

SOURCE_FILES = {
    'daily': 'daily_production.parquet',
    'jobs': 'well_jobs.parquet',
    'failures': 'production_failures.parquet',
}
daily_raw = spark.read.parquet(f'{DATA_DIR}/{SOURCE_FILES["daily"]}')
jobs_raw = spark.read.parquet(f'{DATA_DIR}/{SOURCE_FILES["jobs"]}')
failures_raw = spark.read.parquet(f'{DATA_DIR}/{SOURCE_FILES["failures"]}')
print('Loaded three private Parquet datasets. Only aggregate results will be displayed.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Source verification and data dictionary
# MAGIC 
# MAGIC Official Dataset Handbook reports **5,075,590 daily rows, 15,610 job records and 2,873 failure records**, over 2,905 daily-history wells. Verify these dynamically; different files or revisions may give different counts. Never claim a row count based solely on this document.

# COMMAND ----------

required = {
 'daily': {'uwi','production_date','well_daily_status','hours_down','oil_produced_m3','water_produced_m3','lost_oil_m3',
           'pump_efficiency_pct','pcp_pump_rpm','pcp_pump_rod_torque_ftlb','pcp_pump_electric_current_a'},
 'jobs': {'uwi','start_date'},
 'failures': {'uwi','failure_start_date'}
}
for name, frame in [('daily',daily_raw),('jobs',jobs_raw),('failures',failures_raw)]:
    missing = required[name] - set(frame.columns)
    if missing:
        raise ValueError(f'{name}: missing expected columns: {sorted(missing)}')
source_audit = [
    ('Daily production', daily_raw.count(),daily_raw.select('uwi').where(F.col('uwi').isNotNull()).distinct().count()),
    ('Maintenance jobs', jobs_raw.count(),jobs_raw.select('uwi').where(F.col('uwi').isNotNull()).distinct().count()),
    ('Equipment failures',failures_raw.count(),failures_raw.select('uwi').where(F.col('uwi').isNotNull()).distinct().count()),
]
display(spark.createDataFrame(source_audit, ['source','rows','non_null_unique_wells']))
# Exact official counts are a provenance cross-check, not an assumption in transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data-quality audit: missing keys, duplicates, invalid operating readings
# MAGIC 
# MAGIC WellGuard excludes **all rows in repeated `(uwi, production_date)` groups**, even exact duplicates, rather than selecting an arbitrary row. Missing values stay missing. Negative volumes, out-of-range hours and pump-efficiency values outside **0–1** are withheld from those measurements (the official handbook stores `_pct` as a fraction).

# COMMAND ----------

keyed = (daily_raw
    .withColumn('production_date',F.to_date('production_date'))
    .withColumn('uwi',F.trim(F.col('uwi').cast('string'))))
missing_key_rows = keyed.where(F.col('uwi').isNull() | (F.col('uwi')=='') | F.col('production_date').isNull()).count()
valid_keys = keyed.where(F.col('uwi').isNotNull() & (F.col('uwi')!='') & F.col('production_date').isNotNull())
duplicate_groups = valid_keys.groupBy('uwi','production_date').count().filter('count > 1').select('uwi','production_date')
num_dup_groups = duplicate_groups.count()
duplicate_rows = valid_keys.join(duplicate_groups,on=['uwi','production_date'],how='inner').count()
daily_clean_keys = valid_keys.join(duplicate_groups,on=['uwi','production_date'],how='left_anti')
print({'missing_key_rows':missing_key_rows,'duplicate_well_date_groups_excluded':num_dup_groups,
       'rows_in_duplicate_groups':duplicate_rows,'retained_daily_rows':daily_clean_keys.count()})
assert daily_clean_keys.groupBy('uwi','production_date').count().filter('count>1').limit(1).count()==0

# COMMAND ----------

clean_numeric = [
 'oil_produced_m3','water_produced_m3','lost_oil_m3','hours_down','pump_efficiency_pct',
 'pcp_pump_rpm','pcp_pump_rod_torque_ftlb','pcp_pump_electric_current_a',
]
d = daily_clean_keys
for c in clean_numeric:
    d = d.withColumn(c,F.col(c).cast('double'))

quality_expressions = [F.count(F.lit(1)).alias('retained_source_rows')]
for c in clean_numeric:
    quality_expressions.append(F.sum(F.when(F.col(c).isNull(),1).otherwise(0)).alias('missing_'+c))
    if c=='hours_down': cond=(F.col(c)<0)|(F.col(c)>24)
    elif c=='pump_efficiency_pct': cond=(F.col(c)<0)|(F.col(c)>1)
    else: cond=(F.col(c)<0)
    quality_expressions.append(F.sum(F.when(cond,1).otherwise(0)).alias('invalid_'+c))
quality_summary = d.agg(*quality_expressions)
display(quality_summary)

def safe_nonnegative(c):return F.when(F.col(c)>=0,F.col(c)).otherwise(F.lit(None).cast('double'))
for c in ['oil_produced_m3','water_produced_m3','lost_oil_m3','pcp_pump_rpm','pcp_pump_rod_torque_ftlb','pcp_pump_electric_current_a']:
    d = d.withColumn(c+'_ok',safe_nonnegative(c))
d = (d.withColumn('hours_down_ok',F.when(F.col('hours_down').between(0,24),F.col('hours_down')))
       .withColumn('pump_efficiency_ok',F.when(F.col('pump_efficiency_pct').between(0,1),F.col('pump_efficiency_pct'))))
# No fabricated sensor readings, no imputation based on future data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Weekly production features (past and current observations only)
# MAGIC 
# MAGIC A model row represents **well × Sunday-ending week**. The date calculation below is equivalent to the project's Sunday `week_end` convention. Weekly aggregates only read production rows **on or before that week_end**. `F.avg` ignores missing values without inventing measurements. Percent downtime-day fractions and hour-based downtime are different measures.

# COMMAND ----------

# Spark dayofweek: Sunday=1, Monday=2, ..., Saturday=7.
d = (d.withColumn('week_end',F.date_add(F.col('production_date'), (8-F.dayofweek('production_date'))%7))
       .withColumn('is_producing',F.when(F.col('well_daily_status')=='Producing',1).otherwise(0))
       .withColumn('is_down_day',F.when(F.col('hours_down_ok')>0,1).otherwise(0)))
weekly = (d.groupBy('uwi','week_end').agg(
    F.countDistinct('production_date').alias('n_days'),
    F.sum('is_producing').alias('production_days'),
    F.sum('is_down_day').alias('down_days'),
    F.sum('hours_down_ok').alias('downtime_hours'),
    F.sum('oil_produced_m3_ok').alias('oil_total_m3'),
    F.sum('water_produced_m3_ok').alias('water_total_m3'),
    F.sum('lost_oil_m3_ok').alias('lost_oil_total_m3'),
    F.avg('oil_produced_m3_ok').alias('oil_daily_avg'),
    F.avg('pump_efficiency_ok').alias('eff_avg'),
    F.avg('pcp_pump_rpm_ok').alias('rpm_avg'),
    F.avg('pcp_pump_rod_torque_ftlb_ok').alias('torque_avg'),
    F.avg('pcp_pump_electric_current_a_ok').alias('current_avg'),
    F.count('oil_produced_m3_ok').alias('valid_oil_days'),
    F.count('lost_oil_m3_ok').alias('valid_lost_oil_days'),
 )
 .withColumn('down_fraction',F.when(F.col('n_days')>0,F.col('down_days')/F.col('n_days')))
 .withColumn('watercut',F.when((F.col('oil_total_m3')+F.col('water_total_m3'))>0,
     F.col('water_total_m3')/(F.col('water_total_m3')+F.col('oil_total_m3'))))
 .withColumn('lost_per_week',F.col('lost_oil_total_m3')))
weekly.createOrReplaceTempView('wg_weekly_source')
print('Well-week rows:',weekly.count(),'Unique wells:',weekly.select('uwi').distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Trailing operating trends and well-specific baselines
# MAGIC 
# MAGIC The **reference WellGuard model** uses trailing two-week historical averages. The later V9 experiment also evaluates consecutive-week trends and earlier baseline deviations. This notebook implements a representative Spark exploration of those candidates. A direct numerical comparison must use the project's Python runner and verify identical feature definitions.

# COMMAND ----------

per_well = Window.partitionBy('uwi').orderBy('week_end')
w = weekly.withColumn('prior_week',F.lag('week_end').over(per_well))
w = w.withColumn('gap_from_prior_week',F.datediff('week_end','prior_week'))
w = w.withColumn('new_segment',F.when((F.col('prior_week').isNull())|(F.col('gap_from_prior_week')!=7),1).otherwise(0))
w = w.withColumn('segment_id',F.sum('new_segment').over(per_well.rowsBetween(Window.unboundedPreceding,Window.currentRow)))
continuous = Window.partitionBy('uwi','segment_id').orderBy('week_end')
inputs = ['down_fraction','downtime_hours','oil_daily_avg','lost_per_week','watercut',
          'eff_avg','rpm_avg','torque_avg','current_avg','production_days']
for name in inputs:
    w = w.withColumn(name+'_2w',F.avg(name).over(continuous.rowsBetween(-1,0)))
for name in ['oil_daily_avg','down_fraction','rpm_avg','torque_avg','current_avg','eff_avg','lost_per_week']:
    w = w.withColumn(name+'_change_1w',F.col(name)-F.lag(name).over(continuous))
    w = w.withColumn(name+'_history_std_4w',F.stddev_pop(name).over(continuous.rowsBetween(-4,-1)))
    # The fitted V9 Python model uses a PAST 8-week MEDIAN; this is an exploratory Spark equivalent.
    w = w.withColumn(name+'_prior_median_8w',F.expr('percentile_approx('+name+', 0.5)').over(continuous.rowsBetween(-8,-1)))
    w = w.withColumn(name+'_past_baseline_delta_8w',F.col(name)-F.col(name+'_prior_median_8w'))
w = w.withColumn('month',F.month('week_end'))
# Caution: consecutive segment prevents lag/rolling windows from bridging missing weeks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Maintenance history — strict prior-date join
# MAGIC 
# MAGIC Only distinct recorded job **start dates before the prediction week_end** enter these candidate features. A `0` means no such start is in the supplied log, *not* that no maintenance took place. The dataset does not include point-in-time publication timestamps, so delayed record entry remains a potential limitation.

# COMMAND ----------

job_dates = (jobs_raw.select(F.trim(F.col('uwi').cast('string')).alias('job_uwi'),
                             F.to_date('start_date').alias('job_start'))
             .filter('job_uwi IS NOT NULL AND job_uwi != "" AND job_start IS NOT NULL')
             .dropDuplicates(['job_uwi','job_start']))
keys=w.select('uwi','week_end').distinct()
join_condition = (F.col('uwi')==F.col('job_uwi')) & (F.col('job_start')<F.col('week_end'))
job_history = (keys.join(job_dates,join_condition,'left')
  .groupBy('uwi','week_end').agg(
     F.sum(F.when(F.col('job_start')>=F.date_sub(F.col('week_end'),30),1).otherwise(0)).cast('int').alias('job_starts_30d'),
     F.sum(F.when(F.col('job_start')>=F.date_sub(F.col('week_end'),90),1).otherwise(0)).cast('int').alias('job_starts_90d'),
     F.max('job_start').alias('last_job_start')))
w = (w.join(job_history,['uwi','week_end'],'left')
 .withColumn('days_since_last_job_start',F.when(F.col('last_job_start').isNotNull(),
     F.least(F.lit(365),F.datediff('week_end','last_job_start')))))
assert w.filter(F.col('last_job_start')>=F.col('week_end')).limit(1).count()==0
print('Maintenance date cutoff verified for',w.count(),'well-week rows.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Recorded-failure target (evaluation only): next 1–14 days
# MAGIC 
# MAGIC A positive target means a **recorded** equipment failure whose start date is strictly after `week_end` and no more than 14 days later. **Never** use `target_14d`, `next_failure_date`, or future failure records as model predictors. Do not evaluate outcomes for weeks lacking a complete 14-day observation window. The benchmark uses a final labeled test week of **2026-10-04**.

# COMMAND ----------

f_events = (failures_raw
   .select(F.trim(F.col('uwi').cast('string')).alias('failure_uwi'),
           F.to_date('failure_start_date').alias('failure_date'))
   .where('failure_uwi IS NOT NULL AND failure_uwi != "" AND failure_date IS NOT NULL')
   .dropDuplicates(['failure_uwi','failure_date']))
future_cond = ((F.col('uwi')==F.col('failure_uwi')) &
               (F.col('failure_date')>F.col('week_end')) &
               (F.col('failure_date')<=F.date_add(F.col('week_end'),14)))
future_outcomes = (w.select('uwi','week_end').join(F.broadcast(f_events),future_cond,'left')
  .groupBy('uwi','week_end').agg(F.min('failure_date').alias('next_failure_date')))
prior_cond = ((F.col('uwi')==F.col('failure_uwi')) & (F.col('failure_date')<=F.col('week_end')))
prior_outcomes = (w.select('uwi','week_end').join(F.broadcast(f_events),prior_cond,'left')
  .groupBy('uwi','week_end').agg(F.max('failure_date').alias('prior_failure_date')))
w = (w.join(future_outcomes,['uwi','week_end'],'left')
      .join(prior_outcomes,['uwi','week_end'],'left')
      .withColumn('target_14d',F.when(F.col('next_failure_date').isNotNull(),1).otherwise(0))
      .withColumn('days_since_failure',F.datediff('week_end','prior_failure_date'))
      .withColumn('failure_in_previous_90d',F.when(F.col('days_since_failure').between(0,90),1).otherwise(0))
      .withColumn('log_days_since_failure',F.log1p(F.coalesce(F.least(F.col('days_since_failure'),F.lit(365)),F.lit(365)))))
assert w.filter(F.col('next_failure_date')<=F.col('week_end')).limit(1).count()==0
assert w.filter(F.col('next_failure_date')>F.date_add(F.col('week_end'),14)).limit(1).count()==0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fixed chronological splits and eligibility (the original WellGuard research setup)
# MAGIC 
# MAGIC Reference: **48,865 train**, **11,149 validation**, **22,580 test** well-week rows in the saved Python benchmark. The exact numbers below should be derived from *these* source Parquet files; minor Spark/Pandas implementation differences must be reconciled before reporting equivalence.
# MAGIC 
# MAGIC Split rules: training Jun 2025–Mar 1 2026; validation Mar 22–May 17 2026; test Jun 7–Oct 4 2026. The unused weeks provide a gap greater than the 14-day horizon. Model selection uses validation only. Tests are not reselected or tuned based on test results.

# COMMAND ----------

TRAIN_START, TRAIN_END = date(2025,6,1),date(2026,3,1)
VAL_START, VAL_END = date(2026,3,22),date(2026,5,17)
TEST_START, TEST_END = date(2026,6,7),date(2026,10,4)
assert (VAL_START-TRAIN_END).days>14 and (TEST_START-VAL_END).days>14
assert (date(2026,10,18)-TEST_END).days>=14
eligible = (w.filter((F.col('n_days')>=6)&(F.col('production_days')>=3)&
          (F.col('gap_from_prior_week')==7)&
          (~F.col('days_since_failure').between(0,14) | F.col('days_since_failure').isNull())&
          F.col('week_end').between(F.lit('2025-06-01'),F.lit('2026-10-04'))))
split_df = eligible.withColumn('period',F.when(F.col('week_end').between('2025-06-01','2026-03-01'),'Train')
    .when(F.col('week_end').between('2026-03-22','2026-05-17'),'Validation')
    .when(F.col('week_end').between('2026-06-07','2026-10-04'),'Test')
    .otherwise('Gap / not evaluated'))
split_df.createOrReplaceTempView('wg_modeling_split_audit')
display(split_df.groupBy('period').agg(F.count('*').alias('well_weeks'),F.sum('target_14d').alias('positive_well_weeks'),
  F.countDistinct('uwi').alias('distinct_wells')).orderBy('period'))
assert split_df.filter("period='Test' AND week_end > DATE'2026-10-04'").limit(1).count()==0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. SQL checks and audit verdict
# MAGIC 
# MAGIC These are **aggregate-only** outputs. Detailed individual wells and model scores must remain in the authorized private workspace. The full WellGuard LightGBM training, ROC/PR curves, 25/50/100 inspection-policy comparisons and original-data audits live in `research_source/` and are run separately on an authorized machine.

# COMMAND ----------

# Aggregate-only SQL. This notebook intentionally does not CREATE public tables,
# persist well-level results, train a model or upload private data to GitHub.
q = """
SELECT period,
       COUNT(*) AS well_week_rows,
       SUM(target_14d) AS recorded_positive_well_weeks,
       ROUND(AVG(CAST(target_14d AS DOUBLE)), 5) AS observed_positive_rate,
       COUNT(DISTINCT uwi) AS distinct_wells
FROM wg_modeling_split_audit
WHERE period IN ('Train','Validation','Test')
GROUP BY period
ORDER BY CASE period WHEN 'Train' THEN 1 WHEN 'Validation' THEN 2 ELSE 3 END
"""
display(spark.sql(q))
print('AUDIT NOTES:')
print('- Checked: key uniqueness, strict date cutoffs, and chronological gaps within this notebook.')
print('- Still required: full source-to-model parity test, late-entry assessment and external-field validation.')
print('- No source records were exported from this notebook.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. How this notebook fits the competition submission
# MAGIC 
# MAGIC - **Official deliverables:** GitHub repository, title/team/research question, source code, methods/validation, 5-minute video link, presentation slides, external references and AI-use disclosure. Databricks itself is **optional**, not a condition of entry.
# MAGIC - **Team action:** Import this notebook into Databricks, attach an approved compute session, configure `private_data_dir` and run cells in order **only if authorized**. Record your actual run status, errors and derived counts. The existence of this file does **not** prove it was executed.
# MAGIC - **Private vs public:** Commit this notebook (code only) to the public repository. Never commit source Parquet files, private tables, restricted prints/screenshots, model scores or workspace access details.
# MAGIC - **Implementation:** Dashboard preview entry point: `streamlit_app.py`; full local application: `research_source/app.py`. Actual modeling: `research_source/scripts/02_failure_ranking.py` and `16_experimental_model_upgrade.py`.
# MAGIC - **Research limitation:** Later V9 results (AP about 0.2023, event recall about 26.44% @25 reviews/week) are **exploratory on a previously examined historical test period** and are not independent prospective evidence or demonstrated avoided losses.
