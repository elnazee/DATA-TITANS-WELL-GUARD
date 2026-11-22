"""Audits competition Parquet data using the standard PyArrow reader.
Usage: python scripts/01_data_audit.py --data-dir data --out-dir outputs
Writes only summary JSON and a small downtime-category aggregate CSV to output dir.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--out-dir',type=Path,default=Path('outputs'))
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    prod=pd.read_parquet(args.data_dir/'daily_production.parquet',columns=['uwi','production_date','well_daily_status','hours_down','hours_on_prod','lost_oil_m3','oil_produced_m3','downtime_code_1','pump_efficiency_pct','pcp_pump_rpm','pcp_pump_electric_current_a','pcp_pump_rod_torque_ftlb'])
    jobs=pd.read_parquet(args.data_dir/'well_jobs.parquet')
    fail=pd.read_parquet(args.data_dir/'production_failures.parquet')
    for df,col in [(prod,'production_date'),(jobs,'start_date'),(fail,'failure_start_date')]:df[col]=pd.to_datetime(df[col])
    duplicates=int(prod.duplicated(['uwi','production_date']).sum());same_exact=int(prod.duplicated().sum())
    # Are well-date duplicates exact duplicates across the selected audit columns?
    conflicts=int(sum(g.drop_duplicates().shape[0]>1 for _,g in prod.loc[prod.duplicated(['uwi','production_date'],keep=False)].groupby(['uwi','production_date'])))
    loss=prod['lost_oil_m3'].clip(lower=0).fillna(0)
    reason=prod.assign(lost_positive=loss).groupby('downtime_code_1',dropna=False).agg(days=('uwi','size'),estimated_lost_oil_m3=('lost_positive','sum')).sort_values('estimated_lost_oil_m3',ascending=False)
    reason.to_csv(args.out_dir/'aggregate_downtime_reasons.csv')
    fail_month=fail.groupby(fail.failure_start_date.dt.strftime('%Y-%m')).size().to_dict()
    # Same-date joins may multiply multiple jobs per well-date. Count distinct failures matched.
    fkeys=fail[['uwi','failure_start_date']].drop_duplicates()
    jkeys=jobs[['uwi','start_date']].drop_duplicates()
    same_date=fkeys.merge(jkeys,left_on=['uwi','failure_start_date'],right_on=['uwi','start_date'],how='inner')
    result={'production_rows':len(prod),'unique_wells':int(prod.uwi.nunique()),'duplicate_well_dates':duplicates,'exact_duplicates_in_selected_columns':same_exact,'conflicting_well_dates_in_selected_columns':conflicts,'production_range':[str(prod.production_date.min().date()),str(prod.production_date.max().date())],
     'failure_rows':len(fail),'failure_wells':int(fail.uwi.nunique()),'jobs_rows':len(jobs),'jobs_wells':int(jobs.uwi.nunique()),
     'failure_counts_by_month':fail_month,'same_date_failure_job_overlap_distinct_well_dates':len(same_date),
     'total_estimated_lost_oil_m3':float(loss.sum()),'production_status_counts':{str(k):int(v) for k,v in prod.well_daily_status.value_counts(dropna=False).items()},
     'sensor_nonnull_rate_on_producers':{c:float(pd.to_numeric(prod.loc[prod.well_daily_status.eq('Producing'),c],errors='coerce').notna().mean()) for c in ['pump_efficiency_pct','pcp_pump_rpm','pcp_pump_electric_current_a','pcp_pump_rod_torque_ftlb']},
     'data_warning':'2023-24 have virtually no documented failures; do not treat these as failure-free labels.'}
    (args.out_dir/'audit_summary.json').write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__':main()
