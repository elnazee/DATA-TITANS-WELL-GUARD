"""Private source-data quality gate (requires pyarrow; original files stay local).

This script quantifies known ambiguities without changing the modeling rules.
Aggregates are written to gitignored outputs/source_quality_gate.json.
The estimated financial fields are *not* realized costs/savings.
"""
from pathlib import Path
import argparse,json
import pandas as pd
import numpy as np


def quality(data_dir: Path)->dict:
    p=pd.read_parquet(data_dir/'daily_production.parquet',columns=['uwi','production_date','well_daily_status','pump_efficiency_pct','hours_down','oil_produced_m3','lost_oil_m3'])
    f=pd.read_parquet(data_dir/'production_failures.parquet',columns=['uwi','failure_start_date'])
    j=pd.read_parquet(data_dir/'well_jobs.parquet',columns=['uwi','start_date','actual_final_job_cost','field_est_total'])
    eff=pd.to_numeric(p['pump_efficiency_pct'],errors='coerce')
    valid=eff.dropna()
    n_exact=int(p.duplicated().sum())
    repeated=int(p.duplicated(['uwi','production_date']).sum())
    j_final=pd.to_numeric(j['actual_final_job_cost'],errors='coerce')
    job_est=pd.to_numeric(j['field_est_total'],errors='coerce')
    starts=pd.to_datetime(f['failure_start_date'],errors='coerce')
    failure_years={str(k):int(v) for k,v in starts.dt.year.value_counts().sort_index().items()}
    report={
        'data_provenance':'Restricted competition datasets; private audit only',
        'daily_rows':len(p),'daily_distinct_wells':int(p.uwi.nunique()),
        'duplicate_exact_rows':n_exact,'repeat_well_date_rows':repeated,
        'pump_efficiency_nonmissing':len(valid),
        'pump_efficiency_fraction_0_to_1':int(valid.between(0,1).sum()),
        'pump_efficiency_above_1_to_100':int((valid.gt(1) & valid.le(100)).sum()),
        'pump_efficiency_above_100':int((valid>100).sum()),
        'pump_efficiency_below_0':int((valid<0).sum()),
        'pump_efficiency_quantiles':{str(q):float(valid.quantile(q)) for q in (.01,.25,.5,.75,.95,.99)} if len(valid) else {},
        'failure_rows':len(f),'failure_years':failure_years,
        'job_rows':len(j),'job_final_actual_cost_nonmissing':int(j_final.notna().sum()),
        'job_field_estimated_cost_nonmissing':int(job_est.notna().sum()),
        'job_field_estimated_cost_negative':int((job_est<0).sum()),
    }
    # This is a genuine policy gate: conflicting duplicate dates change source semantics.
    duplicate_groups=p[p.duplicated(['uwi','production_date'],keep=False)].groupby(['uwi','production_date'],dropna=False)
    report['conflicting_well_date_groups']=int(sum(g.drop_duplicates().shape[0]>1 for _,g in duplicate_groups))
    report['decision']='Proceed after EXCLUDING all duplicated well-days; rerun model and update benchmark, subject to sensor-unit cautions' if report['conflicting_well_date_groups'] else 'Proceed with caution; test telemetry-unit assumptions and logging semantics'
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    report=quality(a.data_dir)
    out=a.out_dir/'source_quality_gate.json';out.write_text(json.dumps(report,indent=2))
    print('Saved PRIVATE quality report:',out)
    print('Dataset row counts:',report['daily_rows'],report['job_rows'],report['failure_rows'])
    print('Pump efficiency fractional values:',report['pump_efficiency_fraction_0_to_1'],
          'values above 1 up to 100:',report['pump_efficiency_above_1_to_100'])
    print('Duplicate conflicting well-days:',report['conflicting_well_date_groups'])
    print(report['decision'])
    if report['conflicting_well_date_groups']:
        print('[QUALITY] Conflicting well-days will be EXCLUDED by both analytic pipelines; no values will be imputed.')

if __name__=='__main__':main()
