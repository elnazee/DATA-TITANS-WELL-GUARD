"""Confidential replay cohort audit; exports aggregates only, no individual UWI.

Sensor availability is NOT equivalent to pump type or health. This test is exploratory.
"""
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=Path('outputs'))
 a=p.parse_args()
 d=pd.read_csv(a.out_dir/'private_test_alerts.csv',parse_dates=['week_end'])
 sensor_cols=['eff_avg_2w','rpm_avg_2w','torque_avg_2w','current_avg_2w']
 d['channels_observed']=d[sensor_cols].notna().sum(axis=1)
 d['coverage_cohort']=pd.cut(d['channels_observed'],bins=[-1,.5,2.5,4.5],labels=['0 channels','1–2 channels','3–4 channels'])
 agg=[]
 for key,g in d.groupby('coverage_cohort',observed=True):
  if len(g)<100:continue
  count=int(g.target_14d.sum())
  agg.append({'cohort':str(key),'well_weeks':len(g),'positive_well_weeks':count,
       'positive_rate':float(g.target_14d.mean()),
       'ap':float(average_precision_score(g.target_14d,g.relative_risk_score)) if count>0 else None,
       'ap_lift_over_prevalence':float(average_precision_score(g.target_14d,g.relative_risk_score)/g.target_14d.mean()) if count>0 else None,
       'roc_auc':float(roc_auc_score(g.target_14d,g.relative_risk_score)) if g.target_14d.nunique()>1 else None})
 report={'cohort_definition':'Count of nonmissing two-week mean efficiency, RPM, rod torque, motor current',
         'limits':'Sensor channel availability is a proxy for instrumentation/reporting, not validated equipment type. Cohort labels and failure recording may differ. No live calibration.',
         'source':'June–October 2026 test snapshots from private historical replay','cohorts':agg}
 (a.out_dir/'cohort_robustness.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
