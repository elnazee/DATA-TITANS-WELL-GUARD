"""Build weekly operating history from REAL daily_production.parquet.

No fabricated data, no imputation of missing sensors, no future outcome labels.
Records with ambiguous well/date duplicates are excluded from this descriptive
weekly history and counted in a private quality report.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS=['uwi','production_date','well_daily_status','oil_produced_m3','water_produced_m3',
         'hours_down','lost_oil_m3','pump_efficiency_pct','pcp_pump_rpm',
         'pcp_pump_rod_torque_ftlb','pcp_pump_electric_current_a']

def make_weekly(raw: pd.DataFrame):
    d=raw[COLUMNS].copy()
    d['production_date']=pd.to_datetime(d.production_date,errors='coerce').dt.normalize()
    invalid_keys=int((d.uwi.isna()|d.production_date.isna()).sum())
    d=d[d.uwi.notna()&d.production_date.notna()]
    duplicate=d.duplicated(['uwi','production_date'],keep=False)
    duplicated_groups=int(d.loc[duplicate,['uwi','production_date']].drop_duplicates().shape[0])
    # Count exact-match duplicate rows separately from conflicting rows.
    exact_pairs=d[duplicate].duplicated(subset=COLUMNS,keep=False)
    conflicting_groups=0
    if duplicate.any():
        groups=d.loc[duplicate].groupby(['uwi','production_date'],sort=False)
        conflicting_groups=int(sum(len(g.drop_duplicates())>1 for _,g in groups))
        # No arbitrary choice among conflicting well/day observations.
        d=d[~duplicate].copy()
    for c in COLUMNS[3:]:d[c]=pd.to_numeric(d[c],errors='coerce')
    bad_hour=int((~d.hours_down.between(0,24)&d.hours_down.notna()).sum())
    bad_eff=int((~d.pump_efficiency_pct.between(0,1)&d.pump_efficiency_pct.notna()).sum())
    d['downtime_ok']=d.hours_down.where(d.hours_down.between(0,24))
    d['oil_ok']=d.oil_produced_m3.where(d.oil_produced_m3.ge(0))
    d['water_ok']=d.water_produced_m3.where(d.water_produced_m3.ge(0))
    d['lost_ok']=d.lost_oil_m3.where(d.lost_oil_m3.ge(0))
    d['eff_ok']=d.pump_efficiency_pct.where(d.pump_efficiency_pct.between(0,1))
    d['rpm_ok']=d.pcp_pump_rpm.where(d.pcp_pump_rpm.ge(0))
    d['torque_ok']=d.pcp_pump_rod_torque_ftlb.where(d.pcp_pump_rod_torque_ftlb.ge(0))
    d['current_ok']=d.pcp_pump_electric_current_a.where(d.pcp_pump_electric_current_a.ge(0))
    d['week_end']=d.production_date+pd.to_timedelta(6-d.production_date.dt.weekday,unit='D')
    d['producer_day']=d.well_daily_status.eq('Producing').astype('int8')
    d['missing_pump_channels']=d[['eff_ok','rpm_ok','torque_ok','current_ok']].isna().sum(axis=1)
    grouped=d.groupby(['uwi','week_end'],observed=True,sort=False)
    w=grouped.agg(days=('production_date','nunique'),producing_days=('producer_day','sum'),
        oil_total_m3=('oil_ok','sum'),water_total_m3=('water_ok','sum'),
        estimated_lost_oil_m3=('lost_ok','sum'),hours_down=('downtime_ok','sum'),
        oil_day_count=('oil_ok','count'),water_day_count=('water_ok','count'),
        downtime_day_count=('downtime_ok','count'),lost_oil_day_count=('lost_ok','count'),
        rpm_avg_2w=('rpm_ok','mean'),eff_avg_2w=('eff_ok','mean'),
        torque_avg_2w=('torque_ok','mean'),current_avg_2w=('current_ok','mean'),
        missing_pump_channels_daily_avg=('missing_pump_channels','mean')).reset_index()
    for total,count in [('oil_total_m3','oil_day_count'),('water_total_m3','water_day_count'),
                        ('hours_down','downtime_day_count'),('estimated_lost_oil_m3','lost_oil_day_count')]:
        w[total]=w[total].where(w[count]>0)
    w['oil_daily_avg_2w']=w['oil_total_m3'].div(w.oil_day_count).where(w.oil_day_count>0)
    w['lost_per_week_2w']=w.estimated_lost_oil_m3
    w['down_fraction_2w']=w.hours_down.div(w.days*24).clip(0,1)
    w['watercut_2w']=w.water_total_m3.div(w.oil_total_m3+w.water_total_m3).where((w.oil_total_m3+w.water_total_m3)>0)
    w.sort_values(['uwi','week_end'],inplace=True)
    # UI trends summarize weekly values. Model training computes its own exact
    # trailing-two-week features and remains separate from these display features.
    report={'source_rows':len(raw),'valid_source_rows_after_key_and_duplicate_checks':len(d),
        'missing_uwi_or_date_rows':invalid_keys,'duplicate_well_date_groups_excluded':duplicated_groups,
        'conflicting_duplicate_groups':conflicting_groups,'bad_downtime_hours_rows':bad_hour,
        'out_of_fraction_range_pump_efficiency_rows':bad_eff,
        'weekly_history_rows':len(w),'unique_wells_in_weekly_history':int(w.uwi.nunique()),
        'warning':'Descriptive weekly values for dashboard; NOT model features or calibrated fault diagnoses. Suspicious source values were withheld from relevant metrics without conversion.'}
    return w, report

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    print('Reading real daily production source...')
    raw=pd.read_parquet(a.data_dir/'daily_production.parquet',columns=COLUMNS)
    w,report=make_weekly(raw)
    w.to_parquet(a.out_dir/'private_weekly_history.parquet',index=False)
    (a.out_dir/'private_real_history_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('Private weekly history written; do not upload outputs/.')
if __name__=='__main__':main()
