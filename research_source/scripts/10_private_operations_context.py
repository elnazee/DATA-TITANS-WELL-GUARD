"""Create PRIVATE full-fleet coordinate and observational maintenance context.

Only run on a competition-authorized local computer. Outputs are restricted,
contain actual well identifiers and coordinates and MUST NOT be uploaded or committed.
Run: python scripts/10_private_operations_context.py --data-dir data --out-dir outputs

This script produces private well-specific output; do not publish generated CSVs or maps.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

LOCATION_COLUMNS=['uwi','production_date','surface_latitude','surface_longitude','province','field_operating_area','facility_name','well_daily_status']
PRODUCTION_COLUMNS=['uwi','production_date','oil_produced_m3','hours_down']
JOB_COLUMNS=['uwi','start_date','job_type','field_est_total']


def normalize_locations(raw:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    """Latest source snapshot per well. Contradictory locations withheld, not imputed."""
    d=raw.copy()
    d['production_date']=pd.to_datetime(d['production_date'],errors='coerce')
    for c in ['surface_latitude','surface_longitude']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['uwi','production_date']).sort_values(['uwi','production_date'])
    d['valid_geo']=d.surface_latitude.between(48,61)&d.surface_longitude.between(-140,-96)
    valid=d[d.valid_geo].copy()
    # Detect meaningful (> about 0.005 degrees) coordinate disagreement for one UWI.
    diff=valid.groupby('uwi').agg(lat_span=('surface_latitude',lambda x:x.max()-x.min()),
                                   lon_span=('surface_longitude',lambda x:x.max()-x.min()))
    unstable=set(diff.index[(diff.lat_span>.005)|(diff.lon_span>.005)])
    # For conflicting latest same-day metadata, prefer to flag rather than pick arbitrarily.
    latest_day=valid.groupby('uwi')['production_date'].transform('max')
    latest=valid[valid.production_date.eq(latest_day)].copy()
    latest_conflicts=set()
    for well,grp in latest.groupby('uwi'):
        if len(grp[['surface_latitude','surface_longitude']].drop_duplicates())>1:latest_conflicts.add(well)
    skip=unstable|latest_conflicts
    latest=latest[~latest.uwi.isin(skip)].sort_values(['uwi','production_date'])
    result=latest.drop_duplicates('uwi',keep='last').rename(columns={'surface_latitude':'lat','surface_longitude':'lon',
        'field_operating_area':'area','facility_name':'facility','well_daily_status':'status','production_date':'source_date'})
    result['province']=result['province'].fillna('Unknown')
    result['area']=result['area'].fillna('Unknown')
    result['facility']=result['facility'].fillna('Unknown')
    n_all=int(d.uwi.nunique())
    report={'unique_wells_in_source':n_all,'valid_coordinate_wells':int(valid.uwi.nunique()),
            'excluded_coordinate_conflict_wells':len(skip), 'mappable_unique_wells':len(result),
            'unmappable_unique_wells':n_all-len(result),
            'note':'Map attributes are the latest known source metadata, not historical as-of features. Scores are available only for eligible scored producers.'}
    return result[['uwi','lat','lon','province','area','facility','status','source_date']],report


def maintenance_windows(production:pd.DataFrame,jobs:pd.DataFrame,min_days=20)->pd.DataFrame:
    """Daily 30-day pre/post observational comparison around job *start* date.

    Does not imply attribution to maintenance. Excludes duplicate well-day groups,
    negative oil or implausible downtime, and incomplete daily coverage.
    All dates relative to start_date; during-job period is excluded (day zero).
    """
    p=production.copy();j=jobs.copy()
    p['production_date']=pd.to_datetime(p.production_date,errors='coerce').dt.normalize()
    j['start_date']=pd.to_datetime(j.start_date,errors='coerce').dt.normalize()
    p['oil_produced_m3']=pd.to_numeric(p.oil_produced_m3,errors='coerce')
    p['hours_down']=pd.to_numeric(p.hours_down,errors='coerce')
    p=p[p.uwi.notna()&p.production_date.notna()]
    # Duplicate well-days have ambiguous grain; exclude whole groups from this descriptive comparison.
    p=p[~p.duplicated(['uwi','production_date'],keep=False)]
    p.loc[~p.oil_produced_m3.ge(0),'oil_produced_m3']=np.nan
    p.loc[~p.hours_down.between(0,24),'hours_down']=np.nan
    j=j[j.uwi.notna()&j.start_date.notna()].copy()
    all_jobs={well:np.sort(g.start_date.to_numpy(dtype='datetime64[ns]')) for well,g in j.groupby('uwi')}
    producer_groups={well:frame.sort_values('production_date') for well,frame in p.groupby('uwi')}
    rows=[]
    for well,g in j.groupby('uwi'):
        pwell=producer_groups.get(well)
        if pwell is None or pwell.empty:continue
        dates=pwell.production_date.to_numpy(dtype='datetime64[ns]')
        oil=pwell.oil_produced_m3.to_numpy(dtype=float)
        down=pwell.hours_down.to_numpy(dtype=float)
        jobdates=all_jobs[well]
        for record in g.itertuples():
            t=np.datetime64(record.start_date,'ns')
            before=(dates>=t-np.timedelta64(30,'D'))&(dates<t)
            after=(dates>t)&(dates<=t+np.timedelta64(30,'D'))
            bef_o=oil[before]; aft_o=oil[after]; bef_d=down[before];aft_d=down[after]
            nb=int(np.isfinite(bef_o).sum());na=int(np.isfinite(aft_o).sum())
            # The event start and next 30 days may contain additional jobs; flag rather than attribute.
            concurrent=int(((jobdates>=t-np.timedelta64(30,'D'))&(jobdates<=t+np.timedelta64(30,'D'))).sum())>1
            if nb<min_days or na<min_days:continue
            oil_b=float(np.nanmean(bef_o));oil_a=float(np.nanmean(aft_o));
            d_b=float(np.nanmean(bef_d)) if np.isfinite(bef_d).any() else np.nan
            d_a=float(np.nanmean(aft_d)) if np.isfinite(aft_d).any() else np.nan
            cost=pd.to_numeric(getattr(record,'field_est_total',np.nan),errors='coerce')
            rows.append({'uwi':well,'job_start_date':record.start_date,'job_type':getattr(record,'job_type',None),
                         'estimated_cost_cad':cost,'oil_before_m3d':oil_b,'oil_after_m3d':oil_a,
                         'oil_change_m3d':oil_a-oil_b,'downtime_before_hours_day':d_b,'downtime_after_hours_day':d_a,
                         'days_oil_before':nb,'days_oil_after':na,'has_other_job_within_30d':concurrent,
                         'warning':'DESCRIPTIVE only; job-date timing, operating changes, and concurrent work can confound'})
    return pd.DataFrame.from_records(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    # Read projected source columns with PyArrow on authorized machine.
    raw=pd.read_parquet(args.data_dir/'daily_production.parquet',columns=list(dict.fromkeys(LOCATION_COLUMNS+PRODUCTION_COLUMNS)))
    loc,report=normalize_locations(raw[LOCATION_COLUMNS]);loc.to_csv(args.out_dir/'private_well_locations.csv',index=False)
    jobs=pd.read_parquet(args.data_dir/'well_jobs.parquet',columns=JOB_COLUMNS)
    effect=maintenance_windows(raw[PRODUCTION_COLUMNS],jobs,min_days=20)
    effect.to_csv(args.out_dir/'private_maintenance_windows.csv',index=False)
    report.update({'maintenance_jobs_in_source':int(len(jobs)),'observational_30d_jobs_with_coverage':int(len(effect)),
        'jobs_with_other_work_in_plusminus30d':int(effect.has_other_job_within_30d.sum()) if not effect.empty else 0,
        'private_outputs':['private_well_locations.csv','private_maintenance_windows.csv'],
        'source_confidentiality':'Restricted: NEVER upload these CSVs to public GitHub or into an external AI service.'})
    (args.out_dir/'private_v6_operations_audit.json').write_text(json.dumps(report,indent=2,default=str))
    print('PRIVATE full-fleet map / 30-day descriptive maintenance audit:',json.dumps(report,indent=2))
if __name__=='__main__':main()
