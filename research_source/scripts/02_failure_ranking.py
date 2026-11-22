"""Chronological, evidence-first ranking of recorded equipment failures.

Standard run:
    python scripts/02_failure_ranking.py --data-dir data --out-dir outputs

Internal testing only (requires existing proprietary local pickle files):
    python scripts/02_failure_ranking.py --weekly-cache path.pkl --failures-pickle path.pkl

The model is a retrospective research prototype, not a calibrated risk estimate.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,early_stopping
from sklearn.metrics import average_precision_score,roc_auc_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.scheduling import schedule_inspections

FEATURES=[
    'down_fraction_2w','downtime_hours_2w','oil_daily_avg_2w','lost_per_week_2w',
    'watercut_2w','eff_avg_2w','rpm_avg_2w','torque_avg_2w',
    'current_avg_2w','production_days_2w','failure_in_previous_90d',
    'log_days_since_failure','month']


def weekly_from_source(data_dir:Path)->pd.DataFrame:
    cols=['uwi','production_date','well_daily_status','hours_down','lost_oil_m3',
          'oil_produced_m3','water_produced_m3','pump_efficiency_pct','pcp_pump_rpm',
          'pcp_pump_rod_torque_ftlb','pcp_pump_electric_current_a']
    print('Reading daily production source with PyArrow...')
    d=pd.read_parquet(data_dir/'daily_production.parquet',columns=cols)
    d['production_date']=pd.to_datetime(d.production_date)
    dup=d.duplicated(['uwi','production_date'],keep=False)
    if dup.any():
        groups=d.loc[dup].groupby(['uwi','production_date'],sort=False)
        conflicts=int(sum(g.drop_duplicates().shape[0]>1 for _,g in groups))
        ambiguous=int(d.loc[dup,['uwi','production_date']].drop_duplicates().shape[0])
        print(f'[QUALITY] Excluding {ambiguous} ambiguous/repeated well-date groups ({conflicts} conflicting); no arbitrary row selected.')
        # Same conservative duplicate policy as the descriptive history script.
        d=d.loc[~dup].copy()
    d['week_end']=d.production_date+pd.to_timedelta(6-d.production_date.dt.dayofweek,unit='D')
    d['is_producing']=d.well_daily_status.eq('Producing').astype('int8')
    d['is_down']=d.hours_down.gt(0).astype('int8')
    for key in ['oil_produced_m3','water_produced_m3','pump_efficiency_pct','pcp_pump_rpm','pcp_pump_rod_torque_ftlb','pcp_pump_electric_current_a','hours_down','lost_oil_m3']:
        d[key]=pd.to_numeric(d[key],errors='coerce')
    d['oil_pos']=d.oil_produced_m3.clip(lower=0)
    d['lost_pos']=d.lost_oil_m3.clip(lower=0)
    d['eff_clean']=d.pump_efficiency_pct.where(d.pump_efficiency_pct.between(0,1))
    d['rpm_clean']=d.pcp_pump_rpm.where(d.pcp_pump_rpm.ge(0))
    d['torque_clean']=d.pcp_pump_rod_torque_ftlb.where(d.pcp_pump_rod_torque_ftlb.ge(0))
    d['current_clean']=d.pcp_pump_electric_current_a.where(d.pcp_pump_electric_current_a.ge(0))
    w=d.groupby(['uwi','week_end'],sort=False,observed=True).agg(
        n_days=('production_date','size'),production_days=('is_producing','sum'),
        down_days=('is_down','sum'),downtime_hours=('hours_down','sum'),
        oil_total=('oil_pos','sum'),lost_oil_total=('lost_pos','sum'),
        water_total=('water_produced_m3','sum'),water_n=('water_produced_m3','count'),
        oil_n=('oil_produced_m3','count'),eff_sum=('eff_clean','sum'),eff_n=('eff_clean','count'),
        rpm_sum=('rpm_clean','sum'),rpm_n=('rpm_clean','count'),
        torque_sum=('torque_clean','sum'),torque_n=('torque_clean','count'),
        current_sum=('current_clean','sum'),current_n=('current_clean','count')).reset_index()
    w.sort_values(['uwi','week_end'],inplace=True,ignore_index=True)
    for name in ('eff','rpm','torque','current'):
        w[f'{name}_avg']=w[f'{name}_sum'].div(w[f'{name}_n']).where(w[f'{name}_n']>0)
    for name,n in [('oil','oil_n'),('water','water_n')]:
        w[f'{name}_daily_avg']=w[f'{name}_total'].div(w[n]).where(w[n]>0)
    w['down_fraction']=w.down_days.div(w.n_days)
    w['lost_per_week']=w.lost_oil_total
    w['watercut']=w.water_total.div(w.water_total+w.oil_total).where((w.water_total+w.oil_total)>0)
    hist=['down_fraction','downtime_hours','oil_daily_avg','lost_per_week','watercut','eff_avg','rpm_avg','torque_avg','current_avg','production_days']
    for c in hist:
        w[f'{c}_2w']=w.groupby('uwi',sort=False)[c].transform(lambda x:x.rolling(2,min_periods=1).mean())
    w['date_gap_from_prior_week']=(w.week_end-w.groupby('uwi',sort=False).week_end.shift(1)).dt.days
    w['year']=w.week_end.dt.year
    w['month']=w.week_end.dt.month
    return w


def load_failures(args):
    f=pd.read_pickle(args.failures_pickle) if args.failures_pickle else pd.read_parquet(args.data_dir/'production_failures.parquet')
    f['failure_start_date']=pd.to_datetime(f['failure_start_date'])
    return f[['uwi','failure_start_date']].drop_duplicates().sort_values('failure_start_date')


def score_events(test_rows,pred_col,events,top_n):
    """Recall among unique future events with at least one eligible prior test snapshot."""
    ev=events[events.failure_start_date.between('2026-06-08','2026-10-18')].copy()
    ev['event_key']=ev.uwi.astype(str)+'|'+ev.failure_start_date.astype(str)
    matches=test_rows[['uwi','week_end']].merge(ev,on='uwi',how='inner')
    matches['days_before']=(matches.failure_start_date-matches.week_end).dt.days
    eligible=set(matches.loc[matches.days_before.between(1,14),'event_key'])
    selected=schedule_inspections(test_rows,capacity=top_n,score_col=pred_col)
    found=selected[['uwi','week_end']].merge(ev,on='uwi',how='inner')
    found['days_before']=(found.failure_start_date-found.week_end).dt.days
    caught=set(found.loc[found.days_before.between(1,14),'event_key']) & eligible
    return {'eligible_events':len(eligible),'caught_events':len(caught),'event_recall':len(caught)/len(eligible) if eligible else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('data'));p.add_argument('--out-dir',type=Path,default=Path('outputs'))
    p.add_argument('--weekly-cache',type=Path,help='Internal test: precomputed private weekly features')
    p.add_argument('--failures-pickle',type=Path,help='Internal test: private failure dataframe')
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    w=pd.read_pickle(args.weekly_cache) if args.weekly_cache else weekly_from_source(args.data_dir)
    f=load_failures(args)
    w=w.sort_values('week_end').reset_index(drop=True)
    cols=w[['uwi','week_end']]
    next_fail=pd.merge_asof(cols,f,left_on='week_end',right_on='failure_start_date',by='uwi',direction='forward',allow_exact_matches=False)
    prev_fail=pd.merge_asof(cols,f,left_on='week_end',right_on='failure_start_date',by='uwi',direction='backward',allow_exact_matches=True)
    w['until_next_failure']=(next_fail.failure_start_date-w.week_end).dt.days
    w['since_last_failure']=(w.week_end-prev_fail.failure_start_date).dt.days
    w['target_14d']=w.until_next_failure.between(1,14).astype('int8')
    w['failure_in_previous_90d']=w.since_last_failure.between(0,90).astype('int8')
    w['log_days_since_failure']=np.log1p(w.since_last_failure.clip(lower=0,upper=365)).fillna(np.log1p(365))
    # Testing ends 2026-10-04; available data cover all 14 future days of the labels.
    eligible=(w.n_days>=6)&(w.production_days>=3)&w.week_end.between('2025-06-01','2026-10-04') & w.date_gap_from_prior_week.eq(7)&~w.since_last_failure.between(0,14)
    d=w[eligible].copy(); train=d.week_end.le('2026-03-01'); val=d.week_end.between('2026-03-22','2026-05-17'); test=d.week_end.between('2026-06-07','2026-10-04')
    if min(train.sum(),val.sum(),test.sum())==0:raise ValueError('One of the time splits is empty.')
    features=d[FEATURES].replace([np.inf,-np.inf],np.nan)
    y=d.target_14d.astype(int)
    model=LGBMClassifier(n_estimators=350,learning_rate=.04,num_leaves=12,max_depth=5,min_child_samples=250,subsample=.8,colsample_bytree=.85,reg_lambda=5,verbosity=-1,random_state=42,n_jobs=2)
    model.fit(features[train],y[train],eval_set=[(features[val],y[val])],eval_metric='average_precision',callbacks=[early_stopping(30,verbose=False)])
    t=d.loc[test].copy();t['relative_risk_score']=model.predict_proba(features[test])[:,1]
    t['past_failure_baseline']=t.failure_in_previous_90d.astype(float)
    t['recency_baseline_score']=-t.log_days_since_failure.astype(float)
    ap=lambda s:float(average_precision_score(t.target_14d,s))
    test_metrics={'test_positive_rate':float(t.target_14d.mean()),'model_ap':ap(t.relative_risk_score),'model_roc_auc':float(roc_auc_score(t.target_14d,t.relative_risk_score)),
                  'past_failure_ap':ap(t.past_failure_baseline),
                  'recency_baseline_ap':ap(t.recency_baseline_score),
                  'recency_baseline_roc_auc':float(roc_auc_score(t.target_14d,t.recency_baseline_score)),
                  'downtime_ap':ap(t.down_fraction_2w.fillna(0)),
                  'train_well_weeks':int(train.sum()),'val_well_weeks':int(val.sum()),'test_well_weeks':int(test.sum()),'train_positive_well_weeks':int(y[train].sum()),'val_positive_well_weeks':int(y[val].sum()),'test_positive_well_weeks':int(t.target_14d.sum()),'model_best_iteration':int(model.best_iteration_)}
    for n in (25,50,100):
        for name,col in [('model','relative_risk_score'),('baseline','past_failure_baseline'),('recency','recency_baseline_score')]:
            top=schedule_inspections(t,capacity=n,score_col=col)
            test_metrics[f'{name}_top{n}_well_week_precision']=float(top.target_14d.mean())
            test_metrics[f'{name}_top{n}_event_coverage']=score_events(t,col,f,n)
    # Sensitivity: freeze the same decision weeks and labels, and age the
    # production inputs and failure-history cutoff by one full calendar week.
    operating = [c for c in FEATURES if c not in {
        'failure_in_previous_90d', 'log_days_since_failure', 'month'}]
    delayed = w[['uwi', 'week_end', *FEATURES[:-1]]].copy()
    delayed['week_end'] += pd.Timedelta(days=7)
    delayed_features = (d[['uwi','week_end','month']].merge(
        delayed,on=['uwi','week_end'],how='left',validate='one_to_one')
        .set_axis(d.index)[FEATURES].replace([np.inf,-np.inf],np.nan))
    alternative_features = {'operating_only': features[operating],
                            'seven_day_delayed_inputs': delayed_features}
    for name, matrix in alternative_features.items():
        alt = LGBMClassifier(**model.get_params())
        alt.fit(matrix[train], y[train], eval_set=[(matrix[val], y[val])],
                eval_metric='average_precision',
                callbacks=[early_stopping(30,verbose=False)])
        t[name] = alt.predict_proba(matrix[test])[:,1]
    t['downtime_baseline'] = t.down_fraction_2w.fillna(0)
    t['production_loss_baseline'] = t.lost_per_week_2w.fillna(0)
    t['delayed_recency_baseline'] = -delayed_features.loc[t.index,'log_days_since_failure']
    comparisons = {}
    for name in ('relative_risk_score','recency_baseline_score','operating_only',
                 'seven_day_delayed_inputs','delayed_recency_baseline',
                 'downtime_baseline','production_loss_baseline'):
        comparisons[name] = {'average_precision': ap(t[name]),
                             'event_coverage': {str(cap):score_events(t,name,f,cap)
                                                for cap in (25,50,100)}}
    test_metrics['comparison'] = comparisons
    (args.out_dir/'aggregate_model_metrics.json').write_text(json.dumps(test_metrics,indent=2))
    # LightGBM TreeSHAP contributions are in margin/log-odds units, not causal effects.
    # Keep contributions PRIVATE: they are associated with confidential well identifiers.
    contrib = model.booster_.predict(features[test], pred_contrib=True,
                                      num_iteration=model.best_iteration_)
    important = np.argsort(np.abs(contrib[:, :-1]), axis=1)[:, ::-1][:, :3]
    for j in range(3):
        t[f'driver_{j+1}'] = [FEATURES[k] for k in important[:, j]]
        t[f'driver_{j+1}_margin'] = contrib[np.arange(len(t)),important[:, j]]
    private_cols=['uwi','week_end','relative_risk_score','target_14d',
         'down_fraction_2w','downtime_hours_2w','oil_daily_avg_2w',
         'lost_per_week_2w','eff_avg_2w','rpm_avg_2w',
         'torque_avg_2w','current_avg_2w','watercut_2w',
         'failure_in_previous_90d','log_days_since_failure','recency_baseline_score',
         'driver_1','driver_1_margin','driver_2','driver_2_margin',
         'driver_3','driver_3_margin']
    t[private_cols].to_csv(args.out_dir/'private_test_alerts.csv',index=False)
    # Separate, label-free post-holdout scoring. This does NOT evaluate these weeks
    # or assert any new event-recall result. Prediction at a snapshot uses only
    # measurements through that snapshot and prior recorded failures.
    recent_eligible=(w.n_days>=6)&(w.production_days>=3)&(w.week_end>'2026-10-04') & \
         w.date_gap_from_prior_week.eq(7)&~w.since_last_failure.between(0,14)
    recent=w.loc[recent_eligible].copy()
    if not recent.empty:
        matrix=recent[FEATURES].replace([np.inf,-np.inf],np.nan)
        recent['relative_risk_score']=model.predict_proba(matrix)[:,1]
        recent_contrib=model.booster_.predict(matrix,pred_contrib=True,num_iteration=model.best_iteration_)
        ranked=np.argsort(np.abs(recent_contrib[:,:-1]),axis=1)[:,::-1][:,:3]
        for j in range(3):
            recent[f'driver_{j+1}']=[FEATURES[k] for k in ranked[:,j]]
            recent[f'driver_{j+1}_margin']=recent_contrib[np.arange(len(recent)),ranked[:,j]]
        recent['score_origin']='POST_HOLDOUT_INFERENCE_NOT_VALIDATED'
        recent_cols=['uwi','week_end','relative_risk_score','score_origin',
             'down_fraction_2w','downtime_hours_2w','oil_daily_avg_2w',
             'lost_per_week_2w','eff_avg_2w','rpm_avg_2w',
             'torque_avg_2w','current_avg_2w','watercut_2w',
             'failure_in_previous_90d','log_days_since_failure',
             'driver_1','driver_1_margin','driver_2','driver_2_margin',
             'driver_3','driver_3_margin']
        recent[recent_cols].to_csv(args.out_dir/'private_recent_model_scores.csv',index=False)
        print('[MODEL] Post-holdout scores: ',len(recent),'eligible well-weeks across',recent.week_end.nunique(),'weeks; NO future labels used.')
    else:
        pd.DataFrame(columns=['uwi','week_end','relative_risk_score','score_origin']).to_csv(args.out_dir/'private_recent_model_scores.csv',index=False)
        print('[MODEL] No eligible source observations after the retrospective holdout; no scores invented.')
    joblib.dump({'model':model,'feature_order':FEATURES,'trained_through':'2026-03-01','validation_through':'2026-05-17'},args.out_dir/'private_historical_model.joblib')
    print(json.dumps(test_metrics,indent=2));print('Saved private alert replay, model, and aggregate metrics to',args.out_dir)
    print('NOT a calibrated probability, not a prospective forecast, not a claim of avoided failures.')

if __name__=='__main__':main()
