"""Isolated V9 candidate modeling study. Does not modify existing WellGuard models.

Run locally with real Strathcona Parquet files:
  python scripts/16_experimental_model_upgrade.py --data-dir data \
    --out-dir outputs/experiments/v9_upgrade

Any holdout result is EXPLORATORY because the historical June-Oct test has
already been used for model/policy investigation in earlier releases.
"""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.model_experiments import (add_past_only_trends,add_prior_maintenance,
   check_reference_boundaries,split_masks,run_policy_evaluation,classification_metrics,
   ENGINEERED_FEATURES,MAINTENANCE_FEATURES,SPLIT_DATES)
# Source is loaded from the unchanged V8.x pipeline; no modified target rules.
import importlib.util as util
SOURCE=Path(__file__).with_name('02_failure_ranking.py')
source_spec=util.spec_from_file_location('wg90_frozen_model_source',SOURCE)
source=util.module_from_spec(source_spec)
source_spec.loader.exec_module(source)


def candidate_feature_sets():
    base=list(source.FEATURES)
    trends=base+list(ENGINEERED_FEATURES)
    enriched=trends+list(MAINTENANCE_FEATURES)
    return {
        'reference_features_lightgbm':base,
        'trends_lightgbm':trends,
        'trends_maintenance_lightgbm':enriched,
        'trends_maintenance_xgboost':enriched,
        'trends_maintenance_catboost':enriched,
    }


def fit_candidate(model_name,feature_names,d,masks,seed:int):
    X=d[feature_names].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    # Imputer is learned from TRAINING exclusively; no val/test statistics.
    train_medians=X.loc[masks['train']].median(numeric_only=True)
    fill_dict={c:(366.0 if c=='days_since_last_job_start' else float(train_medians[c]) if pd.notna(train_medians[c]) else -999.0) for c in feature_names}
    X=X.fillna(fill_dict).astype(float)
    train_y=d.loc[masks['train'],'target_14d'].astype(int)
    val_y=d.loc[masks['validation'],'target_14d'].astype(int)
    if model_name.endswith('lightgbm'):
        from lightgbm import LGBMClassifier,early_stopping
        model=LGBMClassifier(n_estimators=350,learning_rate=.04,num_leaves=12,max_depth=5,
                min_child_samples=100,subsample=.8,colsample_bytree=.85,reg_lambda=5,
                verbosity=-1,random_state=seed,n_jobs=2)
        model.fit(X.loc[masks['train']],train_y,
            eval_set=[(X.loc[masks['validation']],val_y)],eval_metric='average_precision',
            callbacks=[early_stopping(30,verbose=False)])
    elif model_name.endswith('xgboost'):
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.04,
            min_child_weight=10,subsample=.85,colsample_bytree=.85,reg_lambda=5,
            objective='binary:logistic',eval_metric='aucpr',tree_method='hist',
            random_state=seed,n_jobs=2)
        model.fit(X.loc[masks['train']],train_y,
                  eval_set=[(X.loc[masks['validation']],val_y)],verbose=False)
        # Fixed boosting budget; no test-driven hyperparameter selection.
    elif model_name.endswith('catboost'):
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=350,depth=5,learning_rate=.04,
             l2_leaf_reg=5,loss_function='Logloss',eval_metric='PRAUC',
             random_seed=seed,thread_count=2,verbose=False)
        model.fit(X.loc[masks['train']],train_y,
                  eval_set=(X.loc[masks['validation']],val_y),early_stopping_rounds=30)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    valid_score=model.predict_proba(X.loc[masks['validation']])[:,1]
    return model,X,np.asarray(valid_score),fill_dict


def run_experiment(data_dir:Path,out_dir:Path,*,weekly_cache=None,failures_pickle=None,jobs_pickle=None,
                   candidates=None,seed=42,overwrite=False):
    data_dir=Path(data_dir);out_dir=Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'Experiment already exists: {out_dir}. Choose a new --out-dir; the original run is preserved.')
    if out_dir.resolve()==(data_dir.parent/'outputs').resolve():
        raise ValueError('Experiment output must be a subfolder, never the reference outputs root.')
    out_dir.mkdir(parents=True,exist_ok=True)
    print('[1/6] Loading original weekly history...')
    w=pd.read_pickle(weekly_cache) if weekly_cache else source.weekly_from_source(data_dir)
    class Params:pass
    p=Params();p.failures_pickle=failures_pickle;p.data_dir=data_dir
    f=source.load_failures(p)
    j=pd.read_pickle(jobs_pickle) if jobs_pickle else pd.read_parquet(data_dir/'well_jobs.parquet',columns=['uwi','start_date'])
    check_reference_boundaries(w,f)
    print('[2/6] Building consecutive-week trends and strictly past-only maintenance features...')
    w=add_prior_maintenance(add_past_only_trends(w),j)
    # Freeze target-building to original project convention: outcome strictly in
    # the next 1–14 days, all predictors sourced through the as-of snapshot.
    w=w.sort_values('week_end',kind='mergesort').reset_index(drop=True)
    keys=w[['uwi','week_end']].copy()
    next_f=pd.merge_asof(keys,f,left_on='week_end',right_on='failure_start_date',
                           by='uwi',direction='forward',allow_exact_matches=False)
    prev_f=pd.merge_asof(keys,f,left_on='week_end',right_on='failure_start_date',
                           by='uwi',direction='backward',allow_exact_matches=True)
    w['until_next_failure']=(next_f.failure_start_date-w.week_end).dt.days
    w['since_last_failure']=(w.week_end-prev_f.failure_start_date).dt.days
    w['target_14d']=w.until_next_failure.between(1,14).astype('int8')
    w['failure_in_previous_90d']=w.since_last_failure.between(0,90).astype('int8')
    w['log_days_since_failure']=np.log1p(w.since_last_failure.clip(lower=0,upper=365)).fillna(np.log1p(365))
    eligible=(w.n_days>=6)&(w.production_days>=3)&w.week_end.between('2025-06-01','2026-10-04')&w.date_gap_from_prior_week.eq(7)&~w.since_last_failure.between(0,14)
    d=w.loc[eligible].copy()
    masks=split_masks(d)
    if any(int(m.sum())<20 for m in masks.values()):
        raise ValueError('Missing or insufficient reference train/validation/test observations')
    if d.loc[masks['train'],'target_14d'].nunique()<2 or d.loc[masks['validation'],'target_14d'].nunique()<2:
        raise ValueError('Both classes required in training and validation')
    print('[3/6] Fit reference-features and upgraded candidates using TRAIN ONLY; select on VALIDATION...')
    all_candidates=candidate_feature_sets()
    candidates=list(candidates) if candidates else list(all_candidates)
    unknown=set(candidates)-set(all_candidates)
    if unknown:raise ValueError(f'Unknown requested candidates: {sorted(unknown)}')
    metrics=[]; fitted={}; skipped={}
    for name in candidates:
        if name.endswith('xgboost') and not util.find_spec('xgboost'):
            skipped[name]='Optional xgboost not installed';continue
        if name.endswith('catboost') and not util.find_spec('catboost'):
            skipped[name]='Optional catboost not installed';continue
        try:
            model,X,val_scores,fill=fit_candidate(name,all_candidates[name],d,masks,seed)
            va=d.loc[masks['validation'],['uwi','week_end','target_14d']].copy()
            va['risk_candidate']=val_scores
            classification=classification_metrics(va.target_14d,val_scores)
            policy=run_policy_evaluation(va,f,capacity=25,cooldown_weeks=0)
            rec={'candidate':name,'feature_count':len(all_candidates[name]),
                'validation_ap':classification['ap'],'validation_roc_auc':classification['roc_auc'],
                'validation_event_recall25':policy['event_recall'],
                'validation_events_flagged25':policy['caught_events'],
                'validation_eligible_events':policy['eligible_events'],
                'validation_well_week_precision25':policy['well_week_precision']}
            metrics.append(rec);fitted[name]=(model,X,fill)
            print('  ',name,rec['validation_ap'],rec['validation_event_recall25'])
        except (ImportError,RuntimeError) as err:
            skipped[name]=f'{type(err).__name__}: {err}'
    if not metrics:raise RuntimeError(f'No candidate was successfully trained: {skipped}')
    valid_table=pd.DataFrame(metrics).sort_values(['validation_ap','validation_event_recall25','candidate'],ascending=[False,False,True])
    chosen=str(valid_table.iloc[0]['candidate'])
    model,X,fill=fitted[chosen]
    print('[4/6] Select cooldown policy (0,1,2 weeks) on VALIDATION only...')
    val=d.loc[masks['validation'],['uwi','week_end','target_14d']].copy()
    val['risk_candidate']=model.predict_proba(X.loc[masks['validation']])[:,1]
    policies=[]
    for cooldown in (0,1,2):
        policy=run_policy_evaluation(val,f,capacity=25,cooldown_weeks=cooldown)
        policies.append({'cooldown_weeks':cooldown,**policy})
    # Stable tie break: prefer less cooldown and higher precision.
    picked=sorted(policies,key=lambda x:(-(x['event_recall'] if x['event_recall'] is not None else -1),
                                       -(x['well_week_precision'] if x['well_week_precision'] is not None else -1),x['cooldown_weeks']))[0]
    print('[5/6] Evaluate one validation-selected candidate/policy on the PREVIOUSLY EXPLORED test period...')
    test=d.loc[masks['test'],['uwi','week_end','target_14d']].copy()
    test['risk_candidate']=model.predict_proba(X.loc[masks['test']])[:,1]
    holdout_class=classification_metrics(test.target_14d,test.risk_candidate)
    holdout_policy=run_policy_evaluation(test,f,capacity=25,cooldown_weeks=int(picked['cooldown_weeks']))
    fixed_policy=run_policy_evaluation(test,f,capacity=25,cooldown_weeks=0)
    # The model and any experiment are stored entirely outside outputs root.
    valid_table.to_csv(out_dir/'CANDIDATES_VALIDATION_ONLY.csv',index=False)
    pd.DataFrame(policies).to_csv(out_dir/'POLICIES_VALIDATION_ONLY.csv',index=False)
    test.to_csv(out_dir/'PRIVATE_EXPERIMENT_HOLDOUT_SCORES.csv',index=False)
    # Raw tree-model split importance is descriptive, not causal or SHAP.
    try:
        importances=np.asarray(model.feature_importances_,dtype=float)
        fi=pd.DataFrame({'feature':all_candidates[chosen],'tree_importance':importances})
        fi.sort_values('tree_importance',ascending=False).to_csv(out_dir/'CANDIDATE_FEATURE_IMPORTANCE.csv',index=False)
    except (AttributeError,ValueError,TypeError):
        pass  # Optional estimator may not expose comparable importance.
    import joblib
    joblib.dump({'model':model,'feature_names':all_candidates[chosen],
                 'imputer_training_only':fill,'splits':SPLIT_DATES,
                 'selected_by':'validation_ap_then_event_recall',
                 'validation_policy_cooldown_weeks':int(picked['cooldown_weeks']),
                 'research_status':'EXPLORATORY_TEST_ALREADY_REUSED'},out_dir/'PRIVATE_SELECTED_CANDIDATE.joblib')
    report={
      'status':'EXPLORATORY_NOT_PROSPECTIVELY_VALIDATED',
      'source':'Original competition data or explicitly supplied private test caches',
      'reference_model_modified':False,
      'reference_metrics_modified':False,
      'models_selected_using_test_data':False,
      'holdout_previously_explored_by_team':True,
      'splits':SPLIT_DATES,
      'target':'Recorded failure starting 1–14 days after Sunday prediction snapshot',
      'selection_rule':'Choose best validation average precision; tie-break with validation distinct-event recall at 25 reviews/week',
      'policy_rule':'Choose validation distinct-event recall @25/week among cooldowns 0,1,2; then selected well-week precision; then smaller cooldown',
      'feature_engineering':'Past-only consecutive calendar-week trends; no new features from future observations or future failure labels.',
      'maintenance_feature_warning':'Past recorded job starts only; publication and knowledge timestamps unavailable, so retrospective-entry bias cannot be ruled out.',
      'efficiency_warning':'Source pump_efficiency_pct units are not automatically rescaled. Reference cleaning (0–1) retained; source quality report must be reviewed before treating this sensor as reliable.',
      'selected_model':chosen,
      'selected_cooldown_weeks':int(picked['cooldown_weeks']),
      'candidate_validation_results':metrics,
      'policy_validation_results':policies,
      'test_previous_holdout':{**holdout_class,'selected_policy':holdout_policy,'no_cooldown':fixed_policy},
      'skipped_optional_models':skipped,
      'claims_permitted':'Validation-selected candidate performance is exploratory; later test was previously examined. Do NOT replace published benchmark, claim a new independent improvement, or imply avoided failures.'
    }
    (out_dir/'EXPERIMENT_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=str),encoding='utf-8')
    print('[6/6] Saved separate experimental files to',out_dir)
    print('Selected candidate:',chosen,'validation AP:',next(x['validation_ap'] for x in metrics if x['candidate']==chosen))
    print('Test AP:',holdout_class['ap'],'event recall @25:',holdout_policy['event_recall'],'(EXPLORATORY, previously examined test)')
    print('Reference metrics/model: UNCHANGED')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs/experiments/v9_upgrade'))
    p.add_argument('--weekly-cache',type=Path,help='Internal controlled-fixture testing only')
    p.add_argument('--failures-pickle',type=Path,help='Internal controlled-fixture testing only')
    p.add_argument('--jobs-pickle',type=Path,help='Internal controlled-fixture testing only')
    p.add_argument('--models',default='reference_features_lightgbm,trends_lightgbm,trends_maintenance_lightgbm,trends_maintenance_xgboost,trends_maintenance_catboost')
    p.add_argument('--overwrite-experiment',action='store_true',help='Overwrite only the specified isolated experiment folder')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    run_experiment(a.data_dir,a.out_dir,weekly_cache=a.weekly_cache,
        failures_pickle=a.failures_pickle,jobs_pickle=a.jobs_pickle,
        candidates=[v.strip() for v in a.models.split(',') if v.strip()],
        seed=a.seed,overwrite=a.overwrite_experiment)

if __name__=='__main__':main()
