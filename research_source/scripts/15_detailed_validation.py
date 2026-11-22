"""Produce actual private ROC/PR curves, month diagnostics, week-block uncertainty,
original-Parquet source checks and an as-of feature replay.

Runs ONLY on the authorized machine with original, real source Parquet and a genuine
private_test_alerts.csv produced by scripts/02_failure_ranking.py.
No curves, metrics or passing source checks are invented when evidence is missing.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.validation_lab import evaluate_private_replay, benchmark_consistency
from scripts import __doc__ as _unused  # scripts namespace is present in Python 3.3+

EXPECTED = json.loads((ROOT / 'results/source_fingerprint.json').read_text(encoding='utf-8'))
METRIC_COLS=['down_fraction_2w','downtime_hours_2w','oil_daily_avg_2w',
             'lost_per_week_2w','eff_avg_2w','rpm_avg_2w','torque_avg_2w',
             'current_avg_2w','watercut_2w']


def check_sources(data_dir:Path):
    """Fast original source identity check: exact hashes, Parquet metadata and row totals."""
    import pyarrow.parquet as pq
    result={}
    for name,truth in EXPECTED.items():
        path=data_dir/name
        if not path.is_file():raise FileNotFoundError(f'Missing original file: {name}')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        pf=pq.ParquetFile(path)
        result[name]={'sha256':h.hexdigest(),'matches_original_archive':h.hexdigest()==truth['sha256'],
                      'parquet_rows':pf.metadata.num_rows,'expected_rows':truth['rows'],
                      'rows_match':pf.metadata.num_rows==truth['rows'],
                      'row_groups':pf.metadata.num_row_groups}
    if not all(v['matches_original_archive'] and v['rows_match'] for v in result.values()):
        raise AssertionError('Original competition data hashes or Parquet record counts differ from the audited archive.')
    return result


def compare_numerics(a,b,columns,atol=2e-6):
    """Verify values, missingness and row identity; summary only, never expose UWI."""
    keys=['uwi','week_end']
    left=a[keys+columns].copy();right=b[keys+columns].copy()
    for z in (left,right):
        z['uwi']=z['uwi'].astype('string')
        z['week_end']=pd.to_datetime(z['week_end'])
    m=left.merge(right,on=keys,how='left',indicator=True,validate='one_to_one',suffixes=('_model','_source'))
    failures={}
    for col in columns:
        x=pd.to_numeric(m[col+'_model'],errors='coerce').to_numpy(dtype=float)
        y=pd.to_numeric(m[col+'_source'],errors='coerce').to_numpy(dtype=float)
        bad=~((np.isnan(x)&np.isnan(y))|np.isclose(x,y,rtol=2e-5,atol=atol,equal_nan=True))
        failures[col]=int(bad.sum())
    return {'compared_well_weeks':len(m),'missing_replayed_weeks':int((m['_merge']!='both').sum()),
            'feature_mismatches':failures,
            'passed':bool((m['_merge']=='both').all() and not any(failures.values()))}


def verify_source_features(data_dir:Path,test:pd.DataFrame,sampled_wells:int=16):
    """Sample REAL wells deterministically and recalculate their exact source features.
    A prefix-only replay verifies that later daily readings cannot change earlier ones.
    This is sampled evidence, not a 100%-of-rows proof.
    """
    import tempfile
    from scripts import __path__ as _scripts_path
    import importlib.util
    name=ROOT/'scripts/02_failure_ranking.py'
    spec=importlib.util.spec_from_file_location('wg_rank_audit',name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    available=test['uwi'].astype(str).drop_duplicates().sort_values().to_numpy()
    if len(available)==0:raise ValueError('No test wells to replay')
    if len(available)>sampled_wells:
        sel=np.linspace(0,len(available)-1,sampled_wells,dtype=int)
        sample=available[sel].tolist()
    else:sample=available.tolist()
    cols=['uwi','production_date','well_daily_status','hours_down','lost_oil_m3',
          'oil_produced_m3','water_produced_m3','pump_efficiency_pct','pcp_pump_rpm',
          'pcp_pump_rod_torque_ftlb','pcp_pump_electric_current_a']
    daily=pd.read_parquet(data_dir/'daily_production.parquet',columns=cols,
                          filters=[('uwi','in',sample)])
    if daily.empty:raise AssertionError('The original daily-production file had no matches for sampled test wells')
    with tempfile.TemporaryDirectory(prefix='wg_asof_') as tmp:
        tmp=Path(tmp)
        daily.to_parquet(tmp/'daily_production.parquet',index=False)
        w=mod.weekly_from_source(tmp)
        sampled=test[test.uwi.astype(str).isin(sample)].copy()
        full=compare_numerics(sampled,w,METRIC_COLS)
        cutoff=pd.Timestamp('2026-07-05')
        before=daily[pd.to_datetime(daily.production_date)<=cutoff].copy()
        before.to_parquet(tmp/'daily_production.parquet',index=False)
        prefix=mod.weekly_from_source(tmp)
        earlier=w[w.week_end.le(cutoff)]
        future=compare_numerics(earlier,prefix,METRIC_COLS)
    return {'sampled_wells':len(sample),'raw_rows_sampled':len(daily),
            'scored_week_feature_replay':full,
            'future_record_invariance_through':'2026-07-05',
            'earlier_features_unchanged_when_future_removed':future,
            'passed':bool(full['passed'] and future['passed'])}


def week_cluster_uncertainty(df,iterations=600,seed=2026):
    x=df.copy();x['week_end']=pd.to_datetime(x.week_end)
    labels=x.target_14d.astype(int).to_numpy()
    pred=x.relative_risk_score.astype(float).to_numpy()
    baseline=x.recency_baseline_score.astype(float).to_numpy()
    dates=x.week_end.to_numpy()
    weeks=np.unique(dates)
    if len(weeks)<3 or len(np.unique(labels))<2:raise ValueError('Insufficient weeks or class labels for bootstrap')
    indices=[np.flatnonzero(dates==week) for week in weeks]
    rng=np.random.default_rng(seed)
    draws={'model_ap':[],'baseline_ap':[],'ap_difference':[],
           'model_roc_auc':[],'baseline_roc_auc':[]}
    for _ in range(iterations):
        sample=np.concatenate([indices[j] for j in rng.integers(len(weeks),size=len(weeks))])
        yy=labels[sample]
        if len(np.unique(yy))<2:continue
        ap=average_precision_score(yy,pred[sample]);base=average_precision_score(yy,baseline[sample])
        draws['model_ap'].append(ap)
        draws['baseline_ap'].append(base)
        draws['ap_difference'].append(ap-base)
        draws['model_roc_auc'].append(roc_auc_score(yy,pred[sample]))
        draws['baseline_roc_auc'].append(roc_auc_score(yy,baseline[sample]))
    if len(draws['model_ap'])<int(iterations*0.9):
        raise ValueError('Too few successful block bootstrap replicates')
    points={'model_ap':average_precision_score(labels,pred),
            'baseline_ap':average_precision_score(labels,baseline),
            'ap_difference':average_precision_score(labels,pred)-average_precision_score(labels,baseline),
            'model_roc_auc':roc_auc_score(labels,pred),
            'baseline_roc_auc':roc_auc_score(labels,baseline)}
    return {'method':'test Sunday snapshot cluster bootstrap; sampling weeks with replacement',
            'weeks':len(weeks),'requested_replicates':iterations,'valid_replicates':len(draws['model_ap']),
            'point':{k:float(v) for k,v in points.items()},
            'descriptive_95_percentile_bounds':{k:[float(v) for v in np.quantile(vs,[.025,.975])]
                                               for k,vs in draws.items()},
            'limitations':'Exploratory historical resampling: repeat wells/events, temporal dependence and prior test use. NOT prospective uncertainty; NOT independent test validation.'}


def write_private_charts(directory:Path,analysis:dict,monthly:pd.DataFrame,uncertainty:dict,source:dict,bench_diff:list):
    """Self-contained offline HTML, no external scripts; no individual well IDs."""
    directory.mkdir(parents=True,exist_ok=True)
    fig=make_subplots(rows=2,cols=2,
        subplot_titles=['Precision–recall curves','ROC curves','Month-level average precision','Week-block uncertainty'],
        specs=[[{},{}],[{},{}]],vertical_spacing=.14,horizontal_spacing=.11)
    for label,color in [('model','#2ce2dd'),('baseline','#f5ad51')]:
        pr=analysis['curves'][label+'_pr'];roc=analysis['curves'][label+'_roc']
        fig.add_trace(go.Scatter(x=pr.recall,y=pr.precision,mode='lines',name=('WellGuard' if label=='model' else 'Recency baseline'),
                                 legendgroup=label,line=dict(color=color,width=2.5)),1,1)
        fig.add_trace(go.Scatter(x=roc.fpr,y=roc.tpr,mode='lines',showlegend=False,
                                 legendgroup=label,line=dict(color=color,width=2.5)),1,2)
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color='#637784',dash='dash'),
                             name='Chance ROC',showlegend=False),1,2)
    for k,label,color in [('model_ap','WellGuard','#2ce2dd'),('recency_ap','Recency','#f5ad51')]:
        fig.add_trace(go.Bar(x=monthly.month,y=monthly[k],name=label,marker_color=color,
                             showlegend=False),2,1)
    points=uncertainty['point'];bounds=uncertainty['descriptive_95_percentile_bounds']
    names=['Model AP','Recency AP','AP difference','Model ROC-AUC'];cols=['model_ap','baseline_ap','ap_difference','model_roc_auc']
    for i,col in enumerate(cols):
        fig.add_trace(go.Scatter(x=[points[col]],y=[names[i]],mode='markers',showlegend=False,
            marker=dict(size=13,color='#2ce2dd' if i in (0,2,3) else '#f5ad51'),
            error_x=dict(type='data',symmetric=False,
                array=[bounds[col][1]-points[col]],arrayminus=[points[col]-bounds[col][0]],visible=True,
                color='#a2b9c9',thickness=2, width=5)),2,2)
    fig.update_xaxes(title_text='Recall',row=1,col=1,range=[0,1])
    fig.update_yaxes(title_text='Precision',row=1,col=1,range=[0,1])
    fig.update_xaxes(title_text='False positive rate',row=1,col=2,range=[0,1])
    fig.update_yaxes(title_text='True positive rate',row=1,col=2,range=[0,1])
    fig.update_yaxes(title_text='Average precision',row=2,col=1)
    fig.update_xaxes(title_text='Metric value (week-block percentiles)',row=2,col=2)
    fig.update_layout(template='plotly_dark',paper_bgcolor='#071521',plot_bgcolor='#0c2230',font=dict(color='#e0edf5'),
                      height=920,barmode='group',legend=dict(orientation='h',y=1.08),
                      title='WellGuard AI — Private Model Validation | Original Holdout Replay',
                      margin=dict(l=65,r=40,t=110,b=75))
    fig.write_html(str(directory/'PRIVATE_ROC_PR_MONTHLY_UNCERTAINTY.html'),include_plotlyjs=True,full_html=True)
    for name,curve in analysis['curves'].items():curve.to_csv(directory/(name+'.csv'),index=False)
    monthly.to_csv(directory/'monthly_performance.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--out-dir',type=Path,default=ROOT/'outputs')
    p.add_argument('--iterations',type=int,default=600)
    p.add_argument('--sample-wells',type=int,default=16)
    args=p.parse_args()
    target=args.out_dir/'private_detailed_validation'
    target.mkdir(parents=True,exist_ok=True)
    report={'source':'ORIGINAL PRIVATE PARQUET AND ORIGINAL LOCAL TEST REPLAY',
            'reference_dates':{'train':'2025-06-01 to 2026-03-01',
                               'validation':'2026-03-22 to 2026-05-17',
                               'test':'2026-06-07 to 2026-10-04'},
            'verified_on_this_computer':False,'status':'BLOCKED'}
    try:
        report['original_source_identity']=check_sources(args.data_dir)
        metrics_path=args.out_dir/'aggregate_model_metrics.json'
        replay=args.out_dir/'private_test_alerts.csv'
        if not replay.is_file() or not metrics_path.is_file():
            raise FileNotFoundError('Missing real private test replay or saved model metrics. Run run_historical_model_windows.bat first.')
        saved=json.loads(metrics_path.read_text(encoding='utf-8'))
        test=pd.read_csv(replay,dtype={'uwi':'string'},parse_dates=['week_end'])
        detail=evaluate_private_replay(test)
        diffs=benchmark_consistency(saved,detail['metrics'])
        report['saved_metrics_replay_disagreements']=diffs
        if diffs:raise AssertionError('Saved metrics and actual test predictions disagree: '+'; '.join(diffs))
        import importlib.util
        loc=ROOT/'scripts/12_audit_data_leakage.py'
        spec=importlib.util.spec_from_file_location('wg_integrity',loc)
        auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
        import pyarrow.parquet as pq
        f=pd.read_parquet(args.data_dir/'production_failures.parquet',columns=['uwi','failure_start_date'])
        report['historical_split']=auditor.temporal_split_checks()
        report['failure_label_and_past_history_replay']=auditor.check_labels_and_past_history(test,f)
        report['sampled_original_daily_feature_replay']=verify_source_features(args.data_dir,test,args.sample_wells)
        if not report['sampled_original_daily_feature_replay']['passed']:
            raise AssertionError('Original daily features do not reproduce on sampled wells')
        uncertainty=week_cluster_uncertainty(test,args.iterations)
        report['test_metrics_recomputed']=detail['metrics']
        report['uncertainty']=uncertainty
        report['month_count']=len(detail['months'])
        write_private_charts(target,detail,detail['months'],uncertainty,report['original_source_identity'],diffs)
        report['verified_on_this_computer']=True
        report['status']='PASS_NARROW_TEMPORAL_AND_SOURCE_REPLAY'
        report['residual_limitations']=['Same wells can occur in train and test',
             'Failure-record completeness/reporting bias not resolved',
             'Published holdout has been consulted for model and policy ideas',
             'No causal proof of avoided failure or realized production savings',
             'Weekly-block intervals are descriptive and cannot quantify prospective generalization']
        print('COMPLETE — source hashes, source feature sample, test labels, holdout metrics, curves, months and uncertainty verified.')
    except Exception as exc:
        report['failure_reason']=str(exc)
        print('BLOCKED —',exc,file=sys.stderr)
        traceback.print_exc()
    finally:
        (target/'validation_evidence.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('Evidence status:',report['status'])
    print('Report:',target/'validation_evidence.json')
    if report['status'].startswith('PASS'):print('Private charts:',target/'PRIVATE_ROC_PR_MONTHLY_UNCERTAINTY.html')
    return 0 if report['status'].startswith('PASS') else 2

if __name__=='__main__':raise SystemExit(main())
