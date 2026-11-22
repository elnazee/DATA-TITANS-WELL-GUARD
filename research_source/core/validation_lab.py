"""WellGuard model validation and operational-decision research, no invented results.

This module deliberately separates the locked historical reference from proposed new
split experiments. Inputs are private source-derived outputs when available; otherwise
only the previously saved aggregate benchmark is displayed and clearly attributed.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

HORIZON_DAYS = 14
REFERENCE_SPLITS = {
    'train_start': '2025-06-01', 'train_end': '2026-03-01',
    'validation_start': '2026-03-22', 'validation_end': '2026-05-17',
    'test_start': '2026-06-07', 'test_end': '2026-10-04',
}
CAPACITIES = (25, 50, 100)


def validate_split_plan(*, train_start, train_end, validation_start, validation_end,
                        test_start, test_end, label_available_through=None, horizon_days=14):
    """Preview-only chronology check: DOES NOT train or change official model metrics."""
    names = ('train_start','train_end','validation_start','validation_end','test_start','test_end')
    dates = [pd.Timestamp(x).normalize() for x in (train_start, train_end, validation_start,
              validation_end, test_start, test_end)]
    d = dict(zip(names,dates))
    issues=[]
    invalid=False
    if not d['train_start'] <= d['train_end']:
        invalid=True
        issues.append('Training start must precede training end.')
    if not d['validation_start'] <= d['validation_end']:
        invalid=True
        issues.append('Validation start must precede validation end.')
    if not d['test_start'] <= d['test_end']:
        invalid=True
        issues.append('Test start must precede test end.')
    gap_train = (d['validation_start']-d['train_end']).days
    gap_test = (d['test_start']-d['validation_end']).days
    if gap_train <= horizon_days:
        invalid=True
        issues.append(f'Train→validation boundary must exceed {horizon_days} days (currently {gap_train}).')
    if gap_test <= horizon_days:
        invalid=True
        issues.append(f'Validation→test boundary must exceed {horizon_days} days (currently {gap_test}).')
    if label_available_through is not None:
        covered = pd.Timestamp(label_available_through).normalize()
        if d['test_end'] + pd.Timedelta(days=horizon_days) > covered:
            invalid=True
            issues.append('Last test prediction would need failure outcomes beyond the declared available-through date.')
    else:
        issues.append('The failure-log coverage end has not been certified; test-label availability must be checked before any new evaluation.')
    return {'valid_temporal_order': not invalid,
            'train_to_validation_gap_days':gap_train,
            'validation_to_test_gap_days':gap_test,
            'warnings':issues,
            'preview_only':True,
            'retraining_or_reference_metric_changes':False}


def evaluation_capacity_rows(m:dict):
    result=[]
    for cap in CAPACITIES:
        model=m.get(f'model_top{cap}_event_coverage') or {}
        base=m.get(f'recency_top{cap}_event_coverage') or {}
        if not model:
            # Model metrics can also be published under the detailed comparison section.
            comp=m.get('comparison') or {}
            model=(comp.get('relative_risk_score') or {}).get('event_coverage',{}).get(str(cap),{})
            base=(comp.get('recency_baseline_score') or {}).get('event_coverage',{}).get(str(cap),{})
        if model.get('caught_events') is None:continue
        n=model.get('eligible_events')
        if n is None or n<=0:continue
        if base and base.get('eligible_events') != n:
            raise ValueError('Model and recency baseline use different event denominators.')
        hits=int(model['caught_events']); baseline=int(base['caught_events']) if base.get('caught_events') is not None else None
        if hits>n or hits<0 or (baseline is not None and (baseline>n or baseline<0)):
            raise ValueError('Invalid caught-event counts in benchmark.')
        result.append({'reviews_per_week':cap,'eligible_events':int(n),'model_caught':hits,
                       'model_event_recall':hits/n,
                       'recency_caught':baseline,
                       'recency_event_recall':(baseline/n if baseline is not None else None),
                       'selected_well_week_precision':m.get(f'model_top{cap}_well_week_precision'),
                       'additional_events_vs_recency':(hits-baseline if baseline is not None else None)})
    return pd.DataFrame(result)


def evaluate_private_replay(replay:pd.DataFrame):
    """Derive curves and month-level variation, never publish well-level rows."""
    cols={'week_end','uwi','target_14d','relative_risk_score','recency_baseline_score'}
    missing=cols-set(replay.columns)
    if missing:raise ValueError(f'Missing private replay columns: {sorted(missing)}')
    df=replay.copy()
    df['week_end']=pd.to_datetime(df.week_end,errors='raise')
    if df[['uwi','week_end']].duplicated().any():raise ValueError('Duplicate well-week in private replay.')
    if df.target_14d.isna().any() or not set(pd.unique(df.target_14d)).issubset({0,1}):
        raise ValueError('Invalid test labels.')
    if df[['relative_risk_score','recency_baseline_score']].isna().any().any():
        raise ValueError('Private test replay has missing scores.')
    if len(df)<2 or df.target_14d.nunique()!=2:raise ValueError('Both target classes are required.')
    if df.week_end.min()<pd.Timestamp(REFERENCE_SPLITS['test_start']) or df.week_end.max()>pd.Timestamp(REFERENCE_SPLITS['test_end']):
        raise ValueError('Private replay includes nonreference weeks; do not mix inference with evaluation.')
    y=df.target_14d.astype(int).to_numpy()
    p=df.relative_risk_score.to_numpy(dtype=float)
    b=df.recency_baseline_score.to_numpy(dtype=float)
    pr,rc,_=precision_recall_curve(y,p)
    fpr,tpr,_=roc_curve(y,p)
    pr_b,rc_b,_=precision_recall_curve(y,b)
    fpr_b,tpr_b,_=roc_curve(y,b)
    curves={'model_pr':pd.DataFrame({'recall':rc,'precision':pr}),
            'baseline_pr':pd.DataFrame({'recall':rc_b,'precision':pr_b}),
            'model_roc':pd.DataFrame({'fpr':fpr,'tpr':tpr}),
            'baseline_roc':pd.DataFrame({'fpr':fpr_b,'tpr':tpr_b})}
    months=[]
    for ym,g in df.groupby(df.week_end.dt.to_period('M')):
        if g.target_14d.nunique()!=2:continue
        months.append({'month':str(ym),'well_weeks':len(g),'positive_rate':g.target_14d.mean(),
                       'model_ap':average_precision_score(g.target_14d,g.relative_risk_score),
                       'recency_ap':average_precision_score(g.target_14d,g.recency_baseline_score)})
    metrics={'test_observations':len(df),'positive_observations':int(y.sum()),
             'positive_rate':float(y.mean()), 'model_ap':float(average_precision_score(y,p)),
             'recency_ap':float(average_precision_score(y,b)),
             'model_roc_auc':float(roc_auc_score(y,p))}
    return {'metrics':metrics, 'curves':curves, 'months':pd.DataFrame(months)}


def benchmark_consistency(saved:dict,replay_metrics:dict,ap_tol=1e-5,auc_tol=1e-5):
    checks={
        'model_ap':ap_tol,
        'recency_ap':ap_tol,
        'model_roc_auc':auc_tol,
    }
    aliases={'recency_ap':'recency_baseline_ap'}
    mismatches=[]
    for metric,tol in checks.items():
        left_key=aliases.get(metric,metric)
        if left_key not in saved:continue
        if abs(float(saved[left_key])-float(replay_metrics[metric]))>tol:
            mismatches.append(f'{left_key} differs between saved metrics and private test replay.')
    if saved.get('test_well_weeks') is not None and int(saved['test_well_weeks'])!=replay_metrics['test_observations']:
        mismatches.append('Test observation count differs from saved benchmark.')
    return mismatches


def planning_scenario(*, additional_flagged:int, fraction_actionable:float, intervention_success:float,
                      value_per_prevented_event_cad:float):
    """Hypothetical upper decision value ONLY; not field savings nor a causal estimate."""
    if additional_flagged<0 or not (0<=fraction_actionable<=1) or not (0<=intervention_success<=1) or value_per_prevented_event_cad<0:
        raise ValueError('Out-of-range scenario assumptions.')
    return {'additional_historical_events_flagged':additional_flagged,
            'hypothetical_actions_justified':additional_flagged*fraction_actionable,
            'hypothetical_successful_interventions':additional_flagged*fraction_actionable*intervention_success,
            'scenario_value_cad':additional_flagged*fraction_actionable*intervention_success*value_per_prevented_event_cad,
            'is_realized_savings':False,
            'notes':'Not a savings forecast; recorded historical alerts do not establish prevention or dollar value. Same review budget in both methods.'}


def render_candidate_upgrade(st,out:Path):
    """Read the separate V9 experiment, never replace the locked V8.9 metrics."""
    st.markdown('### V9 — Advanced ML experiments (separate from official benchmark)')
    path=out/'experiments'/'v9_upgrade'/'EXPERIMENT_RESULTS.json'
    if not path.is_file():
        st.info('No real-source V9 experiment has been run on this computer. The reference benchmark above remains unchanged. Run RUN_ML_EXPERIMENTS.bat to test trends, past-only maintenance features, LightGBM/XGBoost/CatBoost, and validation-chosen inspection cooldowns.')
        st.caption('Optional XGBoost/CatBoost require their optional packages; LightGBM experiments run with the existing setup. Results are NOT invented when files are missing.')
        return
    try:
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('status')!='EXPLORATORY_NOT_PROSPECTIVELY_VALIDATED' or r.get('reference_model_modified') is not False:
            st.error('Candidate results have unexpected status or provenance. Do not use them to replace the locked reference.')
            return
        st.warning('EXPLORATORY experiment. Candidate and cooldown were chosen on validation only, but the June–October test period was already examined in earlier WellGuard iterations. These are not new independent wins.')
        st.write('**Selected candidate:**',r.get('selected_model','Unknown'),'| **Review cooldown:**',r.get('selected_cooldown_weeks','?'),'weeks')
        st.caption('Comparison on the validation period; model selection uses this table only. Greater AP is better but improvements are not guaranteed.')
        vals=pd.DataFrame(r.get('candidate_validation_results') or [])
        if not vals.empty:
            show=['candidate','feature_count','validation_ap','validation_roc_auc','validation_event_recall25','validation_events_flagged25','validation_eligible_events']
            st.dataframe(vals[[c for c in show if c in vals]].rename(columns={
                'candidate':'Candidate','feature_count':'Predictors',
                'validation_ap':'Validation AP','validation_roc_auc':'Validation ROC-AUC',
                'validation_event_recall25':'Validation recall @25','validation_events_flagged25':'Events flagged @25',
                'validation_eligible_events':'Eligible validation events'}),hide_index=True,use_container_width=True)
        policies=pd.DataFrame(r.get('policy_validation_results') or [])
        if not policies.empty:
            show=['cooldown_weeks','caught_events','eligible_events','event_recall','well_week_precision']
            st.dataframe(policies[[c for c in show if c in policies]].rename(columns={
                'cooldown_weeks':'Cooldown (weeks)','caught_events':'Validation events flagged',
                'eligible_events':'Validation eligible events','event_recall':'Validation event recall',
                'well_week_precision':'Validation selected well-week precision'}),hide_index=True,use_container_width=True)
        hold=r.get('test_previous_holdout') or {}
        if hold:
            a,b,c=st.columns(3)
            a.metric('Exploratory holdout AP',f"{hold.get('ap',float('nan')):.4f}")
            a2=hold.get('selected_policy') or {}
            b.metric('Exploratory event recall @25',f"{100*a2.get('event_recall',float('nan')):.1f}%")
            c.metric('Events flagged @25',a2.get('caught_events','—'))
            st.caption('Holdout labels and predictions are real private records if the experiment ran successfully. Because this period was previously inspected, these values cannot be called an untouched independent test.')
        feature_file=path.parent/'CANDIDATE_FEATURE_IMPORTANCE.csv'
        if feature_file.is_file():
            try:
                fi=pd.read_csv(feature_file).head(12)
                if {'feature','tree_importance'}.issubset(fi.columns):
                    st.markdown('#### Candidate tree-feature importance (descriptive)')
                    st.bar_chart(fi.set_index('feature')['tree_importance'])
                    st.caption('Split importance is NOT a causal explanation or proof of a physical failure mechanism.')
            except (OSError,ValueError) as err:
                st.warning(f'Feature importance file could not be read: {err}')
        if r.get('skipped_optional_models'):
            st.info('Skipped models: '+', '.join(f'{k}: {v}' for k,v in r['skipped_optional_models'].items()))
        st.caption('No future sensor values or future failure outcomes are used as predictors. Maintenance uses only earlier recorded job start dates, but retrospective-entry bias remains possible.')
    except (OSError,ValueError,KeyError,TypeError) as err:
        st.error('Unable to validate separate experiment results: '+str(err))


def render_technical_appendix(st,go,root:Path,out:Path):
    st.markdown('## Model Validation & Decision Lab')
    st.caption('A reproducible view of historical evidence, the locked reference split, operational trade-offs, and data-integrity checks. No automatic retraining.')
    path=out/'aggregate_model_metrics.json'
    if path.exists():
        metrics=json.loads(path.read_text(encoding='utf-8'))
        st.success('Metrics source: locally generated private model evaluation.')
    else:
        metrics=json.loads((root/'results'/'source_benchmark.json').read_text(encoding='utf-8'))
        st.warning('Metrics source: packaged reference benchmark, NOT a reproduction on this computer. Run the historical model and temporal audit for source verification.')
    a,b,c,d=st.columns(4)
    a.metric('ROC-AUC',f"{metrics.get('model_roc_auc',float('nan')):.4f}")
    b.metric('Average precision',f"{metrics.get('model_ap',float('nan')):.4f}")
    c.metric('Failure-recency AP',f"{metrics.get('recency_baseline_ap',float('nan')):.4f}")
    d.metric('Positive rate',f"{100*metrics.get('test_positive_rate',float('nan')):.2f}%")
    st.caption('ROC-AUC is discrimination, not 80% accuracy. Average precision is a ranking metric, not a failure probability.')
    st.markdown('### Locked chronological development plan')
    st.dataframe(pd.DataFrame([
        {'Stage':'Train','First Sunday':'2025-06-01','Last Sunday':'2026-03-01','Eligible well-weeks':metrics.get('train_well_weeks')},
        {'Stage':'Validation','First Sunday':'2026-03-22','Last Sunday':'2026-05-17','Eligible well-weeks':metrics.get('val_well_weeks')},
        {'Stage':'Test','First Sunday':'2026-06-07','Last Sunday':'2026-10-04','Eligible well-weeks':metrics.get('test_well_weeks')},
    ]),hide_index=True,use_container_width=True)
    st.info('21 calendar days separate each split boundary, exceeding the 14-day future-outcome horizon. The test was later explored for additional policies: those explorations are NOT independent validation.')
    st.markdown('### Inspection capacity frontier — same historical predictions')
    rows=evaluation_capacity_rows(metrics)
    if rows.empty:
        st.warning('No evaluated inspection-capacity results found. Do not invent recall values.')
    else:
        cap=st.selectbox('Illustrative weekly review budget',CAPACITIES,index=0,key='validation_cap')
        sel=rows[rows.reviews_per_week.eq(cap)].iloc[0]
        t1,t2,t3=st.columns(3)
        t1.metric('Distinct eligible events flagged',f"{int(sel.model_caught):,} / {int(sel.eligible_events):,}")
        t2.metric('Event recall',f"{100*sel.model_event_recall:.1f}%")
        t3.metric('Additional flagged vs baseline',int(sel.additional_events_vs_recency))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=rows.reviews_per_week,y=100*rows.model_event_recall,name='WellGuard',mode='lines+markers',line=dict(color='#19d1ce',width=3)))
        fig.add_trace(go.Scatter(x=rows.reviews_per_week,y=100*rows.recency_event_recall,name='Failure-recency baseline',mode='lines+markers',line=dict(color='#e9af5b',width=3)))
        fig.update_layout(template='plotly_dark',title='Historical event coverage vs weekly review budget',xaxis_title='Reviews per week',yaxis_title='Eligible event recall (%)',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=320)
        st.plotly_chart(fig,use_container_width=True)
        st.caption('The review budget changes shortlist size, not model training or the official ROC-AUC/AP. More flagged events ≠ avoided failures.')
    st.markdown('### What information adds value, and what happens when data are stale?')
    comp=metrics.get('comparison') or {}
    experiments=[]
    names={'relative_risk_score':'Full WellGuard','operating_only':'Operating measurements only',
           'recency_baseline_score':'Recorded failure recency','seven_day_delayed_inputs':'Operating inputs 7 days old',
           'delayed_recency_baseline':'Recency with 7-day delay','downtime_baseline':'Downtime rule',
           'production_loss_baseline':'Production-loss rule'}
    for key,label in names.items():
        detail=comp.get(key) or {}
        if not detail:continue
        summary=(detail.get('event_coverage') or {}).get('25') or {}
        experiments.append({'Approach':label,'Average precision':detail.get('average_precision'),
                            'Events flagged (25/week)':summary.get('caught_events'),
                            'Eligible events':summary.get('eligible_events')})
    if experiments:
        st.dataframe(pd.DataFrame(experiments).round(4),hide_index=True,use_container_width=True)
        delayed=(comp.get('seven_day_delayed_inputs') or {}).get('event_coverage',{}).get('25',{})
        recency=(comp.get('delayed_recency_baseline') or {}).get('event_coverage',{}).get('25',{})
        if delayed and recency and delayed.get('caught_events')<recency.get('caught_events'):
            st.warning(f"Observed data-delay failure mode: with 7-day-old operating inputs, WellGuard flagged {delayed['caught_events']} eligible events at 25 reviews/week vs {recency['caught_events']} for the delayed recency baseline. A freshness-aware fallback is a pilot recommendation—not a field-validated policy.")
    else:
        st.info('Saved comparison experiments are unavailable. Run the reproducible historical model locally.')
    st.markdown('### Evaluation curves and monthly stability — private holdout replay')
    replay=out/'private_test_alerts.csv'
    if replay.is_file():
        try:
            detail=evaluate_private_replay(pd.read_csv(replay,dtype={'uwi':'string'},parse_dates=['week_end']))
            contradictions=benchmark_consistency(metrics,detail['metrics'])
            if contradictions:
                st.error('STOP — saved model metrics disagree with the private test replay: '+'; '.join(contradictions))
            else:st.success('Saved ROC-AUC, AP, and observation count reconcile to the private replay.')
            left,right=st.columns(2)
            for col,kind in ((left,'pr'),(right,'roc')):
                fig=go.Figure()
                for label,color in (('model','#19d1ce'),('baseline','#e9af5b')):
                    data=detail['curves'][f'{label}_{kind}']
                    xcol,ycol=('recall','precision') if kind=='pr' else ('fpr','tpr')
                    fig.add_trace(go.Scatter(x=data[xcol],y=data[ycol],mode='lines',name='WellGuard' if label=='model' else 'Failure-recency',line=dict(color=color,width=2)))
                fig.update_layout(template='plotly_dark',title='Precision–recall curve' if kind=='pr' else 'ROC curve',height=310,xaxis_title='Recall' if kind=='pr' else 'False-positive rate',yaxis_title='Precision' if kind=='pr' else 'True-positive rate',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
                col.plotly_chart(fig,use_container_width=True)
            if not detail['months'].empty:
                st.dataframe(detail['months'].rename(columns={'month':'Month','well_weeks':'Eligible observations','positive_rate':'Positive fraction','model_ap':'WellGuard AP','recency_ap':'Recency AP'}).round(4),hide_index=True,use_container_width=True)
                st.caption('Monthly AP is descriptive: small time groups and prevalence shifts make month-to-month comparisons unstable.')
        except (ValueError,KeyError,OSError) as e:st.error(f'Private replay validation failed: {e}')
    else:
        st.info('Precision–recall/ROC curves and monthly test stability require `outputs/private_test_alerts.csv`, generated from the actual source. No illustrative curves are used.')
    st.markdown('### Uncertainty of the AP comparison')
    uncertainty_path=out/'private_uncertainty_aggregate.json'
    if uncertainty_path.exists():
        result=json.loads(uncertainty_path.read_text(encoding='utf-8'))
        interval=result.get('descriptive_percentile_95_interval',[])
        if len(interval)==2:
            st.metric('Model AP − recency AP',f"{result['ap_difference_model_minus_recency']:+.4f}")
            st.caption(f"Descriptive week-block 95% interval: [{interval[0]:+.4f}, {interval[1]:+.4f}], based on {result.get('valid_draws','?')} re-samples. Not an independently validated confidence guarantee.")
    else:
        st.info('Uncertainty interval not computed here. Run `python scripts/14_private_uncertainty.py --out-dir outputs` locally after the source replay.')
    # V8.9: signed private full-source diagnostic panel. Never claim an audit pass
    # from a static benchmark or from a missing real-data evidence file.
    st.markdown('### Detailed source replay, ROC/PR and uncertainty evidence')
    detail_file = out/'private_detailed_validation'/'validation_evidence.json'
    detail_html = out/'private_detailed_validation'/'PRIVATE_ROC_PR_MONTHLY_UNCERTAINTY.html'
    if detail_file.is_file():
        try:
            evidence=json.loads(detail_file.read_text(encoding='utf-8'))
            status=evidence.get('status','BLOCKED')
            if status=='PASS_NARROW_TEMPORAL_AND_SOURCE_REPLAY':
                st.success('Original source identity, sampled feature replay, future-data invariance, historical labels and saved test metrics: PASSED locally.')
            else:
                st.error('Detailed source replay blocked: '+str(evidence.get('failure_reason','See evidence JSON')))
            if evidence.get('test_metrics_recomputed'):
                st.json(evidence['test_metrics_recomputed'])
            if evidence.get('uncertainty'):
                u=evidence['uncertainty']
                delta=u['point']['ap_difference'];lo,hi=u['descriptive_95_percentile_bounds']['ap_difference']
                st.caption(f'Week-block AP uplift {delta:+.4f}; descriptive percentile interval [{lo:+.4f}, {hi:+.4f}] across {u["valid_replicates"]} valid draws (NOT prospective certainty).')
            if status=='PASS_NARROW_TEMPORAL_AND_SOURCE_REPLAY' and detail_html.is_file():
                st.download_button('Download PRIVATE interactive ROC / PR / month / uncertainty report',
                                  data=detail_html.read_bytes(),
                                  file_name='WellGuard_PRIVATE_Original_Data_Validation.html',
                                  mime='text/html',key='download_private_model_validation')
        except (OSError,ValueError,KeyError) as err:
            st.error('Private validation evidence unreadable: '+str(err))
    else:
        st.info('Detailed real-source report not yet run on this computer. Run RUN_DETAILED_VALIDATION.bat after the historical model is ready. The dashboard will not invent a passing audit.')
        saved_path=root/'results'/'advanced_validation_public.json'
        if saved_path.is_file():
            saved=json.loads(saved_path.read_text(encoding='utf-8'))
            months=saved.get('month_stability') or {}
            if months:
                st.caption('Previously saved aggregate monthly AP — not newly reproduced on this computer:')
                st.dataframe(pd.DataFrame([{'Month':k,'Eligible observations':v['observations'],
                        'WellGuard AP':v['model_ap'],'Recency AP':v['baseline_ap']}
                        for k,v in months.items()]).round(4),hide_index=True,use_container_width=True)
            delta=saved.get('week_cluster_ap_difference_sensitivity') or {}
            if delta.get('percentile_95_interval'):
                st.caption(f'Previously saved week-resampled AP uplift {delta["point_difference"]:+.4f}, exploratory interval {delta["percentile_95_interval"]}. This is not a new source verification.')

    st.markdown('### Evidence and leakage gate')
    audit_path=out/'private_temporal_audit.json'
    quality_path=out/'source_quality_gate.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        if audit.get('status')=='PASS':st.success('Past-failure and 14-day event-label replay: PASS. Reporting/proxy leakage and test-reuse risks still apply.')
        else:st.warning('Temporal audit not passed: '+str(audit.get('status','unknown')))
    else:st.warning('Record-level temporal audit missing. Run CHECK_MODEL_INTEGRITY.bat before making definitive claims.')
    if quality_path.exists():
        quality=json.loads(quality_path.read_text(encoding='utf-8'))
        st.write(f"Source quality: {quality.get('conflicting_well_date_groups','?')} conflicting well-date groups; {quality.get('pump_efficiency_above_1_to_100','?')} pump-efficiency values above the modeled fractional unit range.")
    else:st.warning('Source quality report missing; original-data unit and duplicate issues are not signed off.')
    st.warning('Test-period model/policy exploration is retrospective. Failure reporting before 2025 is sparse. Neither issue is solved by a better dashboard score.')
    with st.expander('Explore hypothetical chronological splits — does NOT retrain or alter reference metrics'):
        a,b=st.columns(2)
        with a:
            tr_end=st.date_input('Proposed training end',value=date(2026,3,1),key='lab_train_end')
            val_start=st.date_input('Proposed validation start',value=date(2026,3,22),key='lab_val_start')
            val_end=st.date_input('Proposed validation end',value=date(2026,5,17),key='lab_val_end')
        with b:
            te_start=st.date_input('Proposed test start',value=date(2026,6,7),key='lab_test_start')
            te_end=st.date_input('Proposed test end',value=date(2026,10,4),key='lab_test_end')
            st.caption('Outcome availability and reporting completeness require a separate source-data check.')
        proposal=validate_split_plan(train_start=date(2025,6,1),train_end=tr_end,validation_start=val_start,validation_end=val_end,test_start=te_start,test_end=te_end)
        if proposal['valid_temporal_order']:st.success('Temporal order and 14-day split gaps pass. This is NOT a certified or trained new model.')
        else:st.error('Invalid or overlapping proposed split; do not use for training.')
        for msg in proposal['warnings']:st.caption('• '+msg)
    render_candidate_upgrade(st,out)
    st.markdown('### Business decision sensitivity — hypothetical only')
    st.caption('41 additional historical events flagged at 25 reviews/week compared with recency baseline. A flagged event is NOT an avoided failure.')
    if not rows.empty:
        reference=rows[rows.reviews_per_week.eq(25)]
        if not reference.empty:
            extra=int(reference.iloc[0].additional_events_vs_recency)
            p1,p2,p3=st.columns(3)
            actionable=p1.slider('Fraction of extra alerts actionable (%)',0,100,25,5,key='lab_actionable')/100
            success=p2.slider('Action succeeds (%)',0,100,25,5,key='lab_success')/100
            value=p3.number_input('Hypothetical CAD benefit per prevented event',min_value=0.0,value=10000.0,step=1000.0,key='lab_value')
            scenario=planning_scenario(additional_flagged=extra,fraction_actionable=actionable,intervention_success=success,value_per_prevented_event_cad=value)
            st.metric('Illustrative gross scenario value (NOT verified savings)',f"CAD {scenario['scenario_value_cad']:,.0f}")
            st.caption('No field intervention outcomes, causal effect, actual final maintenance cost, or oil recovery measured. Intervention costs are not included; the fixed review budget is equal across models.')
    st.markdown('### Decision for research claims')
    st.success('Supported: the saved model improved retrospective prioritization over a failure-recency baseline at equal review capacity.')
    st.error('Not supported: demonstrated reductions in actual failures, oil losses, maintenance costs, or proven financial ROI.')


def render_validation_lab(st,go,root:Path,out:Path):
    """Five stacked research sections plus optional original advanced appendix."""
    from core.validation_story import render_validation_story
    render_validation_story(st,go,root,out)
    # Technical diagnostics are intentionally kept separate from judge-facing pages.
    # Local analysts can enable them with WELLGUARD_SHOW_DIAGNOSTICS=1.
    import os
    if os.environ.get('WELLGUARD_SHOW_DIAGNOSTICS') == '1':
        with st.expander("Local research diagnostics",expanded=False):
            render_technical_appendix(st,go,root,out)
