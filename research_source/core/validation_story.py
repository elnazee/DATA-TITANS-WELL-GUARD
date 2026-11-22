"""Five vertically stacked data-science sections for WellGuard Model Validation Lab.

Never invent missing private results; report V9 exploratory outcomes only when
the saved real-data experiment can be loaded. The global dashboard date does not
change or retrain the locked chronological reference evaluation.
"""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from core.validation_lab import REFERENCE_SPLITS, evaluation_capacity_rows, evaluate_private_replay, benchmark_consistency

ACCENT='#2ad8dc'
GREEN='#34dbad'
AMBER='#f4c16a'
BLUE='#8188ef'

def read_object(path:Path)->dict:
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value,dict) else {}
    except (FileNotFoundError,OSError,json.JSONDecodeError,ValueError):
        return {}

def get_story_data(root:Path,out:Path)->dict:
    saved=read_object(root/'results'/'source_benchmark.json')
    live=read_object(out/'aggregate_model_metrics.json')
    benchmark=live if all(k in live for k in ('model_ap','model_roc_auc','test_well_weeks')) else saved
    exp=read_object(out/'experiments'/'v9_upgrade'/'EXPERIMENT_RESULTS.json')
    if not (exp.get('status')=='EXPLORATORY_NOT_PROSPECTIVELY_VALIDATED'
            and exp.get('reference_model_modified') is False
            and exp.get('reference_metrics_modified') is False
            and exp.get('models_selected_using_test_data') is False):
        exp={}
    fingerprint=read_object(root/'results'/'source_fingerprint.json')
    quality=read_object(out/'source_quality_gate.json')
    audit=read_object(out/'audit_summary.json')
    return dict(benchmark=benchmark, saved=saved, live=bool(live and benchmark is live),
                experiment=exp, fingerprint=fingerprint, quality=quality, audit=audit,
                evidence=read_object(out/'private_detailed_validation'/'validation_evidence.json'),
                temporal=read_object(out/'private_temporal_audit.json'))

def v9_comparison(benchmark:dict, exp:dict)->dict:
    """Never equate equal summary counts with verified row-wise membership."""
    hold=exp.get('test_previous_holdout') or {}
    selected=hold.get('selected_policy') or {}
    fixed=hold.get('no_cooldown') or {}
    if not hold or not selected:return {'available':False}
    def good_number(x):
        return isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x)
    if not good_number(hold.get('ap')) or not good_number(selected.get('event_recall')):
        return {'available':False}
    same_summary=(hold.get('observations')==benchmark.get('test_well_weeks')
                  and hold.get('positives')==benchmark.get('test_positive_well_weeks')
                  and selected.get('eligible_events')==
                  (benchmark.get('model_top25_event_coverage') or {}).get('eligible_events'))
    if not (0<=selected['event_recall']<=1) or hold['ap']<0 or hold['ap']>1:
        return {'available':False}
    return dict(available=True,hold=hold,selected=selected,fixed=fixed,
                ap=float(hold['ap']),recall=float(selected['event_recall']),
                same_summary=bool(same_summary),
                baseline_ap=float(benchmark.get('model_ap',float('nan'))),
                baseline_recall=float((benchmark.get('model_top25_event_coverage') or {}).get('event_recall',float('nan'))))

def _safe(s):return html.escape(str(s),quote=True)
def _fmt_pct(value,decimals=1):
    return f'{100*value:.{decimals}f}%' if isinstance(value,(float,int)) and np.isfinite(value) else '—'
def _fnum(value,decimals=4):
    return f'{value:.{decimals}f}' if isinstance(value,(float,int)) and np.isfinite(value) else '—'

def _section(st,index,name,desc):
    st.markdown(f'''<div class="wgds-section"><span class="wgds-number">{index:02d}</span>
        <div><div class="wgds-section-title">{_safe(name)}</div><div class="wgds-section-sub">{_safe(desc)}</div></div></div>''',unsafe_allow_html=True)

def _card(st,title,value,detail='',tone='cyan'):
    color={'cyan':'#5ee2f2','green':'#66dfb5','amber':'#ffd184','purple':'#b1abff'}[tone]
    st.markdown(f'''<div class="wgds-card"><div class="wgds-label">{_safe(title)}</div>
    <div class="wgds-value" style="color:{color}">{_safe(value)}</div>
    <div class="wgds-detail">{_safe(detail)}</div></div>''',unsafe_allow_html=True)

def _sub(st,title,desc=''):
    st.markdown(f'<div class="wgds-subhead">{_safe(title)}</div>'+(f'<div class="wgds-detail wgds-below">{_safe(desc)}</div>' if desc else ''),unsafe_allow_html=True)

def _fig_style(fig,title,*,height=310):
    fig.update_layout(title=dict(text=title,font=dict(color='#e5f5fa',size=14)),
        template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#bad1df',size=11),height=height,margin=dict(l=12,r=12,t=47,b=30),
        legend=dict(orientation='h',y=-0.24),hovermode='x unified')
    fig.update_xaxes(showgrid=True,gridcolor='#2a4557',zeroline=False)
    fig.update_yaxes(showgrid=True,gridcolor='#2a4557',zeroline=False)
    return fig

def _load_private_curves(out:Path,bench:dict):
    file=out/'private_test_alerts.csv'
    if not file.is_file():return None,'Private test prediction rows not present on this machine.'
    try:
        frame=pd.read_csv(file,dtype={'uwi':'string'},parse_dates=['week_end'])
        detail=evaluate_private_replay(frame)
        issues=benchmark_consistency(bench,detail['metrics'])
        if issues:return None,'Private replay disagrees with the saved metrics: '+'; '.join(issues)
        return detail,None
    except (ValueError,KeyError,OSError,TypeError) as ex:
        return None,f'Private prediction validation failed: {ex}'

def load_v9_private_curves(out:Path, exp:dict, reference_frame:pd.DataFrame|None):
    """Load candidate predictions only if their keys/labels match the reference replay.

    Never chart an inferred V9 ROC from a scorecard, and do not equate aggregate
    matching counts with a row-by-row matched test population.
    """
    if not exp:return None,'V9 experiment has not been loaded from a signed local report.'
    path=out/'experiments'/'v9_upgrade'/'PRIVATE_EXPERIMENT_HOLDOUT_SCORES.csv'
    if not path.is_file():return None,'V9 private test prediction rows are not available.'
    if reference_frame is None:return None,'Reference private test predictions are required for a matched before/after curve.'
    try:
        new=pd.read_csv(path,dtype={'uwi':'string'},parse_dates=['week_end'])
        keys=['uwi','week_end'];needed=keys+['target_14d','risk_candidate']
        if not set(needed).issubset(new):raise ValueError('Missing required V9 prediction fields.')
        old=reference_frame.copy()
        old['uwi']=old['uwi'].astype('string')
        new['uwi']=new['uwi'].astype('string')
        old['week_end']=pd.to_datetime(old['week_end'],errors='raise')
        new['week_end']=pd.to_datetime(new['week_end'],errors='raise')
        if not set(keys+['target_14d','relative_risk_score']).issubset(old):
            raise ValueError('Missing required reference prediction fields.')
        if new[keys].duplicated().any() or old[keys].duplicated().any():
            raise ValueError('Duplicate well-week keys in model comparison.')
        newkeys=new[keys].sort_values(keys).reset_index(drop=True)
        oldkeys=old[keys].sort_values(keys).reset_index(drop=True)
        if not newkeys.equals(oldkeys):raise ValueError('Well-week observation membership differs between versions.')
        joined=old[keys+['target_14d','relative_risk_score']].merge(
            new[keys+['target_14d','risk_candidate']],on=keys,how='inner',validate='one_to_one',suffixes=('_before','_after'))
        if (joined.target_14d_before!=joined.target_14d_after).any():
            raise ValueError('Reference and V9 outcome labels differ.')
        y=joined.target_14d_before.astype(int).to_numpy()
        oldpred=joined.relative_risk_score.to_numpy(dtype=float)
        newpred=joined.risk_candidate.to_numpy(dtype=float)
        if not np.isfinite(newpred).all() or not np.isfinite(oldpred).all():raise ValueError('Missing or nonfinite model scores.')
        if len(set(y))!=2:raise ValueError('Both target classes required for ROC comparison.')
        hold=exp.get('test_previous_holdout') or {}
        if len(joined)!=hold.get('observations') or sum(y)!=hold.get('positives'):
            raise ValueError('V9 source prediction counts differ from experiment report.')
        # Check computed AP agrees with displayed result before plotting.
        if abs(average_precision_score(y,newpred)-float(hold.get('ap',-1)))>0.00005:
            raise ValueError('Recomputed V9 AP differs from saved experiment report.')
        return dict(joined=joined,curves={
            'roc_before':roc_curve(y,oldpred),'roc_after':roc_curve(y,newpred),
            'pr_before':precision_recall_curve(y,oldpred),'pr_after':precision_recall_curve(y,newpred),
        }),None
    except (ValueError,KeyError,OSError,TypeError) as err:return None,str(err)


def render_validation_story(st,go,root:Path,out:Path):
    css=root/'assets'/'model_validation_story.css'
    if css.is_file():st.markdown('<style>'+css.read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)
    data=get_story_data(root,out)
    bench=data['benchmark']; exp=data['experiment']; comparison=v9_comparison(bench,exp)
    st.markdown('''<div class="wgds-hero"><div class="wgds-overline">WELLGUARD AI · DATA SCIENCE / MODEL VALIDATION</div>
        <div class="wgds-title">From Raw Well Data to Defensible Decisions</div>
        <div class="wgds-tagline">Five connected stages of the research — data preparation, model development,
        improvements, scientific evaluation, and inspection value.</div>
        <div class="wgds-path">01 Source Data &nbsp;›&nbsp; 02 Model Development &nbsp;›&nbsp; 03 What We Improved &nbsp;›&nbsp; 04 Scientific Evidence &nbsp;›&nbsp; 05 Operational Impact</div></div>''',unsafe_allow_html=True)

    # 01 Sources, quality and reproducibility
    _section(st,1,'Data Sources & Engineering','Where the information came from, how it was cleaned, and how daily records became weekly prediction examples.')
    fp=data['fingerprint']
    q=data['quality'];audit=data['audit']
    counts=[('Daily operating rows','daily_production.parquet','5,075,590 in official source'),
            ('Maintenance records','well_jobs.parquet','Recorded maintenance jobs'),
            ('Failure-log records','production_failures.parquet','Historical recorded events')]
    cols=st.columns(4)
    for col,(title,key,detail) in zip(cols,counts):
        with col:
            value=fp.get(key,{}).get('rows')
            _card(st,title,f'{value:,}' if isinstance(value,int) else '—',detail)
    with cols[3]:
        total=sum(int(bench.get(k,0) or 0) for k in ('train_well_weeks','val_well_weeks','test_well_weeks'))
        _card(st,'Split well-week observations',f'{total:,}' if total else '—','Training + validation + test; not the number of wells','purple')
    st.markdown('''<div class="wgds-process">
        <span>Parquet sources</span><b>→</b><span>Dates & record audit</span><b>→</b>
        <span>Duplicate handling</span><b>→</b><span>Weekly features</span><b>→</b>
        <span>Leakage-aware split</span></div>''',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        _sub(st,'Data-quality checks')
        checks=[('Repeated well-date rows',q.get('repeat_well_date_rows',audit.get('duplicate_well_dates'))),
                ('Conflicting well-date groups',q.get('conflicting_well_date_groups',audit.get('conflicting_well_dates_in_selected_columns'))),
                ('Pump-efficiency readings > 1 and ≤ 100',q.get('pump_efficiency_above_1_to_100')),
                ('Unique recorded production wells',q.get('daily_distinct_wells',audit.get('unique_wells')))]
        reviewed=[{'Quality check':n,'Recorded value':f'{v:,}'} for n,v in checks if isinstance(v,int)]
        if reviewed:
            st.dataframe(pd.DataFrame(reviewed),hide_index=True,use_container_width=True)
        else:
            st.markdown('**Reviewed during preprocessing:** repeated well-days, source-date alignment, efficiency units, missing measurements, and eligible well coverage.')
    with right:
        _sub(st,'Tools & reproducibility')
        st.markdown('**Python / Pandas / PyArrow** — source preparation; **LightGBM** — historical failure ranking; **Streamlit / Plotly** — interactive dashboard.')
    st.divider()

    # 02 Model design and safeguarded chronology
    _section(st,2,'Model Development','The research objective, information available to the model, and the separated historical evaluation periods.')
    st.markdown('''<div class="wgds-research"><strong>Research question</strong><div>How can predictive maintenance and production optimization analytics help Strathcona Resources identify high-priority wells, prioritize inspections, and support data-driven decisions to reduce operational losses?</div>
    <small>Measurable ML objective: rank wells at risk of a recorded failure within 1–14 days, then evaluate a fixed weekly review budget.</small></div>''',unsafe_allow_html=True)
    splitcols=st.columns(3)
    specs=[('TRAIN','train_well_weeks','Jun 1, 2025 – Mar 1, 2026','green'),
           ('VALIDATE','val_well_weeks','Mar 22 – May 17, 2026','amber'),
           ('TEST','test_well_weeks','Jun 7 – Oct 4, 2026','purple')]
    for col,(name,key,dates,tone) in zip(splitcols,specs):
        with col:_card(st,name,f"{int(bench[key]):,}" if bench.get(key) is not None else '—',dates+' · well-weeks',tone)
    st.markdown('**Time-aware validation:** A 21-day separation between each development period exceeds the 14-day outcome horizon.')
    c1,c2=st.columns(2)
    with c1:
        _sub(st,'What the model learns from')
        st.markdown('**Operating condition** — oil production, RPM, motor current, torque, downtime.  \n**Historical changes** — week-to-week shifts, variability, deviation from each well’s prior baseline.  \n**Earlier recorded activity** — past failure history and maintenance-job start dates.')
    with c2:
        _sub(st,'Leakage safeguards and caveats')
        st.markdown('**Training only** for fitting; **validation only** for candidate and policy selection; later test for retrospective assessment.  \n**No future failure labels** as model features. **Earlier maintenance dates** only, but the actual data-entry time is not always available.')
    st.divider()

    # 03 Improvements without conflating model and inspection-policy changes
    _section(st,3,'Feature Engineering & Model Extensions','Compare the reference approach with richer historical signals and inspection policies.')
    c1,c2=st.columns(2)
    with c1:
        _sub(st,'Before · initial approach')
        st.markdown('• Recent operating averages  \n• Absolute measurement values  \n• Historical failure context  \n• Default model-score ranking for 25 reviews/week')
    with c2:
        _sub(st,'After · enriched candidate')
        st.markdown('• Weekly changes and four-week variability  \n• Deviations from each well’s earlier baseline  \n• Past-only 30/90-day job-start history  \n• Cooldown strategy chosen on validation data')
    ref_rec=(bench.get('model_top25_event_coverage') or {}).get('event_recall')
    if comparison.get('available'):
        a,b,c,d=st.columns(4)
        with a:_card(st,'Reference AP',_fnum(bench.get('model_ap')),'Historical benchmark')
        with b:_card(st,'Candidate AP · exploratory',_fnum(comparison.get('ap')),'Later model experiment','green')
        with c:_card(st,'Reference event recall @25',_fmt_pct(ref_rec),'Original policy')
        with d:_card(st,'Candidate event recall @25',_fmt_pct(comparison.get('recall'),2),'Selected review policy','green')
        fixed=comparison.get('fixed') or {}; selected=comparison['selected']
        _sub(st,'Model and inspection-policy comparison','Historical results at 25 reviews per week')
        rows=[{'Approach':'Reference features and policy', 'Distinct events flagged':(bench.get('model_top25_event_coverage') or {}).get('caught_events'), 'Event recall':_fmt_pct(ref_rec)},
              {'Approach':'Extended model with unchanged policy', 'Distinct events flagged':fixed.get('caught_events'), 'Event recall':_fmt_pct(fixed.get('event_recall'))},
              {'Approach':'Extended model with selected policy', 'Distinct events flagged':selected.get('caught_events'), 'Event recall':_fmt_pct(selected.get('event_recall'))}]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('Candidate results are exploratory. See the methodology for test-period reuse and comparison limitations.')
    else:
        a,b=st.columns(2)
        with a:_card(st,'Reference average precision',_fnum(bench.get('model_ap')),'Historical benchmark')
        with b:_card(st,'Reference event recall @25',_fmt_pct(ref_rec),'Distinct eligible recorded events','green')
    importance_file=out/'experiments'/'v9_upgrade'/'CANDIDATE_FEATURE_IMPORTANCE.csv'
    if comparison.get('available') and importance_file.is_file():
        try:
            importance=pd.read_csv(importance_file)
            if {'feature','tree_importance'}.issubset(importance.columns):
                importance['tree_importance']=pd.to_numeric(importance.tree_importance,errors='coerce')
                importance=importance.dropna(subset=['tree_importance']).nlargest(10,'tree_importance')
                if not importance.empty:
                    _sub(st,'Which historical features influenced the upgraded candidate?')
                    fig=go.Figure(go.Bar(x=importance.tree_importance[::-1],y=importance.feature[::-1],orientation='h',marker_color=GREEN))
                    _fig_style(fig,'Top tree-model split importance (noncausal)',height=335)
                    fig.update_xaxes(title='Split importance')
                    st.plotly_chart(fig,use_container_width=True)
                    st.caption('Importance comes from the fitted private candidate; it does not prove physical causes of failure.')
        except (OSError,ValueError,TypeError):
            pass
    st.divider()

    # 04 Scientific evidence 
    _section(st,4,'Scientific Evaluation','Predictive quality, monthly stability, uncertainty, and the known limits of historical evaluation.')
    ref_metrics=bench
    d1,d2,d3,d4=st.columns(4)
    with d1:_card(st,'Reference ROC-AUC',_fnum(ref_metrics.get('model_roc_auc')),'Measures ranking separation, not accuracy')
    with d2:_card(st,'Reference average precision',_fnum(ref_metrics.get('model_ap')),'Failure-recency AP: '+_fnum(ref_metrics.get('recency_baseline_ap')),'green')
    with d3:_card(st,'Reference 25-review recall',_fmt_pct(ref_rec),'From distinct eligible recorded events','amber')
    with d4:_card(st,'Test positive well-week rate',_fmt_pct(ref_metrics.get('test_positive_rate'),2),'Why accuracy alone is misleading','purple')
    private,why=_load_private_curves(out,bench)
    reference_frame=None
    if private is not None:
        try:
            reference_frame=pd.read_csv(out/'private_test_alerts.csv',dtype={'uwi':'string'},parse_dates=['week_end'])
        except (OSError,ValueError):reference_frame=None
    upgraded,upgrade_why=load_v9_private_curves(out,exp,reference_frame)
    if private is not None:
        left,right=st.columns(2)
        with left:
            fig=go.Figure()
            for key,name,color in [('model_roc','Reference',ACCENT),('baseline_roc','Failure recency',AMBER)]:
                frame=private['curves'][key];fig.add_trace(go.Scatter(x=frame.fpr,y=frame.tpr,name=name,mode='lines',line=dict(color=color,width=2.5)))
            if upgraded is not None:
                roc_new=upgraded['curves']['roc_after']
                fig.add_trace(go.Scatter(x=roc_new[0],y=roc_new[1],name='V9 exploratory · matched',mode='lines',line=dict(color=GREEN,width=2.8)))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random',line=dict(dash='dash',color='#8192a4')))
            fig.update_xaxes(title='False positive rate');fig.update_yaxes(title='True positive rate');_fig_style(fig,'ROC · source-derived reference')
            st.plotly_chart(fig,use_container_width=True)
        with right:
            fig=go.Figure()
            for key,name,color in [('model_pr','Reference',ACCENT),('baseline_pr','Failure recency',AMBER)]:
                frame=private['curves'][key];fig.add_trace(go.Scatter(x=frame.recall,y=frame.precision,name=name,mode='lines',line=dict(color=color,width=2.5)))
            if upgraded is not None:
                pr_new=upgraded['curves']['pr_after']
                fig.add_trace(go.Scatter(x=pr_new[1],y=pr_new[0],name='V9 exploratory · matched',mode='lines',line=dict(color=GREEN,width=2.8)))
            fig.update_xaxes(title='Recall');fig.update_yaxes(title='Precision');_fig_style(fig,'Precision–recall · source-derived reference')
            st.plotly_chart(fig,use_container_width=True)
    else:
        fig=go.Figure(go.Bar(x=['WellGuard reference','Failure-recency baseline'],
            y=[bench.get('model_ap',0),bench.get('recency_baseline_ap',0)],
            marker_color=[ACCENT,AMBER],text=[_fnum(bench.get('model_ap')),_fnum(bench.get('recency_baseline_ap'))],textposition='outside'))
        fig.update_yaxes(title='Average precision')
        _fig_style(fig,'Reference vs baseline · historical test',height=310)
        st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        _sub(st,'Monthly historical average precision')
        months=private['months'] if private is not None else pd.DataFrame()
        if months.empty:
            saved=read_object(root/'results'/'advanced_validation_public.json').get('month_stability') or {}
            months=pd.DataFrame([{'month':k,'model_ap':v.get('model_ap'),'recency_ap':v.get('baseline_ap')} for k,v in saved.items()])
        if not months.empty:
            fig=go.Figure()
            for col,name,color in [('model_ap','Reference model',ACCENT),('recency_ap','Failure-recency rule',AMBER)]:
                if col in months:
                    fig.add_trace(go.Scatter(x=months['month'],y=months[col],name=name,mode='lines+markers',line=dict(color=color,width=2.8)))
            fig.update_yaxes(title='Average precision');_fig_style(fig,'Historical test stability',height=285)
            st.plotly_chart(fig,use_container_width=True)

    with right:
        _sub(st,'Evaluation scope')
        st.markdown('**Historical test:** June–October 2026.  \n**Positive well-week prevalence:** '+_fmt_pct(bench.get('test_positive_rate'),2)+'  \n**Baseline:** failure-recency ranking.  \n**Outcome:** distinct eligible recorded failures.')
        u=read_object(out/'private_uncertainty_aggregate.json')
        if u.get('descriptive_percentile_95_interval') and len(u['descriptive_percentile_95_interval'])==2:
            vals=u['descriptive_percentile_95_interval']
            st.metric('Reference AP difference vs recency',_fnum(u.get('ap_difference_model_minus_recency')))
            st.caption(f'Historical week-block sensitivity range: {vals[0]:+.4f} to {vals[1]:+.4f} (descriptive).')
    st.divider()

    # 05 Impact and inspection capacity
    _section(st,5,'Operational Decision Support','Convert historical risk rankings into a practical inspection-prioritization workflow.')
    capacity=st.selectbox('Weekly inspection capacity', [25,50,100],index=0,key='story_capacity')
    cap_rows=evaluation_capacity_rows(bench)
    if cap_rows.empty:st.warning('No historical capacity records in the saved benchmark.')
    else:
        sel=cap_rows.loc[cap_rows.reviews_per_week.eq(capacity)].iloc[0]
        a,b,c=st.columns(3)
        with a:_card(st,'Historical failure events flagged',f'{int(sel.model_caught):,} / {int(sel.eligible_events):,}',f'At {capacity} weekly reviews')
        with b:_card(st,'Event recall',_fmt_pct(sel.model_event_recall),'Recorded eligible events','green')
        with c:_card(st,'Additional flags vs recency',f'+{int(sel.additional_events_vs_recency)}','Same review capacity; not avoided failures','amber')
        chart=go.Figure()
        chart.add_trace(go.Bar(x=cap_rows.reviews_per_week.astype(str),y=cap_rows.model_caught,marker_color=ACCENT,name='Reference model'))
        chart.add_trace(go.Bar(x=cap_rows.reviews_per_week.astype(str),y=cap_rows.recency_caught,marker_color=AMBER,name='Recency baseline'))
        _fig_style(chart,'Distinct eligible recorded events flagged by weekly review budget',height=310)
        chart.update_layout(barmode='group');chart.update_xaxes(title='Reviews per week');chart.update_yaxes(title='Events flagged')
        st.plotly_chart(chart,use_container_width=True)
    st.markdown('''<div class="wgds-process wgds-actions"><span>1. Prioritize wells</span><b>→</b>
      <span>2. Review measurements & failure evidence</span><b>→</b>
      <span>3. Assign an inspection for operator review</span></div>''',unsafe_allow_html=True)
    st.caption('Interpretation: flagged historical events support inspection planning; operational and economic benefits require field validation.')
