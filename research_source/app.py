"""WellGuard AI — REAL competition records only (private local application).

There is deliberately NO synthetic/demo fallback in this application.
Missing official data or generated outputs produce a visible error, never fake wells.

Start: python -m streamlit run app.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.scheduling import schedule_inspections
from core.real_operations import health_snapshot, SENSOR_METRICS
from core.ai_investigator import explain as investigate, QUESTIONS
from core.command_center_page import render_command_center
from core.table_utils import arrow_safe_view
from core.maintenance_context import add_prior_job_counts
from core.queue_display import friendly_queue_table

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
OUT=ROOT/'outputs'
SOURCE_FILES=('daily_production.parquet','well_jobs.parquet','production_failures.parquet')

st.set_page_config(page_title='WellGuard AI | Real Operations',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
# The enterprise CSS changes presentation only; private source data remain unchanged.
THEME = ROOT / 'assets' / 'enterprise_theme.css'
st.markdown('<style>' + THEME.read_text(encoding='utf8') + '</style>', unsafe_allow_html=True)

@st.cache_data(show_spinner='Loading real operating history…')
def weekly_history():
    x=pd.read_parquet(OUT/'private_weekly_history.parquet')
    x['uwi']=x.uwi.astype(str)
    x['week_end']=pd.to_datetime(x.week_end)
    return x

@st.cache_data(show_spinner='Loading verified well locations…')
def locations():
    x=pd.read_csv(OUT/'private_well_locations.csv',dtype={'uwi':'string'},parse_dates=['source_date'])
    x.uwi=x.uwi.astype(str)
    return x

@st.cache_data(show_spinner=False)
def maintenance():
    p=OUT/'private_maintenance_windows.csv'
    if not p.exists():return pd.DataFrame()
    x=pd.read_csv(p,dtype={'uwi':'string'},parse_dates=['job_start_date'])
    x.uwi=x.uwi.astype(str)
    return x

@st.cache_data(show_spinner=False)
def model_replay():
    paths=[OUT/'private_test_with_context.csv' if (OUT/'private_test_with_context.csv').exists() else OUT/'private_test_alerts.csv',
           OUT/'private_recent_model_scores.csv']
    frames=[]
    for p in paths:
        if p.exists():
            df=pd.read_csv(p,parse_dates=['week_end'],dtype={'uwi':'string'})
            if not df.empty:
                df['uwi']=df.uwi.astype(str)
                if 'score_origin' not in df:df['score_origin']='RETROSPECTIVE_HOLDOUT'
                frames.append(df)
    if not frames:return pd.DataFrame()
    result = pd.concat(frames,ignore_index=True).drop_duplicates(['uwi','week_end'],keep='last')
    # The post-holdout scoring file does not include the maintenance context
    # generated for earlier test weeks. Recompute this as-of field from the
    # original source for ALL score weeks, without retraining the risk model.
    jobs = pd.read_parquet(DATA/'well_jobs.parquet', columns=['uwi','start_date'])
    return add_prior_job_counts(result, jobs)

@st.cache_data(show_spinner=False)
def failures():
    x=pd.read_parquet(DATA/'production_failures.parquet')
    x['uwi']=x.uwi.astype(str)
    x['failure_start_date']=pd.to_datetime(x.failure_start_date,errors='coerce')
    return x

@st.cache_data(show_spinner=False)
def quality_reports():
    reports={}
    for name in ('private_real_history_audit.json','private_v6_operations_audit.json','source_quality_gate.json','aggregate_model_metrics.json'):
        p=OUT/name
        if p.exists():
            try:reports[name]=json.loads(p.read_text())
            except (OSError,ValueError):pass
    return reports


def asof_snapshot(hist:pd.DataFrame,cutoff)->pd.DataFrame:
    """One real row per well: latest observed week no later than cutoff."""
    h=hist[hist.week_end.le(pd.Timestamp(cutoff))]
    return h.sort_values(['uwi','week_end']).drop_duplicates('uwi',keep='last').copy()


def real_summary_table(frame:pd.DataFrame,key:str,columns:list[str]|None=None):
    raw = arrow_safe_view(frame, columns)
    display = friendly_queue_table(raw) if key == 'queue' else raw.copy()
    q=st.text_input('Search records',key='search_'+key)
    if q:
        matched=display.astype(str).apply(lambda col:col.str.contains(q,regex=False,case=False)).any(axis=1)
        raw=raw.loc[matched]
        display=display.loc[matched]
    st.caption(f'{len(display):,} **actual competition records or derived real-data summaries**. Exported rows are restricted competition data.')
    if key == 'queue':
        st.caption('Model scores are relative ranking values—not calibrated failure probabilities. '
                   'Downtime means the average share of observed days with any downtime, not hours down. '
                   'A zero maintenance count means no distinct recorded job-start date within the preceding 90 days; '
                   'it does not prove no work occurred. Private exports retain the original feature names and values.')
    st.dataframe(display,hide_index=True,use_container_width=True,height=365)
    st.download_button('Export PRIVATE filtered CSV',raw.to_csv(index=False).encode('utf-8'),
                       file_name=f'PRIVATE_WELLGUARD_{key}.csv',mime='text/csv',key='csv_'+key)


def plotted_line(frame:pd.DataFrame,y:str,header:str):
    x=frame[['week_end',y]].dropna()
    if x.empty:st.info('No valid recorded measurements for this variable and date range.');return
    fig=px.line(x,x='week_end',y=y,markers=True,title=header,template='plotly_dark')
    fig.update_traces(line_color='#25d3d0')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)


missing=[x for x in SOURCE_FILES if not (DATA/x).is_file()]
outputs_missing=[x for x in ('private_well_locations.csv','private_weekly_history.parquet') if not (OUT/x).is_file()]
if missing or outputs_missing:
    st.error('Private source files and prepared local outputs are required to open the full dashboard.')
    if missing:st.markdown('Missing official source files: '+', '.join(f'`data/{x}`' for x in missing))
    if outputs_missing:st.markdown('Run local preparation to create: '+', '.join(f'`outputs/{x}`' for x in outputs_missing))
    st.code('python scripts/10_private_operations_context.py --data-dir data --out-dir outputs\n'
            'python scripts/11_prepare_real_history.py --data-dir data --out-dir outputs\n'
            'python -m streamlit run app.py',language='powershell')
    st.stop()

weekly=weekly_history()
geo=locations()
if weekly.empty or geo.empty:
    st.error('The selected source files do not contain a usable well-level view. Review local data preparation.')
    st.stop()
reports=quality_reports()
score=model_replay()
maint=maintenance()

# Navigation query values are local-only and validated against the app's own pages.
pages=['Executive Overview','Full Fleet Map','Equipment Health','Failure Records',
       'Inspection Scheduler','Model Validation Lab','Maintenance Effectiveness','Production & Downtime',
       'Data Reliability','AI Well Investigator']
if 'wg_page' in st.query_params:
    target=st.query_params.get('wg_page')
    if target in pages:st.session_state.wg_page=target
    uwi=st.query_params.get('wg_well')
    if uwi and str(uwi) in set(weekly.uwi.astype(str)):st.session_state.chosen_well=str(uwi)
    st.query_params.clear()
# Important: Streamlit forbids changing a radio widget's session_state key
# after that widget is instantiated in the current run. Navigation buttons
# record an intent; the intent is applied on the NEXT rerun, before st.radio.
from core.navigation import apply_pending_navigation
apply_pending_navigation(st.session_state, pages, weekly.uwi.astype(str))
with st.sidebar:
    st.markdown('''<div class="wg-side-brand"><div class="wg-side-logo-mark">◇</div><div><div class="wg-side-wordmark">WELLGUARD <b>AI</b></div><div class="wg-side-sub">OPERATIONS INTELLIGENCE</div></div></div><div class="wg-side-section">Workspace</div>''',unsafe_allow_html=True)
    nav_icons={'Executive Overview':'⌂','Full Fleet Map':'◫','Equipment Health':'⌁',
               'Failure Records':'⚠','Inspection Scheduler':'▤','Model Validation Lab':'◈','Maintenance Effectiveness':'⚒',
               'Production & Downtime':'▥','Data Reliability':'◇','AI Well Investigator':'✧'}
    page=st.radio('Workspace',pages,format_func=lambda p:f"{nav_icons.get(p,'•')}   {p}",
                  label_visibility='collapsed',key='wg_page')
    st.markdown('<div class="wg-side-section" style="margin-top:18px">Historical filters</div>',unsafe_allow_html=True)
    weeks=sorted(weekly.week_end.dt.date.unique(),reverse=True)
    if len(score):
        pref=score.week_end.max().date()
        default=weeks.index(pref) if pref in weeks else 0
    else:default=0
    cutoff=st.selectbox('Historical as-of week',weeks,index=default)
    areas=['All',*sorted(geo.area.fillna('Unknown').astype(str).unique())]
    area_choice=st.selectbox('Operating area',areas)
    provinces=['All',*sorted(geo.province.fillna('Unknown').astype(str).unique())]
    province_choice=st.selectbox('Province',provinces)
    st.caption('Historical well-level records · Risk scores are relative rankings.')

snap=asof_snapshot(weekly,cutoff)
view=snap.merge(geo[['uwi','lat','lon','area','facility','province','status','source_date']],on='uwi',how='left',validate='one_to_one')
view['age_days']=(pd.Timestamp(cutoff)-view.week_end).dt.days
if area_choice!='All':view=view[view.area.eq(area_choice)]
if province_choice!='All':view=view[view.province.eq(province_choice)]
if not len(view):
    st.warning('There are no recorded wells for the current filters. Try another date or area.')
    st.stop()

score_week=pd.DataFrame()
if len(score):
    score_week=score[score.week_end.dt.date.eq(cutoff)].copy()
    if len(score_week):
        view=view.merge(score_week[['uwi','relative_risk_score']].drop_duplicates('uwi'),on='uwi',how='left',validate='one_to_one')
    else:
        view['relative_risk_score']=np.nan
else:view['relative_risk_score']=np.nan

if 'chosen_well' not in st.session_state:st.session_state.chosen_well=str(view.uwi.iloc[0])
if st.session_state.chosen_well not in set(weekly.uwi):st.session_state.chosen_well=str(view.uwi.iloc[0])

if page!='Executive Overview':
    st.markdown('''<div class="wg-enterprise-bar"><div class="wg-brand-row"><div class="wg-logo-mark">◇</div><div><div class="wg-logo">WELLGUARD <em>AI</em></div><div class="wg-subbrand">OPERATIONS INTELLIGENCE</div></div></div><div class="wg-pill">● REAL SOURCE DATA · PRIVATE</div></div><div class="wg-breadcrumb">WELLGUARD  ›  Operations Intelligence  ›  Real source-data views</div>''',unsafe_allow_html=True)

if page=='Executive Overview':
    # When private training is complete, its metrics supersede old published
    # benchmark numbers (which may have used different duplicate handling).
    benchmark_path=OUT/'aggregate_model_metrics.json'
    if not benchmark_path.exists():benchmark_path=ROOT/'results'/'source_benchmark.json'
    try:
        benchmark=json.loads(benchmark_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        benchmark={}
    # The capacity selector is rendered next to the retrospective recall card.
    # On widget change, Streamlit makes its value available before the rerun.
    capacity=st.session_state.get('wg_capacity_shared',st.session_state.get('command_capacity',25))
    if 'command_capacity' not in st.session_state:
        st.session_state.command_capacity=capacity
    if capacity not in (25,50,100):
        capacity=25
    render_command_center(weekly,view,geo,score_week,cutoff,benchmark,failures(),capacity=capacity)

elif page=='Full Fleet Map':
    from core.fleet_geography_page import render_fleet_geography
    render_fleet_geography(view,cutoff,root=ROOT)

elif page=='Equipment Health':
    st.subheader('Equipment health — actual recorded operating trends')
    wells=sorted(view.uwi.astype(str).unique())
    idx=wells.index(st.session_state.chosen_well) if st.session_state.chosen_well in wells else 0
    st.session_state.chosen_well=st.selectbox('Well identifier',wells,index=idx)
    hist=weekly[weekly.uwi.eq(st.session_state.chosen_well)&weekly.week_end.le(pd.Timestamp(cutoff))].sort_values('week_end')
    result=health_snapshot(hist,cutoff)
    if not result['has_snapshot']:st.info('No observed measurements by selected date.')
    else:
        c1,c2,c3=st.columns(3)
        c1.metric('Recorded pump channels',f"{result['pump_channels_available']}/4")
        c2.metric('Last recorded week',result['observed_week'])
        c3.metric('Age of latest record',f"{result['age_days']} days")
        tbl=pd.DataFrame(result['metrics'])
        st.dataframe(tbl[['name','unit','current','baseline','change_pct','flag']],hide_index=True,use_container_width=True)
        metric=st.selectbox('Measurement',list(SENSOR_METRICS),format_func=lambda x:x[1])
        plotted_line(hist,metric[0],f"Actual {metric[1]} — {st.session_state.chosen_well}")
        real_summary_table(hist,'health',['uwi','week_end','days','producing_days','oil_daily_avg_2w','lost_per_week_2w','hours_down',metric[0]])
        if result['data_quality']!='AVAILABLE WITH CAVEATS':st.warning(result['data_quality'])
        st.caption('Changes are descriptive cues, not fault diagnostics; efficiency source units remain subject to verification.')

elif page=='Failure Records':
    from core.failure_intelligence_page import render_failure_intelligence
    # All failure intelligence widgets derive from the SAME source event scope.
    # Sidebar area/province are global filters; the selected historical cutoff
    # and the independent analysis-period selector both apply before KPIs/charts.
    permitted = view.uwi.astype(str) if (area_choice!='All' or province_choice!='All') else None
    area_lookup = geo[['uwi','area']].drop_duplicates('uwi').set_index('uwi').area.to_dict()
    render_failure_intelligence(failures=failures(),weekly=weekly,scores=score,cutoff=cutoff,
                                permitted_wells=permitted,area_lookup=area_lookup)

elif page=='Inspection Scheduler':
    st.subheader('Retrospective risk-ranked inspection reviews')
    st.caption('Historical model rankings from the authorized local research dataset.')
    if score.empty:st.error('No model scores are available. Run the historical model pipeline to enable this view.')
    else:
        score_weeks=sorted(score.week_end.dt.date.unique(),reverse=True)
        week=st.selectbox('Scored historical week',score_weeks)
        capacity=st.select_slider('Weekly review capacity (scenario, not reported staffing)',options=[10,25,50,100],value=25)
        cooldown=st.slider('Repeat-review cooldown (exploratory)',0,3,0)
        # Compute prior selections up to selected week to apply an actual cooldown rule.
        eligible=score[score.week_end.dt.date.le(week)].copy()
        chosen=schedule_inspections(eligible,capacity=capacity,cooldown_weeks=cooldown,through_date=week)
        chosen=chosen[chosen.week_end.dt.date.eq(week)].copy()
        if area_choice!='All':
            chosen=chosen.merge(geo[['uwi','area']],on='uwi',how='left')
            chosen=chosen[chosen.area.eq(area_choice)]
        st.metric('Selected wells for review',len(chosen))
        st.caption('Model output is an uncalibrated ranking. Cooldown policies were explored retrospectively and have not demonstrated prospective uplift.')
        if not chosen.empty:
            chosen=chosen.sort_values('review_rank')
            st.plotly_chart(px.bar(chosen.head(25),x='uwi',y='relative_risk_score',title='Highest-ranked recorded wells',template='plotly_dark'),use_container_width=True)
            st.session_state.chosen_well=st.selectbox('Review candidate well',chosen.uwi.astype(str).tolist())
            real_summary_table(chosen,'queue',['review_rank','week_end','uwi','relative_risk_score','oil_daily_avg_2w','lost_per_week_2w','down_fraction_2w','prior_jobs_90d'])
        metrics=reports.get('aggregate_model_metrics.json')
        if metrics:
            st.markdown('#### Historical model validation')
            st.dataframe(pd.DataFrame([{'Metric':'Average precision','Result':metrics.get('model_ap')},
                                       {'Metric':'ROC-AUC','Result':metrics.get('model_roc_auc')},
                                       {'Metric':'Failure recency baseline AP','Result':metrics.get('recency_baseline_ap')}]),hide_index=True)

elif page=='Model Validation Lab':
    from core.validation_lab import render_validation_lab
    render_validation_lab(st,go,ROOT,OUT)

elif page=='Maintenance Effectiveness':
    st.subheader('Maintenance — observed before and after recorded jobs')
    st.caption('All jobs and production changes are taken from the actual two source files. The comparison is descriptive, NOT causal.')
    if maint.empty:st.error('No verified maintenance-window file. Run scripts/10_private_operations_context.py; no service jobs will be invented.')
    else:
        j=maint[maint.job_start_date.le(pd.Timestamp(cutoff)-pd.Timedelta(days=30))].copy()
        if area_choice!='All':j=j[j.uwi.isin(view.uwi)]
        choice=st.selectbox('Maintenance job type',['All',*sorted(j.job_type.dropna().astype(str).unique())])
        if choice!='All':j=j[j.job_type.eq(choice)]
        only_clean=st.checkbox('Exclude concurrent jobs within ±30 days',value=False)
        if only_clean:j=j[~j.has_other_job_within_30d.astype(bool)]
        if j.empty:st.info('No source job records meet the filters and sufficient pre/post observation requirement.')
        else:
            c1,c2,c3=st.columns(3)
            c1.metric('Eligible observed jobs',len(j))
            c2.metric('Median observed oil-rate change',f"{j.oil_change_m3d.median():+.2f} m³/day")
            c3.metric('Median field-estimated cost',f"CAD {j.estimated_cost_cad.median():,.0f}")
            agg=j.groupby('job_type',as_index=False).agg(before=('oil_before_m3d','median'),after=('oil_after_m3d','median'))
            fig=go.Figure([go.Bar(x=agg.job_type,y=agg.before,name='Prior 30 days'),go.Bar(x=agg.job_type,y=agg.after,name='Following 30 days')])
            fig.update_layout(barmode='group',title='Observed median oil rate near job starts',template='plotly_dark')
            st.plotly_chart(fig,use_container_width=True)
            real_summary_table(j,'maintenance',['uwi','job_start_date','job_type','estimated_cost_cad','oil_before_m3d','oil_after_m3d','oil_change_m3d','days_oil_before','days_oil_after','has_other_job_within_30d'])
            st.warning('These are field-estimated, not actual job costs. Before/after changes cannot be attributed to the job without causal evaluation.')

elif page=='Production & Downtime':
    st.subheader('Historical production and downtime — recorded weekly totals')
    hist=weekly[weekly.week_end.le(pd.Timestamp(cutoff))].copy()
    hist=hist[hist.uwi.isin(view.uwi)]
    periods=st.slider('Most recent weeks to inspect',4,52,16,4)
    recent_weeks=sorted(hist.week_end.unique())[-periods:]
    hist=hist[hist.week_end.isin(recent_weeks)]
    by_week=hist.groupby('week_end',as_index=False).agg(oil=('oil_total_m3','sum'),downtime=('hours_down','sum'),loss=('estimated_lost_oil_m3','sum'))
    t1,t2=st.columns(2)
    t1.plotly_chart(px.line(by_week,x='week_end',y='oil',title='Weekly recorded oil volume (m³)',template='plotly_dark'),use_container_width=True)
    t2.plotly_chart(px.line(by_week,x='week_end',y='downtime',title='Weekly recorded downtime (hours)',template='plotly_dark'),use_container_width=True)
    st.plotly_chart(px.bar(by_week,x='week_end',y='loss',title='Weekly estimated lost oil (m³)',template='plotly_dark'),use_container_width=True)
    real_summary_table(hist,'production',['uwi','week_end','days','producing_days','oil_total_m3','water_total_m3','hours_down','estimated_lost_oil_m3'])

elif page=='Data Reliability':
    st.subheader('Source data quality — verifiable observations only')
    st.caption('Record completeness is not calibrated model confidence, and an absent sensor is not a confirmed physical fault.')
    st.metric('Mapped wells with valid coordinates',f'{len(geo):,}')
    st.metric('Wells in real weekly history',f'{weekly.uwi.nunique():,}')
    st.metric('Recorded score rows for holdout',f'{len(score):,}')
    for name,report in reports.items():
        with st.expander(name,expanded=name=='private_real_history_audit.json'):
            st.json(report)
    st.warning('Pump efficiency units, inconsistent well-days, and changes in failure reporting require domain-owner verification. No missing values are replaced with invented measurements.')

elif page=='AI Well Investigator':
    st.subheader('Evidence-based well investigator — genuine recorded history')
    st.caption('This is an auditable rule-based evidence assistant; it does not invent figures or provide autonomous equipment diagnosis.')
    wells=sorted(view.uwi.astype(str).unique())
    idx=wells.index(st.session_state.chosen_well) if st.session_state.chosen_well in wells else 0
    st.session_state.chosen_well=st.selectbox('Investigate source well',wells,index=idx)
    well=st.session_state.chosen_well
    hist=weekly[weekly.uwi.eq(well)&weekly.week_end.le(pd.Timestamp(cutoff))].sort_values('week_end')
    row=hist.iloc[-1].to_dict()
    matching=score[(score.uwi.eq(well))&score.week_end.le(pd.Timestamp(cutoff))] if len(score) else pd.DataFrame()
    if not matching.empty:
        # Past-only join: no score from a later prediction date.
        matching=matching.sort_values('week_end')
        row.update({k:v for k,v in matching.iloc[-1].dropna().to_dict().items() if k != "week_end"})
        if matching.iloc[-1].week_end!=hist.iloc[-1].week_end:
            st.info('Latest model score is from a different earlier week than the current observation. It is NOT a current or calibrated failure probability.')
    else:row['relative_risk_score']=np.nan
    question=st.selectbox('Ask about the selected well',QUESTIONS)
    if st.button('Investigate selected source records',type='primary'):
        result=investigate(question,row,hist,as_of=cutoff)
        st.markdown('#### '+result['title'])
        for line in result['answer']:st.write('• '+line)
        with st.expander('Evidence ledger — exact source fields',expanded=True):
            for fact in result['evidence']:st.caption(fact)
        st.warning(' '.join(result['limitations']))
    st.markdown('#### Actual historical measurements')
    metric=st.selectbox('Show source metric',list(SENSOR_METRICS),format_func=lambda x:x[1],key='ai_metric')
    plotted_line(hist,metric[0],f'{metric[1]} through {cutoff}')
    real_summary_table(hist.tail(30),'ai_history',['uwi','week_end',metric[0],'oil_daily_avg_2w','hours_down','lost_per_week_2w'])

st.divider()
st.caption('PRIVATE · DATA TITANS · Real source records only · No fabricated data · Competition use subject to Strathcona permission terms')
