"""Third WellGuard page: interactive Failure Intelligence & Recorded Causes.

The design follows the approved enterprise concept but does NOT copy any illustrative
counts, ranking, reasons, warning dates or sensor curves from the image.
"""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .failure_intelligence import (
    PERIODS, scoped_failures, analysis_statistics, annual_counts, pareto_types,
    recorded_causes, event_history, pre_failure_observations, early_warning_check,
)

CYAN='#2bdbda'; AMBER='#f6c262'; PURPLE='#8b76f6'; INK='#0a1c28'
MEASUREMENTS = {
    'Oil rate (2-week avg)': ('oil_daily_avg_2w', 'Oil production (m³/day)'),
    'Pump RPM (2-week avg)': ('rpm_avg_2w', 'Pump speed (RPM)'),
    'Rod torque (2-week avg)': ('torque_avg_2w', 'Rod torque (source unit)'),
    'Motor current (2-week avg)': ('current_avg_2w', 'Motor current (A)'),
    'Pump efficiency (source unit)': ('eff_avg_2w', 'Pump efficiency (source unit)'),
}

def esc(x): return html.escape(str(x), quote=True)

def date_str(value):
    if value is None or pd.isna(value): return '—'
    return pd.Timestamp(value).strftime('%Y-%m-%d')

def pct(value): return '—' if value is None or pd.isna(value) else f'{100*float(value):.1f}%'

def _card(label, value, foot, icon):
    st.markdown(f'''<div class="wg-failure-kpi"><div class="wg-failure-kpi-heading"><span class="wg-failure-icon">{icon}</span><b>{esc(label)}</b></div>
       <div class="wg-failure-kpi-value">{esc(value)}</div><div class="wg-failure-kpi-foot">{esc(foot)}</div></div>''', unsafe_allow_html=True)

def _chart_style(fig, *, height=225, margin=None):
    fig.update_layout(height=height, template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#d0e5ed',size=11),
                      margin=margin or dict(t=18,l=25,r=15,b=35),
                      legend=dict(orientation='h',yanchor='bottom',y=1.0,x=0,font=dict(size=10)),
                      hoverlabel=dict(bgcolor='#112938',font_size=12))
    fig.update_xaxes(gridcolor='#254353',zeroline=False,showline=False)
    fig.update_yaxes(gridcolor='#254353',zeroline=False,showline=False)
    return fig

def _year_plot(counts):
    fig=go.Figure(go.Bar(x=counts.year.astype(str),y=counts.events,text=counts.events,
                         textposition='outside',marker_color=PURPLE,
                         hovertemplate='Year %{x}<br>%{y:,} recorded events<extra></extra>'))
    fig.update_layout(showlegend=False)
    return _chart_style(fig,height=224)

def _pareto_plot(types):
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    fig.add_trace(go.Bar(x=types.failure_type,y=types['count'],name='Recorded events',
                         marker_color=CYAN,text=types['count'],textposition='outside',
                         hovertemplate='%{x}<br>%{y:,} records<extra></extra>'),secondary_y=False)
    fig.add_trace(go.Scatter(x=types.failure_type,y=types.cumulative_pct,
                             name='Cumulative share',mode='lines+markers+text',
                             line=dict(color=AMBER,width=2),marker=dict(size=6),
                             text=[f'{v:.1f}%' for v in types.cumulative_pct],
                             textposition='top center',
                             hovertemplate='%{x}<br>%{y:.1f}% cumulative<extra></extra>'),secondary_y=True)
    fig.update_yaxes(title_text='Event count',secondary_y=False)
    fig.update_yaxes(title_text='Cumulative %',range=[0,110],secondary_y=True,ticksuffix='%')
    fig.update_layout(barmode='group')
    return _chart_style(fig,height=224,margin=dict(t=25,l=25,r=27,b=45))

def _cause_plot(causes):
    fig=go.Figure(go.Pie(labels=causes.cause,values=causes['count'],hole=.65,
                         textinfo='none',marker=dict(colors=[CYAN,'#8f78ed','#558de7','#f2a65a','#8896a4','#386979']),
                         hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>'))
    fig.update_layout(showlegend=True,legend=dict(orientation='v',x=1.04,y=.9,font=dict(size=10)))
    return _chart_style(fig,height=224,margin=dict(t=4,l=8,r=25,b=10))

def _pre_failure_plot(history, event_date, field, label):
    if history.empty or field not in history.columns:
        return None
    x=history.copy()
    x['week_end']=pd.to_datetime(x.week_end)
    x['days_before_failure']=(x.week_end-pd.Timestamp(event_date).normalize()).dt.days
    measured=pd.to_numeric(x[field],errors='coerce')
    down=100*pd.to_numeric(x.get('down_fraction_2w',pd.Series(np.nan,index=x.index)),errors='coerce')
    if measured.notna().sum()==0 and down.notna().sum()==0:
        return None
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    if measured.notna().any():
        fig.add_trace(go.Scatter(x=x.days_before_failure,y=measured,mode='lines+markers',
                                 name=label,line=dict(color=CYAN,width=2.8),marker=dict(size=6),
                                 customdata=x.week_end.dt.strftime('%Y-%m-%d'),
                                 hovertemplate='Snapshot %{customdata}<br>%{y:.2f}<extra></extra>'),secondary_y=False)
    if down.notna().any():
        fig.add_trace(go.Scatter(x=x.days_before_failure,y=down,mode='lines+markers',
                                 name='Days with downtime (%)',line=dict(color=PURPLE,width=1.8),
                                 fill='tozeroy',fillcolor='rgba(140,118,246,.17)',
                                 hovertemplate='%{x} days before event<br>%{y:.1f}%<extra></extra>'),secondary_y=True)
    fig.add_vline(x=0,line=dict(color='#f27272',dash='dash',width=2))
    fig.update_xaxes(range=[-30,.5],title_text='Days before failure')
    fig.update_yaxes(title_text=label,secondary_y=False)
    fig.update_yaxes(title_text='Days with downtime (%)',secondary_y=True,range=[0,100],ticksuffix='%')
    return _chart_style(fig,height=225,margin=dict(t=25,l=28,r=35,b=40))


def render_failure_intelligence(*, failures:pd.DataFrame, weekly:pd.DataFrame, scores:pd.DataFrame,
                                cutoff, permitted_wells=None, area_lookup=None):
    """Render page 3 from actual private source records and as-of score history."""
    st.markdown('''<div class="wg-failure-heading"><div class="wg-failure-heading-icon">⚠</div>
    <div><div class="wg-failure-title">Failure Intelligence &amp; Recorded Causes</div>
    <div class="wg-failure-sub">Explore recorded equipment failures, recurring events, and operating conditions preceding failure.</div></div></div>''',unsafe_allow_html=True)

    # Reserve the KPI row ahead of the period controls, matching the approved design.
    # Widget state still determines the numbers: nothing is rendered until the
    # selected period has been applied to the source event cohort.
    kpi_spot = st.container()
    # Period selection is the PRIMARY filter for charts, KPIs and event table.
    pleft, pright = st.columns([1.03, 1.18], gap='small')
    with pleft:
        st.markdown('<div class="wg-failure-section-cap">ANALYSIS PERIOD · Applies to every chart, KPI and event table</div>',unsafe_allow_html=True)
        period=st.radio('Analysis period',PERIODS, index=2, horizontal=True,
                        label_visibility='collapsed',key='failure_analysis_period')
    with pright:
        st.markdown('''<div class="wg-failure-reporting"><b>ⓘ &nbsp; Data reporting context</b><div>
        Recorded equipment failures are very sparse in 2021–2024 despite production activity.
        This may reflect differences in reporting coverage—not proof of failure-free operation.
        <strong>2025–2026 is recommended</strong> for failure-pattern analysis; earlier data remain available for exploration.</div></div>''',unsafe_allow_html=True)

    f = scoped_failures(failures,cutoff=cutoff,period=period,permitted_wells=permitted_wells)
    stats = analysis_statistics(f)
    with kpi_spot:
        labels = st.columns(4,gap='small')
        with labels[0]:_card('Recorded Failures',f"{stats['events']:,}",'Events in selected period','⚙')
        with labels[1]:_card('Affected Wells',f"{stats['wells']:,}",'Unique wells with ≥1 recorded event','♧')
        with labels[2]:_card('Rod-Related Failures',pct(stats['rod_share']),
                             f"{stats['rod_events']:,} recorded events",'◇')
        with labels[3]:_card('Repeat-Failure Wells',f"{stats['repeat_wells']:,}",
                             f"{pct(stats['repeat_well_share'])} of affected wells · ≥2 distinct dates",'⟳')

    period_short = period.replace(' (Recommended)','')
    a,b,c=st.columns([.92,1.12,1.1],gap='small')
    with a:
        st.markdown(f'<div class="wg-failure-panel-title">▥ &nbsp; Recorded Failures by Year · {esc(period_short)}</div>',unsafe_allow_html=True)
        counts=annual_counts(f,period,cutoff)
        if counts.empty: st.info('No recorded years within the selected cutoff.')
        else: st.plotly_chart(_year_plot(counts),use_container_width=True,key='fail_year_chart')
    with b:
        st.markdown('<div class="wg-failure-panel-title">◈ &nbsp; Failure Types · Pareto Analysis</div>',unsafe_allow_html=True)
        types=pareto_types(f,max_types=4)
        if types.empty: st.info('No recorded failure types in this period.')
        else: st.plotly_chart(_pareto_plot(types),use_container_width=True,key='fail_pareto_chart')
    with c:
        st.markdown('<div class="wg-failure-panel-title">▤ &nbsp; Recorded Failure Causes</div>',unsafe_allow_html=True)
        causes=recorded_causes(f,max_causes=5)
        if causes.empty: st.info('No recorded failure causes in this period.')
        else:
            st.plotly_chart(_cause_plot(causes),use_container_width=True,key='fail_causes_chart')
            with st.expander('Cause counts and exact shares'):
                st.dataframe(causes.rename(columns={'cause':'Recorded cause','count':'Events','share_pct':'Share (%)'}).round(1),
                             hide_index=True,use_container_width=True)

    left,center,right=st.columns([.74,1.68,1.04],gap='small')
    with left:
        st.markdown('<div class="wg-failure-panel-title">⟳ &nbsp; Repeat-Failure Analysis</div>',unsafe_allow_html=True)
        st.markdown(f'''<div class="wg-repeat-box"><span>Wells with ≥2 failure dates</span><strong>{stats['repeat_wells']:,}</strong>
          <small>{esc(pct(stats['repeat_well_share']))} of affected wells</small></div>
          <div class="wg-repeat-box"><span>Additional recorded failure dates</span><strong>{stats['additional_event_dates']:,}</strong><small>After each well's first recorded date</small></div>
          <div class="wg-repeat-box"><span>Median interval between repeat dates</span><strong>{stats['median_repeat_gap_days']:.0f} days</strong><small>Observed intervals only · no causality inferred</small></div>'''
          if stats['median_repeat_gap_days'] is not None else
          f'''<div class="wg-repeat-box"><span>Wells with ≥2 failure dates</span><strong>{stats['repeat_wells']:,}</strong></div>
          <div class="wg-repeat-box"><span>Additional recorded failure dates</span><strong>{stats['additional_event_dates']:,}</strong></div>
          <div class="wg-repeat-box"><span>Median interval</span><strong>—</strong></div>''',unsafe_allow_html=True)

    with center:
        st.markdown('<div class="wg-failure-panel-title">⌁ &nbsp; Pre-Failure Operating Investigation</div>',unsafe_allow_html=True)
        if f.empty:
            st.info('No recorded events under these filters.')
            selected=None
        else:
            rich=event_history(f).reset_index(drop=True)
            options=rich._event_key.astype(str).tolist()
            # A table-row selection is stored as an intent and applied *before*
            # creating the selectbox, avoiding Streamlit widget-state errors.
            pending=st.session_state.pop('wg_failure_pending_event',None)
            if pending in options:
                st.session_state.wg_failure_selected_key=pending
            if st.session_state.get('wg_failure_selected_key') not in options:
                st.session_state.wg_failure_selected_key=options[0]
            lookup=rich.set_index('_event_key',drop=False)
            def event_label(key):
                r=lookup.loc[key]
                return f"{date_str(r.failure_start_date)} · {r.uwi} · {r.get('failure_type','Unspecified')}"
            choice=st.selectbox('Select a recorded failure event',options,format_func=event_label,
                                key='wg_failure_selected_key')
            selected=lookup.loc[choice]
            st.markdown(f'''<div class="wg-failure-event-box"><b>Well (UWI): {esc(selected.uwi)}</b>
               <span>Failure: {esc(selected.get('failure_type','Not recorded'))} &nbsp;·&nbsp; Recorded cause: {esc(selected.get('cause','Not recorded'))}</span>
               <span>Event: {esc(date_str(selected.failure_start_date))}</span></div>''',unsafe_allow_html=True)
            metric=st.selectbox('Operating measurement',list(MEASUREMENTS),key='wg_failure_metric')
            field,label=MEASUREMENTS[metric]
            history=pre_failure_observations(weekly,str(selected.uwi),selected.failure_start_date,30)
            plot=_pre_failure_plot(history,selected.failure_start_date,field,label)
            if plot is None:
                st.info('No valid weekly operating measurements within the 30 days preceding this event.')
            else:st.plotly_chart(plot,use_container_width=True,key='fail_prior_telemetry_chart')
            st.caption('Historical weekly snapshots use two-week aggregated sensor values. Only snapshots before the recorded event are plotted; these patterns are not a diagnosed cause.')

    with right:
        st.markdown('<div class="wg-failure-panel-title">◎ &nbsp; Model Early-Warning Check</div>',unsafe_allow_html=True)
        if selected is None:
            st.info('Select a recorded failure to inspect its earlier model rankings.')
        else:
            cap = st.session_state.get('wg_capacity_shared',25)
            if cap not in (25,50,100):cap=25
            st.caption(f'Well (UWI): **{selected.uwi}** · {date_str(selected.failure_start_date)}')
            result=early_warning_check(scores,str(selected.uwi),selected.failure_start_date,
                                       capacity=cap,horizon_days=14)
            state={'FLAGGED':'Flagged before failure','NOT_SELECTED':'Not selected in top reviews',
                   'NO_SCORE':'No eligible model score'}[result['status']]
            st.markdown(f'<div class="wg-early-state {"wg-flagged" if result["selected_for_review"] else ""}">{esc(state)}</div>',unsafe_allow_html=True)
            if result['status']!='NO_SCORE':
                r1,r2=st.columns(2,gap='small')
                with r1:st.metric('Historical model rank',f"#{result['rank']} / {result['eligible_count']:,}")
                with r2:st.metric('Scoring week',date_str(result['scored_week']))
                r3,r4=st.columns(2,gap='small')
                with r3:st.metric('First eligible alert',date_str(result['first_alert_week']))
                with r4:st.metric('Advance lead time',f"{result['lead_days']} days" if result['lead_days'] is not None else '—')
                st.caption(f"Top {cap} selected: **{'Yes' if result['selected_for_review'] else 'No'}** · "
                           f"Relative model score: {result['relative_risk_score']:.4f} · "
                           f"Source: {result['score_origin'].replace('_',' ').title()}.")
                if result['score_origin'] and 'POST_HOLDOUT' in result['score_origin']:
                    st.info('Post-holdout historical score; not part of the independently evaluated holdout recall.')
            else:
                st.info('No eligible historical score for this well within 1–14 days before the event. '
                        'This is **not** evidence that the well was low-risk.')
            st.caption('A retrospective top-ranked selection is not a confirmed advance diagnosis or a prevented failure. '
                       'Warning lead time is bounded to the 14-day research horizon.')

    st.markdown(f'<div class="wg-failure-panel-title wg-failure-table-head">▤ &nbsp; Recorded Failure Events · {esc(period_short)}</div>',unsafe_allow_html=True)
    if f.empty:
        st.info('No events found for this historical period and the current filters.')
        return
    records=event_history(f).reset_index(drop=True)
    search=st.text_input('Search recorded events by well, equipment type, cause, or area',key='failure_search',placeholder='Search well ID, type, cause…')
    if search:
        mask=records[[c for c in ('uwi','failure_type','cause') if c in records]].astype(str).apply(
            lambda col: col.str.contains(search,case=False,regex=False)).any(axis=1)
        records=records.loc[mask].copy()
    display_cols=[c for c in ('uwi','failure_start_date','failure_type','cause','days_since_previous_failure') if c in records]
    shown=records[display_cols].copy()
    shown=shown.rename(columns={'uwi':'Well (UWI)','failure_start_date':'Event date',
                                'failure_type':'Failure type','cause':'Recorded cause',
                                'days_since_previous_failure':'Days since previous failure'})
    shown['Operating area']=records.uwi.map(area_lookup or {}).fillna('Unknown')
    if 'Event date' in shown:shown['Event date']=pd.to_datetime(shown['Event date']).dt.strftime('%Y-%m-%d')
    shown.index=range(1,len(shown)+1)
    st.caption(f'{len(shown):,} recorded source events match the selection. Click a row to update both investigations. '
               'Data shown here are restricted to authorized competition use.')
    # The generated event_key is never published or treated as a feature.
    event_table=st.dataframe(shown,hide_index=False,use_container_width=True,height=290,
                             on_select='rerun',selection_mode='single-row',key='wg_failure_log_table')
    try:
        chosen_rows=event_table.selection.rows
    except (AttributeError,KeyError):chosen_rows=[]
    if chosen_rows:
        idx=chosen_rows[0]
        if 0<=idx<len(records):
            chosen_event=str(records.iloc[idx]._event_key)
            signature=(period, str(cutoff),chosen_event)
            if st.session_state.get('wg_last_failure_table_choice')!=signature:
                st.session_state.wg_last_failure_table_choice=signature
                if st.session_state.get('wg_failure_selected_key')!=chosen_event:
                    st.session_state.wg_failure_pending_event=chosen_event
                    st.rerun()
    # Raw private export preserves underlying recorded values, including the event date.
    export_cols=[c for c in ('uwi','failure_start_date','failure_type','cause','equipment_type') if c in records]
    st.download_button('Export PRIVATE filtered failure records',records[export_cols].to_csv(index=False).encode('utf-8'),
                       file_name='PRIVATE_WELLGUARD_failure_intelligence.csv',mime='text/csv',key='failure_private_export')
    st.caption('Recorded causes are source labels, not independently validated root causes. '
               'Repeated events are counted by distinct well and event date. Historical model scores are uncalibrated.')
