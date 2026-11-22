"""Render first page in user-approved WellGuard Operations Command Center layout.

The screenshot was a STYLE reference only. Every displayed number comes from real
historical data or a separately identified aggregate validation report.
"""
from __future__ import annotations
import html
from urllib.parse import quote
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .command_center import build_command_center
from .navigation import queue_navigation

CYAN='#24d3d5'; AMBER='#f3be66'; PALE='#b9d1df'; DARK='#091a26'

def esc(value):
    return html.escape(str(value), quote=True)

def number(value, decimals=0, unit=''):
    if value is None or not np.isfinite(float(value)):
        return '—'
    return f'{float(value):,.{decimals}f}{unit}'

def kpi(label, value, icon, foot='', amber=False):
    style=' wg-kpi-amber' if amber else ''
    st.markdown(f'''<div class="wg-kpi{style}">
      <div class="wg-kpi-top"><span class="wg-kpi-icon">{esc(icon)}</span>{esc(label)}</div>
      <div class="wg-kpi-value">{esc(value)}</div>
      <div class="wg-kpi-foot">{esc(foot)}</div></div>''', unsafe_allow_html=True)

def _chart_layout(fig, height=240, l=10, r=25):
    fig.update_layout(template='plotly_dark',height=height,margin=dict(l=l,r=r,t=7,b=37),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif',color='#b6d1db',size=11),
        xaxis=dict(gridcolor='#243e4d',zeroline=False),
        yaxis=dict(gridcolor='#243e4d',zeroline=False),
        hoverlabel=dict(bgcolor='#163343',font_size=12),
        legend=dict(orientation='h',y=1.18,x=0.0,font=dict(size=10)))
    return fig

def _area_chart(frame, scored):
    fig=go.Figure()
    if not frame.empty:
        f=frame.iloc[::-1]
        fig.add_trace(go.Bar(x=f.value,y=f.area,orientation='h',
            marker=dict(color=CYAN,line=dict(width=0)),
            text=[str(int(x)) if scored else number(x,0) for x in f.value],
            textposition='outside',textfont=dict(color='#c6eff4',size=10),
            customdata=f[['area']].to_numpy(),
            hovertemplate='%{y}: %{x:,.1f}' + (' priority reviews' if scored else ' m³ estimated lost oil')+'<extra></extra>'))
    _chart_layout(fig, height=238,l=18,r=32)
    fig.update_layout(showlegend=False,yaxis=dict(title='',automargin=True,categoryorder='array',categoryarray=(frame.iloc[::-1].area.tolist() if not frame.empty else [])),xaxis_title='Count of selected wells' if scored else 'Estimated lost oil (m³)')
    return fig

def _production_chart(trend):
    fig=go.Figure()
    if not trend.empty:
        w=trend.sort_values('week_end')
        fig.add_trace(go.Bar(x=w.week_end,y=w.lost,name='Estimated loss (m³)',
            marker_color=AMBER,opacity=.84,yaxis='y2',hovertemplate='%{x|%d %b %Y}<br>Estimated lost oil %{y:,.1f} m³<extra></extra>'))
        fig.add_trace(go.Scatter(x=w.week_end,y=w.oil,mode='lines+markers',
            name='Oil production (m³)',line=dict(color=CYAN,width=3),marker=dict(size=5),
            fill='tozeroy',fillcolor='rgba(36,211,213,0.07)',
            hovertemplate='%{x|%d %b %Y}<br>Oil produced %{y:,.1f} m³<extra></extra>'))
    _chart_layout(fig,height=238,l=15,r=55)
    fig.update_layout(yaxis=dict(title='Oil produced (m³)',rangemode='tozero',gridcolor='#294454'),
       yaxis2=dict(title='Lost oil (m³)',overlaying='y',side='right',rangemode='tozero',showgrid=False),
       xaxis=dict(tickformat='%b %d',gridcolor='#263e4a'),
       barmode='overlay',legend=dict(orientation='h',x=0.05,y=1.15))
    return fig

def _table_html(queue, scored):
    if queue.empty:
        return '<div class="wg-muted-msg">No real recorded wells are available to list for this week and filter.</div>'
    rows=[]
    for _,v in queue.head(10).iterrows():
        well=str(v.get('uwi',''))
        label=str(v.get('priority_tier',''))
        style='high' if label=='High review' else 'loss' if label=='Loss-only' else 'review' if label=='Review' else ''
        link=f'?wg_page=Equipment+Health&wg_well={quote(well,safe="")}'
        reason=esc(v.get('reason','Recorded history available'))
        rows.append(f'''<tr><td>{int(v['review_rank'])}</td>
          <td><a class="wg-well-link" href="{link}" target="_self">{esc(well)}</a></td>
          <td title="{esc(v.get('area','Unknown'))}">{esc(v.get('area','Unknown'))}</td>
          <td><span class="wg-tier {style}">{esc(label)}</span></td>
          <td title="{reason}">{reason}</td>
          <td><a class="wg-table-action" href="{link}" target="_self">Review evidence →</a></td></tr>''')
    return '''<div class="wg-priority-scroller"><table class="wg-priority-table"><colgroup>
       <col style="width:5%"><col style="width:20%"><col style="width:16%"><col style="width:14%"><col style="width:29%"><col style="width:16%"></colgroup>
       <thead><tr><th>#</th><th>Well</th><th>Area</th><th>Review tier</th><th>Observed evidence</th><th>Next step</th></tr></thead>
       <tbody>'''+''.join(rows)+'''</tbody></table></div>'''

def render_command_center(weekly, view, geo, score_week, cutoff, model_report, source_failures, *, capacity=25):
    d=build_command_center(weekly,view,geo,score_week,cutoff,model_report=model_report,capacity=capacity)
    # Reference image carries hypothetical figures; those are NEVER copied.
    st.markdown('''<div class="wg-command-header"><div class="wg-crumbline">
       <div class="wg-crumb">OPERATIONS INTELLIGENCE &nbsp;/&nbsp; <b>EXECUTIVE OVERVIEW</b></div>
       <div class="wg-source-tag">● &nbsp; REAL HISTORICAL WELL DATA · LOCAL</div></div>
       <div class="wg-title-row"><div class="wg-pump-mark">◈</div><div>
       <div class="wg-title-text">Operations Command Center</div>
       <div class="wg-title-sub">Executive overview for inspection prioritization using real historical well data.</div>
       </div></div></div>''',unsafe_allow_html=True)
    cols=st.columns(4,gap='small')
    with cols[0]:kpi(d['eligible_label'],f"{d['eligible_count']:,}",'♜','Actual observations at selected cutoff')
    with cols[1]:
        v=f"{d['selected_reviews']:,}" if d['has_scores'] else '—'
        kpi(f'Top {capacity} review candidates' if d['has_scores'] else 'Model alerts not yet available',v,'⚠',
            f'{capacity} reviews/week is an illustrative capacity' if d['has_scores'] else 'First launch trains locally; reload when complete',amber=True)
    with cols[2]:kpi('Estimated weekly lost oil',number(d['weekly_lost'],1,' m³'),'▣','Selected real well-week records')
    with cols[3]:kpi('Recorded weekly downtime',number(d['weekly_downtime'],0,' h'),'◷','Selected real well-week records')
    benchmark=d['benchmark']
    coverage=benchmark.get(f'model_top{capacity}_event_coverage') or {}
    recall=coverage.get('event_recall')
    events=coverage.get('caught_events'); eligible_events=coverage.get('eligible_events')
    recency_baseline_events=(benchmark.get(f'recency_top{capacity}_event_coverage') or {}).get('caught_events')
    first_area = str(d['area_rank'].area.iloc[0]) if not d['area_rank'].empty else '—'
    area_title='Most priority reviews' if d['has_scores'] else 'Highest estimated lost oil area'
    age=d['median_freshness_days']
    freshness=f"{age:,.0f} days" if age is not None else '—'
    rec_val=f'{recall*100:.1f}%' if isinstance(recall,(float,int)) else '—'
    event_note=(f'{events} / {eligible_events} eligible events flagged'
                if events is not None and eligible_events else 'Validation not available')
    if recency_baseline_events is not None and events is not None:
        event_note+=f' · failure-recency rule: {recency_baseline_events}'
    # Display the review-capacity dropdown immediately beside the historical
    # event-recall metric. The dropdown is bound to Streamlit session state;
    # its updated value is available at the start of the next script rerun.
    insights=st.columns(3,gap='small')
    with insights[0]:
        st.markdown(f'''<div class="wg-insight-panel">
          <div class="wg-insight-label">⌖ &nbsp; {esc(area_title)}</div>
          <div class="wg-insight-value">{esc(first_area)}</div>
          <div class="wg-insight-badge">{esc(d['rank_detail'])}</div></div>''',unsafe_allow_html=True)
    with insights[1]:
        st.markdown(f'''<div class="wg-insight-panel">
          <div class="wg-insight-label">▤ &nbsp; Median record age</div>
          <div class="wg-insight-value">{esc(freshness)}</div>
          <div class="wg-insight-badge">{d['current_week_coverage']:,} of {d['total_visible_wells']:,} wells have records ≤7 days old</div></div>''',unsafe_allow_html=True)
    with insights[2]:
        left_metric, right_picker = st.columns([1.55,1],gap='small',vertical_alignment='center')
        with left_metric:
            st.markdown(f'''<div class="wg-insight-panel wg-recall-panel">
              <div class="wg-insight-label">◎ &nbsp; Historical event recall</div>
              <div class="wg-insight-value">{esc(rec_val)}</div>
              <div class="wg-insight-badge">{esc(event_note)}<br>Fleet-wide historical test · not current-week or area-filtered performance</div></div>''',unsafe_allow_html=True)
        with right_picker:
            def keep_review_capacity():
                st.session_state.wg_capacity_shared=st.session_state.command_capacity
                st.session_state.fleet_review_capacity=st.session_state.command_capacity
            st.selectbox('Reviews / week', [25,50,100], key='command_capacity',
                on_change=keep_review_capacity,
                help='Illustrative inspection-review capacity, not Strathcona staffing. '
                     'Replays the same historical June–October 2026 test at 25, 50 or 100 reviews/week.')
    if d['selected_partial']:
        st.warning(f'Partial reporting week ending {pd.Timestamp(cutoff):%Y-%m-%d}: a maximum of {d["source_days_in_selected_week"]} of 7 calendar days is represented. The KPI totals are partial; the trend chart excludes incomplete weeks to avoid a false production collapse.')
    if d['has_scores'] and 'score_origin' in score_week:
        origins=score_week.score_origin.dropna().astype(str).unique()
        if any('POST_HOLDOUT' in x for x in origins):
            st.caption('Latest available as-of rankings, not evaluated prospective predictions. Historical recall percentages refer to the separate June–October 2026 test period.')
    if not d['has_scores']:
        st.info('No trained risk scores exist for this exact selected week. The area chart and review-candidate table are based on observed estimated lost oil only—not predictive failure alerts. Run `run_historical_model_windows.bat` and choose a scored historical week to see the true model-ranked inspection queue.')
    st.markdown('<div class="wg-private-notice"><b>Confidential source-data view:</b> Real well identifiers and records are displayed locally. Do not publish screenshots or individual-well exports without organizer permission. Daily/weekly figures use recorded values for the selected period; historical validation is separate.</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap='small')
    with c1:
        st.markdown(f'<div class="wg-section-head"><span>▥</span>{esc(d["rank_label"])}</div>',unsafe_allow_html=True)
        st.markdown('<div class="wg-chart-hint">Select a bar to filter the table below; select it again to clear.</div>',unsafe_allow_html=True)
        selection=st.plotly_chart(_area_chart(d['area_rank'],d['has_scores']),use_container_width=True,
                key='command_area_chart',on_select='rerun',selection_mode='points')
    with c2:
        st.markdown('<div class="wg-section-head"><span>⌁</span>Production vs Estimated Loss Trend</div>',unsafe_allow_html=True)
        st.markdown('<div class="wg-chart-hint">Actual weekly totals from recorded history; shared weekly intervals.</div>',unsafe_allow_html=True)
        st.plotly_chart(_production_chart(d['trend']),use_container_width=True,key='command_production_chart')
    filtered=d['queue']
    try:
        points=selection.selection.points
        if points:
            area=points[0].get('y')
            if area: filtered=filtered[filtered.area.eq(str(area))]
    except (AttributeError, KeyError, IndexError, TypeError):pass
    left,right=st.columns([1.8,1],gap='small')
    with left:
        title='Top Inspection Priorities' if d['has_scores'] else 'Observed Production-Loss Review Candidates'
        subtitle='Ranked by actual LightGBM scores for this historical week.' if d['has_scores'] else 'Not predictive: ranked by recorded estimated lost oil, until historical model scores are available.'
        st.markdown(f'<div class="wg-priority-wrap"><div class="wg-section-head"><span>♜</span>{esc(title)}</div><div class="wg-panel-subtitle">{esc(subtitle)}</div>{_table_html(filtered,d["has_scores"])}</div>',unsafe_allow_html=True)
        if len(d['queue'])>10:st.caption(f'Showing 10 of {len(d["queue"])} actual selected records; chart filter may narrow the view.')
        with st.expander('View complete private underlying records / export'):
            cols=['review_rank','uwi','area','province','facility','week_end','relative_risk_score','reason','estimated_lost_oil_m3','hours_down']
            cols=[c for c in cols if c in filtered.columns]
            st.dataframe(filtered[cols],hide_index=True,use_container_width=True)
            st.download_button('Export PRIVATE review candidates', filtered[cols].to_csv(index=False).encode('utf-8'),
                               file_name='PRIVATE_wellguard_overview_candidates.csv',mime='text/csv')
    with right:
        st.markdown('''<div class="wg-action-panel"><div class="wg-section-head"><span>ϟ</span>Recommended Next Action</div>
           <div class="wg-action-copy">Select a real well, examine its recorded trends and model drivers, then decide whether field review is justified. No automatic fault diagnosis or claims of avoided losses.</div>
           <div class="wg-action-chain"><span>Open Well Detail</span> › <span>Review Evidence</span> › <span>Decide Inspection</span></div></div>''',unsafe_allow_html=True)
        options=d['queue'].uwi.astype(str).tolist()
        if options:
            selected=st.selectbox('Choose a recorded well',options,key='command_selected_well')
            b1,b2=st.columns(2,gap='small')
            if b1.button('Open Well Detail',key='command_open_health',use_container_width=True):
                queue_navigation(st.session_state, 'Equipment Health', selected)
                st.rerun()
            if b2.button('AI Evidence',key='command_open_ai',use_container_width=True):
                queue_navigation(st.session_state, 'AI Well Investigator', selected)
                st.rerun()
        st.markdown('<div class="wg-section-head" style="margin-top:9px"><span>◷</span>Recent Recorded Failures</div>',unsafe_allow_html=True)
        recent=source_failures.copy()
        if not recent.empty:
            recent=recent[recent.failure_start_date.le(pd.Timestamp(cutoff)) & recent.uwi.astype(str).isin(view.uwi.astype(str))]
            recent=recent.sort_values('failure_start_date',ascending=False)
            query=st.text_input('Search failures',placeholder='Search well, type or date...',key='command_recent_search',label_visibility='collapsed')
            if query:
                valid=recent.astype(str).apply(lambda x:x.str.contains(query,case=False,regex=False)).any(axis=1)
                recent=recent.loc[valid]
            lines=[]
            for _,event in recent.head(5).iterrows():
                u=esc(event.get('uwi',''))
                kind=esc(event.get('failure_type',event.get('equipment_type','Recorded failure')))
                dt=event.get('failure_start_date')
                when=esc(pd.Timestamp(dt).strftime('%Y-%m-%d')) if pd.notna(dt) else '—'
                lines.append(f'<tr><td>{u}</td><td title="{kind}">{kind}</td><td>{when}</td></tr>')
            table='<table class="wg-event-table"><thead><tr><th>Well</th><th>Recorded event</th><th>Date</th></tr></thead><tbody>'+''.join(lines)+'</tbody></table>' if lines else '<div class="wg-muted-msg">No matching historical recorded events.</div>'
            st.markdown(table,unsafe_allow_html=True)
        else:
            st.caption('No source failure events available.')
    st.markdown('<div class="wg-lower-tag">Source: private competition Parquet files and local source-derived model replay. Risk ranking is retrospective and uncalibrated; design-reference values were not imported.</div>',unsafe_allow_html=True)
