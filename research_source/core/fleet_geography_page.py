"""Streamlit presentation of the private WellGuard geographic inspection workspace."""
from __future__ import annotations

import base64
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st
from core.fleet_geography import CAPACITIES, OVERLAYS, prepare_fleet, overlay_figure
from core.navigation import queue_navigation
from core.table_utils import arrow_safe_view


def _esc(x):
    return escape(str(x), quote=True)


def _metric(label:str, value, foot:str, icon='◇'):
    st.markdown(f'<div class="wg-kpi wg-fleet-kpi"><div class="wg-kpi-top"><span class="wg-kpi-icon">{_esc(icon)}</span>{_esc(label)}</div>'
                f'<div class="wg-kpi-value">{_esc(value)}</div><div class="wg-kpi-foot">{_esc(foot)}</div></div>',
                unsafe_allow_html=True)


def _go(page:str,well:str):
    queue_navigation(st.session_state,page,well)
    st.rerun()


def _value(x,unit='',digits=1):
    n=pd.to_numeric(x,errors='coerce')
    if pd.isna(n):return 'Not recorded'
    return f'{n:,.{digits}f}{unit}'


def _source_age(row):
    v=pd.to_numeric(row.get('age_days'),errors='coerce')
    return f'{int(v):,} days since last recorded week' if pd.notna(v) else 'No freshness measurement'


def render_fleet_geography(view:pd.DataFrame, cutoff, *, root:Path):
    """Render Page 2. `view` is source-derived and already filtered for as-of/area/province."""
    # Apply any map facility-click request before the selectbox exists.
    pending=st.session_state.pop('wg_fleet_pending_facility_focus',None)
    if pending:
        st.session_state.fleet_facility_focus=str(pending)
        st.session_state.fleet_facility_pick=str(pending)
    source=view.copy()
    if 'uwi' not in source:
        st.error('Missing real source UWI data');return
    has_any_scores=('relative_risk_score' in source and source.relative_risk_score.notna().any())
    st.markdown('''<div class="wg-crumbline"><div class="wg-crumb">OPERATIONS INTELLIGENCE &nbsp; / &nbsp; <b>FLEET MAP</b></div>
    <div class="wg-source-tag">● REAL HISTORICAL WELL DATA · PRIVATE</div></div>
    <div class="wg-fleet-title"><span class="wg-fleet-pin">⌖</span><div><div class="wg-title-text">Fleet Geography &amp; Inspection Planning</div>
    <div class="wg-title-sub">Where are priority wells located, and how do they relate to operating areas and facilities?</div></div></div>''',
                unsafe_allow_html=True)
    current_cap=st.session_state.get('fleet_review_capacity',st.session_state.get('command_capacity',25))
    if current_cap not in CAPACITIES:current_cap=25
    if 'fleet_review_capacity' not in st.session_state:st.session_state.fleet_review_capacity=current_cap
    def sync_review_capacity():
        st.session_state.wg_capacity_shared=st.session_state.fleet_review_capacity
        st.session_state.command_capacity=st.session_state.fleet_review_capacity
    # Compute initial metrics without display restrictions, to avoid presenting
    # changed denominator as if it represented the entire fleet.
    initial=prepare_fleet(source,capacity=current_cap)
    stats=initial['stats']
    columns=st.columns(4)
    with columns[0]:_metric('Mapped assets',f"{stats['mapped_before_display_filter']:,}",
                            f"Valid locations • {stats['coordinate_coverage_pct']:.1f}% of scoped historical wells",'◈')
    with columns[1]:_metric('Facilities represented',f"{stats['facilities']:,}",
                            'Distinct named facilities with located assets','▥')
    with columns[2]:_metric('Operating areas',f"{stats['areas']:,}",
                            'Geographic areas with mapped wells','⌖')
    with columns[3]:
        if stats['has_scores']:
            val=f"{stats['mapped_priority_count']:,} / {stats['mapped_before_display_filter']:,}"
            label='Mapped review shortlist'
            foot=f"{stats['total_selected_reviews']} source-model selections at {current_cap} slots/week"
        else:val='—';label='Mapped review shortlist';foot='No score file for this historical week'
        _metric(label,val,foot,'◎')

    st.markdown('<div class="wg-fleet-toolbar-label">Map overlay <span>Color wells by recorded operating context</span></div>',unsafe_allow_html=True)
    c_overlay,c_capacity=st.columns([3.45,1.0])
    with c_overlay:
        options=list(OVERLAYS)
        suggested=0 if has_any_scores else 1
        # Never imply a missing model score is a low-risk well.
        overlay=st.radio('Map overlay',options,index=suggested,horizontal=True,
                         label_visibility='collapsed',key='fleet_overlay')
    with c_capacity:
        capacity=st.selectbox('Review capacity (scenario)',CAPACITIES,
                              key='fleet_review_capacity',format_func=lambda n:f'{n} reviews/week',on_change=sync_review_capacity)
    statuses=sorted(initial['mapped'].status.unique())
    facilities=sorted(initial['mapped'].loc[initial['mapped'].facility!='Unknown','facility'].unique())
    c_status,c_facility,c_flag=st.columns([1.8,1.6,1])
    with c_status:
        selected_statuses=st.multiselect('Last-known well status',statuses,default=statuses,key='fleet_statuses')
    with c_facility:
        # Facility focus is separate from the dropdown key and can be changed
        # safely by map clicks or table buttons on subsequent reruns.
        focus=st.session_state.get('fleet_facility_focus','All facilities')
        options=['All facilities']+facilities
        if focus not in options:focus='All facilities';st.session_state.fleet_facility_focus=focus
        if st.session_state.get('fleet_facility_pick') not in options:
            st.session_state['fleet_facility_pick']=focus
        def update_facility_focus():
            st.session_state.fleet_facility_focus=st.session_state.fleet_facility_pick
        picked=st.selectbox('Focus facility',options,key='fleet_facility_pick',on_change=update_facility_focus)
        focus=picked
    with c_flag:
        show_facilities=st.checkbox('Show facility groups',value=True,key='fleet_facility_markers')
    res=prepare_fleet(source,capacity=capacity,statuses=selected_statuses,facility_focus=focus)
    data=res['mapped'];sub=res['stats']
    if not has_any_scores:
        st.info('No model rankings for this exact selected historical week. The map still shows real wells, locations and observations, but it will not invent inspection priorities. Run the historical model and select a scored week.')
    if focus!='All facilities':
        st.caption(f'Facility focus: **{focus}**. Priority selections remain based on the original eligible fleet scope, not recomputed from this facility.')
    if sub['mapped_count']<sub['mapped_before_display_filter']:
        st.caption(f"Map displays {sub['mapped_count']:,} of {sub['mapped_before_display_filter']:,} geolocated wells after status/facility filters.")

    c_map,c_detail=st.columns([2.13,1.05],gap='small')
    with c_map:
        st.markdown('<div class="wg-fleet-section-title">◈ &nbsp; GEOGRAPHIC INSPECTION MAP</div>',unsafe_allow_html=True)
        bg_path=root/'assets'/'western_canada_base.png'
        bg='data:image/png;base64,'+base64.b64encode(bg_path.read_bytes()).decode('ascii') if bg_path.exists() else None
        fig=overlay_figure(data,overlay,base64_map=bg)
        if show_facilities and len(res['facilities']):
            # Facility markers use real centroids of mapped source wells;
            # they do not imply that the facility itself has a risk score.
            import plotly.graph_objects as go
            groups=res['facilities']
            if has_any_scores: groups=groups[groups.priority_wells.ge(2)]
            else: groups=groups[groups.mapped_wells.ge(4)]
            if len(groups):
                fig.add_trace(go.Scatter(x=groups.lon,y=groups.lat,mode='markers',name='Facility groups',
                    customdata=groups[['facility','area','priority_wells','mapped_wells']].fillna('').astype(str).values,
                    marker=dict(symbol='circle-open',size=18,color='#46f3df',line=dict(width=2,color='#46f3df')),
                    hovertemplate='Facility: %{customdata[0]}<br>Area: %{customdata[1]}<br>Selected: %{customdata[2]}<br>Mapped: %{customdata[3]}<extra></extra>'))
        selected=st.plotly_chart(fig,use_container_width=True,on_select='rerun',selection_mode='points',key='wg_geo_map')
        try:
            point=selected.selection.points[0]
            # Facility marker uses a different customdata signature; well
            # IDs are validated before updating the shared well selection.
            well=str(point.get('customdata',[None])[0])
            selection_token=(str(cutoff),overlay,well)
            if st.session_state.get('wg_fleet_map_last_selection')!=selection_token:
                st.session_state.wg_fleet_map_last_selection=selection_token
                if well in set(data.uwi):
                    st.session_state.chosen_well=well
                elif well in set(res['facilities'].facility.astype(str)):
                    # A facility marker contains its name in customdata[0].
                    # Apply focus in the next rerun before the dropdown exists.
                    st.session_state.wg_fleet_pending_facility_focus=well
                    st.rerun()
        except (AttributeError,KeyError,IndexError,TypeError):pass
        st.markdown('<div class="wg-fleet-legend">● <b>Highlighted:</b> selected model review wells, if scores exist. &nbsp; ◯ Named facility groups use mapped-well centroids. &nbsp; Zoom and pan to explore.</div>',unsafe_allow_html=True)
    with c_detail:
        st.markdown('<div class="wg-fleet-section-title">◇ &nbsp; SELECTED WELL</div>',unsafe_allow_html=True)
        options=sorted(data.uwi.astype(str))
        if not options:
            st.warning('No recorded wells match the current map filters.')
        else:
            chosen=st.session_state.get('chosen_well','')
            if chosen not in options:chosen=options[0]
            # Set selector before widget creation so marker click and dropdown
            # selection stay synchronized without Streamlit's session error.
            st.session_state['wg_fleet_well_pick']=chosen
            def handle_well_pick():
                st.session_state.chosen_well=st.session_state.wg_fleet_well_pick
            picked_well=st.selectbox('Recorded well identifier',options,key='wg_fleet_well_pick',on_change=handle_well_pick)
            row=data.loc[data.uwi.eq(picked_well)].iloc[0]
            risk_val=row.get('relative_risk_score')
            priority='Selected for review' if bool(row.priority_flag) else ('Scored — outside shortlist' if pd.notna(risk_val) else 'Not scored')
            age=pd.to_numeric(row.get('age_days'),errors='coerce')
            freshness='Stale data' if pd.notna(age) and age>90 else 'Observed history'
            st.markdown(f'<div class="wg-fleet-well-header"><div class="wg-fleet-well-id">{_esc(picked_well)}</div>'
                        f'<div class="wg-fleet-well-meta">{_esc(row.area)} &nbsp; • &nbsp; {_esc(row.province)} &nbsp; • &nbsp; {_esc(row.facility)}</div>'
                        f'<div class="wg-fleet-well-tier">{_esc(priority)}</div></div>',unsafe_allow_html=True)
            def record_line(a,b):
                st.markdown(f'<div class="wg-fleet-record"><span>{_esc(a)}</span><strong>{_esc(b)}</strong></div>',unsafe_allow_html=True)
            record_line('Last known status',row.status)
            record_line('Observed through',pd.Timestamp(row.week_end).strftime('%Y-%m-%d') if pd.notna(row.week_end) else '—')
            record_line('Historical freshness',_source_age(row))
            record_line('Model score (not probability)',_value(risk_val,digits=3))
            record_line('Estimated lost oil (well-week)',_value(row.get('estimated_lost_oil_m3'),' m³'))
            record_line('Recorded downtime (well-week)',_value(row.get('hours_down'),' h',0))
            if pd.notna(age) and age>90:
                st.warning('This well has no recent weekly observation. Its last-known status is not proof of current activity.')
            b1,b2=st.columns(2)
            with b1:
                if st.button('Equipment Health ↗',use_container_width=True,key='wg_fleet_go_health'):_go('Equipment Health',picked_well)
            with b2:
                if st.button('AI Evidence ↗',use_container_width=True,key='wg_fleet_go_ai'):_go('AI Well Investigator',picked_well)
            if st.button('Open Inspection Scheduler →',type='primary',use_container_width=True,key='wg_fleet_go_scheduler'):
                _go('Inspection Scheduler',picked_well)
            st.caption('Locations and status are last-known source metadata; they may postdate the historical prediction snapshot. Model score is a relative ranking, not a calibrated probability.')

    st.markdown('<div class="wg-fleet-bottom-label">GEOGRAPHIC INTELLIGENCE & FIELD REVIEW PLANNING</div>',unsafe_allow_html=True)
    b1,b2,b3=st.columns([1.1,1.05,1],gap='small')
    with b1:
        st.markdown('<div class="wg-fleet-section-title">▥ &nbsp; FACILITY CONCENTRATION</div>',unsafe_allow_html=True)
        fac=res['facilities'].head(7).copy()
        if not len(fac):st.info('No named facilities in this filter.')
        else:
            if has_any_scores:
                show=fac[['facility','area','priority_wells','mapped_wells','priority_share']].rename(columns={
                    'facility':'Facility','area':'Area','priority_wells':'Priority wells',
                    'mapped_wells':'Mapped wells','priority_share':'Share (%)'})
            else:
                show=fac[['facility','area','mapped_wells']].rename(columns={'facility':'Facility','area':'Area','mapped_wells':'Mapped wells'})
            st.dataframe(show,hide_index=True,use_container_width=True,height=238)
            st.caption('Counts are from displayed mapped wells; no inferred facility failures.')
    with b2:
        st.markdown('<div class="wg-fleet-section-title">▥ &nbsp; OPERATING AREA SUMMARY</div>',unsafe_allow_html=True)
        area=res['areas'].head(7).copy()
        if not len(area):st.info('No mapped operating areas.')
        else:
            if has_any_scores:
                show=area[['area','priority_wells','mapped_wells','priority_share']].rename(columns={
                    'area':'Area','priority_wells':'Priority','mapped_wells':'Mapped','priority_share':'Share (%)'})
            else:
                show=area[['area','mapped_wells']].rename(columns={'area':'Area','mapped_wells':'Mapped'})
            st.dataframe(show,hide_index=True,use_container_width=True,height=238)
            st.caption('The area list is for geographic coordination, not model accuracy.')
    with b3:
        st.markdown('<div class="wg-fleet-section-title">◎ &nbsp; GEOGRAPHIC INSPECTION PLAN</div>',unsafe_allow_html=True)
        with st.container(border=True):
            if has_any_scores:
                st.markdown(f'**{stats["total_selected_reviews"]} model-ranked wells selected** for the {capacity}-review scenario.')
                st.markdown(f'**{stats["mapped_priority_count"]}** have mappable locations before additional display filters.')
                st.markdown(f'**{stats["priority_facilities"]} named facilities** and **{stats["priority_areas"]} operating areas** represented.')
            else:
                st.markdown('**No validated inspection shortlist** for the selected historical week.')
                st.markdown('Run the model and choose a scored week to plan reviews from actual rankings.')
            st.markdown('Plan grouping is by recorded area/facility. **No route or travel-time optimization is claimed.**')
            if st.button('View Inspection Scheduler →',use_container_width=True,key='wg_fleet_go_scheduler_bottom'):
                _go('Inspection Scheduler',st.session_state.get('chosen_well',''))
    with st.expander('Complete restricted well-coordinate table · search and export',expanded=False):
        q=st.text_input('Search source well records',key='wg_fleet_search')
        cols=['uwi','province','area','facility','status','lat','lon','source_date','week_end','age_days',
              'relative_risk_score','priority_flag','hours_down','estimated_lost_oil_m3']
        full=arrow_safe_view(data,cols)
        if q:
            mask=full.astype(str).apply(lambda col:col.str.contains(q,case=False,regex=False)).any(axis=1)
            full=full.loc[mask]
        st.caption(f'{len(full):,} private source-derived well rows · not for public redistribution')
        st.dataframe(full,hide_index=True,use_container_width=True,height=300)
        st.download_button('Export PRIVATE filtered CSV',data=full.to_csv(index=False).encode('utf-8'),
                           file_name='PRIVATE_WellGuard_Fleet_Geography.csv',mime='text/csv',key='wg_fleet_export')
    st.caption('Research prototype · real source records only · latest-known geographic metadata may differ from historical as-of status · do not redistribute restricted well coordinates or measurements.')
