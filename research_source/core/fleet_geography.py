"""Real-only geographic planning calculations for the private WellGuard app.

This module uses *only* records provided by the official source-preparation
scripts. Missing model scores never become fake risks or fake priority wells.
The map and area summaries are NOT a validated route optimization algorithm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAPACITIES = (25, 50, 100)
OVERLAYS = ('Failure risk (model)', 'Operating status', 'Estimated loss',
            'Recorded downtime', 'Data freshness')


def _clean_label(series: pd.Series) -> pd.Series:
    val = series.fillna('').astype(str).str.strip()
    return val.where(~val.str.lower().isin({'', 'nan', 'none', 'null', 'unknown'}), 'Unknown')


def prepare_fleet(view: pd.DataFrame, *, capacity: int = 25,
                  statuses: list[str] | None = None,
                  facility_focus: str | None = None) -> dict:
    """Build scoped fleet summary and map without fabricating absent data.

    Priority ranking is determined *before* changing map-status visibility or
    selecting a facility, so toggling markers cannot silently change the
    declared model's top-N list. Global area/province filters come from `view`.
    """
    if capacity not in CAPACITIES:
        raise ValueError('Capacity must be a validated comparison of 25, 50, or 100')
    d = view.copy()
    if 'uwi' not in d:
        raise ValueError('Expected source-derived `uwi` column')
    d['uwi'] = d['uwi'].astype(str)
    # One record per well, latest as of the chosen historical cutoff.
    if d['uwi'].duplicated().any():
        raise ValueError('Fleet view must contain one row per UWI')
    for col in ('lat','lon','relative_risk_score','age_days','hours_down','estimated_lost_oil_m3'):
        if col not in d: d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors='coerce')
    for col in ('area','facility','province','status'):
        if col not in d: d[col] = 'Unknown'
        d[col] = _clean_label(d[col])
    # Do not infer missing coordinates from facility names or other wells.
    valid = d['lat'].between(48,61) & d['lon'].between(-140,-96)
    mapped = d.loc[valid].copy()
    # Important: a model score of 0 is valid; only NA means not scored.
    scored = d.dropna(subset=['relative_risk_score']).sort_values(
        ['relative_risk_score','uwi'], ascending=[False,True]).copy()
    top = scored.head(capacity).copy()
    top['rank'] = np.arange(1,len(top)+1)
    chosen_ids = set(top.uwi)
    mapped['priority_flag'] = mapped.uwi.isin(chosen_ids)
    mapped['is_scored'] = mapped.relative_risk_score.notna()
    if statuses is not None:
        mapped = mapped[mapped.status.isin(statuses)].copy()
    if facility_focus and facility_focus != 'All facilities':
        mapped = mapped[mapped.facility.eq(facility_focus)].copy()
    # Status and facility display filters do not re-rank model selections.
    displayed_top = mapped.loc[mapped.priority_flag].copy()
    # Exclude the Unknown bucket from named facility concentration, but keep
    # missing facility counts visible in a dedicated summary field.
    facility_data = mapped[mapped.facility.ne('Unknown')].copy()
    fac_all = facility_data.groupby(['facility','area'],dropna=False).agg(
        mapped_wells=('uwi','nunique'), lat=('lat','mean'), lon=('lon','mean')).reset_index()
    if len(fac_all):
        fac_priority = displayed_top[displayed_top.facility.ne('Unknown')].groupby(
            ['facility','area'],dropna=False).uwi.nunique().rename('priority_wells').reset_index()
        facilities = fac_all.merge(fac_priority,on=['facility','area'],how='left')
        facilities['priority_wells'] = facilities.priority_wells.fillna(0).astype(int)
        facilities['priority_share'] = (100*facilities.priority_wells/facilities.mapped_wells).round(1)
        facilities = facilities.sort_values(['priority_wells','mapped_wells','facility'],
                                            ascending=[False,False,True])
    else:
        facilities = pd.DataFrame(columns=['facility','area','mapped_wells','lat','lon','priority_wells','priority_share'])
    area_all = mapped.groupby('area').uwi.nunique().rename('mapped_wells').reset_index()
    area_top = displayed_top.groupby('area').uwi.nunique().rename('priority_wells').reset_index()
    areas = area_all.merge(area_top,how='left',on='area')
    if len(areas):
        areas['priority_wells'] = areas.priority_wells.fillna(0).astype(int)
        areas['priority_share'] = (100*areas.priority_wells/areas.mapped_wells).round(1)
        areas = areas.sort_values(['priority_wells','mapped_wells','area'],ascending=[False,False,True])
    n_total = len(d)
    n_valid = int(valid.sum())
    has_scores = len(scored)>0
    # We do not call the count of historical source assets 'currently active'.
    stats = {
        'mapped_count': int(len(mapped)),
        'mapped_before_display_filter': n_valid,
        'scoped_historical_wells': n_total,
        'coordinate_coverage_pct': 100*n_valid/n_total if n_total else np.nan,
        'facilities': int(mapped.loc[mapped.facility!='Unknown','facility'].nunique()),
        'areas': int(mapped.loc[mapped.area!='Unknown','area'].nunique()),
        'stale_90d': int((mapped.age_days>90).sum()),
        'status_missing': int((mapped.status=='Unknown').sum()),
        'unknown_facility': int((mapped.facility=='Unknown').sum()),
        'has_scores': has_scores,
        'scored_wells_in_scope': int(len(scored)),
        'total_selected_reviews': int(len(top)) if has_scores else None,
        'mapped_priority_count': int(len(displayed_top)) if has_scores else None,
        'priority_facilities': int(displayed_top.loc[displayed_top.facility.ne('Unknown'),'facility'].nunique()),
        'priority_areas': int(displayed_top.loc[displayed_top.area.ne('Unknown'),'area'].nunique()),
        'requested_capacity': int(capacity),
        'facility_focus': facility_focus or 'All facilities',
    }
    return {'mapped':mapped,'top':top,'displayed_top':displayed_top,
            'facilities':facilities,'areas':areas,'stats':stats}


def overlay_figure(data: pd.DataFrame, overlay: str,
                   *, base64_map: str | None = None,
                   show_facilities: bool = True):
    """Create a private, offline Plotly chart using real source coordinates only."""
    import plotly.graph_objects as go
    if overlay not in OVERLAYS:raise ValueError('Unsupported overlay')
    fig = go.Figure()
    if data.empty:
        fig.add_annotation(text='No valid well coordinates for these filters',xref='paper',yref='paper',
                           x=.5,y=.5,showarrow=False,font=dict(color='#cfe8ed',size=15))
    else:
        x = data.copy()
        cd = x[['uwi','area','facility','status']].fillna('Unknown').astype(str).to_numpy()
        hover = '<b>%{customdata[0]}</b><br>Area: %{customdata[1]}<br>Facility: %{customdata[2]}<br>Status: %{customdata[3]}<extra></extra>'
        if overlay == 'Failure risk (model)':
            other=x[x.relative_risk_score.isna()]
            scored=x[x.relative_risk_score.notna() & ~x.priority_flag]
            priority=x[x.priority_flag]
            for subset,name,color,size,opacity in [
                (other,'No score','#5d7584',5,.35),
                (scored,'Scored wells','#58d3bb',7,.65),
                (priority,'Selected for review','#ffbc62',13,1)]:
                if len(subset):
                    fig.add_trace(go.Scatter(x=subset.lon,y=subset.lat,customdata=subset[['uwi','area','facility','status']].fillna('Unknown').astype(str).to_numpy(),
                                             mode='markers',name=name,hovertemplate=hover,
                                             marker=dict(size=size,color=color,opacity=opacity,
                                                         line=dict(color='#f8f5ef',width=1.5 if name=='Selected for review' else 0))))
        elif overlay=='Operating status':
            palette={'Producing':'#2ad1b3','Injecting':'#f6c566','Suspended':'#e6875e','SOLD':'#9871db','Abandoned':'#8ca4b7'}
            for status,g in x.groupby('status',sort=True):
                fig.add_trace(go.Scatter(x=g.lon,y=g.lat,mode='markers',name=str(status),
                    customdata=g[['uwi','area','facility','status']].fillna('Unknown').astype(str).to_numpy(),
                    marker=dict(size=7,color=palette.get(status,'#6aa6c3'),opacity=.84),hovertemplate=hover))
        else:
            variable={'Estimated loss':'estimated_lost_oil_m3','Recorded downtime':'hours_down',
                      'Data freshness':'age_days'}[overlay]
            have=x[x[variable].notna()]
            missing=x[x[variable].isna()]
            if len(have):
                fig.add_trace(go.Scatter(x=have.lon,y=have.lat,mode='markers',name=overlay,
                    customdata=have[['uwi','area','facility','status']].fillna('Unknown').astype(str).to_numpy(),
                    marker=dict(size=8,color=have[variable],colorscale='Turbo',opacity=.85,showscale=True,
                                colorbar=dict(title={'Estimated loss':'m³','Recorded downtime':'hours',
                                                     'Data freshness':'days'}[overlay],thickness=10)),hovertemplate=hover))
            if len(missing):
                fig.add_trace(go.Scatter(x=missing.lon,y=missing.lat,mode='markers',name='No valid observation',
                    customdata=missing[['uwi','area','facility','status']].fillna('Unknown').astype(str).to_numpy(),
                    marker=dict(size=5,color='#607583',opacity=.35),hovertemplate=hover))
    # A local image is used as a *visual* geographic background. Coordinates
    # never leave the user's computer to obtain third-party map tiles.
    if base64_map:
        fig.add_layout_image(dict(source=base64_map,xref='x',yref='y',x=-126,y=61,
                                  sizex=25,sizey=13,sizing='stretch',opacity=.83,layer='below'))
    # Use the documented West-Canada map extent. Zoom / box selection
    # permits inspection of the actual selected subset in the interactive app.
    fig.update_layout(template='plotly_dark',height=425,paper_bgcolor='#081b27',plot_bgcolor='#102b3a',
                      margin=dict(l=7,r=10,t=5,b=12),dragmode='pan',
                      legend=dict(orientation='h',y=-.08,x=0,font=dict(size=10),bgcolor='rgba(0,0,0,0)'),
                      xaxis=dict(range=[-126,-101],gridcolor='#304c5a',zeroline=False,title='',showticklabels=False),
                      yaxis=dict(range=[48,61],gridcolor='#304c5a',zeroline=False,title='',showticklabels=False,scaleanchor='x',scaleratio=1))
    return fig
