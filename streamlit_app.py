"""WellGuard | full public dashboard for DATA TITANS.

Saved benchmark figures are retrospective research aggregates; all individual
well records, locations, job events and inspector screens are explicitly
SYNTHETIC demonstration data. No restricted competition data or well-level
model predictions are hosted, reconstructed, or inferred.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from demo_data import make_demo, schedule_demo, ZONES

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="WellGuard | DATA TITANS", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
BENCH = json.loads((ROOT / "public_data/benchmark.json").read_text(encoding="utf8"))
MONTHS = pd.read_csv(ROOT / "public_data/monthly_summary.csv")
PAGES = ["Executive Overview", "Fleet Map", "Equipment Health", "Failure Records", "Inspection Scheduler",
         "Model Validation Lab", "Maintenance Effectiveness", "Production & Downtime", "Data Reliability", "Well Investigator"]
NAV = ["⌂", "◫", "⌁", "⚠", "▤", "◈", "⚒", "▥", "◇", "✧"]
TEAL, BLUE, AMBER, RED = "#4DD7CC", "#76A9F0", "#F0C177", "#F48083"

st.markdown('''<style>
:root{color-scheme:dark}
.stApp{background:#0a1422;color:#ebf3f8}
[data-testid="stHeader"]{background:#0a1422}
[data-testid="stSidebar"]{background:#101e2e;border-right:1px solid #23374c}
.block-container{max-width:1520px;padding:1.65rem 2.4rem 4rem}
h1,h2,h3{letter-spacing:-.025em}
h1{font-size:2rem!important;font-weight:780!important}
h2{font-size:1.5rem!important}
h3{font-size:1.1rem!important}
p,li{color:#c2d2df}
div[data-testid="stMetric"]{padding:15px 17px;background:linear-gradient(145deg,#132a3a,#101c2e);border:1px solid #294259;border-radius:14px}
div[data-testid="stMetricValue"]{color:#f1fbff;font-weight:780}
div[data-testid="stMetricLabel"]{color:#a6c5d3}
[data-testid="stDataFrame"]{border:1px solid #273c4f;border-radius:10px}
.wg-brand{font-size:.74rem;font-weight:800;letter-spacing:.15em;color:#6adcd3;margin-bottom:12px}
.wg-hero{border:1px solid #2c5365;border-radius:18px;padding:26px 30px;margin-bottom:19px;background:radial-gradient(ellipse at 96% -45%,#215166,#152c40 54%,#101e30 100%)}
.wg-hero h1{color:#f4fcff;margin:5px 0 8px!important;font-size:2.1rem!important}
.wg-hero p{font-size:1rem;color:#c7deeb;margin:0;max-width:870px;line-height:1.6}
.wg-eyebrow{font-weight:800;letter-spacing:.15em;text-transform:uppercase;font-size:.70rem;color:#78e5db}
.wg-notice{border:1px solid #395465;border-radius:10px;background:#132638;color:#cbdfe8;padding:10px 14px;margin:8px 0 22px;font-size:.85rem}
.wg-section{font-size:.77rem;letter-spacing:.1em;text-transform:uppercase;color:#75d9d4;font-weight:750;border-top:1px solid #273d50;padding-top:18px;margin:25px 0 10px}
.wg-note{color:#a8becc;font-size:.84rem;line-height:1.5}
.wg-tag{display:inline-block;border-radius:24px;padding:6px 11px;background:#1c3643;color:#87dad5;font-size:.75rem;letter-spacing:.035em;margin:4px 5px 6px 0;border:1px solid #35606c}
.wg-tile{border:1px solid #29475a;background:#142538;border-radius:12px;padding:14px 17px;min-height:92px}
.wg-tile b{font-size:.85rem;color:#ecf9ff;display:block;margin-bottom:8px}
.wg-tile span{color:#aec6d4;font-size:.82rem;line-height:1.45}
[data-testid="stTabs"]{margin-top:14px}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_demo():
    return make_demo()

FLEET, HISTORY, JOBS = load_demo()
DATES = sorted(pd.to_datetime(HISTORY.week_end.unique()))

def format_chart(fig, title=None, height=370):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#bcd3df",size=12), height=height,
                      margin=dict(l=12,r=12,t=55 if title else 25,b=30),
                      legend=dict(orientation="h",y=-.26), title=dict(text=title or "",font=dict(size=17,color="#eafbff")))
    fig.update_xaxes(gridcolor="#243c4d",zeroline=False)
    fig.update_yaxes(gridcolor="#243c4d",zeroline=False)
    return fig


def section(text):
    st.markdown(f'<div class="wg-section">{text}</div>', unsafe_allow_html=True)


def demo_note():
    st.markdown('<div class="wg-notice"><b>PUBLIC DEMONSTRATION:</b> Individual wells, scores, sensor values, locations, maintenance and loss records on this page are <b>fictional</b>. Historical research performance is shown separately and is labeled as a saved aggregate.</div>', unsafe_allow_html=True)


def show_table(df: pd.DataFrame, name: str, height=370):
    st.dataframe(df,hide_index=True,use_container_width=True,height=height)
    st.download_button(f"Download simulated {name} CSV", data=df.to_csv(index=False).encode("utf8"),
                       file_name=f"WELLGUARD_SYNTHETIC_{name.replace(' ','_')}.csv",mime="text/csv",key=f"download_{name}")


def metric_with_caption(label, value, detail):
    st.metric(label,value)
    st.caption(detail)

with st.sidebar:
    st.markdown('<div class="wg-brand">◇ WELLGUARD <span style="color:#8aa8b7"> / DATA TITANS</span></div>',unsafe_allow_html=True)
    st.caption("RESEARCH • INSPECTION DECISIONS • OPERATIONS")
    page = st.radio("Workspace", PAGES, format_func=lambda s: f"{NAV[PAGES.index(s)]}  {s}",label_visibility="collapsed")
    st.divider()
    st.caption("ILLUSTRATIVE OPERATIONS FILTERS")
    as_of = st.selectbox("Synthetic as-of week", DATES, index=len(DATES)-1, format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    zone = st.selectbox("Example operating sector", ["All sectors",*ZONES])
    capacity = st.select_slider("Weekly inspection capacity", options=[25,50,100],value=25)
    st.caption("Capacity changes the demo inspection queue and the saved retrospective benchmark comparison.")
    st.divider()
    st.caption("No source wells, protected datasets or current field recommendations are available in this public application.")

as_of = pd.Timestamp(as_of)
snapshot = HISTORY[HISTORY.week_end.eq(as_of)].copy()
if zone != "All sectors": snapshot = snapshot[snapshot.zone.eq(zone)]
visible_history = HISTORY[(HISTORY.week_end.le(as_of)) & (HISTORY.well_id.isin(snapshot.well_id))].copy()
all_visible = set(snapshot.well_id)
queue = snapshot.sort_values(["relative_risk_score","well_id"],ascending=[False,True]).head(capacity).copy()
queue.insert(0,"review_rank",np.arange(1,len(queue)+1))

st.markdown('<div class="wg-brand">DATA TITANS  /  DSMLC DATATHON 2026  /  WELLGUARD</div>',unsafe_allow_html=True)
st.markdown(f'''<div class="wg-hero"><div class="wg-eyebrow">{page.upper()} &nbsp; • &nbsp; OPERATIONS INTELLIGENCE</div>
<h1>Predict risk. Prioritize inspections. Support production decisions.</h1>
<p>A research prototype for prioritizing heavy-oil well inspections using historical operating patterns, chronological validation, and operational context.</p>
</div>''',unsafe_allow_html=True)

if page == "Executive Overview":
    section("Retrospective research results  |  saved aggregate benchmark")
    a,b,c,d = st.columns(4)
    a.metric("Reference ROC-AUC",f"{BENCH['model_roc_auc']:.3f}")
    b.metric("Reference avg. precision",f"{BENCH['model_ap']:.3f}")
    c.metric("Recorded eligible events","643")
    d.metric("Test well-weeks",f"{BENCH['test_well_weeks']:,}")
    st.caption("Saved aggregate evaluation from the private, time-split research pipeline; not recomputed from data in this public deployment.")
    section("Inspection-capacity decision  |  historical backtest")
    key = str(capacity)
    model = BENCH["comparison"]["relative_risk_score"]["event_coverage"][key]
    baseline = BENCH["comparison"]["recency_baseline_score"]["event_coverage"][key]
    x,y,z = st.columns(3)
    x.metric(f"{capacity} weekly reviews",f"{model['caught_events']:,} / 643",f"{model['event_recall']*100:.1f}% of eligible events flagged")
    y.metric("Failure-recency ranking",f"{baseline['caught_events']:,} / 643",f"{baseline['event_recall']*100:.1f}% flagged")
    z.metric("Additional events flagged",f"+{model['caught_events']-baseline['caught_events']}","Retrospective identification, not prevention")
    capacities=[25,50,100]
    comp = pd.DataFrame({"Weekly review slots":capacities,
                         "WellGuard ranking":[BENCH['comparison']['relative_risk_score']['event_coverage'][str(n)]['caught_events'] for n in capacities],
                         "Recency baseline":[BENCH['comparison']['recency_baseline_score']['event_coverage'][str(n)]['caught_events'] for n in capacities]})
    fig = go.Figure()
    fig.add_bar(name="WellGuard ranking",x=comp['Weekly review slots'].astype(str),y=comp['WellGuard ranking'],marker_color=TEAL)
    fig.add_bar(name="Recency baseline",x=comp['Weekly review slots'].astype(str),y=comp['Recency baseline'],marker_color=BLUE)
    fig.update_layout(barmode="group")
    st.plotly_chart(format_chart(fig,"Recorded events flagged within the eligible retrospective set"),use_container_width=True)
    st.caption("The benchmark does not prove inspections prevent failures. Increasing slots changes the ranked review budget, not prediction accuracy or staffing actually deployed.")
    section("Explore the complete operating workflow")
    col1,col2,col3=st.columns(3)
    for col,title,body in [(col1,"01 / Prioritize","Sort fictional wells using transparent relative-risk rankings and 25, 50 or 100 review slots."),
                            (col2,"02 / Investigate","Inspect demonstration sensor history, job context and reason codes before any operational action."),
                            (col3,"03 / Evaluate","Review chronological validation, data reliability and the distinction between observed historical losses and avoidable losses.")]:
        with col: st.markdown(f'<div class="wg-tile"><b>{title}</b><span>{body}</span></div>',unsafe_allow_html=True)
    demo_note()
    st.subheader("Fictional fleet snapshot")
    p,q,r,s=st.columns(4)
    p.metric("Fictional wells in scope",f"{len(snapshot):,}")
    q.metric("Example priority queue",f"{len(queue):,}")
    r.metric("Simulated downtime",f"{snapshot.downtime_hours.sum():,.0f} h")
    s.metric("Simulated lost oil",f"{snapshot.estimated_lost_oil_m3.sum():,.0f} m³")
    st.caption("The four metrics immediately above are synthetic interface examples and are not competition observations.")

elif page == "Fleet Map":
    demo_note()
    section("Illustrative fleet footprint")
    st.caption("The scatter coordinates are an invented diagram, NOT real geographic coordinates. Display positions have no relation to Strathcona facilities or well sites.")
    data = snapshot.merge(FLEET[['well_id','equipment','x','y']],on="well_id",how="left")
    color_by=st.selectbox("Colour demo map by",["zone","relative_risk_score","equipment"])
    fig=px.scatter(data,x="x",y="y",color=color_by,hover_name="well_id",size="oil_rate_m3d",size_max=17,
                   hover_data={"zone":True,"oil_rate_m3d":':.1f',"relative_risk_score":':.3f',"x":False,"y":False},
                   color_continuous_scale="Tealrose" if color_by=="relative_risk_score" else None)
    fig.update_xaxes(visible=False);fig.update_yaxes(visible=False,scaleanchor="x",scaleratio=1)
    st.plotly_chart(format_chart(fig,"Fictional sectors and relative positions",520),use_container_width=True)
    g=data.groupby("zone",as_index=False).agg(wells=("well_id","nunique"),mean_demo_risk=("relative_risk_score","mean"),example_lost_oil_m3=("estimated_lost_oil_m3","sum"))
    show_table(g.round(3),"fleet_sector_summary",260)

elif page == "Equipment Health":
    demo_note()
    section("Illustrative monitoring and trends")
    if not len(snapshot): st.info("No demonstration wells match the filters."); st.stop()
    selected=st.selectbox("Choose fictional well",sorted(snapshot.well_id.tolist()))
    hist=visible_history[visible_history.well_id.eq(selected)].sort_values("week_end")
    current=hist.iloc[-1]; prior=hist.iloc[-2] if len(hist)>1 else current
    p,q,r,s=st.columns(4)
    p.metric("Illustrative oil rate",f"{current.oil_rate_m3d:.1f} m³/d",f"{current.oil_rate_m3d-prior.oil_rate_m3d:+.1f} vs prior week")
    q.metric("Pump RPM", "Missing" if pd.isna(current.pump_rpm) else f"{current.pump_rpm:.0f}")
    r.metric("Demo downtime",f"{current.downtime_hours:.1f} h")
    s.metric("Relative demo ranking score",f"{current.relative_risk_score:.3f}")
    choice=st.selectbox("Charted reading",["oil_rate_m3d","pump_rpm","pump_efficiency","tubing_pressure_kpa","rod_torque_ftlb","downtime_hours"])
    st.plotly_chart(format_chart(px.line(hist,x="week_end",y=choice,markers=True,color_discrete_sequence=[TEAL]),
                                   f"{selected} — simulated {choice.replace('_',' ')}"),use_container_width=True)
    st.caption("Missing values are left missing; simulated trends are not sensor fault diagnoses.")
    show_table(hist[["week_end","well_id",choice,"relative_risk_score","demo_job_record"]].tail(16),"equipment_health")

elif page == "Failure Records":
    demo_note()
    section("Example failure-event explorer")
    f=visible_history[visible_history.demo_failure_record].copy()
    a,b,c=st.columns(3)
    a.metric("Simulated event records",len(f))
    b.metric("Fictional wells with events",f.well_id.nunique())
    c.metric("Synthetic history weeks",visible_history.week_end.nunique())
    if len(f):
        fig=px.bar(f.groupby("week_end",as_index=False).size(),x="week_end",y="size",color_discrete_sequence=[RED])
        st.plotly_chart(format_chart(fig,"Fictional recorded events by week"),use_container_width=True)
        st.caption("No causal failure diagnostics are inferred from these synthetic records.")
        show_table(f[["well_id","zone","week_end","relative_risk_score","oil_rate_m3d","downtime_hours"]].sort_values("week_end",ascending=False),"failure_examples")
    else: st.info("No illustrative failure records match these filters.")
    st.divider()
    st.subheader("Actual research data availability")
    st.write("The restricted competition source contains **2,873 recorded failure events**. Failure-reporting completeness and dates vary; this public dashboard does not expose individual event records.")

elif page == "Inspection Scheduler":
    demo_note()
    section("Risk-ranked review queue — simulated well records")
    cooldown=st.slider("Repeat-review cooldown (weeks)",0,3,0,help="Demo policy: temporarily exclude wells reviewed recently; queue recalculated on the fictional full fleet history.")
    ranking=schedule_demo(HISTORY,as_of,capacity,zone,cooldown)
    p,q,r=st.columns(3)
    p.metric("Review slots per week",capacity)
    q.metric("Fictional wells selected",len(ranking))
    r.metric("Minimum selected rank score",f"{ranking.relative_risk_score.min():.3f}" if len(ranking) else "—")
    st.caption("Scores are relative, uncalibrated demo values. The ranked table is an interface example and does NOT represent actual wells chosen by the research model.")
    if len(ranking):
        fig=px.bar(ranking.head(30),x="well_id",y="relative_risk_score",color="zone",hover_data=["oil_rate_m3d","downtime_hours"])
        st.plotly_chart(format_chart(fig,f"Top {min(30,len(ranking))} fictional inspection candidates"),use_container_width=True)
        search=st.text_input("Filter fictional well ID",placeholder="DEMO-00...")
        out=ranking[ranking.well_id.str.contains(search,case=False,regex=False)] if search else ranking
        show_table(out[["review_rank","well_id","zone","relative_risk_score","oil_rate_m3d","downtime_hours","estimated_lost_oil_m3"]],"inspection_queue")
    section("Research benchmark at same review capacity  |  not simulated")
    m=BENCH["comparison"]["relative_risk_score"]["event_coverage"][str(capacity)]
    base=BENCH["comparison"]["recency_baseline_score"]["event_coverage"][str(capacity)]
    u,v,w=st.columns(3)
    u.metric("Historical events flagged",f"{m['caught_events']} / {m['eligible_events']}")
    v.metric("Recency baseline",f"{base['caught_events']} / {base['eligible_events']}")
    w.metric("Additional historical flags",m['caught_events']-base['caught_events'])
    st.caption("These are saved cohort-level test results, not the outcomes of the fictional queue displayed above. There is no demonstrated intervention benefit.")

elif page == "Model Validation Lab":
    section("Saved reference evaluation  |  actual retrospective aggregates")
    a,b,c,d=st.columns(4)
    a.metric("ROC-AUC",f"{BENCH['model_roc_auc']:.4f}")
    b.metric("Average precision",f"{BENCH['model_ap']:.4f}")
    c.metric("Recency ROC-AUC",f"{BENCH['recency_baseline_roc_auc']:.4f}")
    d.metric("Recency avg. precision",f"{BENCH['recency_baseline_ap']:.4f}")
    st.caption("Saved aggregate chronological test; random record splits were not used. Figures are not freshly recomputed within this public app.")
    st.markdown('<span class="wg-tag">48,865 train well-weeks</span><span class="wg-tag">11,149 validation well-weeks</span><span class="wg-tag">22,580 test well-weeks</span><span class="wg-tag">21-day split buffers</span><span class="wg-tag">1–14 day failure horizon</span>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        fig=go.Figure()
        fig.add_bar(x=["Average precision","ROC-AUC"],y=[BENCH['model_ap'],BENCH['model_roc_auc']],name="WellGuard",marker_color=TEAL)
        fig.add_bar(x=["Average precision","ROC-AUC"],y=[BENCH['recency_baseline_ap'],BENCH['recency_baseline_roc_auc']],name="Recency",marker_color=BLUE)
        fig.update_layout(barmode="group")
        st.plotly_chart(format_chart(fig,"Relative model discrimination"),use_container_width=True)
    with b:
        fig=go.Figure()
        fig.add_scatter(x=MONTHS.month,y=MONTHS.model_ap,name="WellGuard AP",mode="lines+markers",line_color=TEAL)
        fig.add_scatter(x=MONTHS.month,y=MONTHS.recency_ap,name="Recency AP",mode="lines+markers",line_color=BLUE)
        st.plotly_chart(format_chart(fig,"Monthly saved average precision"),use_container_width=True)
    section("Capacity sensitivity and baseline checks")
    rows=[]
    for n in (25,50,100):
        for name,key in [("WellGuard","relative_risk_score"),("Failure-recency","recency_baseline_score"),("Operating-only","operating_only"),("Seven-day-delayed inputs","seven_day_delayed_inputs")]:
            obj=BENCH['comparison'][key]['event_coverage'][str(n)]
            rows.append({"Weekly capacity":n,"Method":name,"Eligible events":obj['eligible_events'],"Events flagged":obj['caught_events'],"Event coverage (%)":round(obj['event_recall']*100,2)})
    show_table(pd.DataFrame(rows),"reference_validation_comparison",320)
    st.info("ROC-AUC and average precision evaluate historical ranking performance; they do not establish calibrated failure probabilities. Event identification is not failure prevention or recovered revenue.")

elif page == "Maintenance Effectiveness":
    demo_note()
    section("Maintenance history explorer")
    available_jobs=JOBS[(JOBS.week_end.le(as_of)) & (JOBS.well_id.isin(all_visible))].copy()
    options=["All",*sorted(JOBS.job_type.unique())]
    job_type=st.selectbox("Fictional job type",options)
    if job_type!="All":available_jobs=available_jobs[available_jobs.job_type.eq(job_type)]
    p,q,r=st.columns(3)
    p.metric("Fictional maintenance jobs",len(available_jobs))
    q.metric("Median field-estimated cost",f"CAD {available_jobs.field_estimated_cost_cad.median():,.0f}" if len(available_jobs) else "—")
    r.metric("Example median oil-rate change",f"{(available_jobs.after_oil_m3d-available_jobs.before_oil_m3d).median():+.1f} m³/day" if len(available_jobs) else "—")
    if len(available_jobs):
        agg=available_jobs.groupby("job_type",as_index=False).agg(before=("before_oil_m3d","median"),after=("after_oil_m3d","median"))
        fig=go.Figure()
        fig.add_bar(x=agg.job_type,y=agg.before,name="Prior-window example",marker_color=BLUE)
        fig.add_bar(x=agg.job_type,y=agg.after,name="Following-window example",marker_color=TEAL)
        fig.update_layout(barmode="group")
        st.plotly_chart(format_chart(fig,"Fictional pre/post maintenance rates"),use_container_width=True)
        show_table(available_jobs.sort_values("week_end",ascending=False),"maintenance_events")
    st.warning("Even with real observations, before/after changes would be descriptive—not causal maintenance effectiveness. Real job costs were largely unavailable in the source material.")

elif page == "Production & Downtime":
    demo_note()
    section("Production, downtime and estimated losses  |  simulated operating history")
    count=st.slider("Weeks to display",4,24,12,4)
    recent_dates=sorted(visible_history.week_end.unique())[-count:]
    agg=visible_history[visible_history.week_end.isin(recent_dates)].groupby("week_end",as_index=False).agg(
        oil_volume_m3=("oil_volume_m3","sum"),downtime_hours=("downtime_hours","sum"),estimated_lost_oil_m3=("estimated_lost_oil_m3","sum"))
    a,b,c=st.columns(3)
    a.metric("Fictional oil volume",f"{agg.oil_volume_m3.sum():,.0f} m³")
    b.metric("Fictional downtime",f"{agg.downtime_hours.sum():,.0f} h")
    c.metric("Fictional estimated lost oil",f"{agg.estimated_lost_oil_m3.sum():,.0f} m³")
    col1,col2=st.columns(2)
    with col1:st.plotly_chart(format_chart(px.area(agg,x="week_end",y="oil_volume_m3",color_discrete_sequence=[TEAL]),"Example weekly oil volume"),use_container_width=True)
    with col2:st.plotly_chart(format_chart(px.line(agg,x="week_end",y="downtime_hours",markers=True,color_discrete_sequence=[AMBER]),"Example weekly downtime"),use_container_width=True)
    st.plotly_chart(format_chart(px.bar(agg,x="week_end",y="estimated_lost_oil_m3",color_discrete_sequence=[RED]),"Example weekly estimated lost oil"),use_container_width=True)
    st.caption("Production loss values are not actual revenue, and are not a measure of recoverable production. Nothing here estimates avoided loss from an inspection.")
    show_table(agg,"production_weekly_summary",300)

elif page == "Data Reliability":
    section("Official dataset information  |  source-level aggregate facts")
    p,q,r=st.columns(3)
    p.metric("Daily source rows","5,075,590")
    q.metric("Maintenance jobs","15,610")
    r.metric("Recorded failures","2,873")
    st.caption("Aggregated dataset inventory from competition materials. Source records are not bundled or served.")
    st.subheader("Data quality and model limits")
    quality=pd.DataFrame([
        ("Pump RPM",52.5),("Pump efficiency",66.8),("Rod torque",74.0),("Motor current",82.2),
        ("Tubing pressure",47.4),("Lost oil",67.5)])
    quality.columns=["Field","Missing rows (%)"]
    st.plotly_chart(format_chart(px.bar(quality,x="Field",y="Missing rows (%)",color_discrete_sequence=[AMBER]),"Recorded source missingness (source audit)"),use_container_width=True)
    st.markdown("**Source warnings:** failure-reporting coverage changes over time; actual final job costs are overwhelmingly missing; pressure has placeholder values; field units and data granularity need domain checks. A missing sensor is not necessarily a fault.")
    demo_note()
    st.subheader("Synthetic sensor completeness")
    fields=["pump_rpm","pump_efficiency","rod_torque_ftlb","tubing_pressure_kpa"]
    coverage=pd.DataFrame({"Field":fields,"Synthetic coverage (%)":[round(snapshot[f].notna().mean()*100,1) for f in fields]})
    st.plotly_chart(format_chart(px.bar(coverage,x="Field",y="Synthetic coverage (%)",color_discrete_sequence=[TEAL]),"Example app telemetry coverage"),use_container_width=True)
    show_table(coverage,"example_sensor_completeness",245)

elif page == "Well Investigator":
    demo_note()
    section("Evidence-based investigator  |  rule-based, explainable demonstration")
    if snapshot.empty:st.info("No demonstration wells match these filters.");st.stop()
    selected=st.selectbox("Select fictional well to investigate",sorted(snapshot.well_id.tolist()),key="investigator_well")
    hist=visible_history[visible_history.well_id.eq(selected)].sort_values("week_end")
    r=hist.iloc[-1]
    question=st.selectbox("Investigation question",[
        "Why is this well high on the review list?","What changed over the past four weeks?",
        "Are any sensor readings missing?","What is the maintenance context?",
        "What can we infer about production losses?","What should an operator verify?"])
    if st.button("Investigate available example records",type="primary"):
        lines=[]
        if "review list" in question:
            rank=int(snapshot.relative_risk_score.rank(ascending=False,method="min")[snapshot.well_id.eq(selected)].iloc[0])
            lines=[f"Relative illustrative rank: {rank} of {len(snapshot)} in this sector selection.",
                   f"Example ranking score: {r.relative_risk_score:.3f}; not a calibrated failure probability.",
                   f"Latest simulated downtime: {r.downtime_hours:.1f} hours."]
        elif "changed" in question:
            past=hist.iloc[max(0,len(hist)-4)]
            lines=[f"Simulated oil-rate change: {r.oil_rate_m3d-past.oil_rate_m3d:+.1f} m³/day.",
                   f"Simulated downtime change: {r.downtime_hours-past.downtime_hours:+.1f} hours.",
                   "These are trends, not evidence of a confirmed mechanical root cause."]
        elif "missing" in question:
            fields=["pump_rpm","pump_efficiency","tubing_pressure_kpa","rod_torque_ftlb"]
            absent=[f.replace('_',' ') for f in fields if pd.isna(r[f])]
            lines=["Missing simulated channels: "+(", ".join(absent) if absent else "none of the four displayed channels"),
                   "A missing channel cannot establish equipment condition."]
        elif "maintenance" in question:
            n=len(JOBS[(JOBS.well_id.eq(selected)) & (JOBS.week_end.le(as_of)) & (JOBS.week_end.gt(as_of-pd.Timedelta(days=90)))])
            lines=[f"Fictional prior maintenance jobs recorded in 90 days: {n}.","Later jobs are excluded; job counts are context, not proof of a cause."]
        elif "losses" in question:
            lines=[f"Fictional example lost-oil estimate: {r.estimated_lost_oil_m3:.1f} m³ in this displayed week.",
                   "No amount of estimated lost oil is automatically recoverable or equivalent to financial savings."]
        else:
            lines=["Review the latest telemetry and available maintenance records.",
                   "Verify instrument status, data freshness and operational constraints.",
                   "A human specialist must evaluate any work order or intervention."]
        st.markdown("#### Findings based only on fictional displayed records")
        for line in lines:st.write("• "+line)
        with st.expander("Evidence ledger",expanded=True):
            st.json({"fictional_well_id":selected,"as_of":str(as_of.date()),
                     "relative_risk_score":float(r.relative_risk_score),
                     "oil_rate_m3d":float(r.oil_rate_m3d),"downtime_hours":float(r.downtime_hours),
                     "source":"deterministic synthetic demonstration data"})
    st.subheader("Supporting simulated operating trends")
    plot=px.line(hist,x="week_end",y=["oil_rate_m3d","downtime_hours"],markers=True)
    st.plotly_chart(format_chart(plot,"Fictional oil-rate and downtime trends"),use_container_width=True)
    st.caption("The investigator is a transparent rules-and-evidence explainer, not a live generative model or a field diagnosis engine.")

st.divider()
st.caption("WellGuard • DATA TITANS • Public competition showcase. All well-level screens use entirely synthetic demonstration records. Saved research aggregates are retrospective and do not imply verified prevention, savings, deployment or live model inference. Source datasets are restricted and intentionally excluded.")
