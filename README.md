# WellGuard | DATA TITANS — DSMLC Datathon 2026

**Predict Risk · Prioritize Inspections · Support Better Production Decisions**

### Team DATA TITANS

- **Saad Gul**
- **Dia Toredi**
- **Syed Ali**
- **Elnaz Ebrahimi**

### Research Question

How can predictive maintenance and production optimization analytics help Strathcona Resources identify high-priority wells, prioritize inspections, and support data-driven decisions to reduce operational losses?

## Project Resources

| Resource | Link |
|---|---|
| **Live Interactive Dashboard** | [Open WellGuard Dashboard](https://data-titans-wellguard.streamlit.app/) |
| **Five-Minute Demonstration Video** | [Watch Video](https://drive.google.com/file/d/1k9JjCHW6yomQqg_amaE4EvXxowXeUAj9/view) |
| **Final Competition Presentation** | [View Presentation](https://docs.google.com/presentation/d/1iyn9YzzsCQj7LIBIeJg3cTDimuVDbJBs/edit) |

## 1. Project Overview

**WellGuard** is a predictive maintenance and operational decision-support research prototype designed to help prioritize oil and gas well inspections using historical operating data, equipment performance indicators, and recorded failures.

The project addresses a practical operational challenge: **When inspection resources are limited, which wells should receive attention first?**

WellGuard brings together historical failure-risk rankings, inspection scheduling, production monitoring, maintenance analysis, and explainable evidence to support engineering review. The predictive research evaluates risk of a **recorded failure within the following 1–14 days** using chronologically later test observations.

This is a **retrospective research prototype**, not a live operational model or a demonstrated failure-prevention intervention.

## 2. Interactive Dashboard

The public Streamlit application includes **10 interactive sections**:

| Dashboard Section | Key Functionality |
|---|---|
| **Executive Overview** | Saved historical performance, inspection scenarios, and simulated fleet indicators |
| **Fleet Map** | Fictional asset visualization and risk filters; demo coordinates are not real well locations |
| **Equipment Health** | Synthetic pump, pressure, efficiency, production, and downtime trends |
| **Failure Records** | Synthetic example events and historical dataset-level counts |
| **Inspection Scheduler** | Demo inspection queues with 25, 50, and 100 weekly review slots |
| **Model Validation Lab** | Saved ROC-AUC, average precision, chronological testing, and baseline comparisons |
| **Maintenance Effectiveness** | Synthetic maintenance activity and non-causal pre/post comparisons |
| **Production & Downtime** | Synthetic production, downtime, and estimated-loss exploration |
| **Data Reliability** | Dataset-level source coverage and simulated telemetry completeness |
| **Well Investigator** | Rules-based explanation of fictional well rankings and operating changes |

### Inspection Capacity Optimization

The Inspection Scheduler allows users to explore three illustrative operational scenarios:

- **25 reviews per week:** A focused high-priority shortlist under constrained inspection capacity.
- **50 reviews per week:** Expanded inspection coverage.
- **100 reviews per week:** Broader coverage of the eligible demonstration fleet.

Users can change capacity and filters, inspect ranking evidence, and compare the selected capacity with **separate historical research results**. Demo rankings are synthetic; historical benchmark counts are not recalculated from the fictional wells.

## 3. Dataset Overview

The competition provided three primary datasets for the research analysis:

| Dataset | Records |
|---|---:|
| Daily Production | 5,075,590 |
| Maintenance Jobs | 15,610 |
| Recorded Equipment Failures | 2,873 |

The analytical workflow includes data cleaning, date alignment, historical feature engineering, chronological validation, and failure-risk ranking.

## 4. Predictive Modeling and Validation

WellGuard evaluates historical operating measurements to rank wells by their relative risk of a recorded failure in the following 14 days. The reference analysis uses chronological training, validation, and test periods to reduce future-information leakage.

### Historical Data Splits

| Dataset | Period | Well-Week Observations |
|---|---|---:|
| Training | June 1, 2025 – March 1, 2026 | 48,865 |
| Validation | March 22 – May 17, 2026 | 11,149 |
| Testing | June 7 – October 4, 2026 | 22,580 |

### Saved Historical Test Performance

| Metric | WellGuard Reference Model | Failure-Recency Baseline |
|---|---:|---:|
| ROC-AUC | **0.8055** | 0.7243 |
| Average Precision | **0.1637** | 0.1030 |
| Distinct eligible events flagged — 25 reviews/week | **116 / 643** | 75 / 643 |
| Distinct eligible events flagged — 50 reviews/week | **180 / 643** | 114 / 643 |
| Distinct eligible events flagged — 100 reviews/week | **289 / 643** | 188 / 643 |

### Key Finding

At **25 weekly reviews**, WellGuard's retrospective ranking flagged **116 eligible recorded failure events**, compared with **75** for the failure-recency baseline: **41 additional historical events flagged at the same review capacity**.

These are retrospective results, **not 41 prevented failures** or proven recovered production. The denominator of 643 represents eligible test events, not all reported failures. Scores are relative rankings rather than calibrated failure probabilities, and no prospective intervention was evaluated.

## 5. Research Methodology

1. **Data Preparation:** Clean and align daily production, maintenance, and failure records.
2. **Feature Engineering:** Develop past-only operating indicators, sensor trends, variability measures, and failure context.
3. **Predictive Modeling:** Evaluate a LightGBM-based reference model using chronological data splits.
4. **Model Validation:** Measure classification performance and compare rankings against a failure-recency baseline.
5. **Inspection Prioritization:** Assess performance at fixed weekly review capacities.
6. **Operational Investigation:** Review production, downtime, maintenance, and equipment indicators to support decisions.

Further details are documented in [Methodology and Validation](docs/METHODOLOGY_AND_VALIDATION.md).

## 6. Public Dashboard and Data Handling

**Historical research aggregates:** Saved aggregate model-performance results and retrospective comparisons are provided in `public_data/`. They are displayed as previously computed research summaries, **not recalculated online from restricted source records**. Publication is subject to applicable organizer permissions.

**Synthetic interactive examples:** Individual demo wells, identifiers, rankings, positions, scores, sensor measurements, maintenance records, and example failures are fictional, reproducibly generated by `demo_data.py`. They are not anonymized or reconstructed Strathcona well-level observations.

The application distinguishes research benchmarks from synthetic interactive examples. The Well Investigator organizes evidence through transparent rules; it does not autonomously diagnose equipment faults or authorize repairs.

## 7. Technology Stack

| Technology | Application |
|---|---|
| Python | Data processing and application development |
| Pandas / NumPy | Analysis and transformation |
| LightGBM | Historical failure-risk reference modeling |
| Scikit-learn | Model evaluation |
| Streamlit | Interactive application |
| Plotly | Interactive charts |
| Databricks / SQL | Reproducibility materials and analytical workflow |
| GitHub | Source-code hosting and documentation |

## 8. Run or Deploy the Dashboard

**Live application:** [data-titans-wellguard.streamlit.app](https://data-titans-wellguard.streamlit.app/)

**Streamlit Community Cloud configuration:**

- **GitHub repository:** `Saadgul35/DATA-TITANS-WellGuard`
- **Branch:** `main`
- **App entrypoint:** `streamlit_app.py`
- **Dependencies:** `requirements.txt`

**Run locally:**

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The public app requires no private competition Parquet files or protected model artifacts.

## 9. Repository Structure

| File / Directory | Description |
|---|---|
| `streamlit_app.py` | Public 10-section WellGuard dashboard |
| `demo_data.py` | Reproducible synthetic demonstration records |
| `requirements.txt` | Python dependencies |
| `public_data/` | Saved aggregate research results |
| `research_source/` | Research scripts and private-data application source; not the public entrypoint |
| `databricks/` | Notebook and SQL reproducibility materials |
| `docs/` | Methodology, data-handling guidance, and references |
| `tests/` | Application and data checks |
| `.streamlit/` | Streamlit configuration |
| `AI_USE_DISCLOSURE.md` | Competition-required AI-use disclosure |

## 10. Limitations and Future Development

Historical failure reporting changes over time, some sensors are incompletely recorded, and source costs are generally estimates rather than actual expenditures. Observed changes around maintenance are not evidence of causal maintenance effects.

The reference model's relative rankings are **not calibrated failure probabilities**. Future work could include prospective validation, integration with operational systems, field-inspection feedback, model monitoring, and verification of realized operational benefits.

Restricted competition datasets and actual well-level source records are not included in this public repository. Synthetic well-level examples are used in the interactive dashboard.

## AI-Use Disclosure

The project's disclosure is provided in [AI_USE_DISCLOSURE.md](AI_USE_DISCLOSURE.md).

---

**DATA TITANS | DSMLC Datathon 2026**

*WellGuard — Predict Risk. Prioritize Reviews. Verify Evidence.*
