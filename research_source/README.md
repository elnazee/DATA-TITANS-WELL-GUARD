# Research source — authorized local Python pipeline

This folder contains Python source for WellGuard's data preparation, retrospective modelling, time-aware validation, private-data audits and advanced research extensions. It contains no original competition datasets or private prediction exports.

The public website is deployed from **repository-root `streamlit_app.py`**, not the `app.py` in this source folder. The `app.py` here implements the full well-level dashboard, requiring authorized source Parquet data and locally generated operational/model outputs.

## Key files

- `scripts/02_failure_ranking.py` — original LightGBM ranking pipeline and chronological evaluation.
- `scripts/12_audit_data_leakage.py` — time-window and recorded-label checks.
- `scripts/15_detailed_validation.py` — private replay/validation routines.
- `scripts/16_experimental_model_upgrade.py` — separate candidate-model study.
- `core/validation_story.py` — dashboard research stages and evidence charts.

On an authorized computer with the three original Parquet files under `data/`:

```bash
pip install -r requirements.txt
python scripts/02_failure_ranking.py --data-dir data --out-dir outputs
```

Additional locally generated operational histories and model outputs are needed to run the full local dashboard. See the separate private team package for Windows launch scripts and setup guidance.

**Data-sharing requirement:** Keep all raw Parquet files, well-level prediction exports, locally generated `outputs/` and private keys out of GitHub. The saved `results/` JSON files contain only aggregate analysis. For interpretation and research caveats, see `../docs/METHODOLOGY_AND_VALIDATION.md`.
