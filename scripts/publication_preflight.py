"""Lightweight public-submission preflight. Supplement with manual review."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[1]
ignore={'.git','__pycache__','.venv','venv','.pytest_cache','.ipynb_checkpoints'}
forbidden_extensions={'.parquet','.pkl','.pickle','.joblib','.db','.sqlite','.pem','.key'}
required=['streamlit_app.py','demo_data.py','requirements.txt','README.md','public_data/benchmark.json','databricks/WellGuard_Databricks_Reproducible_Pipeline.ipynb','research_source/scripts/02_failure_ranking.py']
errors=[]
for rel in required:
    if not (ROOT/rel).is_file():errors.append(f'Missing required file: {rel}')
for f in ROOT.rglob('*'):
    if not f.is_file() or any(p in ignore for p in f.relative_to(ROOT).parts):continue
    rel=f.relative_to(ROOT)
    parts=set(rel.parts)
    if parts & {'data','outputs','logs'}:errors.append(f'Forbidden private directory: {rel}')
    if f.suffix.lower() in forbidden_extensions:errors.append(f'Forbidden private/binary suffix: {rel}')
    if f.name in {'secrets.toml','.env'} or f.name.startswith('.env.'):errors.append(f'Forbidden credentials: {rel}')
    if 'private_' in f.name.lower() and f.suffix.lower() in {'.csv','.json','.parquet'}:errors.append(f'Suspected private output: {rel}')
    if f.stat().st_size>15*1024*1024:errors.append(f'Review unexpectedly large file: {rel}')
    if f.suffix.lower()=='.ipynb':
        try:
            nb=json.loads(f.read_text(encoding='utf-8'))
            for i,c in enumerate(nb.get('cells',[])):
                if c.get('cell_type')=='code' and (c.get('outputs') or c.get('execution_count') is not None):
                    errors.append(f'Notebook code cell {i} contains outputs/execution_count: {rel}')
        except Exception as exc:errors.append(f'Unreadable notebook: {rel}: {exc}')
if errors:
    print('FAIL: public file review found:')
    for e in errors:print(' -',e)
    sys.exit(1)
print('PASS: automated file-list and notebook-output checks; manually review file content and permissions before publishing.')
