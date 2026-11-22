"""Import only the three official Parquet sources into a local, gitignored data/ folder.

Use only with the supplied competition ZIP on an authorized competition computer.
The script does not create any sample wells, simulated signals, or example outcomes.
"""
from __future__ import annotations
from pathlib import Path
import argparse, zipfile, shutil, zlib

NAMES=('daily_production.parquet','well_jobs.parquet','production_failures.parquet')

def extract(archive:Path, target:Path):
    if not archive.is_file(): raise FileNotFoundError(archive)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        found={}
        for member in z.infolist():
            if member.is_dir():continue
            name=Path(member.filename).name
            if name in NAMES:
                if name in found:raise ValueError(f'Ambiguous archive: {name} appears more than once')
                found[name]=member
        missing=set(NAMES)-set(found)
        if missing:raise ValueError(f'Missing in official archive: {sorted(missing)}')
        for name in NAMES:
            out=target/name
            info=found[name]
            if out.exists():
                if out.stat().st_size == info.file_size:
                    crc=0
                    with out.open('rb') as existing:
                        for chunk in iter(lambda: existing.read(1024*1024), b''):
                            crc=zlib.crc32(chunk, crc)
                    if (crc & 0xffffffff) == info.CRC:
                        print('Already verified; preserving source file:', name)
                        continue
                raise FileExistsError(f'Existing {name} failed the ZIP size/checksum test; verify or move it before re-importing: {out}')
            part=out.with_name(out.name + '.importing')
            try:
                with z.open(info) as src,part.open('wb') as dst:
                    shutil.copyfileobj(src,dst,1024*1024)
                if part.stat().st_size!=info.file_size:
                    raise IOError(f'Unexpected extracted size for {name}')
                part.replace(out)
            finally:
                if part.exists(): part.unlink()
            print('Imported',name, f'{out.stat().st_size:,} bytes')
    print('Private competition data imported locally. Do not commit data/ or outputs/.')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--zip',dest='zip_path',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    a=p.parse_args();extract(a.zip_path,a.data_dir)
