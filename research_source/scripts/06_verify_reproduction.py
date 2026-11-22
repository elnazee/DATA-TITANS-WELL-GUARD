"""Compare the raw-data rerun with the public release benchmark."""
import argparse
import json
import math
from pathlib import Path


def differences(expected, actual, key=''):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f'{key}: expected a mapping']
        return [error for name, value in expected.items()
                for error in differences(value, actual.get(name), f'{key}.{name}')]
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        if math.isclose(expected, actual, abs_tol=1e-7, rel_tol=0):
            return []
    elif expected == actual:
        return []
    return [f'{key}: expected {expected!r}, received {actual!r}']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = []
    for shipped, rerun in [('source_benchmark.json', 'aggregate_model_metrics.json'),
                           ('advanced_validation_public.json', 'advanced_validation.json')]:
        expected = json.loads((root / 'results' / shipped).read_text())
        actual = json.loads((args.out_dir / rerun).read_text())
        errors.extend(differences(expected, actual, rerun))
    if errors:
        print('\n'.join(errors))
        raise SystemExit('SOURCE-TO-RESULT VERIFICATION FAIL')
    print('SOURCE-TO-RESULT VERIFICATION PASS: source model, comparators, lag, and policy metrics')


if __name__ == '__main__':
    main()
