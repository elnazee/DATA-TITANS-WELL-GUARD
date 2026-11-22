"""Prepare a pandas table for Streamlit/Arrow without duplicate column labels.

Selecting a metric already requested in a generic list must not repeat its
column; genuinely repeated columns in the underlying frame are retained with
unambiguous suffixes, rather than discarded silently.
"""
from __future__ import annotations
from collections import Counter
import pandas as pd


def arrow_safe_view(frame: pd.DataFrame, requested: list[str] | None = None) -> pd.DataFrame:
    if requested is None:
        positions = list(range(len(frame.columns)))
    else:
        # Deduplicate the REQUEST; do not suppress distinct source columns.
        wanted = list(dict.fromkeys(c for c in requested if c in frame.columns))
        positions = [i for col in wanted for i, name in enumerate(frame.columns) if name == col]
    result = frame.iloc[:, positions].copy()
    seen: Counter[str] = Counter()
    labels = []
    for original in result.columns:
        name = str(original)
        seen[name] += 1
        labels.append(name if seen[name] == 1 else f"{name} [source duplicate {seen[name]}]")
    result.columns = labels
    assert result.columns.is_unique, 'Display labels must be distinct for Arrow conversion'
    return result
