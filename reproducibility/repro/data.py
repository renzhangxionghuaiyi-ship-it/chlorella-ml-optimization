"""Shared input schema for the revised Chlorella analyses."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cultivation_records.csv"
DAY_COLUMN = "time(d)"
ID_COLUMNS = (
    "source_row", "experiment", "factor", "level", "gradient", "band",
    "bottle_id", "condition_id", "replicate_id", "original_day_label",
)


def load_data(path=DATA):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(frame.columns[19:]) != ID_COLUMNS or frame.shape != (630, 29):
        raise ValueError("Expected the published 630-row dataset with 19 measurement columns and 10 metadata columns.")
    if not np.isfinite(frame.iloc[:, :19].to_numpy(float)).all():
        raise ValueError("The measurement columns must be complete and finite.")
    if not np.array_equal(frame.source_row, np.arange(1, 631)):
        raise ValueError("Keep the published row order for the record-level split.")
    groups = frame.groupby("bottle_id")
    day_sets = groups[DAY_COLUMN].agg(lambda days: tuple(sorted(days)))
    if len(groups) != 90 or not day_sets.map(lambda days: days == tuple(range(1, 8))).all():
        raise ValueError("Expected 90 flasks measured once on each of days 1 through 7.")
    if frame.condition_id.nunique() != 80 or not groups.condition_id.nunique().eq(1).all():
        raise ValueError("Expected 80 distinct cultivation-condition groups.")
    settings = list(frame.columns[:3]) + list(frame.columns[4:17])
    if frame.groupby(settings, dropna=False).condition_id.nunique().max() != 1:
        raise ValueError("Matching cultivation settings must share a condition ID.")
    replicate_keys = frame[["condition_id", "bottle_id", "replicate_id"]].drop_duplicates()
    if len(replicate_keys) != 90 or replicate_keys.duplicated(["condition_id", "replicate_id"]).any():
        raise ValueError("Each replicate identifier must identify one flask within its condition.")
    if frame[DAY_COLUMN].ne(frame.original_day_label).sum() != 15:
        raise ValueError("Expected 15 corrected day labels.")
    return frame


def write_legacy_input(destination):
    """Reconstruct the historical 19-column input for the original workflow."""
    data = load_data()
    legacy = data.iloc[:, :19].copy()
    legacy[DAY_COLUMN] = data.original_day_label
    destination = Path(destination)
    destination.parent.mkdir(exist_ok=True, parents=True)
    legacy.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination
