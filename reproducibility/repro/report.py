"""Verify published outputs and export the values used in the revision."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import ROOT
from .record_level import SEED, scores


REFERENCE = ROOT / "reference"


def load_results(mode, folder):
    if mode == "reference":
        metrics = pd.read_csv(REFERENCE / "record_metrics.csv")
        predictions = pd.read_csv(REFERENCE / "record_predictions.csv")
        settings = REFERENCE / "record_model_settings.json"
        grouped = {k: pd.read_csv(REFERENCE / f"{k}_predictions.csv") for k in ("condition", "bottle")}
    else:
        metrics = pd.read_csv(folder / "record" / "metrics.csv")
        predictions = pd.read_csv(folder / "record" / "predictions.csv")
        settings = folder / "record" / "model_settings.json"
        grouped = {k: pd.read_csv(folder / k / "predictions.csv") for k in ("condition", "bottle")}
    return metrics, predictions, json.loads(settings.read_text(encoding="utf-8")), grouped


def key_results(data, metrics, predictions, grouped):
    validation_train, test = train_test_split(data.source_row.to_numpy(), test_size=.2, random_state=SEED)
    train, validation = train_test_split(validation_train, test_size=.2, random_state=SEED)
    indices = {"training": set(train), "validation": set(validation), "test": set(test)}
    assert tuple(map(len, indices.values())) == (403, 101, 126)
    indexed = metrics.set_index(["target", "model"])
    winners = metrics.loc[metrics.groupby("target").validation_r2.idxmax()]
    assert winners.model.eq("CatBoost").all()
    assert indexed.loc[list(zip(winners.target, winners.model)), "selected"].all()
    assert metrics.selected.sum() == len(winners) == 3
    assert len(predictions) == 630 * 12
    assert not predictions.duplicated(["source_row", "target", "model"]).any()
    for (target, model, split), rows in predictions.groupby(["target", "model", "split"]):
        assert set(rows.source_row) == indices[split]
        actual = scores(rows.actual, rows.predicted)
        if split in ("validation", "test"):
            for metric, result in actual.items():
                np.testing.assert_allclose(result, indexed.loc[(target, model), f"{split}_{metric}"], rtol=1e-10, atol=1e-10)
    published_scores = pd.read_csv(REFERENCE / "record_metrics.csv").set_index(["target", "model"]).sort_index()
    values = indexed.sort_index()[published_scores.select_dtypes("number").columns]
    np.testing.assert_allclose(values, published_scores[values.columns], rtol=1e-9, atol=1e-10)

    grouped_rows = []
    for protocol, count, column in [("condition", 80, "condition_id"), ("bottle", 90, "bottle_id")]:
        held_out = grouped[protocol]
        assert len(held_out) == 630 * 3 and held_out.outer_fold.nunique() == count
        assert not held_out.duplicated(["source_row", "target"]).any()
        row_meta = data.set_index("source_row").loc[held_out.source_row]
        assert np.array_equal(held_out.outer_fold, row_meta[column])
        assert np.array_equal(held_out.condition_id, row_meta.condition_id)
        assert np.array_equal(held_out.bottle_id, row_meta.bottle_id)
        assert held_out.groupby("target").size().eq(630).all()
        expected = pd.read_csv(REFERENCE / f"{protocol}_metrics.csv").query("band == 'All'").set_index("target")
        for target, frame in held_out.groupby("target"):
            observed = scores(frame.actual, frame.predicted)
            np.testing.assert_allclose(list(observed.values()), expected.loc[target, list(observed)].astype(float), rtol=1e-9, atol=1e-10)
            grouped_rows.append(dict(protocol=protocol, target=target, folds=count, **observed))
    return winners.sort_values("target"), pd.DataFrame(grouped_rows)


def export_tables(metrics, settings, grouped_metrics, output):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    rows = []
    for model in settings["models"]:
        rows.append({"target": model["target"], "model": model["model"],
                     "settings": json.dumps(model["settings"], sort_keys=True),
                     "retained_features": ", ".join(model["retained_features"]),
                     "ensemble_weights": model["weights"]})
    pd.DataFrame(rows).to_csv(output / "table_s2_model_settings.csv", index=False)
    search_rows = [dict(model=model, **p) for model, params in settings["search_spaces"].items() for p in params]
    pd.DataFrame(search_rows).to_csv(output / "table_s2_search_space.csv", index=False)
    metrics.to_csv(output / "table_s3_record_performance.csv", index=False)
    grouped_metrics.to_csv(output / "grouped_performance.csv", index=False)
