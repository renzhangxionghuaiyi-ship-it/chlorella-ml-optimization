"""Nested CatBoost evaluation by flask and by cultivation condition."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .data import DATA, load_data
from .record_level import SEED, TARGETS, scores, target_values


GRID = (
    (250, 3, .035, 6), (500, 3, .035, 6), (300, 4, .05, 4), (500, 5, .04, 3),
    (350, 5, .06, 8), (350, 6, .05, 5), (500, 6, .035, 8), (300, 7, .05, 8),
)
META = ["source_row", "experiment", "factor", "level", "gradient", "band",
        "condition_id", "bottle_id", "replicate_id"]
OUTER_COLUMNS = {"condition": "condition_id", "bottle": "bottle_id"}


def regressor(candidate):
    iterations, depth, rate, l2 = candidate
    return CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=rate,
                             l2_leaf_reg=l2, verbose=False, allow_writing_files=False,
                             random_seed=SEED, thread_count=1)


def evaluate_fold(task):
    x, raw_y, conditions, train, test, outer_id = task
    inner = [(train[a], train[b]) for a, b in GroupKFold(n_splits=5).split(x[train], groups=conditions[train])]
    for fit_rows, score_rows in inner:
        assert set(conditions[fit_rows]).isdisjoint(conditions[score_rows])
        assert not np.intersect1d(test, np.r_[fit_rows, score_rows]).size

    predictions, tuning = [], []
    for target_idx, target in enumerate(TARGETS):
        inner_y = [target_values(raw_y, target_idx, fit_rows) for fit_rows, _ in inner]
        means = []
        for candidate in GRID:
            values = []
            for (fit_rows, score_rows), y in zip(inner, inner_y):
                model = regressor(candidate).fit(x[fit_rows], y[fit_rows])
                values.append(r2_score(y[score_rows], model.predict(x[score_rows])))
            means.append(float(np.mean(values)))
        best = GRID[int(np.argmax(means))]
        y = target_values(raw_y, target_idx, train)
        fit = regressor(best).fit(x[train], y[train])
        predictions.append(pd.DataFrame({"source_row": test + 1, "target": target,
                                         "actual": y[test], "predicted": fit.predict(x[test]),
                                         "outer_fold": outer_id}))
        tuning.append({"outer_fold": outer_id, "target": target,
                       "selected": best, "inner_mean_r2": means})
    return predictions, tuning


def grouped_summary(predictions):
    metrics = []
    for target, frame in predictions.groupby("target", sort=False):
        for band, subset in [("All", frame), *frame.groupby("band")]:
            metrics.append({"target": target, "band": band, "n": len(subset),
                            **scores(subset.actual, subset.predicted)})
    return pd.DataFrame(metrics)


def run(data, output, protocol, workers):
    frame = load_data(data)
    x = frame.iloc[:, :17].to_numpy(float)
    raw_y = frame.iloc[:, 17:19].to_numpy(float)
    conditions = frame.condition_id.to_numpy()
    outer_ids = frame[OUTER_COLUMNS[protocol]].to_numpy()
    tasks, assignments = [], []
    for train, test in LeaveOneGroupOut().split(x, groups=outer_ids):
        outer_id = int(outer_ids[test[0]])
        assert not np.intersect1d(frame.bottle_id.iloc[train], frame.bottle_id.iloc[test]).size
        if protocol == "condition":
            assert not np.intersect1d(conditions[train], conditions[test]).size
        tasks.append((x, raw_y, conditions, train, test, outer_id))
        assignments.append(dict(outer_fold=outer_id, train_rows=len(train), test_rows=len(test),
                                train_flasks=frame.bottle_id.iloc[train].nunique(),
                                test_flasks=frame.bottle_id.iloc[test].nunique()))
    assert len(tasks) == (80 if protocol == "condition" else 90)
    destination = Path(output)
    destination.mkdir(exist_ok=False, parents=True)
    rows, tuning = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (fold_rows, fold_tuning) in enumerate(pool.map(evaluate_fold, tasks), 1):
            rows.extend(fold_rows)
            tuning.extend(fold_tuning)
            if index % 10 == 0 or index == len(tasks):
                print(f"{protocol}: {index}/{len(tasks)} folds complete", flush=True)
    predicted = pd.concat(rows, ignore_index=True).merge(frame[META], on="source_row", validate="many_to_one")
    predicted = predicted.sort_values(["target", "source_row"])
    assert len(predicted) == 630 * 3 and not predicted.duplicated(["target", "source_row"]).any()
    predicted.to_csv(destination / "predictions.csv", index=False)
    pd.DataFrame(assignments).to_csv(destination / "folds.csv", index=False)
    grouped_summary(predicted).to_csv(destination / "metrics.csv", index=False)
    metadata = {"protocol": protocol, "data_sha256": sha256(Path(data).read_bytes()).hexdigest(),
                "seed": SEED, "grid_columns": ["iterations", "depth", "learning_rate", "l2_leaf_reg"],
                "grid": GRID, "inner_splits": 5, "inner_group": "condition_id", "tuning": tuning}
    (destination / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=DATA)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--protocol", choices=OUTER_COLUMNS, required=True)
    cli.add_argument("--workers", type=int, default=6)
    args = cli.parse_args()
    run(args.data, args.output, args.protocol, args.workers)
