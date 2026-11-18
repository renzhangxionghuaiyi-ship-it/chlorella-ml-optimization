"""Reproduce the record-level comparison with RFECV inside each tuning fold."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real
from xgboost import XGBRegressor

from .data import DATA, load_data


SEED = 42
SEARCH_EVALUATIONS = 15
ESTIMATORS = {"XGBoost": XGBRegressor, "LightGBM": LGBMRegressor, "CatBoost": CatBoostRegressor}
MODELS = (*ESTIMATORS, "Ensemble")
TARGETS = ("Biomass", "Total chlorophyll", "Normalized composite objective")
ENSEMBLE_WEIGHTS = (0.5, 0.5)
FEATURES = (
    "Illumination intensity", "Light time", "Dark time", "Cultivation day",
    "Temperature", "pH", "NH4Cl", "NaNO3", "Urea", "MgSO4", "CaCl2",
    "K2HPO4", "KH2PO4", "Trace elements", "Initial inoculum density",
    "Tris", "Acetic acid",
)
SEARCH_SPACES = {
    "XGBoost": [
        Integer(3, 20, name="max_depth"),
        Integer(100, 500, name="n_estimators"),
        Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
        Real(1e-8, 0.1, prior="log-uniform", name="reg_alpha"),
        Real(1e-8, 0.1, prior="log-uniform", name="reg_lambda"),
    ],
    "LightGBM": [
        Integer(3, 20, name="num_leaves"),
        Integer(100, 500, name="n_estimators"),
        Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
        Real(0.5, 1.0, name="feature_fraction"),
        Real(0.5, 1.0, name="bagging_fraction"),
        Integer(1, 10, name="bagging_freq"),
    ],
    "CatBoost": [
        Integer(3, 10, name="depth"),
        Integer(100, 500, name="iterations"),
        Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
        Real(1e-8, 0.1, prior="log-uniform", name="l2_leaf_reg"),
    ],
}
FIXED = {
    "XGBoost": {"random_state": SEED, "objective": "reg:squarederror", "n_jobs": 1},
    "LightGBM": {"random_state": SEED, "objective": "regression",
                 "metric": "rmse", "verbosity": -1, "n_jobs": 1},
    "CatBoost": {"random_seed": SEED, "loss_function": "RMSE", "verbose": False,
                 "thread_count": 1, "allow_writing_files": False},
}


def cross_validation():
    return KFold(n_splits=5, shuffle=True, random_state=SEED)


def target_values(responses, target, training):
    if target < 2:
        return responses[:, target]
    lo = responses[training].min(axis=0)
    hi = responses[training].max(axis=0)
    if np.any(hi == lo):
        raise ValueError("Composite objective requires variation in both training targets.")
    return ((responses - lo) / (hi - lo)).sum(axis=1)


def select_features(x, y, training, model):
    if model == "Ensemble":
        selector_model = XGBRegressor(
            **FIXED["XGBoost"], n_estimators=100, max_depth=6, learning_rate=0.1
        )
    else:
        selector_model = ESTIMATORS[model](**FIXED[model])
    imputer = SimpleImputer(strategy="median").fit(x[training])
    selector = RFECV(estimator=selector_model, step=1, scoring="r2",
                     cv=cross_validation(), n_jobs=1)
    selector.fit(imputer.transform(x[training]), y[training])
    return imputer, selector


def fit_estimator(name, params, training, labels, evaluation=None):
    # The archived XGBoost model used the full tree budget. LightGBM and
    # CatBoost used early stopping in the fits supplied with evaluation data.
    model = ESTIMATORS[name](**FIXED[name], **params)
    if evaluation is not None and name == "LightGBM":
        model.fit(training, labels, eval_set=[evaluation],
                  callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
    elif evaluation is not None and name == "CatBoost":
        model.fit(training, labels, eval_set=evaluation, early_stopping_rounds=50, verbose=False)
    else:
        model.fit(training, labels)
    return model


def scores(y, prediction):
    return {"r2": float(r2_score(y, prediction)),
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "mae": float(mean_absolute_error(y, prediction))}


def fit_candidate(x, responses, train, validation, target, model):
    folds = []
    for tr, val in cross_validation().split(train):
        training, tuning = train[tr], train[val]
        y = target_values(responses, target, training)
        imputer, selector = select_features(x, y, training, model)
        folds.append({
            "training_rows": training,
            "validation_rows": tuning,
            "train_x": selector.transform(imputer.transform(x[training])),
            "val_x": selector.transform(imputer.transform(x[tuning])),
            "train_y": y[training],
            "val_y": y[tuning],
            "features": [FEATURES[i] for i in selector.get_support(indices=True)],
        })
    print(f"{TARGETS[target]} / {model}: selected features in five training folds", flush=True)

    settings = {}
    predictions = {}
    names = ("XGBoost", "CatBoost") if model == "Ensemble" else (model,)
    for name in names:
        space = SEARCH_SPACES[name]
        evaluated = {}

        def objective(candidate):
            key = tuple(candidate)
            if key not in evaluated:
                params = dict(zip((d.name for d in space), candidate))
                evaluated[key] = []
                for fold in folds:
                    fitted = fit_estimator(name, params, fold["train_x"], fold["train_y"],
                                           (fold["val_x"], fold["val_y"]))
                    evaluated[key].append(fitted.predict(fold["val_x"]))
            return -np.mean([r2_score(f["val_y"], p) for f, p in zip(folds, evaluated[key])])

        result = gp_minimize(objective, space, n_calls=SEARCH_EVALUATIONS, random_state=SEED)
        settings[name] = {d.name: v.item() if isinstance(v, np.generic) else v
                          for d, v in zip(space, result.x)}
        predictions[name] = evaluated[tuple(result.x)]

    y = target_values(responses, target, train)
    imputer, selector = select_features(x, y, train, model)
    train_x = selector.transform(imputer.transform(x[train]))
    validation_x = selector.transform(imputer.transform(x[validation]))
    evaluation = (validation_x, y[validation]) if model != "Ensemble" else None
    fitted = [fit_estimator(name, settings[name], train_x, y[train], evaluation) for name in names]
    validation_prediction = np.mean([reg.predict(validation_x) for reg in fitted], axis=0)
    fold_results = []
    for index, fold in enumerate(folds):
        predicted = np.mean([predictions[name][index] for name in names], axis=0)
        fold_results.append({
            "fold": index + 1,
            "training_source_rows": (fold["training_rows"] + 2).tolist(),
            "validation_source_rows": (fold["validation_rows"] + 2).tolist(),
            "retained_features": fold["features"],
            "metrics": scores(fold["val_y"], predicted),
        })

    info = {
        "target": TARGETS[target],
        "model": model,
        "settings": settings,
        "retained_features": [FEATURES[i] for i in selector.get_support(indices=True)],
        "weights": ENSEMBLE_WEIGHTS if model == "Ensemble" else None,
        "actual_tree_counts": {
            name: int(count)
            for name, reg in zip(names, fitted)
            if (count := getattr(reg, "tree_count_", getattr(reg, "n_estimators_", None))) is not None
        },
        "folds": fold_results,
        "cv": {k: float(np.mean([f["metrics"][k] for f in fold_results])) for k in ("r2", "rmse", "mae")},
        "validation": scores(y[validation], validation_prediction),
    }
    return target, model, info, imputer, selector, fitted


def main(data, output, workers):
    frame = load_data(data)
    x = frame.iloc[:, :17].to_numpy(float)
    responses = frame.iloc[:, 17:19].to_numpy(float)
    train_validation, test = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=SEED)
    train, validation = train_test_split(train_validation, test_size=0.2, random_state=SEED)
    assert (len(train), len(validation), len(test)) == (403, 101, 126)
    output.mkdir(parents=True, exist_ok=False)
    trained = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_candidate, x, responses, train, validation, t, m)
                   for t in range(len(TARGETS)) for m in MODELS]
        for future in as_completed(futures):
            t, m, info, imputer, selector, fitted = future.result()
            trained[t, m] = (info, imputer, selector, fitted)
            print(f"{len(trained)}/{len(futures)} complete: {info['target']} / {m}", flush=True)

    winners = {
        t: max(MODELS, key=lambda m: trained[t, m][0]["validation"]["r2"])
        for t in range(len(TARGETS))
    }
    print(f"Validation-selected models: {winners}", flush=True)
    results = []
    score_rows = []
    prediction_rows = []
    for t in range(len(TARGETS)):
        y = target_values(responses, t, train)
        for m in MODELS:
            info, imputer, selector, fitted = trained[t, m]
            held_out = selector.transform(imputer.transform(x[test]))
            predicted = np.mean([reg.predict(held_out) for reg in fitted], axis=0)
            info["test"] = scores(y[test], predicted)
            for split_name, ix in (("training", train), ("validation", validation), ("test", test)):
                part_x = selector.transform(imputer.transform(x[ix]))
                part_predictions = np.mean([reg.predict(part_x) for reg in fitted], axis=0)
                prediction_rows.append(pd.DataFrame({
                    "source_row": ix + 1, "target": info["target"], "model": m, "split": split_name,
                    "actual": y[ix], "predicted": part_predictions, "residual": y[ix] - part_predictions,
                    "selected": m == winners[t],
                }))
            info["selected"] = m == winners[t]
            results.append(info)
            score_rows.append({
                "target": info["target"], "model": m, "selected": info["selected"],
                **{f"{part}_{metric}": score
                   for part in ("cv", "validation", "test") for metric, score in info[part].items()},
            })
    scope = {
        name: [{"parameter": d.name, "low": d.low, "high": d.high, "prior": d.prior}
               for d in dimensions]
        for name, dimensions in SEARCH_SPACES.items()
    }
    report = {
        "data_sha256": sha256(data.read_bytes()).hexdigest(),
        "versions": {pkg: version(pkg) for pkg in (
            "numpy", "pandas", "scikit-learn", "scikit-optimize", "xgboost", "lightgbm", "catboost"
        )},
        "seed": SEED,
        "search_evaluations": SEARCH_EVALUATIONS,
        "search_spaces": scope,
        "fixed_settings": FIXED,
        "features_in_source_order": list(zip(FEATURES, frame.columns[:17])),
        "source_rows": {name: (ix + 2).tolist()
                        for name, ix in (("training", train), ("validation", validation), ("test", test))},
        "models": results,
    }
    (output / "model_settings.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(score_rows).to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame([{"source_row": int(index + 1), "split": part}
                  for part, rows in (("training", train), ("validation", validation), ("test", test))
                  for index in rows]).to_csv(output / "split_assignments.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=DATA)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    main(args.data, args.output, args.workers)
