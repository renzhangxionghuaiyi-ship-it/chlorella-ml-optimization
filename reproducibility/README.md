# Chlorella cultivation analysis for the revised manuscript

`chlorella_ml_optimization_revised.ipynb` presents the data checks, model comparison, grouped cross-validation, and figure source values for the revision. `chlorella_original_analysis.ipynb` runs the historical analysis used for SHAP, partial dependence, and Pareto candidate selection.

## Run the notebooks

Use Python 3.12 and install the pinned dependencies from this directory.

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `chlorella_ml_optimization_revised.ipynb` and run all cells. Reference mode loads the saved fits' predictions, verifies them against the reported metrics, and produces the tables and vector plots in `outputs/reference_report/`. It requires no network or private files.

For a full refit, set `CHLORELLA_RUN_MODE=fresh` before launching Jupyter or set `mode = "fresh"` in the notebook setup cell. Fresh runs write to `outputs/record`, `outputs/condition`, and `outputs/bottle`. Set `CHLORELLA_OUTPUT_DIR` to a new directory to rerun, or run each command separately with its own output directory. `CHLORELLA_WORKERS` defaults to six. Full model fitting can take several minutes.

```bash
python -m repro.record_level --output outputs/record --workers 6
python -m repro.grouped --protocol condition --output outputs/condition --workers 6
python -m repro.grouped --protocol bottle --output outputs/bottle --workers 6
```

Open `chlorella_original_analysis.ipynb` for the original workflow. It reconstructs the original day labels from the dataset and saves its outputs to `outputs/original/`. Use the original workflow to reproduce the provenance of the manuscript's model interpretation and the conditions selected for the separate laboratory experiment.

## Data

`data/cultivation_records.csv` contains 630 daily observations from 90 flasks, measured once per day on days 1–7. Observations from different flasks may have the same combination of cultivation settings. The verified grouping contains 80 condition IDs. The first 17 columns retain the original predictor order and units. They are followed by biomass and total chlorophyll in mg L⁻¹.

| Field | Meaning |
|---|---|
| `source_row` | One-based data-row index in the archived analysis order. Keep this order to reproduce the record-level split. |
| `experiment`, `factor`, `level`, `gradient`, `band` | Experimental block, varied factor and its tested level, gradient rank, and lower/intermediate/upper position. |
| `condition_id` | Group of flasks sharing cultivation settings, verified against the archived modeling records and the experiment design table. Cultivation day is a predictor and is excluded from this grouping. |
| `bottle_id` | Identifier of one independently cultivated flask. The same ID appears on all seven daily observations for that flask. |
| `replicate_id` | Flask number within the condition group. This distinguishes separately cultivated flasks sharing a setting, including shared settings across experimental blocks. |
| `original_day_label` | Original day before label correction. Fifteen labels differed from the verified sampling order. This column is used only to reconstruct the historical input. |

Metadata columns are excluded from the model predictors. The separate three-strategy laboratory validation is reported in the manuscript and is outside the 630-record modeling dataset.

## Evaluation and outputs

- **Record-level reanalysis.** Two seeded splits reproduce the 403 training, 101 validation, and 126 held-out test records. Within each of five tuning folds, imputation, RFECV, and the composite target scaling use the fold's training portion. The final fit uses training data and the separate validation set for applicable early stopping. The selected model has the highest validation R². The test set provides the held-out performance. The ensemble weights are 0.5 and 0.5.
- **Leave-one-condition-out.** All records from every flask with the selected cultivation setting are held out together. Five-fold tuning within the outer training data is grouped by condition. There are 80 outer folds.
- **Leave-one-bottle-out.** Seven records from a single flask are held out. The same setting can appear in other training flasks. Five-fold tuning within the outer training data is grouped by condition. There are 90 outer folds.

Each grouped protocol produces 630 out-of-fold predictions per target. Metrics are calculated after pooling these predictions. RMSE and MAE are in mg L⁻¹ for the individual targets and dimensionless for the normalized composite target. For the plotting percentages in Figure 4, test RMSE and MAE are divided by the full observed range of the corresponding target. The composite range uses the training-set scaling.

The report folder includes source values for Figures 4–6, the search spaces and fitted settings for Table S2, the record-level scores for Table S3, and the grouped performance metrics. Fresh runs additionally save membership for each split or outer fold, per-record predictions, and search results. The notebook reproduces the statistical quantities and supplies editable plots. The assembled journal artwork uses the manuscript's page layout.
