# Chlorella sp. Multi-Target Machine Learning Optimization

This repository provides the main analysis notebook for a manuscript on interpretable machine-learning-assisted optimization of *Chlorella* sp. cultivation.

The current public repository contains the notebook:

- `chlorella_ml_optimization.ipynb`

The notebook performs data preprocessing, model training, model comparison, SHAP/PDP interpretation, Pareto-guided candidate selection, and generation of analysis outputs for biomass, total chlorophyll, and a normalized biomass-chlorophyll composite objective.

## Manuscript title

Interpretable machine learning enables Pareto-guided optimization of *Chlorella* cultivation for simultaneous biomass accumulation and total chlorophyll production

## Code purpose

The notebook is designed to:

1. Load and preprocess single-factor *Chlorella* cultivation data.
2. Build prediction targets for:
   - Biomass (mg L^-1)
   - Total chlorophyll (mg L^-1)
   - Biomass + total chlorophyll normalized composite objective
3. Compare XGBoost, LightGBM, CatBoost, and an XGBoost-CatBoost voting ensemble.
4. Select the best model according to validation-set performance.
5. Evaluate model performance using R2, RMSE, MAE, normalized RMSE, and normalized MAE.
6. Generate observed-versus-predicted plots and residual diagnostics.
7. Interpret selected models using SHAP feature importance and partial dependence plots.
8. Perform single-objective and Pareto-guided multi-objective optimization.
9. Export figures, model pipelines, performance tables, and Pareto candidate tables.

## Python version

The notebook was developed and tested with:

- Python 3.12.7

## Main dependencies

Install the main Python dependencies before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn scikit-optimize joblib shap xgboost lightgbm catboost jupyter
```

The main packages used by the notebook are:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `scikit-optimize`
- `joblib`
- `shap`
- `xgboost`
- `lightgbm`
- `catboost`

## Input data

The notebook currently expects the input CSV file to be named:

```text
小球藻单因素优化汇总数据1.csv
```

The dataset is not included in this notebook-only repository. To run the notebook, place the CSV file in the same folder as `chlorella_ml_optimization.ipynb`, or provide the file path through the environment variable `CHLORELLA_DATA_CSV`.

Example on Windows PowerShell:

```powershell
$env:CHLORELLA_DATA_CSV="C:\path\to\小球藻单因素优化汇总数据1.csv"
jupyter notebook
```

Example on macOS/Linux:

```bash
export CHLORELLA_DATA_CSV="/path/to/小球藻单因素优化汇总数据1.csv"
jupyter notebook
```

If your data file uses another name, either rename it to `小球藻单因素优化汇总数据1.csv` or update `DATA_CSV_NAME` in the notebook.

## Running order

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Place the input CSV file next to `chlorella_ml_optimization.ipynb`, or set `CHLORELLA_DATA_CSV`.
4. Open the notebook:

```bash
jupyter notebook chlorella_ml_optimization.ipynb
```

5. Run the notebook cells from top to bottom.

## Generated outputs

After running the notebook, results are written to:

```text
chlorella_results/
```

Typical outputs include:

- `model_comparison_all.csv`: model comparison across targets and algorithms.
- `Table1_model_performance.csv`: model-performance summary.
- `Table2_optimal_conditions.csv`: single-objective optimal conditions.
- `Table3_pareto_strategies.csv`: Pareto-guided strategy summary.
- `pareto_solutions.csv`: Pareto candidate solutions.
- `single_objective_optimization.json`: single-objective optimization results.
- `*_pipeline.pkl`: trained preprocessing and model pipelines.
- `Fig_PCC_heatmap.png`: Pearson correlation heatmap.
- `Fig_data_distribution.png`: input and response distribution figure.
- `Fig_scatter_*.png`: observed-versus-predicted plots.
- `Fig_residual_*.png`: residual diagnostic plots.
- `Fig_SHAP_*.png`: SHAP interpretation figures.
- `Fig_PDP_*.png`: one-dimensional and two-dimensional partial dependence plots.
- `Fig_Pareto_*.png`: Pareto-front visualization.
- `Fig_ML_framework.png`: machine-learning workflow figure.

Some cells may also copy generated outputs into a desktop folder named `chlorella_results_分类` for manual inspection.

## Public GUI

The deployed graphical user interface for *Chlorella* cultivation prediction and candidate screening is available at:

http://8.138.251.204/Chlorella-cultivation-optimization

The GUI is intended as a model-assisted decision-support tool. It should not replace experimental validation.

## Reproducibility notes

- Random seed: `42`
- Optimization day: day 7
- Main models: XGBoost, LightGBM, CatBoost, and XGBoost-CatBoost voting ensemble
- Main interpretation methods: SHAP and partial dependence plots
- Main optimization methods: differential evolution and Pareto-guided weighted objective search

Because the current repository contains only the notebook, reproducibility depends on access to the corresponding input CSV file and the same package versions or compatible package versions.

## Suggested citation in manuscript

The source code used for data preprocessing, CatBoost model training, SHAP and PDP analysis, Pareto-guided optimization, and model-output generation is available at:

https://github.com/renzhangxionghuaiyi-ship-it/chlorella-ml-optimization

