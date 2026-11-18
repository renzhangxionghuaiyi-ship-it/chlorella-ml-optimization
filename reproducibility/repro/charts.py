"""Figures 4–6 from the public record-level metrics and predictions."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .record_level import TARGETS


COLORS = {"XGBoost": "#0072B5", "LightGBM": "#20854E", "CatBoost": "#BC3C29", "Ensemble": "#7876B1"}
MODELS = ("CatBoost", "Ensemble", "LightGBM", "XGBoost")
LABELS = ("Biomass", "Total chlorophyll", "Composite")
UNITS = (r"(mg L$^{-1}$)", r"(mg L$^{-1}$)", "(dimensionless)")
DEVELOPMENT = "#5586AC"
TEST = "#D67560"


def style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, folder, name):
    file = folder / f"{name}.pdf"
    fig.savefig(file, bbox_inches="tight")
    return file


def normalized_metrics(metrics, predictions):
    outputs = predictions[predictions.model.eq("CatBoost")]
    ranges = outputs.groupby("target").actual.agg(lambda y: y.max() - y.min())
    result = metrics.copy()
    for column in ("rmse", "mae"):
        result[f"test_n{column}"] = 100 * result[f"test_{column}"] / result.target.map(ranges)
    return result


def model_comparison(metrics):
    fig, plots = plt.subplots(2, 2, figsize=(7.1, 4.5))
    specs = (("validation_r2", "Validation R² (n = 101)", "R²"),
             ("test_r2", "Held-out test R² (n = 126)", "R²"),
             ("test_nrmse", "Held-out test nRMSE (n = 126)", "nRMSE (%)"),
             ("test_nmae", "Held-out test nMAE (n = 126)", "nMAE (%)"))
    x = np.arange(3)
    for letter, ax, (column, title, ylabel) in zip("abcd", plots.flat, specs):
        values = metrics.pivot(index="target", columns="model", values=column).loc[list(TARGETS), list(COLORS)]
        offsets = np.arange(4) * .18 - .27
        for (model, color), offset in zip(COLORS.items(), offsets):
            ax.bar(x + offset, values[model], .166, color=color, label=model, zorder=3)
        is_r2 = column.endswith("r2")
        winners = values.to_numpy().argmax(axis=1) if is_r2 else values.to_numpy().argmin(axis=1)
        ax.scatter(x + offsets[winners], values.to_numpy()[x, winners] + (.02 if is_r2 else .5),
                   marker="*", s=80, color="black", zorder=5)
        ax.set(xticks=x, xticklabels=LABELS, ylabel=ylabel, title=title)
        if is_r2:
            ax.set(ylim=(.48, 1.015), yticks=np.arange(.5, 1.01, .1))
        ax.grid(axis="y", alpha=.35, linestyle="--")
        ax.text(-.13, 1.02, letter, transform=ax.transAxes, weight="bold")
    handles, labels = plots[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .99), ncol=4, frameon=False)
    star = Line2D([], [], marker="*", linestyle="none", color="black", label="Best value for each target")
    fig.legend(handles=[star], loc="lower center", bbox_to_anchor=(.5, .015), frameon=False)
    fig.subplots_adjust(left=.11, right=.98, top=.86, bottom=.14, wspace=.35, hspace=.55)
    return fig


def observed_predicted(predictions):
    fig, axes = plt.subplots(4, 3, figsize=(6.2, 8.45))
    points = predictions[predictions.split.isin(("training", "test"))]
    for col, (target, label, unit) in enumerate(zip(TARGETS, LABELS, UNITS)):
        target_points = points[points.target.eq(target)]
        extrema = target_points[["actual", "predicted"]].to_numpy()
        lo, hi = extrema.min(), extrema.max()
        pad = .045 * (hi - lo)
        limit = (lo - pad, hi + pad)
        axes[0, col].set_title(f"{label} {unit}", pad=11)
        for row, model in enumerate(MODELS):
            ax = axes[row, col]
            values = target_points[target_points.model.eq(model)]
            ax.plot(limit, limit, "--", color="#263846", lw=.8)
            slope, intercept = np.polyfit(values.actual, values.predicted, 1)
            ax.plot(limit, slope * np.asarray(limit) + intercept, color="#705EAE", lw=.95)
            stats = []
            for part, color, prefix in [("training", DEVELOPMENT, "Tr"), ("test", TEST, "Te")]:
                sub = values[values.split.eq(part)]
                ax.scatter(sub.actual, sub.predicted, c=color, s=8, alpha=.66, lw=0)
                stats.append(f"{prefix} R² {r2_score(sub.actual, sub.predicted):.3f}  RMSE {mean_squared_error(sub.actual, sub.predicted)**.5:.3g}")
            ax.text(.035, .95, "\n".join(stats), transform=ax.transAxes, fontsize=5, va="top",
                    bbox=dict(facecolor="white", edgecolor="#D9E0E4", pad=1.5))
            ax.set(xlim=limit, ylim=limit, aspect="equal")
            ax.tick_params(labelsize=6)
            ax.text(-.14, 1.02, "abcdefghijkl"[row*3+col], transform=ax.transAxes, weight="bold")
            ax.grid(alpha=.26)
            if col == 0:
                ax.set_ylabel(model, weight="bold")
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=l, markersize=4)
               for c, l in [(DEVELOPMENT, "Training"), (TEST, "Held-out test")]]
    handles.extend([Line2D([], [], color="#263846", ls="--", label="y = x"),
                    Line2D([], [], color="#705EAE", label="Linear fit")])
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center")
    fig.supxlabel("Observed", fontsize=8, y=.035)
    fig.supylabel("Predicted", fontsize=8, x=.02)
    fig.subplots_adjust(left=.12, right=.99, top=.905, bottom=.075, wspace=.11, hspace=.27)
    return fig


def residual_diagnostics(predictions, metrics):
    fig = plt.figure(figsize=(7.15, 11.5))
    grid = fig.add_gridspec(6, 2, left=.08, right=.99, top=.93, bottom=.06, wspace=.28, hspace=.65)
    order = ("XGBoost", "LightGBM", "CatBoost", "Ensemble")
    lookup = metrics.set_index(["model", "target"]).test_rmse
    for pos, model in enumerate(order):
        for i, (target, label, unit) in enumerate(zip(TARGETS, LABELS, UNITS)):
            row, col = i + 3*(pos//2), pos % 2
            cell = grid[row, col].subgridspec(1, 2, width_ratios=(5, 1), wspace=.04)
            ax, marginal = fig.add_subplot(cell[0, 0]), fig.add_subplot(cell[0, 1])
            rows = predictions[predictions.target.eq(target) & predictions.model.eq(model)]
            rmse = lookup.at[(model, target)]
            bins = np.linspace(rows.residual.min(), rows.residual.max(), 22)
            for is_test, color in [(False, DEVELOPMENT), (True, TEST)]:
                sub = rows[rows.split.eq("test").eq(is_test)]
                ax.scatter(sub.predicted, sub.residual, color=color, s=8, alpha=.65, linewidths=0)
                marginal.hist(sub.residual, bins=bins, orientation="horizontal", color=color, alpha=.55)
            for panel in (ax, marginal):
                panel.axhspan(-rmse, rmse, color="#CFDFE9", alpha=.48, zorder=0)
                panel.axhline(0, lw=.8, color="#263846", ls="--")
                panel.tick_params(labelsize=6)
            marginal.sharey(ax)
            marginal.tick_params(axis="y", left=False, labelleft=False)
            marginal.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, integer=True))
            ax.set_title(f"{model} | {label}", loc="left", fontsize=7)
            ax.set_xlabel(f"Predicted {label.lower()} {unit}", fontsize=6)
            if col == 0:
                ax.set_ylabel("Residual", fontsize=7)
            marginal.set_xlabel("Count", fontsize=6)
            ax.text(.04, .82, f"±1 test RMSE = {rmse:.3g}", transform=ax.transAxes, fontsize=5.6,
                    bbox=dict(facecolor="white", edgecolor="#D9E0E4", pad=1.2))
            ax.text(-.15, 1.04, "abcdefghijkl"[row*2+col], transform=ax.transAxes, weight="bold")
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=l, markersize=4)
               for c, l in [(DEVELOPMENT, "Development (training + validation)"), (TEST, "Held-out test")]]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper center")
    return fig


def render(predictions, metrics, output):
    style()
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    figure_metrics = normalized_metrics(metrics, predictions)
    figure_metrics.to_csv(output / "figure_04_source.csv", index=False)
    predictions.to_csv(output / "figures_05_06_source.csv", index=False)
    plots = [model_comparison(figure_metrics), observed_predicted(predictions), residual_diagnostics(predictions, metrics)]
    for number, figure in zip((4, 5, 6), plots):
        save(figure, output, f"figure_{number:02}")
    return plots
