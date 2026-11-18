"""Plot the four completed local baselines from recorded experiment outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINES = {
    "logistic_regression": "Logistic Regression",
    "gaussian_nb": "Gaussian Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def diagnose_ties(run):
    run = Path(run)
    result = {}
    keys = ["trip_id", "observation_id"]
    for name in ["decision_tree", "random_forest"]:
        scores = pd.read_csv(run / name / "cv_candidate_scores.csv",
                             usecols=keys + ["probability"], dtype={key: "string" for key in keys})
        maximum = scores.groupby(keys).probability.transform("max")
        counts = scores.loc[scores.probability.eq(maximum)].groupby(keys).size()
        result[name] = {
            "observations_with_candidates": len(counts),
            "fraction_with_tied_top_score": float(counts.gt(1).mean()),
            "fraction_candidate_scores_exactly_zero_or_one": float(scores.probability.isin([0.0, 1.0]).mean()),
        }
    (run / "score_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def report(run):
    run = Path(run)
    rows = []
    for name, label in BASELINES.items():
        summary = json.loads((run / name / "cv_summary.json").read_text())
        folds = pd.read_csv(run / name / "cv_folds.csv")
        if len(folds) != 5 or set(folds.fold) != set(range(5)):
            raise ValueError(f"Incomplete five-fold results for {name}")
        if not np.isclose(folds.top1_accuracy.mean(), summary["top1_accuracy"]):
            raise ValueError(f"Fold results disagree with summary for {name}")
        rows.append({"model": label, **summary})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(run / "four_model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6.6), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    y = np.arange(len(comparison))
    values = comparison.top1_accuracy.to_numpy() * 100
    spreads = comparison.top1_accuracy_std.to_numpy() * 100
    colors = ["#3973d6", "#8a67c4", "#d79746", "#139b8b"]
    ax.barh(y, values, color=colors, height=0.55, zorder=3)
    ax.errorbar(values, y, xerr=spreads, fmt="none", ecolor="#243447", capsize=5, linewidth=1.5, zorder=4)
    for position, value in zip(y, values):
        ax.text(1.025, position, f"{value:.2f}%", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=16, fontweight="bold", color="#172b4d", clip_on=False)
    ax.set_yticks(y, comparison.model, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Observation-level Top-1 accuracy (%)", fontsize=11, labelpad=12)
    ax.grid(axis="x", color="#dbe3ec", zorder=0)
    ax.tick_params(axis="both", length=0, colors="#34465d")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.04, 0.94, "PathSnap | Four-model comparison", fontsize=21, fontweight="bold", color="#172b4d")
    fig.text(0.04, 0.885, "Same five trip-based folds • Mean accuracy with ±1 fold standard deviation", fontsize=11, color="#52657b")
    fig.text(0.04, 0.07, "Classifier-only predictions against valid, derived reference labels. Two unreliable training trips excluded.", fontsize=10, color="#52657b")
    fig.text(0.04, 0.035, "Cross-validation results; student-testing, Bayesian tuning and HMM evaluation are not included.", fontsize=10, color="#52657b")
    fig.subplots_adjust(left=0.24, right=0.87, top=0.81, bottom=0.2)
    for extension in ["png", "svg"]:
        fig.savefig(run / f"four_model_comparison.{extension}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return comparison


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    cli.add_argument("--diagnose-ties", action="store_true")
    args = cli.parse_args()
    print(report(args.run).to_string(index=False))
    if args.diagnose_ties:
        print(json.dumps(diagnose_ties(args.run), indent=2))
