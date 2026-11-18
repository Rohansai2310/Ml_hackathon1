"""Export compact measured evaluator results; keep model binaries and predictions local."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def report(model_dir, evaluation_dir, archive):
    model_dir, evaluation_dir, archive = map(Path, [model_dir, evaluation_dir, archive])
    manifest = json.loads((evaluation_dir / "run.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Evaluation must finish before results can be published")
    metrics = json.loads((evaluation_dir / "metrics.json").read_text())
    student = json.loads((model_dir / "student_test_metrics.json").read_text())
    training = json.loads((model_dir / "run.json").read_text())
    table = pd.read_csv(evaluation_dir / "per_trip_metrics.csv", dtype={"trip_id": "string"})
    with_hmm = "hmm_top1_accuracy" in metrics
    archive.mkdir(parents=True, exist_ok=True)
    for filename in ["metrics.json", "per_trip_metrics.csv"]:
        shutil.copy2(evaluation_dir / filename, archive / filename)
    shutil.copy2(evaluation_dir / "run.json", archive / "evaluation_run.json")
    for filename in ["run.json", "selection.json", "configuration.json", "student_test_metrics.json"]:
        shutil.copy2(model_dir / filename, archive / f"model_{filename}")
    frozen_decoder = evaluation_dir / "decoder_configuration.json"
    if frozen_decoder.exists():
        shutil.copy2(frozen_decoder, archive / "decoder_configuration.json")
    if with_hmm:
        route_rows = []
        for trip_id in table.trip_id:
            route = json.loads((evaluation_dir / "predictions" / trip_id / "hmm_route.json").read_text())
            route_rows.append({"trip_id": trip_id, "segments": len(route["route_segments"]),
                "reconstructed_arcs": sum(map(len, route["route_segments"])),
                "breaks": len(route["break_reasons"]),
                "unmatched_points": route["point_arc_ids"].count(-1)})
        pd.DataFrame(route_rows).to_csv(archive / "route_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    positions = np.arange(len(table))
    ax.barh(positions - .25, table.nearest_top1_accuracy * 100, .24, label="Nearest arc", color="#b2bfcc")
    ax.barh(positions, table.top1_accuracy * 100, .24, label="Random Forest", color="#087e86")
    if with_hmm:
        ax.barh(positions + .25, table.hmm_top1_accuracy * 100, .24, label="RF + HMM/Viterbi", color="#d89035")
    ax.set_yticks(positions, table.trip_id)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Observation Top-1 accuracy on valid derived references (%)")
    ax.set_title("Frozen Random Forest | evaluator-test trips")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(archive / "per_trip_accuracy.png", dpi=160)
    plt.close(fig)
    pct = lambda value: f"{value:.2%}"
    lines = [
        "# Random Forest: evaluator-test results", "",
        "Measured after selecting Random Forest. Classifier and decoder settings were fixed before evaluator scoring; no evaluator tuning was used.", "",
        "## Overall results", "",
        "| Measure | Value |", "|---|---:|",
        f"| Evaluator trips | {metrics['trips']} |",
        f"| GPS observations | {metrics['observations']:,} |",
        f"| Valid derived-reference observations | {metrics['known_observations']:,} |",
        f"| Unresolved-reference observations | {metrics['unresolved_observations']:,} |",
        f"| Reference coverage | {pct(metrics['reference_coverage'])} |",
        f"| Correct known-reference observations | {metrics['correct_known_observations']:,} |",
        f"| Pooled observation Top-1 accuracy | **{pct(metrics['top1_accuracy'])}** |",
        f"| Mean per-trip Top-1 accuracy | {pct(metrics['mean_trip_top1_accuracy'])} |",
        f"| Nearest-arc baseline Top-1 | {pct(metrics['nearest_top1_accuracy'])} |",
        f"| Binary candidate Macro F1 | {metrics['macro_f1']:.4f} |",
        f"| Binary candidate log loss | {metrics['log_loss']:.4f} |",
        f"| Known-reference candidate miss rate | {pct(metrics['candidate_miss_rate'])} |",
        f"| Zero-candidate observations (all references) | {metrics['zero_candidate_observations']:,} |",
        "", "Top-1 counts candidate misses and zero-candidate observations with known references as errors. Unresolved references are excluded from accuracy, F1 and log-loss denominators. Macro F1 and log loss are candidate-level binary metrics, not route-level metrics.",
        "", "## Per-trip performance", "",
        "| Trip | GPS points | Known reference | Unresolved | RF Top-1 | Nearest Top-1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if with_hmm:
        index = lines.index("| Binary candidate Macro F1 | " + f"{metrics['macro_f1']:.4f}" + " |")
        lines.insert(index, f"| RF + HMM/Viterbi observation Top-1 | **{pct(metrics['hmm_top1_accuracy'])}** |")
        lines.insert(index + 1, f"| RF + HMM mean per-trip Top-1 | {pct(metrics['mean_trip_hmm_top1_accuracy'])} |")
        lines[-2] += " HMM Top-1 |"
        lines[-1] += "---:|"
    for row in table.itertuples():
        lines.append(f"| {row.trip_id} | {row.observations:,} | {row.known_observations:,} | {row.unresolved_observations:,} | {pct(row.top1_accuracy)} | {pct(row.nearest_top1_accuracy)} |")
        if with_hmm:
            lines[-1] += f" {pct(row.hmm_top1_accuracy)} |"
    lines += [
        "", "![Per-trip classifier comparison](per_trip_accuracy.png)", "",
        "## Training and interpretation", "",
        f"The model was refitted on {len(training['training_trips'])} eligible training trips ({training['training_candidates']:,} candidate rows). It uses 200 trees, seed {training['seed']}, median imputation fitted on eligible training data, and the original baseline classifier settings. Training took {training['training_seconds']:.1f} seconds. Trips 00000033 and 00000054 remain excluded. No student/evaluator rows entered fitting.",
        "",
        f"Previously recorded five-fold CV Top-1 was 85.63% (mean across folds). The refitted model's student-test Top-1 is {pct(student['top1_accuracy'])}. Evaluator Top-1 is pooled across valid-reference observations; these aggregation schemes are different.",
        "",
        "Reference arc IDs were derived by the existing monotone alignment to each .route, with a 50 m reference-validity threshold fixed before evaluation. These are not independent pointwise labels. The candidate search radius is 75 m, so validity filtering makes known-reference candidate recall appear especially high. No evaluator trips were excluded for poor reference coverage; unresolved observations are counted explicitly.",
        "",
        "Predictions for all evaluator trips were saved before the first evaluator .route was read. The saved model and prediction files are SHA-256 tracked. References are used only in the separate evaluation phase. Evaluator outcomes must not be used to retune this benchmark.",
        "", "## Local artifacts and reproduction", "",
        f"- Frozen model: `{model_dir}/pipeline.joblib` (local, ignored by Git).",
        f"- Label-free predictions: `{evaluation_dir}/predictions/<trip_id>/`.",
        f"- Reference-bearing comparisons: `{evaluation_dir}/evaluation/<trip_id>/`.",
        f"- Model SHA-256: `{manifest['model_sha256']}`.", "",
        "```bash",
        "python -m pathsnap.selected_forest \\",
        "  --output outputs/training/random_forest_selected_001 --jobs 2",
        "python -m pathsnap.evaluate_forest \\",
        "  --model-dir outputs/training/random_forest_selected_001 \\",
        "  --data-root .. \\",
        f"  --output {evaluation_dir}" + (" --with-hmm" if with_hmm else ""),
        "```", "",
        "Use a fresh output directory for each run. Obtain the original raw data and rebuild outputs/features first when working from a new clone. Training artifacts and evaluator data are deliberately not uploaded to Git.", "",
    ]
    if with_hmm:
        delta = (metrics['hmm_top1_accuracy'] - metrics['top1_accuracy']) * 100
        change = "improved" if delta >= 0 else "lowered"
        lines += ["## Measured outcome", "",
            f"RF + HMM {change} pooled Top-1 by {abs(delta):.2f} percentage points in this run. All reconstructed segments passed directed-adjacency checks before evaluation. Decoder settings remained fixed; evaluator outcomes were not used to retune them.", ""]
        lines += ["## HMM settings and limits", "",
            "Directed metric shortest paths, partial-arc travel and GPS displacement are used in the transition potential. Empty/disconnected/time-gap segments are explicit; intermediate road arcs are reconstructed. Recorded decoder parameters:", "",
            "```json", json.dumps(manifest['decoder'], indent=2), "```", "",
            "HMM Top-1 measures decoded per-observation choices against the same valid derived references as the classifier baseline. It does not establish an independently verified route-level accuracy score.", ""]
        if frozen_decoder.exists():
            lines += ["The supplied decoder configuration records training-only selection and is bound to the frozen model hash. This evaluator set was already examined; the result is a repeat development benchmark, not a fresh untouched final estimate.", ""]
    (archive / "README.md").write_text("\n".join(lines))
    print(archive / "README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    report(args.model_dir, args.evaluation_dir, args.archive)
