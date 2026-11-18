"""Refit the user-selected Random Forest once, without evaluator data or tuning."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn

from .models import build_model
from .metrics import evaluate, score_candidates
from .train import file_hash, save_configuration, write_json
from .training_handoff import load_feature_dir


def fit_selected(feature_dir, output, jobs=2, seed=42):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new, empty model output directory")
    train, student, sources = load_feature_dir(feature_dir)
    fit = train.labelled
    if set(fit.label.unique()) != {0, 1}:
        raise ValueError("Training requires both candidate classes")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "fitting", "model": "random_forest", "seed": seed, "jobs": jobs,
        "selection_basis": "User selected Random Forest from recorded baseline CV; no new tuning",
        "features": train.features,
        "training_trips": sorted(fit.trip_id.unique().tolist()),
        "training_candidates": len(fit),
        "excluded_training_trips": train.fold_assignments.loc[
            train.fold_assignments.validation_fold.lt(0), "trip_id"].tolist(),
        "input_sha256": {key: file_hash(path) for key, path in sources.items()},
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "evaluator_used_for_fitting_or_selection": False,
    }
    write_json(output / "run.json", metadata)
    train.fold_assignments.to_csv(output / "fold_assignments.csv", index=False)
    model = build_model("random_forest", seed=seed, jobs=jobs)
    save_configuration(output, model, train.features, {})
    print(f"Fitting 200-tree Random Forest: {len(fit):,} candidates, "
          f"{fit.trip_id.nunique()} training trips, jobs={jobs}", flush=True)
    started = perf_counter()
    model.fit(fit[train.features], fit.label.astype(int))
    metadata["training_seconds"] = perf_counter() - started
    print(f"Fit complete in {metadata['training_seconds']:.1f}s; saving model", flush=True)
    temporary = output / "pipeline.pending.joblib"
    joblib.dump(model, temporary, compress=3)
    temporary.replace(output / "pipeline.joblib")
    metadata["model_sha256"] = file_hash(output / "pipeline.joblib")
    started = perf_counter()
    scores = score_candidates(model, student.candidates, train.features)
    metrics, observations = evaluate(student.candidates, student.inventory, scores)
    metrics["inference_seconds"] = perf_counter() - started
    restored = joblib.load(output / "pipeline.joblib")
    check = student.candidates.iloc[:1000]
    np.testing.assert_allclose(score_candidates(model, check, train.features).probability,
                               score_candidates(restored, check, train.features).probability,
                               rtol=0, atol=1e-12)
    scores.to_csv(output / "student_test_candidate_scores.csv", index=False)
    observations.to_csv(output / "student_test_observations.csv", index=False)
    write_json(output / "student_test_metrics.json", metrics)
    metadata["status"] = "complete"
    write_json(output / "run.json", metadata)
    write_json(output / "selection.json", {
        "model": "random_forest", "final_artifact": "pipeline.joblib",
        "model_sha256": metadata["model_sha256"],
        "selection_basis": metadata["selection_basis"], "tuned": False,
    })
    print(f"Student-test Top-1: {metrics['top1_accuracy']:.4%}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="outputs/features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.jobs == 0:
        parser.error("jobs must be nonzero")
    fit_selected(args.feature_dir, args.output, args.jobs, args.seed)


if __name__ == "__main__":
    main()
