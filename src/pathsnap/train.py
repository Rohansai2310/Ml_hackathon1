"""Run with: python -m pathsnap.train --help."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import load_pair
from .metrics import evaluate, score_candidates
from .models import MODELS, build_model, suggest_params


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            # E.g. SimpleImputer's NaN missing-value sentinel, not a metric.
            return str(item)
        return item
    path.write_text(json.dumps(safe(value), indent=2, default=str, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FoldPlan:
    """Materialise one fold at a time instead of retaining five copies of millions of rows."""

    def __init__(self, train, assignments):
        self.train, self.assignments = train, assignments

    def __len__(self):
        return 5

    def __iter__(self):
        for fold in range(5):
            yield self[fold]

    def __getitem__(self, fold):
        train, assignments = self.train, self.assignments
        fit_trips = set(assignments.loc[assignments.validation_fold.ge(0) & assignments.validation_fold.ne(fold), "trip_id"])
        val_trips = set(assignments.loc[assignments.validation_fold.eq(fold), "trip_id"])
        fit = train.candidates.loc[train.candidates.label.notna() & train.candidates.trip_id.isin(fit_trips)].copy()
        val = train.candidates.loc[train.candidates.trip_id.isin(val_trips)].copy()
        val_inventory = train.inventory.loc[train.inventory.trip_id.isin(val_trips)].copy()
        if set(fit.label.unique()) != {0, 1}:
            raise ValueError(f"Fold {fold} training data must contain both labels")
        if not val_inventory.reference_status.eq("known").any():
            raise ValueError(f"Fold {fold} validation data has no resolved reference observations")
        return fit, val, val_inventory


def make_folds(train, seed):
    if train.fold_assignments is not None:
        assignments = train.fold_assignments.copy()
    else:
        inventory = train.inventory.reset_index(drop=True)
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        rows = []
        for fold, (_, indices) in enumerate(splitter.split(inventory, groups=inventory.trip_id)):
            rows.extend({"trip_id": trip, "validation_fold": fold} for trip in sorted(set(inventory.iloc[indices].trip_id)))
        assignments = pd.DataFrame(rows)
    return FoldPlan(train, assignments), assignments


def cross_validate(name, train, folds, seed, jobs, params=None):
    results, predictions = [], []
    for fold, (fit, val, inventory) in enumerate(folds):
        model = build_model(name, seed, jobs, params)
        start = perf_counter()
        model.fit(fit[train.features], fit.label.astype(int))
        training_time = perf_counter() - start
        start = perf_counter()
        scores = score_candidates(model, val, train.features)
        inference_time = perf_counter() - start
        metrics, _ = evaluate(val, inventory, scores)
        print(f"  {name} fold {fold}: Top-1={metrics['top1_accuracy']:.4f}, fit={training_time:.1f}s", flush=True)
        results.append({"fold": fold, **metrics, "training_seconds": training_time, "inference_seconds": inference_time})
        predictions.append(scores.assign(fold=fold))
    table = pd.DataFrame(results)
    summary = {
        key: (float(table[key].mean()) if table[key].notna().any() else None)
        for key in ["top1_accuracy", "top1_accuracy_reference_present", "candidate_miss_rate", "macro_f1", "log_loss", "training_seconds", "inference_seconds"]
    }
    summary["top1_accuracy_std"] = float(table.top1_accuracy.std(ddof=0))
    return summary, table, pd.concat(predictions, ignore_index=True)


def save_configuration(directory, model, features, params):
    write_json(directory / "configuration.json", {
        "features": features, "tuning_parameters": params or {},
        "pipeline_parameters": model.get_params(deep=True),
        "score": "Positive-class probability per candidate; not normalised across candidates",
    })


def refit_and_evaluate(name, train, test, directory, seed, jobs, params=None):
    model = build_model(name, seed, jobs, params)
    fit = train.labelled
    start = perf_counter()
    model.fit(fit[train.features], fit.label.astype(int))
    training_time = perf_counter() - start
    joblib.dump(model, directory / "pipeline.joblib")
    save_configuration(directory, model, train.features, params)
    start = perf_counter()
    scores = score_candidates(model, test.candidates, test.features)
    inference_time = perf_counter() - start
    metrics, observations = evaluate(test.candidates, test.inventory, scores)
    restored = joblib.load(directory / "pipeline.joblib")
    restored_scores = score_candidates(restored, test.candidates, test.features)
    if not np.allclose(scores.probability, restored_scores.probability, rtol=0, atol=1e-12):
        raise RuntimeError("Saved model did not reproduce predictions")
    scores.to_csv(directory / "student_test_candidate_scores.csv", index=False)
    observations.to_csv(directory / "student_test_observations.csv", index=False)
    metrics.update(training_seconds=training_time, inference_seconds=inference_time)
    write_json(directory / "student_test_metrics.json", metrics)
    return metrics


def plot_comparison(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["macro_f1", "top1_accuracy", "log_loss", "inference_seconds"]:
        fig, ax = plt.subplots(figsize=(11, 6))
        table.set_index("model")[[f"cv_{metric}", f"test_{metric}"]].plot.bar(ax=ax)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"CV fold mean and student-testing {metric.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=150)
        plt.close(fig)


def run(args):
    if args.trials < 1 or args.jobs == 0:
        raise ValueError("Trials must be positive and jobs must be nonzero")
    selected_models = args.models or MODELS
    if len(set(selected_models)) != len(selected_models):
        raise ValueError("Model names must be unique")
    if not args.baseline_only and set(selected_models) != set(MODELS):
        raise ValueError("Final selection requires all five models; use --baseline-only for a separate model run")
    if args.feature_dir:
        from .training_handoff import load_feature_dir, FEATURE_DEFINITIONS
        if any([args.train, args.student_test, args.train_inventory, args.student_test_inventory, args.features]):
            raise ValueError("Use --feature-dir or CSV input arguments, not both")
        train, test, source_files = load_feature_dir(args.feature_dir, args.allow_partial_data)
        features_spec = {name: FEATURE_DEFINITIONS[name] for name in train.features}
    else:
        if not all([args.train, args.student_test, args.train_inventory, args.student_test_inventory, args.features]):
            raise ValueError("Supply --feature-dir or all five CSV input arguments")
        features_spec = json.loads(Path(args.features).read_text())
        source_files = {key: getattr(args, key) for key in ["train", "student_test", "train_inventory", "student_test_inventory", "features"]}
    if not isinstance(features_spec, dict) or not features_spec or not all(
        isinstance(v, str) and v.strip() for v in features_spec.values()
    ):
        raise ValueError("Features JSON must map feature names to nonempty descriptions")
    reserved = {"trip_id", "observation_id", "candidate_id", "label", "reference_candidate_id", "reference_status"}
    if reserved.intersection(features_spec):
        raise ValueError("Identifiers and reference columns cannot be model features")
    if not args.feature_dir:
        train, test = load_pair(args.train, args.student_test, args.train_inventory, args.student_test_inventory,
                                list(features_spec), args.allow_partial_data)
    folds, assignments = make_folds(train, args.seed)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory must be empty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {key: file_hash(path) for key, path in source_files.items()}
    if args.resume:
        previous = json.loads((output / "run.json").read_text())
        if previous["input_sha256"] != input_hashes or previous["seed"] != args.seed or previous["features"] != features_spec:
            raise ValueError("Resume input data, seed or features differ from the original run")
        old_folds = pd.read_csv(output / "fold_assignments.csv", dtype={"trip_id": "string"})
        if not old_folds.astype(str).equals(assignments.astype(str)):
            raise ValueError("Resume folds differ from the original run")
    else:
        assignments.to_csv(output / "fold_assignments.csv", index=False)
    metadata = {**vars(args), "input_sha256": input_hashes,
        "features": features_spec, "python": platform.python_version(),
        "input_files": source_files,
        "excluded_training_trips": assignments.loc[assignments.validation_fold.lt(0), "trip_id"].tolist(),
        "selection_metric": "Mean fold observation Top-1 accuracy, counting candidate misses as errors"}
    write_json(output / ("resume.json" if args.resume else "run.json"), metadata)
    (output / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    cv_results = {}
    for name in selected_models:
        print(f"Cross-validating {name}", flush=True)
        directory = output / name
        directory.mkdir(exist_ok=True)
        completed = [directory / filename for filename in ["cv_summary.json", "cv_folds.csv", "cv_candidate_scores.csv"]]
        if args.resume and all(path.exists() for path in completed):
            cv_results[name] = json.loads(completed[0].read_text())
            print(f"  Reusing saved {name} CV results", flush=True)
            continue
        write_json(directory / "execution.json", metadata)
        summary, fold_results, scores = cross_validate(name, train, folds, args.seed, args.jobs)
        cv_results[name] = summary
        fold_results.to_csv(directory / "cv_folds.csv", index=False)
        scores.to_csv(directory / "cv_candidate_scores.csv", index=False)
        write_json(directory / "cv_summary.json", summary)
    if args.baseline_only:
        comparisons = []
        for name in selected_models:
            directory = output / name
            metrics_path = directory / "student_test_metrics.json"
            if args.resume and metrics_path.exists() and (directory / "pipeline.joblib").exists():
                result = json.loads(metrics_path.read_text())
            else:
                result = refit_and_evaluate(name, train, test, directory, args.seed, args.jobs)
            comparisons.append({"model": name, **{f"cv_{k}": v for k, v in cv_results[name].items()},
                                **{f"test_{k}": v for k, v in result.items()}})
        table = pd.DataFrame(comparisons)
        table.to_csv(output / "baseline_comparison.csv", index=False)
        print(f"Baseline run completed: {output.resolve()}. Five-model selection and tuning remain pending.", flush=True)
        return table
    def rank_result(result):
        loss = result["log_loss"] if result["log_loss"] is not None else float("inf")
        return (-result["top1_accuracy"], loss, result["inference_seconds"])
    winner = min(MODELS, key=lambda name: rank_result(cv_results[name]))
    print(f"Bayesian optimisation: {winner}, {args.trials} trials", flush=True)
    study = optuna.create_study(study_name="pathsnap", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=min(10, max(1, args.trials // 3))),
        storage=f"sqlite:///{(output / 'optimisation.sqlite3').resolve()}", load_if_exists=args.resume)

    def objective(trial):
        params = suggest_params(trial, winner)
        summary, fold_results, _ = cross_validate(winner, train, folds, args.seed, args.jobs, params)
        trial.set_user_attr("cv_summary", summary)
        trial.set_user_attr("fold_results", fold_results.to_dict(orient="records"))
        return summary["top1_accuracy"]

    start = perf_counter()
    complete_trials = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    study.optimize(objective, n_trials=max(0, args.trials - complete_trials))
    tuning_seconds = perf_counter() - start
    study.trials_dataframe().to_csv(output / "optimisation_trials.csv", index=False)
    params = study.best_params
    tuned_name = f"{winner}_tuned"
    tuned_dir = output / tuned_name
    tuned_dir.mkdir(exist_ok=True)
    tuned_summary, tuned_folds, tuned_scores = cross_validate(winner, train, folds, args.seed, args.jobs, params)
    tuned_folds.to_csv(tuned_dir / "cv_folds.csv", index=False)
    tuned_scores.to_csv(tuned_dir / "cv_candidate_scores.csv", index=False)
    write_json(tuned_dir / "cv_summary.json", tuned_summary)
    # Keep the baseline if Bayesian search produces a worse CV result.
    use_tuned = rank_result(tuned_summary) < rank_result(cv_results[winner])
    final_name = tuned_name if use_tuned else winner
    write_json(output / "selection.json", {"selected_baseline": winner, "best_parameters": params,
        "best_trial": study.best_trial.number, "tuning_seconds": tuning_seconds,
        "final_model": final_name, "final_artifact": f"{final_name}/pipeline.joblib"})
    comparisons = []
    # Student-testing is evaluated only after baseline selection and tuning finish.
    for label, name, parameters, summary in [
        *((name, name, None, cv_results[name]) for name in MODELS),
        (tuned_name, winner, params, tuned_summary),
    ]:
        print(f"Refitting and evaluating {label}", flush=True)
        metrics_path = output / label / "student_test_metrics.json"
        if args.resume and parameters is None and metrics_path.exists() and (output / label / "pipeline.joblib").exists():
            result = json.loads(metrics_path.read_text())
        else:
            result = refit_and_evaluate(name, train, test, output / label, args.seed, args.jobs, parameters)
        comparisons.append({"model": label, **{f"cv_{k}": v for k, v in summary.items()},
                            **{f"test_{k}": v for k, v in result.items()}})
    table = pd.DataFrame(comparisons)
    table.to_csv(output / "comparison.csv", index=False)
    plot_comparison(table, output)
    print(f"Completed. Results: {output.resolve()}", flush=True)
    return table


def parser():
    cli = argparse.ArgumentParser(description="PathSnap five-model training and Bayesian optimisation")
    cli.add_argument("--feature-dir", help="Existing Person 1 Parquet feature directory")
    cli.add_argument("--train", help="Training candidate CSV")
    cli.add_argument("--student-test", help="Student-testing candidate CSV")
    cli.add_argument("--train-inventory")
    cli.add_argument("--student-test-inventory")
    cli.add_argument("--features", help="JSON mapping feature names to definitions and units")
    cli.add_argument("--output", required=True, help="Empty or new run directory")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--jobs", type=int, default=1, help="Random Forest workers; CV and trials run sequentially")
    cli.add_argument("--trials", type=int, default=30)
    cli.add_argument("--models", nargs="+", choices=MODELS, help="Subset to run with --baseline-only")
    cli.add_argument("--baseline-only", action="store_true", help="CV, refit and evaluate requested models without tuning or final selection")
    cli.add_argument("--resume", action="store_true", help="Reuse complete baseline results after validating input hashes, folds and seed")
    cli.add_argument("--allow-partial-data", action="store_true", help="Development only: allow fewer than 80/10 trips")
    return cli


def main():
    try:
        run(parser().parse_args())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Training stopped: {exc}") from exc


if __name__ == "__main__":
    main()
