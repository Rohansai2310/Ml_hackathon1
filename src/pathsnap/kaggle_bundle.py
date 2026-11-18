"""Build an uploadable Kaggle bundle and import its completed AdaBoost run."""

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from . import config


def create_bundle(feature_dir, output):
    feature_dir, output = Path(feature_dir), Path(output)
    inputs = [feature_dir / name for name in [
        "feature_schema.json", "train_candidates.parquet", "train_points.parquet",
        "student_test_candidates.parquet", "student_test_points.parquet",
    ]]
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"Missing feature input: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    import sklearn
    requirements = "\n".join([
        "numpy>=1.26,<3", "pandas>=2.2,<3", "scipy>=1.11", f"scikit-learn=={sklearn.__version__}",
        "optuna>=4,<5", "matplotlib>=3.8,<4", "joblib>=1.3,<2", "pyarrow>=14",
    ]) + "\n"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for path in inputs:
            archive.write(path, f"features/{path.name}")
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(path, f"src/pathsnap/{path.name}")
        archive.writestr("requirements-kaggle.txt", requirements)
    return output


def import_results(external, local):
    external, local = Path(external), Path(local)
    remote_metadata = json.loads((external / "run.json").read_text())
    local_metadata = json.loads((local / "run.json").read_text())
    for key in ["input_sha256", "seed", "features"]:
        if remote_metadata[key] != local_metadata[key]:
            raise ValueError(f"Kaggle {key} differs from the local run")
    remote_folds = pd.read_csv(external / "fold_assignments.csv", dtype={"trip_id": "string"})
    local_folds = pd.read_csv(local / "fold_assignments.csv", dtype={"trip_id": "string"})
    if not remote_folds.equals(local_folds):
        raise ValueError("Kaggle fold assignments differ from the local run")
    if remote_metadata.get("models") != ["adaboost"] or not remote_metadata.get("baseline_only"):
        raise ValueError("Expected an AdaBoost-only baseline run")
    source = external / "adaboost"
    required = ["cv_summary.json", "cv_folds.csv", "cv_candidate_scores.csv", "pipeline.joblib",
                "configuration.json", "student_test_metrics.json", "student_test_candidate_scores.csv",
                "student_test_observations.csv"]
    if not (external / "environment.txt").is_file():
        raise ValueError("Kaggle environment record is missing")
    if not all((source / name).is_file() for name in required):
        raise ValueError("Kaggle baseline outputs are incomplete")
    metrics = pd.read_csv(source / "cv_folds.csv")
    if len(metrics) != 5 or set(metrics.fold) != set(range(5)):
        raise ValueError("Kaggle run must have completed all five folds")
    destination = local / "adaboost"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Local AdaBoost output is not empty; preserve or move it before importing")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    shutil.copy2(external / "run.json", destination / "execution.json")
    shutil.copy2(external / "environment.txt", destination / "execution_environment.txt")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("create")
    pack.add_argument("--feature-dir", default=str(config.FEATURE_DIR))
    pack.add_argument("--output", required=True)
    merge = sub.add_parser("import-results")
    merge.add_argument("--external-run", required=True)
    merge.add_argument("--local-run", required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            print(create_bundle(args.feature_dir, args.output))
        else:
            print(import_results(args.external_run, args.local_run))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
