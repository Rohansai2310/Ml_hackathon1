"""Synthetic leakage and frozen-model checks; not experiment evidence."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pathsnap import config, data_io
from pathsnap.evaluate_forest import run
from pathsnap.models import build_model
from pathsnap.train import file_hash, save_configuration, write_json


def fixture_run(tmp_path):
    root = tmp_path / "raw"
    for tid in ["eval01", "eval02"]:
        directory = root / "evaluator_test" / tid
        directory.mkdir(parents=True)
        nodes = np.array([[0.0, 51.0], [0.001, 51.0], [0.002, 51.0]])
        np.savetxt(directory / f"{tid}.nodes", nodes)
        np.savetxt(directory / f"{tid}.arcs", [[0, 1], [1, 2]], fmt="%d")
        np.savetxt(directory / f"{tid}.track", [
            [0.0002, 51., 0], [0.0008, 51., 1], [0.0014, 51., 2],
            [0.0014, 51.01, 3],  # Unresolved and outside candidate radius.
        ])
        np.savetxt(directory / f"{tid}.route", [0, 1], fmt="%d")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    features = config.FEATURE_COLUMNS
    rng = np.random.default_rng(42)
    samples = pd.DataFrame(rng.random((20, len(features))), columns=features)
    model = build_model("random_forest", params={"n_estimators": 3, "max_depth": 2})
    model.fit(samples, np.tile([0, 1], 10))
    joblib.dump(model, model_dir / "pipeline.joblib")
    save_configuration(model_dir, model, features, {})
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    hashes = {}
    for name in ["train", "student_test"]:
        path = feature_dir / f"{name}_points.parquet"
        pd.DataFrame({"trip_id": [name]}).to_parquet(path)
        hashes[f"{name}_points"] = file_hash(path)
    write_json(model_dir / "run.json", {
        "status": "complete", "model": "random_forest", "training_trips": ["train"],
        "excluded_training_trips": [], "input_sha256": hashes,
        "model_sha256": file_hash(model_dir / "pipeline.joblib"),
    })
    return model_dir, root, feature_dir


def test_evaluation_freezes_all_predictions_before_reading_references(tmp_path, monkeypatch):
    model, root, features = fixture_run(tmp_path)
    output = tmp_path / "result"
    original_load = data_io._load
    reads = []

    def guard(path, *args, **kwargs):
        if str(path).endswith(".route"):
            for tid in ["eval01", "eval02"]:
                directory = output / "predictions" / tid
                assert (directory / "candidate_scores.csv").is_file()
                assert (directory / "observation_predictions.csv").is_file()
                assert (directory / "hmm_route.json").is_file()
                assert (directory / "hmm_observation_predictions.csv").is_file()
                assert "label" not in pd.read_parquet(directory / "candidates.parquet")
                assert "reference" not in (directory / "observation_predictions.csv").read_text()
            reads.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(data_io, "_load", guard)
    before = file_hash(model / "pipeline.joblib")
    metrics = run(model, root, output, feature_dir=features, with_hmm=True)
    assert len(reads) == 2
    assert metrics["trips"] == 2 and metrics["observations"] == 8
    assert metrics["known_observations"] == 6
    assert metrics["unresolved_observations"] == 2
    assert metrics["zero_candidate_observations"] == 2
    assert metrics["reference_coverage"] == .75
    assert metrics["top1_accuracy"] == pytest.approx(metrics["correct_known_observations"] / 6)
    assert metrics["hmm_top1_accuracy"] == pytest.approx(metrics["hmm_correct_known_observations"] / 6)
    assert file_hash(model / "pipeline.joblib") == before
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="empty"):
        run(model, root, output, feature_dir=features)


def test_evaluator_overlap_is_rejected_before_prediction(tmp_path):
    model, root, features = fixture_run(tmp_path)
    path = model / "run.json"
    metadata = json.loads(path.read_text())
    metadata["training_trips"] = ["eval01"]
    write_json(path, metadata)
    with pytest.raises(ValueError, match="overlap"):
        run(model, root, tmp_path / "result", feature_dir=features)


def test_changed_frozen_model_is_rejected(tmp_path):
    model, root, features = fixture_run(tmp_path)
    with (model / "pipeline.joblib").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        run(model, root, tmp_path / "result", feature_dir=features)
