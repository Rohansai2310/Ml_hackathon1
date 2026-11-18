"""Synthetic verification only: these fixtures are not experiment evidence."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pathsnap.data import IDS, load_pair, validate
from pathsnap.metrics import evaluate, score_candidates
from pathsnap.models import MODELS, build_model
from pathsnap.train import make_folds, parser, run


def sample(trips):
    rows, inventory = [], []
    for trip in trips:
        for observation in range(4):
            obs = str(observation)
            inventory.append(dict(trip_id=trip, observation_id=obs, reference_status="known", reference_candidate_id="A"))
            for candidate, distance, label in [("A", 2.0 + observation, 1), ("B", 20.0 + observation, 0)]:
                rows.append(dict(trip_id=trip, observation_id=obs, candidate_id=candidate,
                                 distance=distance, heading=np.nan if observation == 0 else distance / 2, label=label))
    return pd.DataFrame(rows), pd.DataFrame(inventory)


@pytest.fixture
def partition():
    return validate(*sample([f"T{i}" for i in range(6)]), ["distance", "heading"])


@pytest.mark.parametrize("name", MODELS)
def test_models_and_fold_local_preprocessing(partition, name, tmp_path):
    folds, _ = make_folds(partition, 42)
    fit, val, _ = folds[0]
    fit = fit.copy()
    val = val.copy()
    # A distinctive held-out distribution must not change fitted medians/scales.
    val["distance"] = 1000000.0
    model = build_model(name)
    model.fit(fit[partition.features], fit.label.astype(int))
    np.testing.assert_allclose(model.named_steps["imputer"].statistics_, fit[partition.features].median())
    if "scaler" in model.named_steps:
        expected = fit[partition.features].fillna(fit[partition.features].median()).mean()
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, expected)
    scores = score_candidates(model, val, partition.features)
    assert scores[IDS].equals(val[IDS])
    assert scores.probability.between(0, 1).all()
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    np.testing.assert_array_equal(scores.probability, score_candidates(joblib.load(artifact), val, partition.features).probability)


def test_shared_trip_folds(partition):
    folds, assignments = make_folds(partition, 42)
    assert len(folds) == 5
    assert not assignments.trip_id.duplicated().any()
    assert set(assignments.trip_id) == set(partition.inventory.trip_id)
    for fit, val, inventory in folds:
        assert not set(fit.trip_id) & set(inventory.trip_id)
        assert set(val.trip_id) == set(inventory.trip_id)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "Duplicate"), ("wrong_label", "disagree"),
    ("unknown_observation", "inventory"), ("infinite", "Infinite"),
    ("bad_status", "reference_status"), ("reference_feature", "cannot be features"),
])
def test_invalid_handoffs(mutation, match):
    candidates, inventory = sample(["T0"])
    features = ["distance", "heading"]
    if mutation == "duplicate":
        candidates = pd.concat([candidates, candidates.iloc[:1]])
    elif mutation == "wrong_label":
        candidates.loc[0, "label"] = 0
    elif mutation == "unknown_observation":
        candidates.loc[0, "observation_id"] = "unknown"
    elif mutation == "infinite":
        candidates.loc[0, "distance"] = np.inf
    elif mutation == "bad_status":
        inventory.loc[0, "reference_status"] = "guess"
    else:
        features.append("label")
    with pytest.raises(ValueError, match=match):
        validate(candidates, inventory, features)


def test_misses_unresolved_and_alignment():
    candidates, inventory = sample(["T0"])
    # Observation 1 has candidates but lacks the reference; 2 has zero candidates.
    candidates = candidates.loc[~((candidates.observation_id == "1") & (candidates.candidate_id == "A"))]
    candidates = candidates.loc[candidates.observation_id != "2"].reset_index(drop=True)
    inventory.loc[inventory.observation_id == "3", ["reference_status", "reference_candidate_id"]] = ["unresolved", pd.NA]
    candidates.loc[candidates.observation_id == "3", "label"] = np.nan
    part = validate(candidates, inventory, ["distance", "heading"])
    scores = part.candidates[IDS].copy()
    scores["probability"] = np.where(part.candidates.candidate_id.eq("A"), 0.9, 0.1)
    metrics, observations = evaluate(part.candidates, part.inventory, scores)
    assert metrics["top1_accuracy"] == pytest.approx(1 / 3)
    assert metrics["top1_accuracy_reference_present"] == 1
    assert metrics["candidate_miss_rate"] == pytest.approx(2 / 3)
    assert metrics["zero_candidate_observations"] == 1
    assert metrics["unresolved_observations"] == 1
    assert len(observations) == 4
    assert len(part.labelled) == 3
    with pytest.raises(ValueError, match="aligned"):
        evaluate(part.candidates, part.inventory, scores.iloc[::-1])
    candidates.loc[candidates.observation_id == "3", "label"] = 0
    with pytest.raises(ValueError, match="blank labels"):
        validate(candidates, inventory, ["distance", "heading"])


def write_inputs(tmp_path, overlap=False):
    for name, trips in [("train", [f"T{i}" for i in range(6)]), ("test", ["T0" if overlap else "S0", "S1"])]:
        candidates, inventory = sample(trips)
        candidates.to_csv(tmp_path / f"{name}.csv", index=False)
        inventory.to_csv(tmp_path / f"{name}_inventory.csv", index=False)
    (tmp_path / "features.json").write_text(json.dumps({"distance": "Distance in metres", "heading": "Heading difference in degrees"}))


def test_partition_overlap_and_counts(tmp_path):
    write_inputs(tmp_path, overlap=True)
    args = [tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "train_inventory.csv", tmp_path / "test_inventory.csv", ["distance", "heading"]]
    with pytest.raises(ValueError, match="overlap"):
        load_pair(*args, partial=True)
    write_inputs(tmp_path)
    with pytest.raises(ValueError, match="80 training"):
        load_pair(*args)


def test_full_runner_synthetic_only(tmp_path):
    write_inputs(tmp_path)
    output = tmp_path / "synthetic_verification"
    args = parser().parse_args([
        "--train", str(tmp_path / "train.csv"), "--student-test", str(tmp_path / "test.csv"),
        "--train-inventory", str(tmp_path / "train_inventory.csv"),
        "--student-test-inventory", str(tmp_path / "test_inventory.csv"),
        "--features", str(tmp_path / "features.json"), "--output", str(output),
        "--trials", "2", "--allow-partial-data",
    ])
    table = run(args)
    assert len(table) == 6
    assert set(MODELS).issubset(table.model)
    assert len(pd.read_csv(output / "optimisation_trials.csv")) == 2
    assert len(pd.read_csv(output / "fold_assignments.csv")) == 6
    assert (output / "optimisation.sqlite3").exists()
    selection = json.loads((output / "selection.json").read_text())
    assert (output / selection["final_artifact"]).exists()
    for model in table.model:
        directory = output / model
        assert (directory / "pipeline.joblib").exists()
        assert len(pd.read_csv(directory / "cv_candidate_scores.csv")) == 48
        scores = pd.read_csv(directory / "student_test_candidate_scores.csv")
        assert list(scores.columns) == IDS + ["probability"]
        assert len(scores) == 16
    assert len(list(output.glob("*.png"))) == 4
