"""Adapt Person 1's Parquet tables without changing labels, folds or exclusions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import OBS, validate

FEATURE_DEFINITIONS = dict(zip(config.FEATURE_COLUMNS, [
    "GPS-to-candidate distance (m)", "Clamped fractional position along arc (0–1)",
    "Arc length (m)", "Zero-based candidate distance rank", "Distance above nearest candidate (m)",
    "Candidate count for observation", "Travel-to-arc heading difference (degrees)",
    "Cosine of heading difference", "Local vehicle speed (m/s)", "Previous valid sample gap (s)",
    "Next valid sample gap (s)", "Neighbour mean distance to candidate (m)",
    "Neighbour maximum distance to candidate (m)", "Neighbour minimum distance to candidate (m)",
    "Fraction of valid neighbours within half candidate radius", "Out-degree of arc destination",
    "In-degree of arc source", "Reverse arc exists (0/1)", "Distance to destination node (m)",
    "Distance to source node (m)", "Signed neighbouring progress along arc (m)",
    "Absolute along-arc progress minus neighbouring displacement (m)",
]))


def convert(candidates, points, features):
    rename = {"point_idx": "observation_id", "arc_id": "candidate_id"}
    c = candidates.rename(columns=rename).copy()
    p = points.rename(columns=rename).copy()
    for table in (c, p):
        table["trip_id"] = table.trip_id.astype("string")
        table["observation_id"] = table.observation_id.astype("string")
    if not p.label_valid.isin([True, False]).all() or not c.label_valid.isin([True, False]).all():
        raise ValueError("label_valid must contain booleans without missing values")
    check = c[OBS + ["label_valid"]].merge(p[OBS + ["label_valid"]], on=OBS, how="left", suffixes=("_candidate", "_point"), validate="many_to_one")
    if check.label_valid_point.isna().any() or not check.label_valid_candidate.eq(check.label_valid_point).all():
        raise ValueError("Candidate label validity disagrees with point inventory")
    c["candidate_id"] = c.candidate_id.astype("string")
    c["label"] = c.label.astype(float).where(c.label_valid, np.nan)
    p["reference_status"] = np.where(p.label_valid, "known", "unresolved")
    p["reference_candidate_id"] = p.label_arc.astype("Int64").astype("string").where(p.label_valid, pd.NA)
    partition = validate(c[OBS + ["candidate_id", "label"] + features],
                         p[OBS + ["reference_status", "reference_candidate_id"]], features)
    if "fold" in p:
        mapping = p[["trip_id", "fold"]].drop_duplicates().rename(columns={"fold": "validation_fold"})
        if mapping.trip_id.duplicated().any() or not mapping.validation_fold.isin([-1, 0, 1, 2, 3, 4]).all():
            raise ValueError("Every trip must have exactly one valid fold")
        if "fold" not in c:
            raise ValueError("Candidate fold assignments are missing")
        joined = c[["trip_id", "fold"]].merge(mapping, on="trip_id", validate="many_to_one")
        if not joined.fold.eq(joined.validation_fold).all():
            raise ValueError("Candidate and point fold assignments disagree")
        partition.fold_assignments = mapping.reset_index(drop=True)
    return partition


def load_feature_dir(directory, partial=False):
    directory = Path(directory)
    schema_path = directory / "feature_schema.json"
    schema = json.loads(schema_path.read_text())
    features = schema["feature_columns"]
    if features != config.FEATURE_COLUMNS:
        raise ValueError("Feature schema disagrees with feature-generation code")
    sources = {"schema": str(schema_path)}
    partitions = []
    for name in ("train", "student_test"):
        for kind in ("candidates", "points"):
            sources[f"{name}_{kind}"] = str(directory / f"{name}_{kind}.parquet")
        partitions.append(convert(pd.read_parquet(sources[f"{name}_candidates"]),
                                  pd.read_parquet(sources[f"{name}_points"]), features))
    train, test = partitions
    tr, te = set(train.inventory.trip_id), set(test.inventory.trip_id)
    if tr & te:
        raise ValueError("Training and student-testing trips overlap")
    if not partial and (len(tr), len(te)) != (80, 10):
        raise ValueError("Expected 80 training and 10 student-testing trips")
    if train.fold_assignments is None or set(train.fold_assignments.validation_fold) - {-1} != set(range(5)):
        raise ValueError("Training tables must preserve all five existing folds")
    return train, test, sources
