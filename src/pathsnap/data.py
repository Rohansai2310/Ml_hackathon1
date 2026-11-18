"""Strict, label-aware validation of the teammate data handoff."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

IDS = ["trip_id", "observation_id", "candidate_id"]
OBS = IDS[:2]
INVENTORY_COLUMNS = OBS + ["reference_status", "reference_candidate_id"]


@dataclass
class Partition:
    candidates: pd.DataFrame
    inventory: pd.DataFrame
    features: list[str]
    fold_assignments: pd.DataFrame | None = None

    @property
    def labelled(self):
        mask = self.candidates.label.notna()
        if self.fold_assignments is not None:
            eligible = self.fold_assignments.loc[self.fold_assignments.validation_fold.ge(0), "trip_id"]
            mask &= self.candidates.trip_id.isin(eligible)
        return self.candidates.loc[mask].copy()


def read_csv(path, inventory=False):
    columns = INVENTORY_COLUMNS if inventory else IDS
    return pd.read_csv(Path(path), dtype={c: "string" for c in columns})


def validate(candidates, inventory, features):
    candidates, inventory = candidates.copy(), inventory.copy()
    if not features or len(set(features)) != len(features):
        raise ValueError("Provide a nonempty, unique list of feature columns")
    if set(features) & set(IDS + ["label"] + INVENTORY_COLUMNS):
        raise ValueError("Identifiers and reference fields cannot be features")
    expected = set(IDS + ["label"] + features)
    if set(candidates.columns) != expected:
        raise ValueError(f"Candidate columns must be exactly {sorted(expected)}")
    if set(inventory.columns) != set(INVENTORY_COLUMNS):
        raise ValueError(f"Inventory columns must be exactly {INVENTORY_COLUMNS}")
    for frame, keys in [(candidates, IDS), (inventory, OBS)]:
        for col in keys:
            frame[col] = frame[col].astype("string")
            if frame[col].isna().any() or frame[col].str.strip().eq("").any():
                raise ValueError(f"Missing or empty identifier: {col}")
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate rows for {keys}")
    if inventory.empty:
        raise ValueError("Observation inventory must not be empty")
    if not inventory.reference_status.isin(["known", "unresolved"]).all():
        raise ValueError("reference_status must be known or unresolved")
    inventory["reference_candidate_id"] = inventory.reference_candidate_id.astype("string")
    known = inventory.reference_status.eq("known")
    reference = inventory.reference_candidate_id
    absent = reference.isna() | reference.str.strip().eq("").fillna(True)
    if (known & absent).any() or (~known & ~absent).any():
        raise ValueError("Known references require an ID; unresolved references require a blank ID")
    merged = candidates[IDS].merge(inventory, on=OBS, how="left", validate="many_to_one", indicator=True)
    if merged._merge.ne("both").any():
        raise ValueError("Every candidate observation must appear in the inventory")
    try:
        candidates["label"] = pd.to_numeric(candidates.label, errors="raise")
        for name in features:
            candidates[name] = pd.to_numeric(candidates[name], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Features and labels must be numeric (blank features are imputed)") from exc
    if np.isinf(candidates[features].to_numpy()).any():
        raise ValueError("Infinite feature values are not permitted")
    row_known = merged.reference_status.eq("known").to_numpy()
    labels = candidates.label
    if not labels.loc[row_known].isin([0, 1]).all():
        raise ValueError("Every known-reference candidate requires a binary label")
    if labels.loc[~row_known].notna().any():
        raise ValueError("Unresolved observations must have blank labels")
    expected_labels = merged.candidate_id.eq(merged.reference_candidate_id).fillna(False).astype(int)
    if not np.array_equal(labels.loc[row_known].to_numpy(), expected_labels.loc[row_known].to_numpy()):
        raise ValueError("Labels disagree with inventory reference_candidate_id")
    return Partition(candidates, inventory, list(features))


def load_pair(train_path, test_path, train_inventory, test_inventory, features, partial=False):
    train = validate(read_csv(train_path), read_csv(train_inventory, True), features)
    test = validate(read_csv(test_path), read_csv(test_inventory, True), features)
    train_trips = set(train.inventory.trip_id)
    test_trips = set(test.inventory.trip_id)
    if train_trips & test_trips:
        raise ValueError("Training and student-testing trips overlap")
    if not partial and (len(train_trips), len(test_trips)) != (80, 10):
        raise ValueError("Expected 80 training and 10 student-testing trips; use --allow-partial-data for development")
    if len(train_trips) < 5:
        raise ValueError("Five-fold cross-validation requires at least five training trips")
    if not train.labelled[features].notna().any().all():
        raise ValueError("Each feature must contain a known-reference training value")
    return train, test
