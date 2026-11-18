"""Candidate metrics and observation accuracy with explicit miss accounting."""

import numpy as np
from sklearn.metrics import f1_score, log_loss

from .data import IDS, OBS


def score_candidates(model, candidates, features):
    result = candidates[IDS].copy()
    if candidates.empty:
        result["probability"] = np.array([], dtype=float)
    else:
        positive = list(model.classes_).index(1)
        result["probability"] = model.predict_proba(candidates[features])[:, positive]
    return result


def evaluate(candidates, inventory, scores):
    if not scores[IDS].reset_index(drop=True).equals(candidates[IDS].reset_index(drop=True)):
        raise ValueError("Candidate scores are not aligned with candidate identifiers")
    probability = scores.probability.to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Candidate probabilities must be finite and between zero and one")
    # Lexical candidate ID resolves tied probabilities reproducibly.
    ranked = scores.sort_values(IDS, kind="stable").sort_values("probability", ascending=False, kind="stable")
    top = ranked.drop_duplicates(OBS).rename(columns={"candidate_id": "predicted_candidate_id"})
    observations = inventory.merge(top, on=OBS, how="left", validate="one_to_one")
    observations["candidate_count"] = observations.merge(
        candidates.groupby(OBS).size().rename("count").reset_index(), on=OBS, how="left"
    )["count"].fillna(0).astype(int)
    present = candidates.loc[candidates.label.eq(1), OBS].assign(reference_present=True)
    observations = observations.merge(present, on=OBS, how="left", validate="one_to_one")
    observations["reference_present"] = observations.reference_present.astype("boolean").fillna(False).astype(bool)
    known = observations.reference_status.eq("known")
    eligible = known & observations.reference_present
    observations["correct"] = observations.predicted_candidate_id.eq(observations.reference_candidate_id).fillna(False) & known
    labelled = candidates.label.notna().to_numpy()
    metrics = {
        "top1_accuracy": float(observations.loc[known, "correct"].mean()) if known.any() else None,
        "top1_accuracy_reference_present": float(observations.loc[eligible, "correct"].mean()) if eligible.any() else None,
        "candidate_miss_rate": float((~observations.loc[known, "reference_present"]).mean()) if known.any() else None,
        "macro_f1": float(f1_score(candidates.loc[labelled, "label"], probability[labelled] >= 0.5, labels=[0, 1], average="macro", zero_division=0)) if labelled.any() else None,
        "log_loss": float(log_loss(candidates.loc[labelled, "label"], np.column_stack([1 - probability[labelled], probability[labelled]]), labels=[0, 1])) if labelled.any() else None,
        "known_observations": int(known.sum()),
        "unresolved_observations": int((~known).sum()),
        "reference_present_observations": int(eligible.sum()),
        "zero_candidate_observations": int(observations.candidate_count.eq(0).sum()),
    }
    return metrics, observations
