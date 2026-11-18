import numpy as np
import pandas as pd
import pytest

from pathsnap.train import make_folds
from pathsnap.training_handoff import convert


def handoff():
    candidates, points = [], []
    for fold in [-1, 0, 1, 2, 3, 4]:
        trip = f"{fold + 1:08d}"
        for point in range(3):
            valid = point != 2
            points.append(dict(trip_id=trip, point_idx=point, label_arc=12, label_valid=valid, fold=fold))
            if point == 1:
                continue  # Known reference, no candidates: retained in denominator.
            for arc in [12, 15]:
                candidates.append(dict(trip_id=trip, point_idx=point, arc_id=arc, label=int(arc == 12),
                                       label_valid=valid, fold=fold, distance=np.float32(arc)))
    return pd.DataFrame(candidates), pd.DataFrame(points)


def test_native_handoff_preserves_exclusions_and_inventory():
    part = convert(*handoff(), ["distance"])
    assert part.candidates.distance.dtype == np.float32
    assert part.inventory.reference_status.eq("unresolved").sum() == 6
    assert len(part.inventory) == 18
    assert part.candidates.label.isna().sum() == 12
    assert "00000000" not in set(part.labelled.trip_id)
    folds, assignments = make_folds(part, 999)
    assert assignments.validation_fold.tolist() == [-1, 0, 1, 2, 3, 4]
    for fold, (fit, val, inventory) in enumerate(folds):
        assert set(inventory.trip_id) == {f"{fold + 1:08d}"}
        assert "00000000" not in set(fit.trip_id) | set(val.trip_id)
        assert not set(fit.trip_id) & set(val.trip_id)
        assert len(inventory) == 3 and len(val) == 4


def test_native_handoff_rejects_mismatched_metadata():
    candidates, points = handoff()
    candidates.loc[0, "fold"] = 4
    with pytest.raises(ValueError, match="fold assignments disagree"):
        convert(candidates, points, ["distance"])
    candidates, points = handoff()
    candidates.loc[0, "label_valid"] = False
    with pytest.raises(ValueError, match="validity disagrees"):
        convert(candidates, points, ["distance"])
