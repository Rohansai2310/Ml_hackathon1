"""Run: python -m pytest -q tests   (needs data/raw populated)"""
import numpy as np, pandas as pd, pytest
from pathsnap import config, data_io as io, build
from pathsnap.network import build_geom
from pathsnap.features import build_candidate_table
from pathsnap.labels import align_points_to_route
from pathsnap.geo import Projector

TRIP, PART = "00000048", "train"
pytestmark = pytest.mark.skipif(not (config.DATA_DIR / PART / TRIP).exists(), reason="dataset not extracted")


def test_features_identical_without_route_file():
    """Leakage guard: inference path (with_route=False) gives the same features as training path."""
    a = build_candidate_table(build_geom(io.load_trip(TRIP, PART, with_route=True)))
    b = build_candidate_table(build_geom(io.load_trip(TRIP, PART, with_route=False)))
    pd.testing.assert_frame_equal(a, b)
    assert io.load_trip(TRIP, PART, with_route=False).route is None


def test_feature_columns_present_and_finite():
    df = build_candidate_table(build_geom(io.load_trip(TRIP, PART)))
    assert list(config.FEATURE_COLUMNS) == [c for c in config.FEATURE_COLUMNS if c in df]
    assert np.isfinite(df[config.FEATURE_COLUMNS].to_numpy(float)).all()
    assert (df.groupby("point_idx").size() <= config.MAX_CANDIDATES).all()


def test_projection_is_metric():
    p = Projector(-1.84, 52.95)
    xy = p.to_xy(np.array([[-1.84, 52.95], [-1.84, 52.951]]))
    assert abs(np.hypot(*(xy[1] - xy[0])) - 111.2) < 1.0          # 0.001 deg lat ~ 111 m


def test_labels_monotone_and_on_route():
    tr = io.load_trip(TRIP, PART, with_route=True); g = build_geom(tr)
    pos, lab, ld = align_points_to_route(g, tr.route)
    assert (np.diff(pos) >= 0).all() and set(lab) <= set(tr.route)


def test_folds_are_group_disjoint():
    s = pd.read_csv(config.EDA_DIR / "folds.csv", dtype={"trip_id": str})
    assert s.trip_id.is_unique                                       # a trip has exactly one fold
    tr = pd.read_parquet(config.FEATURE_DIR / "train_points.parquet", columns=["trip_id", "fold"])
    assert (tr.groupby("trip_id").fold.nunique() == 1).all()
