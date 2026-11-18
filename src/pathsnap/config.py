"""Single place for every constant, so results are reproducible and explainable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
DATA_DIR = ROOT / "data" / "raw"                    # data/raw/train, data/raw/student_test
OUT_DIR = ROOT / "outputs"
FEATURE_DIR = OUT_DIR / "features"
EDA_DIR = OUT_DIR / "eda"
MAP_DIR = OUT_DIR / "maps"

SEED = 42
N_FOLDS = 5                 # group-aware CV: all points of one trip stay in one fold

# ---- candidate generation -------------------------------------------------
CAND_RADIUS_M = 75.0        # arcs within this distance of a GPS point become candidates
MAX_CANDIDATES = 40         # keep the K nearest arcs only (bounds table size)

# ---- feature engineering --------------------------------------------------
WINDOW = 3                  # +/- WINDOW neighbouring GPS points used for context features
HEADING_MIN_SPEED = 1.0     # m/s; below this the heading estimate is unreliable

# ---- label quality (data-quality decisions, justified in docs/data_quality_record.md)
LABEL_MAX_DIST_M = 50.0     # a point is a *valid* training label only if it lies within 50 m of its
                            # aligned reference arc (median is ~5 m; larger = route doesn't cover the point)
UNRELIABLE_TRIP_FRAC = 0.30 # trip flagged unreliable if >30% of its points have no valid label
                            # (reference route covers only part of the GPS trace) -> excluded from training/CV

# ---- label derivation -----------------------------------------------------
# The .route file is a list of arcs, NOT one arc per GPS point. We align points
# to route arcs with a monotone dynamic programme (see labels.py).

FEATURE_COLUMNS = [
    "dist_m", "frac_along", "arc_len_m", "dist_rank", "dist_rel_min", "n_cand",
    "heading_diff_deg", "cos_heading", "speed_mps", "dt_prev", "dt_next",
    "win_mean_dist", "win_max_dist", "win_min_dist", "win_hit_frac",
    "out_deg_to", "in_deg_from", "has_reverse", "to_node_dist_m", "from_node_dist_m",
    "along_advance_m", "speed_dist_resid",
]
