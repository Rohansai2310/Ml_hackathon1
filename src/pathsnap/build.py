"""Build the model-ready candidate tables (one parquet per partition) + trip summary + folds."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm
from . import config, data_io as io
from .network import build_geom
from .features import build_candidate_table
from .labels import align_points_to_route

F32 = [c for c in config.FEATURE_COLUMNS if c not in ("n_cand", "dist_rank", "out_deg_to", "in_deg_from", "has_reverse")]


def process_trip(trip_id, partition, root=None, with_labels=True):
    """Returns (candidate_df, points_df). With with_labels=False no .route is read."""
    tr = io.load_trip(trip_id, partition, with_route=with_labels, root=root)
    g = build_geom(tr)
    cand = build_candidate_table(g)
    N = len(g.P)
    pts = pd.DataFrame({"trip_id": trip_id, "point_idx": np.arange(N, dtype=np.int32),
                        "t_sec": g.t, "lon": tr.track[:, 0], "lat": tr.track[:, 1]})
    if with_labels:
        pos, lab, ld = align_points_to_route(g, tr.route)
        valid = ld <= config.LABEL_MAX_DIST_M
        pts["label_arc"] = lab; pts["label_dist_m"] = ld; pts["label_valid"] = valid
        pts["route_pos"] = pos
        m = cand.merge(pts[["point_idx", "label_arc", "label_valid"]], on="point_idx", how="left")
        cand["label"] = (m["arc_id"].values == m["label_arc"].values).astype(np.int8)
        cand["label_valid"] = m["label_valid"].values
        has = np.zeros(N, bool); has[cand.point_idx[cand.label == 1]] = True
        pts["label_in_candidates"] = has
    for c in F32:
        cand[c] = cand[c].astype(np.float32)
    cand["dist_rank"] = cand["dist_rank"].astype(np.int16)
    cand["n_cand"] = cand["n_cand"].astype(np.int16)
    return cand, pts


def make_folds(summary: pd.DataFrame, n_folds=config.N_FOLDS, seed=config.SEED):
    """Group-aware folds: every trip is wholly inside one fold (no point leakage across folds).
    Greedy balancing by #points so folds are similar in size. Unreliable trips -> fold -1."""
    ok = summary[summary.trip_ok].copy()
    rng = np.random.RandomState(seed)
    ok = ok.iloc[rng.permutation(len(ok))].sort_values("n_points", ascending=False, kind="stable")
    load = np.zeros(n_folds); fold = {}
    for tid, n in zip(ok.trip_id, ok.n_points):
        k = int(np.argmin(load)); fold[tid] = k; load[k] += n
    return summary.trip_id.map(fold).fillna(-1).astype(int)


def build_partition(partition, root=None, out_dir=None, with_labels=True, folds=None):
    out_dir = Path(out_dir or config.FEATURE_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    trips = io.list_trips(partition, root)
    writer = None; pts_all = []; t0 = time.time(); n_rows = 0
    for tid in tqdm(trips, desc=f"features[{partition}]"):
        cand, pts = process_trip(tid, partition, root, with_labels)
        if folds is not None:
            cand["fold"] = np.int8(folds.get(tid, -1)); pts["fold"] = np.int8(folds.get(tid, -1))
        tab = pa.Table.from_pandas(cand, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_dir / f"{partition}_candidates.parquet", tab.schema, compression="zstd")
        writer.write_table(tab); n_rows += len(cand); pts_all.append(pts)
    writer.close()
    pts_df = pd.concat(pts_all, ignore_index=True)
    pts_df.to_parquet(out_dir / f"{partition}_points.parquet", index=False)
    return {"partition": partition, "trips": len(trips), "points": int(len(pts_df)),
            "candidate_rows": int(n_rows), "seconds": round(time.time() - t0, 1)}


def write_schema(out_dir=None):
    out_dir = Path(out_dir or config.FEATURE_DIR)
    doc = {
        "candidate_radius_m": config.CAND_RADIUS_M, "max_candidates": config.MAX_CANDIDATES,
        "window": config.WINDOW, "feature_columns": config.FEATURE_COLUMNS,
        "label_max_dist_m": config.LABEL_MAX_DIST_M, "seed": config.SEED, "n_folds": config.N_FOLDS,
        "target": "label (1 if candidate arc == aligned reference arc of that GPS point)",
        "use_rows_where": "label_valid == True (training/CV: also fold >= 0)",
    }
    (out_dir / "feature_schema.json").write_text(json.dumps(doc, indent=2))
