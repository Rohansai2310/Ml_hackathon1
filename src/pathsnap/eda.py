"""Dataset exploration (Dataset Exploration.pdf section 4/8): inventory, quality, network
connectivity, reference-route checks, radius study, plots. Every number is computed from
the raw files - nothing is typed in by hand."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from . import config, data_io as io
from .network import build_geom
from .labels import align_points_to_route
from .candidates import generate_candidates

PARTITIONS = ["train", "student_test"]


def _trip_stats(tid, part, root):
    d = Path(root or config.DATA_DIR) / part / tid
    tr = io.load_trip(tid, part, with_route=True, root=root)
    g = build_geom(tr)
    trk, nodes, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
    N, V, E = len(trk), len(nodes), len(arcs)
    t = trk[:, 2]; dt = np.diff(t)
    step = np.hypot(*np.diff(g.P, axis=0).T)
    spd = np.where(dt > 0, step / np.where(dt > 0, dt, 1), np.nan)
    r = {"trip_id": tid, "partition": part, "n_points": N, "n_nodes": V, "n_arcs": E, "n_route_arcs": len(route)}
    for ext in ("track", "nodes", "arcs", "route"):
        p = d / f"{tid}.{ext}"; r[f"kb_{ext}"] = round(p.stat().st_size / 1024, 1) if p.exists() else np.nan
    # --- GPS quality
    r["nan_values_track"] = int(np.isnan(trk).sum())
    r["invalid_coords"] = int(((np.abs(trk[:, 0]) > 180) | (np.abs(trk[:, 1]) > 90) | ((trk[:, 0] == 0) & (trk[:, 1] == 0))).sum())
    r["dup_timestamps"] = int((dt == 0).sum()); r["time_going_backwards"] = int((dt < 0).sum())
    r["dt_median_s"] = float(np.median(dt)); r["dt_eq_1s_frac"] = float((dt == 1).mean())
    r["dt_max_s"] = float(dt.max()); r["gaps_gt_5s"] = int((dt > 5).sum()); r["gaps_gt_30s"] = int((dt > 30).sum())
    r["duration_min"] = float((t[-1] - t[0]) / 60)
    r["path_len_km"] = float(step.sum() / 1000)
    r["extent_x_km"] = float(np.ptp(g.P[:, 0]) / 1000); r["extent_y_km"] = float(np.ptp(g.P[:, 1]) / 1000)
    r["speed_max_mps"] = float(np.nanmax(spd)); r["speed_gt_60mps"] = int(np.nansum(spd > 60))
    r["stationary_frac"] = float((step < 0.5).mean())
    r["lon_c"] = float(np.median(trk[:, 0])); r["lat_c"] = float(np.median(trk[:, 1])); r["utm_epsg"] = g.proj.epsg
    # --- missing / invalid in network files
    r["nan_values_net"] = int(np.isnan(nodes).sum())
    r["arcs_bad_node_ref"] = int(((arcs < 0) | (arcs >= V)).any(1).sum())
    r["nodes_unreferenced"] = int(V - len(np.unique(arcs)))
    r["self_loop_arcs"] = int((arcs[:, 0] == arcs[:, 1]).sum())
    r["zero_len_arcs"] = int((g.arc_len == 0).sum())
    # --- connectivity (weakly connected components)
    adj = coo_matrix((np.ones(E), (arcs[:, 0], arcs[:, 1])), shape=(V, V))
    nc, lab = connected_components(adj, directed=False)
    sizes = np.bincount(lab)
    r["n_components"] = int(nc); r["largest_comp_frac"] = float(sizes.max() / V)
    r["isolated_nodes"] = int(((g.out_deg + g.in_deg) == 0).sum())
    r["two_way_arc_frac"] = float(g.reverse_exists.mean())
    r["mean_out_deg"] = float(g.out_deg.mean()); r["arc_len_median_m"] = float(np.median(g.arc_len))
    # --- reference route
    r["route_breaks"] = int((arcs[route[:-1], 1] != arcs[route[1:], 0]).sum())
    r["route_revisits"] = int(len(route) - len(np.unique(route)))
    r["route_len_km"] = float(g.arc_len[route].sum() / 1000)
    r["route_over_path"] = r["route_len_km"] / max(r["path_len_km"], 1e-9)
    # --- aligned labels + candidate coverage at the configured radius
    pos, lab_arc, ld = align_points_to_route(g, route)
    r["label_dist_med_m"] = float(np.median(ld)); r["label_dist_p95_m"] = float(np.percentile(ld, 95))
    valid = ld <= config.LABEL_MAX_DIST_M
    r["frac_label_invalid"] = float(1 - valid.mean())
    pi, arc, d_, _ = generate_candidates(g)
    hit = np.zeros(N, bool); hit[pi[arc == lab_arc[pi]]] = True
    r["cand_miss_rate_valid"] = float(1 - hit[valid].mean()) if valid.any() else np.nan
    r["avg_candidates"] = float(len(pi) / N)
    r["trip_ok"] = bool(r["frac_label_invalid"] <= config.UNRELIABLE_TRIP_FRAC)
    return r, dt, spd[np.isfinite(spd)], ld


def trip_table(root=None, out_dir=None, cache=True):
    out_dir = Path(out_dir or config.EDA_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "trip_summary.csv"
    if cache and f.exists():
        return pd.read_csv(f, dtype={"trip_id": str})
    rows = []; DT = []; SPD = []; LD = []
    for part in PARTITIONS:
        for tid in tqdm(io.list_trips(part, root), desc=f"inventory[{part}]"):
            r, dt, spd, ld = _trip_stats(tid, part, root)
            rows.append(r); DT.append(pd.DataFrame({"partition": part, "dt": dt}).sample(frac=min(1, 200000 / max(len(dt), 1)), random_state=0))
            SPD.append(pd.DataFrame({"partition": part, "spd": spd}).sample(frac=min(1, 200000 / max(len(spd), 1)), random_state=0))
            LD.append(pd.DataFrame({"partition": part, "ld": ld}))
    df = pd.DataFrame(rows); df.to_csv(f, index=False)
    pd.concat(DT).to_parquet(out_dir / "_dt_sample.parquet"); pd.concat(SPD).to_parquet(out_dir / "_spd_sample.parquet")
    pd.concat(LD).to_parquet(out_dir / "_labeldist_sample.parquet")
    return df


def partition_inventory(df):
    g = df.groupby("partition")
    inv = pd.DataFrame({
        "trips": g.size(), "gps_points_total": g.n_points.sum(), "points_per_trip_median": g.n_points.median(),
        "points_per_trip_min": g.n_points.min(), "points_per_trip_max": g.n_points.max(),
        "road_nodes_median": g.n_nodes.median(), "road_arcs_median": g.n_arcs.median(),
        "road_arcs_max": g.n_arcs.max(), "route_arcs_median": g.n_route_arcs.median(),
        "trips_with_all_4_files": g.apply(lambda x: int(x[["kb_track", "kb_nodes", "kb_arcs", "kb_route"]].notna().all(axis=1).sum())),
        "trips_unreliable_reference": g.apply(lambda x: int((~x.trip_ok).sum())),
    })
    return inv.T


def missing_value_table(df):
    cols = {"NaN values in .track": "nan_values_track", "Invalid coordinates (0,0 / out of range)": "invalid_coords",
            "Duplicate timestamps": "dup_timestamps", "Timestamps going backwards": "time_going_backwards",
            "GPS gaps > 5 s": "gaps_gt_5s", "GPS gaps > 30 s": "gaps_gt_30s",
            "Implausible speed > 60 m/s (216 km/h)": "speed_gt_60mps", "NaN values in .nodes": "nan_values_net",
            "Arcs referencing a non-existent node": "arcs_bad_node_ref", "Unreferenced (isolated) nodes": "isolated_nodes",
            "Self-loop arcs": "self_loop_arcs", "Zero-length arcs": "zero_len_arcs",
            "Reference-route breaks (arc i end != arc i+1 start)": "route_breaks"}
    rows = []
    for k, c in cols.items():
        rows.append({"check": k, "total_count": int(df[c].sum()), "trips_affected": int((df[c] > 0).sum()),
                     "train_trips_affected": int(((df[c] > 0) & (df.partition == "train")).sum()),
                     "student_test_trips_affected": int(((df[c] > 0) & (df.partition == "student_test")).sum())})
    return pd.DataFrame(rows)


def radius_study(root=None, radii=(25, 50, 75, 100, 150), out_dir=None, cache=True):
    """Candidate Miss Rate vs radius, on points that have a valid reference label (train, reliable trips)."""
    out_dir = Path(out_dir or config.EDA_DIR); f = out_dir / "radius_study.csv"
    if cache and f.exists():
        return pd.read_csv(f)
    acc = {r: [0, 0, 0, 0, 0, 0] for r in radii}  # hits(valid), valid pts, cand rows, pts, hits(all), all pts
    summ = trip_table(root, out_dir)
    for tid in tqdm(summ[(summ.partition == "train") & summ.trip_ok].trip_id, desc="radius-study"):
        tr = io.load_trip(tid, "train", True, root); g = build_geom(tr)
        _, lab, ld = align_points_to_route(g, tr.route); valid = ld <= config.LABEL_MAX_DIST_M
        for r in radii:
            pi, arc, _, _ = generate_candidates(g, radius=r, max_k=10_000)
            hit = np.zeros(len(g.P), bool); hit[pi[arc == lab[pi]]] = True
            a = acc[r]; a[0] += hit[valid].sum(); a[1] += valid.sum(); a[2] += len(pi); a[3] += len(g.P); a[4] += hit.sum(); a[5] += len(hit)
    # miss_valid is 0 for radius >= LABEL_MAX_DIST_M BY CONSTRUCTION (valid labels are <=50 m away);
    # miss_all_points (every point of reliable trips, incl. noisy labels) is the non-circular number.
    out = pd.DataFrame([{"radius_m": r, "candidate_miss_rate": 1 - a[4] / a[5], "miss_rate_valid_labels_only": 1 - a[0] / a[1],
                         "avg_candidates_per_point": a[2] / a[3]} for r, a in acc.items()])
    out.to_csv(f, index=False)
    return out


# ------------------------------------------------------------------ plots
def make_plots(df, out_dir=None, root=None):
    out = Path(out_dir or config.EDA_DIR); out.mkdir(parents=True, exist_ok=True)
    col = {"train": "#1f77b4", "student_test": "#ff7f0e"}
    # 1 points per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(df.n_points.min()), np.log10(df.n_points.max()), 25)
    for p, g in df.groupby("partition"): ax.hist(g.n_points, bins=bins, alpha=.7, label=f"{p} ({len(g)} trips)", color=col[p])
    ax.set_xscale("log"); ax.set_xlabel("GPS observations per trip (log scale)"); ax.set_ylabel("trips"); ax.legend()
    ax.set_title("Trip length distribution"); fig.tight_layout(); fig.savefig(out / "01_points_per_trip.png", dpi=150); plt.close(fig)
    # 2 sampling interval
    dt = pd.read_parquet(out / "_dt_sample.parquet")
    fig, ax = plt.subplots(figsize=(7, 4)); v = dt.dt.clip(upper=30)
    ax.hist(v, bins=np.arange(-0.5, 31.5, 1), color="#1f77b4"); ax.set_yscale("log")
    ax.set_xlabel("time between consecutive GPS points (s, clipped at 30)"); ax.set_ylabel("count (log)")
    ax.set_title(f"Sampling interval - {100*(dt.dt==1).mean():.1f}% of steps are exactly 1 s"); fig.tight_layout()
    fig.savefig(out / "02_sampling_interval.png", dpi=150); plt.close(fig)
    # 3 speed
    sp = pd.read_parquet(out / "_spd_sample.parquet")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(sp.spd.clip(upper=60), bins=60, color="#2ca02c")
    ax.set_yscale("log"); ax.set_xlabel("speed between consecutive points (m/s, clipped at 60)"); ax.set_ylabel("count (log)")
    ax.set_title("GPS-derived speed"); fig.tight_layout(); fig.savefig(out / "03_speed.png", dpi=150); plt.close(fig)
    # 4 network size
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p, g in df.groupby("partition"): ax.scatter(g.n_nodes, g.n_arcs, s=18, alpha=.7, label=p, color=col[p])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("road nodes"); ax.set_ylabel("road arcs (directed)")
    ax.set_title("Road-network size per trip (arcs ~ 2 x nodes)"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "04_network_size.png", dpi=150); plt.close(fig)
    # 5 geographic overview
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, g in df.groupby("partition"): ax.scatter(g.lon_c, g.lat_c, s=25, alpha=.75, label=p, color=col[p])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Trip centroids: trips come from different cities"); fig.tight_layout()
    fig.savefig(out / "05_geographic_overview.png", dpi=150); plt.close(fig)
    # 6 label distance
    ld = pd.read_parquet(out / "_labeldist_sample.parquet")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(ld.ld.clip(upper=200), bins=80, color="#9467bd"); ax.set_yscale("log")
    ax.axvline(config.LABEL_MAX_DIST_M, color="r", ls="--", label=f"valid-label limit {config.LABEL_MAX_DIST_M:.0f} m")
    ax.set_xlabel("distance GPS point -> its aligned reference arc (m, clipped at 200)"); ax.set_ylabel("count (log)")
    ax.set_title("How far GPS points are from the reference route"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "06_label_distance.png", dpi=150); plt.close(fig)
    # 7 reference reliability
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for p, g in df.groupby("partition"): ax.scatter(g.route_over_path, g.frac_label_invalid, s=25, alpha=.8, label=p, color=col[p])
    ax.axhline(config.UNRELIABLE_TRIP_FRAC, color="r", ls="--", lw=1)
    for _, r in df[~df.trip_ok].iterrows(): ax.annotate(r.trip_id, (r.route_over_path, r.frac_label_invalid), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("reference-route length / GPS path length"); ax.set_ylabel("fraction of points >50 m from reference")
    ax.set_title("Reference-route limitation: partial routes"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "07_reference_reliability.png", dpi=150); plt.close(fig)
    # 8 data-quality issue bar
    mv = missing_value_table(df); mv = mv[mv.trips_affected > 0]
    if len(mv):
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(mv) + 1.5)); ax.barh(mv.check, mv.trips_affected, color="#d62728")
        ax.set_xlabel("number of trips affected (of 90)"); ax.set_title("Data-quality issues found"); fig.tight_layout()
        fig.savefig(out / "08_data_quality_issues.png", dpi=150); plt.close(fig)
    # 9 trajectory grid (projected metres) with local road network
    pick = list(df[(df.partition == "train") & df.trip_ok].sort_values("n_points").trip_id.iloc[[2, 20, 40, 60]])
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    for ax, tid in zip(axs.ravel(), pick):
        tr = io.load_trip(tid, "train", True, root); g = build_geom(tr)
        lo, hi = g.P.min(0) - 200, g.P.max(0) + 200
        m = ~((np.maximum(g.A, g.B) < lo).any(1) | (np.minimum(g.A, g.B) > hi).any(1))
        segs = np.stack([g.A[m], g.B[m]], 1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=.5))
        rt = g.A[tr.route]; ax.add_collection(LineCollection(np.stack([g.A[tr.route], g.B[tr.route]], 1), colors="#2ca02c", linewidths=2))
        ax.scatter(g.P[:, 0], g.P[:, 1], s=2, c="#1f77b4", zorder=3)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); ax.set_title(f"trip {tid}  ({len(g.P)} pts)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("GPS (blue) vs road network (grey) vs reference route (green) - train, UTM metres"); fig.tight_layout()
    fig.savefig(out / "09_trajectory_examples.png", dpi=150); plt.close(fig)
    # 10 radius study
    rs = radius_study(root, out_dir=out)
    fig, ax = plt.subplots(figsize=(6.5, 4)); ax.plot(rs.radius_m, 100 * rs.candidate_miss_rate, "o-", color="#d62728")
    ax.set_xlabel("candidate radius (m)"); ax.set_ylabel("candidate miss rate, all points (%)", color="#d62728")
    ax2 = ax.twinx(); ax2.plot(rs.radius_m, rs.avg_candidates_per_point, "s--", color="#1f77b4"); ax2.set_ylabel("avg candidates / point", color="#1f77b4")
    ax.axvline(config.CAND_RADIUS_M, color="k", ls=":"); ax.set_title("Radius trade-off: miss rate vs candidate count (train)")
    fig.tight_layout(); fig.savefig(out / "10_radius_study.png", dpi=150); plt.close(fig)


def feature_report(out_dir=None, n_sample=600_000):
    """Evidence for 'why these features': per-feature univariate ROC-AUC + class means + correlation heatmap."""
    from sklearn.metrics import roc_auc_score
    from .dataset import load_candidates
    out = Path(out_dir or config.EDA_DIR)
    df = load_candidates("train"); df = df[df.label_valid & (df.fold >= 0)]
    df = df.sample(min(n_sample, len(df)), random_state=config.SEED)
    rows = []
    for c in config.FEATURE_COLUMNS:
        x = df[c].values.astype(float)
        a = roc_auc_score(df.label, x)
        rows.append({"feature": c, "auc_univariate": a, "auc_strength": abs(a - 0.5) * 2,
                     "mean_true_arc": x[df.label.values == 1].mean(), "mean_other_arcs": x[df.label.values == 0].mean()})
    r = pd.DataFrame(rows).sort_values("auc_strength", ascending=False)
    r.to_csv(out / "feature_report.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6)); rr = r.iloc[::-1]
    ax.barh(rr.feature, rr.auc_strength, color="#1f77b4"); ax.set_xlabel("|AUC - 0.5| x 2   (single-feature separating power)")
    ax.set_title("Each feature alone: true arc vs other candidates (train)"); fig.tight_layout()
    fig.savefig(out / "11_feature_strength.png", dpi=150); plt.close(fig)
    cm = df[config.FEATURE_COLUMNS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 8)); im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=7); fig.colorbar(im)
    ax.set_title("Feature correlation (Spearman, train sample)"); fig.tight_layout()
    fig.savefig(out / "12_feature_correlation.png", dpi=150); plt.close(fig)
    cls = df.label.value_counts()
    (out / "class_balance.json").write_text(json.dumps({"positive_rows": int(cls.get(1, 0)), "negative_rows": int(cls.get(0, 0)),
                                                        "positive_fraction": float(cls.get(1, 0) / cls.sum())}))
    return r
