"""PathSnap - Person 1 command line.

    python -m pathsnap.cli check                      # is the dataset in the right folder?
    python -m pathsnap.cli explore                    # inventory + quality + plots + tables
    python -m pathsnap.cli build-features             # candidate/feature tables for train + student_test
    python -m pathsnap.cli feature-report             # why-these-features evidence
    python -m pathsnap.cli map --trip 00000048 --partition train
    python -m pathsnap.cli inspect-trip --trip 00000048 --partition train
    python -m pathsnap.cli all                        # everything above, in order
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from . import config, data_io as io, eda, build, maps


def cmd_check(a):
    root = Path(io.dataset_root_check(a.data_root))
    for p in eda.PARTITIONS:
        print(f"{p:13s}: {len(io.list_trips(p, root))} trips  ({root / p})")


def cmd_explore(a):
    io.dataset_root_check(a.data_root)
    df = eda.trip_table(a.data_root, cache=not getattr(a, "refresh", False))
    out = config.EDA_DIR
    inv = eda.partition_inventory(df); inv.to_csv(out / "dataset_inventory.csv"); print("\n== Dataset inventory\n", inv.round(2).to_string())
    mv = eda.missing_value_table(df); mv.to_csv(out / "missing_value_analysis.csv", index=False); print("\n== Missing / invalid values\n", mv.to_string(index=False))
    bad = df[~df.trip_ok][["trip_id", "partition", "n_points", "route_len_km", "path_len_km", "route_over_path", "frac_label_invalid"]]
    print("\n== Trips with unreliable reference route (excluded from training/CV)\n", bad.round(3).to_string(index=False))
    rs = eda.radius_study(a.data_root, cache=not getattr(a, "refresh", False)); print("\n== Candidate radius study (train, reliable trips)\n", rs.round(4).to_string(index=False))
    eda.make_plots(df, root=a.data_root)
    from .docgen import write_dq_record
    write_dq_record(df, rs)
    print(f"\nPlots/tables -> {out}\nData-quality record -> docs/data_quality_record.md")


def cmd_build(a):
    io.dataset_root_check(a.data_root)
    df = eda.trip_table(a.data_root)
    folds = build.make_folds(df[df.partition == "train"])
    fmap = dict(zip(df[df.partition == "train"].trip_id, folds))
    fmap.update({t: -1 for t in df[df.partition == "student_test"].trip_id})
    pd.DataFrame({"trip_id": list(fmap), "fold": list(fmap.values())}).to_csv(config.EDA_DIR / "folds.csv", index=False)
    logs = [build.build_partition(p, a.data_root, folds=fmap) for p in eda.PARTITIONS]
    build.write_schema(); (config.FEATURE_DIR / "build_log.json").write_text(json.dumps(logs, indent=2))
    print(json.dumps(logs, indent=2)); print(f"Tables -> {config.FEATURE_DIR}")


def cmd_feature_report(a):
    r = eda.feature_report(); print(r.round(3).to_string(index=False))


def cmd_map(a):
    f = maps.make_trip_map(a.trip, a.partition, a.data_root); print("Folium map ->", f)


def cmd_inspect(a):
    tr = io.load_trip(a.trip, a.partition, with_route=True, root=a.data_root)
    from .network import build_geom
    from .features import build_candidate_table
    from .labels import align_points_to_route
    g = build_geom(tr); df = build_candidate_table(g); _, lab, ld = align_points_to_route(g, tr.route)
    print(f"trip {a.trip} [{a.partition}]  GPS points={len(g.P)}  nodes={len(tr.nodes)}  arcs={len(tr.arcs)}  route arcs={len(tr.route)}")
    print(f"projection EPSG:{g.proj.epsg}   candidates/point={len(df)/len(g.P):.1f}   valid labels={(ld<=config.LABEL_MAX_DIST_M).mean():.1%}")
    print("\nFirst 3 GPS points and their candidates (nearest first):")
    cols = ["point_idx", "arc_id", "dist_m", "heading_diff_deg", "along_advance_m", "win_mean_dist", "dist_rank"]
    for i in range(3):
        print(f"-- point {i}: reference arc = {lab[i]} (label dist {ld[i]:.1f} m)")
        print(df[df.point_idx == i][cols].head(5).round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="folder containing train/ and student_test/ (default data/raw)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in [("check", cmd_check), ("explore", cmd_explore), ("build-features", cmd_build), ("feature-report", cmd_feature_report)]:
        p = sub.add_parser(n); p.set_defaults(fn=f)
        if n == "explore": p.add_argument("--refresh", action="store_true", help="ignore cached results")
    p = sub.add_parser("map"); p.add_argument("--trip", required=True); p.add_argument("--partition", default="train"); p.set_defaults(fn=cmd_map)
    p = sub.add_parser("inspect-trip"); p.add_argument("--trip", required=True); p.add_argument("--partition", default="train"); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("all"); p.set_defaults(fn=lambda a: [cmd_check(a), cmd_explore(a), cmd_build(a), cmd_feature_report(a), maps.make_trip_map("00000048", "train", a.data_root), maps.make_trip_map("00000056", "student_test", a.data_root)])
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
