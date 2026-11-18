"""Reading raw PathSnap files. One folder per trip: <id>.track/.nodes/.arcs/.route

SAFETY (leakage rule): `load_trip(..., with_route=False)` never touches the .route
file. The API / inference path MUST use with_route=False so ground truth cannot
leak into prediction.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from . import config


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray            # (N,3) lon, lat, t_sec
    nodes: np.ndarray            # (V,2) lon, lat      (row index = node id)
    arcs: np.ndarray             # (E,2) from_node, to_node (row index = arc id)
    route: np.ndarray | None = None   # reference arc ids (chronological) or None
    path: Path | None = field(default=None, repr=False)


def _load(path, dtype=float, ndmin=2):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.empty((0, 0), dtype=dtype)
    return np.loadtxt(p, dtype=dtype, ndmin=ndmin)


def list_trips(partition: str, root: Path | None = None) -> list[str]:
    root = Path(root or config.DATA_DIR)
    d = root / partition
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def load_trip(trip_id: str, partition: str, with_route: bool = False,
              root: Path | None = None) -> Trip:
    root = Path(root or config.DATA_DIR)
    d = root / partition / trip_id
    track = _load(d / f"{trip_id}.track")
    nodes = _load(d / f"{trip_id}.nodes")
    arcs = _load(d / f"{trip_id}.arcs", dtype=int)
    route = None
    if with_route:
        route = _load(d / f"{trip_id}.route", dtype=int, ndmin=1).reshape(-1)
    return Trip(trip_id, partition, track, nodes, arcs, route, d)


def dataset_root_check(root: Path | None = None) -> str:
    """Friendly error if the zip was extracted into the wrong place."""
    root = Path(root or config.DATA_DIR)
    miss = [p for p in ("train", "student_test") if not (root / p).exists()]
    if miss:
        raise SystemExit(
            f"Missing {miss} under {root}.\nExtract 'Student folder.zip' so you have:\n"
            f"  {root}/train/<trip_id>/<trip_id>.track ...\n  {root}/student_test/<trip_id>/...\n"
            f"(or pass --data-root <path containing train/ and student_test/>)")
    return str(root)
