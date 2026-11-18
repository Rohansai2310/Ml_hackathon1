"""Projected, ready-to-query geometry of one trip (points + road network)."""
from dataclasses import dataclass
import numpy as np
from .geo import Projector, bearing_deg
from .data_io import Trip


@dataclass
class TripGeom:
    trip: Trip
    proj: Projector
    P: np.ndarray            # (N,2) projected GPS points (m)
    t: np.ndarray            # (N,) seconds
    node_xy: np.ndarray      # (V,2)
    A: np.ndarray            # (E,2) arc start xy
    B: np.ndarray            # (E,2) arc end xy
    arc_len: np.ndarray      # (E,)
    arc_bearing: np.ndarray  # (E,) compass deg
    out_deg: np.ndarray      # (V,) out-degree of node
    in_deg: np.ndarray       # (V,) in-degree of node
    reverse_exists: np.ndarray  # (E,) bool: arc (b,a) exists in network


def build_geom(trip: Trip) -> TripGeom:
    track = trip.track
    proj = Projector(float(np.median(track[:, 0])), float(np.median(track[:, 1])))
    P = proj.to_xy(track[:, :2])
    node_xy = proj.to_xy(trip.nodes[:, :2])
    arcs = trip.arcs
    A, B = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
    d = B - A
    arc_len = np.hypot(d[:, 0], d[:, 1])
    V = len(node_xy)
    out_deg = np.bincount(arcs[:, 0], minlength=V)
    in_deg = np.bincount(arcs[:, 1], minlength=V)
    key = arcs[:, 0].astype(np.int64) * V + arcs[:, 1]
    rkey = arcs[:, 1].astype(np.int64) * V + arcs[:, 0]
    reverse_exists = np.isin(rkey, key)
    return TripGeom(trip, proj, P, track[:, 2].astype(float), node_xy, A, B, arc_len,
                    bearing_deg(d[:, 0], d[:, 1]), out_deg, in_deg, reverse_exists)
