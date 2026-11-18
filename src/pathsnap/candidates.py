"""Candidate generation: for every GPS point, the road segments (arcs) that could
plausibly be the one the vehicle was on = all arcs within CAND_RADIUS_M, keeping at
most MAX_CANDIDATES nearest. The classifier later scores each (point, arc) pair.

Why: map-matching has ~100k+ arcs per trip, so we cannot classify over all of them.
A radius query turns it into a small ranking problem per point. Cost: if the true arc
is farther than the radius it can never be found -> reported as Candidate Miss Rate.
"""
import numpy as np
import shapely
from shapely import STRtree
from . import config
from .geo import point_segment
from .network import TripGeom


def generate_candidates(g: TripGeom, radius=None, max_k=None):
    """Returns pt_idx, arc_id, dist, t(frac along arc) - sorted by (pt_idx, dist)."""
    radius = radius or config.CAND_RADIUS_M
    max_k = max_k or config.MAX_CANDIDATES
    # 1) crop network to the track bounding box (+radius): big speed-up for 100k-arc nets
    lo, hi = g.P.min(0) - radius, g.P.max(0) + radius
    amin, amax = np.minimum(g.A, g.B), np.maximum(g.A, g.B)
    keep = np.where(~((amax < lo).any(1) | (amin > hi).any(1)))[0]
    if len(keep) == 0:
        z = np.empty(0)
        return z.astype(int), z.astype(int), z, z
    lines = shapely.linestrings(np.stack([g.A[keep], g.B[keep]], axis=1))
    tree = STRtree(lines)
    pts = shapely.points(g.P)
    pi, li = tree.query(pts, predicate="dwithin", distance=radius)
    arc = keep[li]
    d, t = point_segment(g.P[pi], g.A[arc], g.B[arc])
    order = np.lexsort((d, pi))                 # by point, then nearest first
    pi, arc, d, t = pi[order], arc[order], d[order], t[order]
    rank = np.arange(len(pi)) - np.searchsorted(pi, pi, side="left")
    m = rank < max_k
    return pi[m], arc[m], d[m], t[m]
