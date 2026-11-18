"""Turning the reference route into one reference arc per GPS point.

The .route file is a short ordered list of arcs (e.g. 149 arcs for 1239 points), so
there is NO row-to-row correspondence with the track. We assign each GPS point to a
route arc with a monotone dynamic programme:

    choose j_1 <= j_2 <= ... <= j_N  minimising  sum_i  min(dist(P_i, route_arc[j_i]), CAP)

i.e. the vehicle can only move forward along the reference route (a point may stay on
the same arc, or jump ahead, never go back). Solved exactly in O(N*M) with a running
prefix-minimum. This uses ONLY training/student-test data and is never used at
inference. Caveat (documented in the report): this is a derived label, so label noise
exists wherever the GPS jumps by more than the CAP.
"""
import numpy as np
from .geo import point_segment
from .network import TripGeom

CAP_M = 150.0


def align_points_to_route(g: TripGeom, route: np.ndarray):
    """Returns (route_pos[N], label_arc[N], label_dist[N])."""
    N, M = len(g.P), len(route)
    A, B = g.A[route], g.B[route]
    cost = np.empty((N, M), dtype=np.float32)
    for j in range(M):                          # M columns, vectorised over N points
        d, _ = point_segment(g.P, np.repeat(A[j][None], N, 0), np.repeat(B[j][None], N, 0))
        cost[:, j] = d
    raw = cost.copy()
    np.minimum(cost, CAP_M, out=cost)
    back = np.zeros((N, M), dtype=np.int32)
    dp = cost[0].copy()
    ar = np.arange(M)
    for i in range(1, N):
        pm = np.minimum.accumulate(dp)
        idx = np.maximum.accumulate(np.where(dp == pm, ar, 0))   # argmin over k<=j
        back[i] = idx
        dp = cost[i] + pm
    j = int(np.argmin(dp))
    pos = np.empty(N, dtype=int)
    for i in range(N - 1, -1, -1):
        pos[i] = j
        j = back[i, j]
    lab = route[pos]
    return pos, lab, raw[np.arange(N), pos].astype(float)
