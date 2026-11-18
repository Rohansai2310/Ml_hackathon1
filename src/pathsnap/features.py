"""Feature engineering for (GPS point, candidate arc) pairs.  NO labels are used here,
so the exact same function runs at training time and inside the API at inference.

Feature groups (each one answers a question a human would ask when snapping a point):
 1. Distance   : how far is the point from this arc?  (dist_m, dist_rank, dist_rel_min)
 2. Position   : where along the arc, how long is the arc, how close to its ends?
 3. Direction  : does the vehicle's heading agree with the arc direction? Two-way roads
                 give two arcs (a->b, b->a) at the same distance - only direction separates
                 them (heading_diff_deg, cos_heading, along_advance_m).
 4. Motion     : speed and sampling gaps (speed_mps, dt_prev, dt_next).
 5. Context    : does the arc also fit the neighbouring GPS points (+-WINDOW)? A noisy single
                 point may sit nearer a parallel street, but the neighbours vote for the
                 right road (win_* features).
 6. Topology   : junction structure of the arc (out_deg_to, in_deg_from, has_reverse).
"""
import numpy as np
import pandas as pd
from . import config
from .geo import point_segment, bearing_deg, ang_diff_deg
from .network import TripGeom
from .candidates import generate_candidates

MAX_NEIGHBOUR_GAP_S = 10.0


def _neighbour_index(t, off):
    """Index of neighbour i+off; invalid (-> self, mask False) if out of range or far in time."""
    N = len(t)
    j = np.arange(N) + off
    ok = (j >= 0) & (j < N)
    jj = np.clip(j, 0, N - 1)
    ok &= np.abs(t[jj] - t) <= MAX_NEIGHBOUR_GAP_S * abs(off)
    return np.where(ok, jj, np.arange(N)), ok


def point_motion(g: TripGeom):
    """Per-point speed (m/s), heading (deg), time gaps. Central difference where possible."""
    P, t = g.P, g.t
    N = len(P)
    ip, okp = _neighbour_index(t, -1)
    inx, okn = _neighbour_index(t, +1)
    dxy = P[inx] - P[ip]
    dt = t[inx] - t[ip]
    speed = np.where(dt > 0, np.hypot(dxy[:, 0], dxy[:, 1]) / np.where(dt > 0, dt, 1), 0.0)
    head = bearing_deg(dxy[:, 0], dxy[:, 1])
    head_ok = (speed >= config.HEADING_MIN_SPEED) & (dt > 0)
    dt_prev = np.where(okp, t - t[ip], 0.0)
    dt_next = np.where(okn, t[inx] - t, 0.0)
    return speed, head, head_ok, dt_prev, dt_next, np.hypot(dxy[:, 0], dxy[:, 1])


def build_candidate_table(g: TripGeom) -> pd.DataFrame:
    """One row per (point, candidate arc) with all model features. Label-free."""
    N = len(g.P)
    pi, arc, dist, frac = generate_candidates(g)
    n = len(pi)
    if n == 0:
        return pd.DataFrame(columns=["trip_id", "point_idx", "arc_id"] + config.FEATURE_COLUMNS)

    speed, head, head_ok, dt_prev, dt_next, chord = point_motion(g)
    A, B = g.A[arc], g.B[arc]
    AB = B - A
    L = g.arc_len[arc]
    Lsafe = np.where(L > 0, L, 1.0)
    unit = AB / Lsafe[:, None]

    # 1/2 distance & position -------------------------------------------------
    first = np.searchsorted(pi, pi, side="left")
    rank = np.arange(n) - first
    cnt = np.bincount(pi, minlength=N)
    mind = np.minimum.reduceat(dist, np.r_[0, np.cumsum(cnt)[:-1]][cnt > 0])
    min_per_pt = np.zeros(N); min_per_pt[cnt > 0] = mind
    to_d = np.hypot(*(g.P[pi] - B).T)
    fr_d = np.hypot(*(g.P[pi] - A).T)

    # 3 direction ---------------------------------------------------------------
    hd = ang_diff_deg(head[pi], g.arc_bearing[arc])
    ok = head_ok[pi]
    hd = np.where(ok, hd, 90.0)                 # neutral when vehicle (almost) stationary
    cosh = np.where(ok, np.cos(np.radians(hd)), 0.0)

    def s_along(idx):                           # unclamped metres along arc direction
        return np.einsum("ij,ij->i", g.P[idx] - A, unit)

    ip, okp = _neighbour_index(g.t, -1)
    inx, okn = _neighbour_index(g.t, +1)
    adv = s_along(inx[pi]) - s_along(ip[pi])    # signed progress along arc across i-1 -> i+1
    resid = np.abs(adv) - chord[pi]             # ~0 if arc is parallel to the motion

    # 5 context window ----------------------------------------------------------
    dsum = np.zeros(n); dmax = np.zeros(n); dmin = np.full(n, np.inf)
    hits = np.zeros(n); valid = np.zeros(n)
    for off in [o for o in range(-config.WINDOW, config.WINDOW + 1) if o != 0]:
        j, okj = _neighbour_index(g.t, off)
        d_o, _ = point_segment(g.P[j[pi]], A, B)
        m = okj[pi]
        dsum += np.where(m, d_o, 0); dmax = np.maximum(dmax, np.where(m, d_o, 0))
        dmin = np.minimum(dmin, np.where(m, d_o, np.inf))
        hits += np.where(m, d_o <= config.CAND_RADIUS_M / 2, 0); valid += m
    v = np.where(valid > 0, valid, 1)
    wmean = np.where(valid > 0, dsum / v, dist)              # no neighbours -> own distance
    wmin = np.where(np.isfinite(dmin), dmin, dist)
    whit = np.where(valid > 0, hits / v, 0.0)

    df = pd.DataFrame({
        "trip_id": g.trip.trip_id, "point_idx": pi.astype(np.int32), "arc_id": arc.astype(np.int64),
        "dist_m": dist, "frac_along": frac, "arc_len_m": L,
        "dist_rank": rank, "dist_rel_min": dist - min_per_pt[pi], "n_cand": cnt[pi],
        "heading_diff_deg": hd, "cos_heading": cosh,
        "speed_mps": speed[pi], "dt_prev": dt_prev[pi], "dt_next": dt_next[pi],
        "win_mean_dist": wmean, "win_max_dist": dmax, "win_min_dist": wmin, "win_hit_frac": whit,
        "out_deg_to": g.out_deg[g.trip.arcs[arc, 1]], "in_deg_from": g.in_deg[g.trip.arcs[arc, 0]],
        "has_reverse": g.reverse_exists[arc].astype(np.int8),
        "to_node_dist_m": to_d, "from_node_dist_m": fr_d,
        "along_advance_m": adv, "speed_dist_resid": resid,
    })
    return df
