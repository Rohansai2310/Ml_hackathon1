"""Metric HMM/Viterbi map matching, adapted from the person-3 implementation.

Only local bounded shortest paths are cached. Reference routes are never inputs.
Missing observations and disconnected/time-gap segments retain their identities.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_decoder_parameters(path, *, model_sha256):
    """Load training-selected settings bound to the same frozen classifier."""
    record = json.loads(Path(path).read_text())
    if record.get("model_sha256") != model_sha256:
        raise ValueError("Decoder configuration belongs to a different model")
    if record.get("evaluator_used_for_selection") is not False:
        raise ValueError("Decoder configuration must record evaluator-free selection")
    parameters = record["decoder_parameters"]
    allowed = {"beta", "same_arc_jitter_m", "emission_mode", "max_gap_s", "max_speed_mps", "max_transition_distance_m"}
    if not parameters or set(parameters) - allowed:
        raise ValueError("Invalid decoder parameter names")
    for name, value in parameters.items():
        if name == "emission_mode":
            if value not in {"probability", "log_odds"}:
                raise ValueError("Invalid emission mode")
        elif (not isinstance(value, (float, int)) or isinstance(value, bool)
              or not np.isfinite(value) or value < 0 or (name != "same_arc_jitter_m" and value == 0)):
            raise ValueError(f"Invalid decoder parameter: {name}")
    return parameters.copy()


@dataclass
class DecodeResult:
    point_arc_ids: list[int]       # -1 means no candidate; exactly one entry per GPS point
    segment_ids: list[int]         # -1 at unmatched observations
    route_segments: list[list[int]]
    break_reasons: dict[int, str]


class RoadGraph:
    """Directed, metric, parallel-arc-safe graph with a bounded LRU search cache."""

    def __init__(self, arcs, lengths, cache_size=512):
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=float)
        if (self.arcs.ndim != 2 or self.arcs.shape[1] != 2 or (self.arcs < 0).any()
                or self.lengths.shape != (len(self.arcs),)
                or not np.isfinite(self.lengths).all() or (self.lengths < 0).any()):
            raise ValueError("Require nonnegative directed node IDs and finite arc lengths in metres")
        self.adjacency = {}
        for arc_id, (source, target) in enumerate(self.arcs):
            self.adjacency.setdefault(int(source), []).append((int(target), arc_id))
        self.cache_size, self.cache = cache_size, OrderedDict()

    def search(self, source, limit):
        # Round the cache bound up, but enforce the exact limit on each transition.
        bound = float(np.ceil(limit / 100.0) * 100.0)
        key = (int(source), bound)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        distances, parents, queue = {int(source): 0.0}, {}, [(0.0, int(source))]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for target, arc_id in self.adjacency.get(node, ()):
                next_distance = distance + self.lengths[arc_id]
                if next_distance <= bound and next_distance < distances.get(target, np.inf):
                    distances[target], parents[target] = next_distance, arc_id
                    heapq.heappush(queue, (next_distance, target))
        self.cache[key] = (distances, parents)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return distances, parents

    def transition(self, previous_arc, previous_frac, current_arc, current_frac, limit,
                   *, same_arc_jitter_m=0.0):
        """Return a metric distance and intermediate arc IDs, including partial arcs."""
        progress = (current_frac - previous_frac) * self.lengths[previous_arc]
        if previous_arc == current_arc and progress >= -same_arc_jitter_m:
            # Small backwards projection noise is a stationary observation, not
            # permission to traverse an arc in the reverse direction.
            distance = max(0.0, progress)
            return (distance, []) if distance <= limit else (np.inf, [])
        source = int(self.arcs[previous_arc, 1])
        target = int(self.arcs[current_arc, 0])
        distances, parents = self.search(source, limit)
        distance = ((1 - previous_frac) * self.lengths[previous_arc]
                    + distances.get(target, np.inf) + current_frac * self.lengths[current_arc])
        if distance > limit or not np.isfinite(distance):
            return np.inf, []
        path, node = [], target
        while node != source:
            arc_id = parents[node]
            path.append(arc_id)
            node = int(self.arcs[arc_id, 0])
        return float(distance), path[::-1]


def viterbi_decode(candidates: pd.DataFrame, arcs, probabilities, *, arc_lengths,
                   gps_xy, timestamps, beta=50.0, max_gap_s=30.0,
                   max_speed_mps=60.0, max_transition_distance_m=5000.0,
                   same_arc_jitter_m=0.0, emission_mode="probability", return_details=False):
    """Decode label-free candidate scores into observation matches and road segments.

    Transition log potential is -abs(network_metres - GPS_displacement_metres)/beta.
    Backward projection beyond the configured noise tolerance requires a return path.
    Default settings are fixed design choices, not tuned on evaluator results.
    Deterministic ties prefer the lower numeric arc ID after candidate sorting.
    """
    required = {"point_idx", "arc_id", "frac_along"}
    if not required.issubset(candidates):
        raise ValueError(f"Missing candidate columns: {sorted(required - set(candidates))}")
    xy, times = np.asarray(gps_xy, dtype=float), np.asarray(timestamps, dtype=float)
    if (times.ndim != 1 or xy.shape != (len(times), 2) or not np.isfinite(xy).all()
            or not np.isfinite(times).all() or (np.diff(times) <= 0).any()):
        raise ValueError("GPS coordinates and strictly increasing timestamps must be finite and aligned")
    if not all(np.isfinite(value) and value > 0 for value in
               [beta, max_gap_s, max_speed_mps, max_transition_distance_m]):
        raise ValueError("Decoder scales, gap and transition limits must be positive and finite")
    if not np.isfinite(same_arc_jitter_m) or same_arc_jitter_m < 0:
        raise ValueError("Same-arc projection jitter must be finite and nonnegative")
    if emission_mode not in {"probability", "log_odds"}:
        raise ValueError("Unknown emission_mode")
    graph = RoadGraph(arcs, arc_lengths)
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (len(candidates),) or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("One finite probability in [0, 1] is required per candidate")
    data = candidates[["point_idx", "arc_id", "frac_along"]].copy()
    for name, upper in [("point_idx", len(times)), ("arc_id", len(graph.arcs))]:
        values = data[name].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values != np.floor(values)).any()
                or ((values < 0) | (values >= upper)).any()):
            raise ValueError(f"Invalid {name} index")
        data[name] = values.astype(np.int64)
    fractions = data.frac_along.to_numpy(dtype=float)
    if not np.isfinite(fractions).all() or ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("frac_along must be finite and in [0, 1]")
    if data.duplicated(["point_idx", "arc_id"]).any():
        raise ValueError("Duplicate candidate arc for an observation")
    data["emission"] = np.log(np.clip(probs, 1e-12, 1.0))
    if emission_mode == "log_odds":
        clipped = np.clip(probs, 1e-12, 1 - 1e-12)
        data["emission"] = np.log(clipped) - np.log1p(-clipped)
    groups = {int(i): group.sort_values("arc_id", kind="stable") for i, group in data.groupby("point_idx")}
    point_arcs = np.full(len(times), -1, dtype=int)
    segment_ids = np.full(len(times), -1, dtype=int)
    breaks, blocks, states, backs, scores = {}, [], [], [], None

    def finish():
        nonlocal states, backs, scores
        if not states:
            return
        chosen, index = [], int(np.argmax(scores))
        for state_index in range(len(states) - 1, -1, -1):
            point, ids, frac = states[state_index]
            chosen.append((point, int(ids[index]), float(frac[index])))
            if state_index:
                index = int(backs[state_index - 1][index])
        chosen.reverse()
        segment = len(blocks)
        for point, arc_id, _ in chosen:
            point_arcs[point], segment_ids[point] = arc_id, segment
        blocks.append(chosen)
        states, backs, scores = [], [], None

    def limit_between(first, second):
        # Candidate endpoints can sit up to the radius away from the observed GPS.
        return min(max_transition_distance_m,
                   max_speed_mps * (times[second] - times[first]) + 2 * config_radius)

    # Reuse the fixed candidate radius; no evaluator-specific threshold selection.
    from .config import CAND_RADIUS_M as config_radius

    for point in range(len(times)):
        group = groups.get(point)
        if group is None:
            finish()
            breaks[point] = "no_candidates"
            continue
        ids = group.arc_id.to_numpy(dtype=int)
        frac = group.frac_along.to_numpy(dtype=float)
        emissions = group.emission.to_numpy()
        if states and times[point] - times[states[-1][0]] > max_gap_s:
            finish()
            breaks[point] = "time_gap"
        if not states:
            states, scores = [(point, ids, frac)], emissions.copy()
            continue
        previous_point, previous_ids, previous_frac = states[-1]
        limit = limit_between(previous_point, point)
        displacement = float(np.linalg.norm(xy[point] - xy[previous_point]))
        distances = np.full((len(previous_ids), len(ids)), np.inf)
        current_sources = graph.arcs[ids, 0]
        for i, previous_arc in enumerate(previous_ids):
            if not np.isfinite(scores[i]):
                continue
            source = int(graph.arcs[previous_arc, 1])
            reached, _ = graph.search(source, limit)
            middle = np.array([reached.get(int(node), np.inf) for node in current_sources])
            distances[i] = ((1 - previous_frac[i]) * graph.lengths[previous_arc]
                            + middle + frac * graph.lengths[ids])
            progress = (frac - previous_frac[i]) * graph.lengths[previous_arc]
            same_hold = (ids == previous_arc) & (progress >= -same_arc_jitter_m)
            distances[i, same_hold] = np.maximum(0.0, progress[same_hold])
        distances[distances > limit] = np.inf
        totals = scores[:, None] - np.abs(distances - displacement) / beta
        back = np.argmax(totals, axis=0)
        next_scores = totals[back, np.arange(len(ids))] + emissions
        if not np.isfinite(next_scores).any():
            # Backtrack the old block independently; never invent a predecessor.
            finish()
            breaks[point] = "no_reachable_transition"
            states, scores = [(point, ids, frac)], emissions.copy()
        else:
            # Removing a common offset preserves ranking and bounds long-trip scores.
            scores = next_scores - np.max(next_scores)
            states.append((point, ids, frac))
            backs.append(back)
    finish()
    routes = []
    for block in blocks:
        route = [block[0][1]]
        for previous, current in zip(block, block[1:]):
            p, previous_arc, previous_frac = previous
            q, current_arc, current_frac = current
            distance, middle = graph.transition(previous_arc, previous_frac, current_arc,
                                                 current_frac, limit_between(p, q),
                                                 same_arc_jitter_m=same_arc_jitter_m)
            if not np.isfinite(distance):
                raise RuntimeError("Decoded transition cannot be reconstructed")
            if (previous_arc == current_arc and
                    (current_frac - previous_frac) * graph.lengths[previous_arc] >= -same_arc_jitter_m):
                continue
            route.extend(middle)
            route.append(current_arc)
        routes.append(route)
    result = DecodeResult(point_arcs.tolist(), segment_ids.tolist(), routes, breaks)
    return result if return_details else result.point_arc_ids
