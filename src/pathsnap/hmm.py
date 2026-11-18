"""HMM/Viterbi decoding for PathSnap candidate road arcs."""

from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import pandas as pd


def _shortest_paths(arcs: np.ndarray) -> list[dict[int, int]]:
    """Build directed unweighted shortest-path distances between road nodes."""
    adjacency = defaultdict(list)

    for arc_id, (u, v) in enumerate(arcs):
        adjacency[int(u)].append((int(v), arc_id))

    distances = []
    node_count = int(arcs.max()) + 1 if arcs.size else 0

    for source in range(node_count):
        dist = {source: 0}
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for neighbour, _ in adjacency[node]:
                if neighbour not in dist:
                    dist[neighbour] = dist[node] + 1
                    queue.append(neighbour)

        distances.append(dist)

    return distances


def viterbi_decode(
    candidates: pd.DataFrame,
    arcs: np.ndarray,
    probabilities,
    *,
    beta: float = 50.0,
    max_transition_distance: int = 30,
) -> list[int]:
    """
    Decode one road arc per GPS point.

    candidates must contain point_idx, arc_id and frac_along.
    probabilities contains one classifier probability per candidate row.
    """
    required = {"point_idx", "arc_id", "frac_along"}
    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")

    if len(candidates) == 0:
        return []

    probs = np.asarray(probabilities, dtype=float)

    if len(probs) != len(candidates):
        raise ValueError("One probability is required for every candidate row.")

    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities must be finite.")

    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    data = candidates.reset_index(drop=True).copy()
    data["_prob"] = np.clip(probs, 1e-12, 1.0)

    arc_array = np.asarray(arcs, dtype=int)
    if arc_array.ndim != 2 or arc_array.shape[1] != 2:
        raise ValueError("arcs must have shape (number_of_arcs, 2).")

    if data["arc_id"].min() < 0 or data["arc_id"].max() >= len(arc_array):
        raise ValueError("Candidate arc_id is outside the road network.")

    if data["frac_along"].isna().any():
        raise ValueError("frac_along cannot contain missing values.")

    groups = [
        group.sort_values("arc_id").reset_index(drop=True)
        for _, group in data.groupby("point_idx", sort=True)
    ]

    if not groups:
        return []

    shortest = _shortest_paths(arc_array)
    states = []

    for group in groups:
        states.append(group)

    scores = np.log(states[0]["_prob"].to_numpy())
    backpointers = []

    for index in range(1, len(states)):
        previous = states[index - 1]
        current = states[index]

        previous_arcs = previous["arc_id"].to_numpy(dtype=int)
        current_arcs = current["arc_id"].to_numpy(dtype=int)
        previous_frac = previous["frac_along"].to_numpy(dtype=float)
        current_frac = current["frac_along"].to_numpy(dtype=float)

        next_scores = np.full(len(current), -np.inf)
        back = np.zeros(len(current), dtype=int)

        for j, current_arc in enumerate(current_arcs):
            current_start, current_end = arc_array[current_arc]

            for k, previous_arc in enumerate(previous_arcs):
                previous_start, previous_end = arc_array[previous_arc]

                if previous_arc == current_arc:
                    road_distance = max(0.0, current_frac[j] - previous_frac[k])
                else:
                    node_dist = shortest[int(previous_end)].get(
                        int(current_start)
                    )

                    if node_dist is None or node_dist > max_transition_distance:
                        continue

                    road_distance = (
                        (1.0 - previous_frac[k])
                        + node_dist
                        + current_frac[j]
                    )

                transition_score = -abs(road_distance) / max(beta, 1e-9)
                score = scores[k] + transition_score

                if score > next_scores[j]:
                    next_scores[j] = score
                    back[j] = k

        emissions = np.log(current["_prob"].to_numpy())
        next_scores += emissions

        if not np.isfinite(next_scores).any():
            # Disconnected candidates: restart at this point rather than crash.
            next_scores = emissions.copy()
            back = np.zeros(len(current), dtype=int)

        scores = next_scores
        backpointers.append(back)

    state_index = int(np.argmax(scores))
    result = [0] * len(states)

    for i in range(len(states) - 1, -1, -1):
        result[i] = int(states[i].iloc[state_index]["arc_id"])
        if i > 0:
            state_index = int(backpointers[i - 1][state_index])

    return result