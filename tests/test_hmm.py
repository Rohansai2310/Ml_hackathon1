"""Small directed graphs verify actual decoding and route reconstruction."""
import itertools

import numpy as np
import pandas as pd
import pytest

from pathsnap.hmm import RoadGraph, viterbi_decode


def decode(rows, arcs, lengths, probabilities, xy=None, times=None):
    candidates = pd.DataFrame(rows, columns=['point_idx', 'arc_id', 'frac_along'])
    n = int(candidates.point_idx.max()) + 1 if len(candidates) else 3
    return viterbi_decode(candidates, np.array(arcs), probabilities,
                          arc_lengths=lengths, gps_xy=np.array(xy if xy is not None else [[10*i, 0] for i in range(n)]),
                          timestamps=np.array(times if times is not None else range(n)), return_details=True)


def test_distances_are_metres_and_parallel_arcs_choose_shortest():
    graph = RoadGraph([[0, 1], [0, 2], [2, 3], [3, 1], [0, 1]], [1000, 10, 10, 10, 40])
    distances, _ = graph.search(0, 100)
    assert distances[1] == 30
    assert 1000 not in distances.values()
    graph = RoadGraph([[0, 1], [0, 1]], [100, 10])
    assert graph.search(0, 200)[0][1] == 10


def test_partial_arcs_and_backward_travel_require_a_directed_cycle():
    graph = RoadGraph([[0, 1]], [100])
    assert graph.transition(0, .2, 0, .8, 200)[0] == pytest.approx(60)
    assert np.isinf(graph.transition(0, .8, 0, .2, 200)[0])
    graph = RoadGraph([[0, 1], [1, 0]], [100, 100])
    distance, middle = graph.transition(0, .8, 0, .2, 200)
    assert distance == pytest.approx(140) and middle == [1]


def test_global_path_rejects_a_disconnected_high_emission():
    result = decode([(0,0,.5),(1,1,.5),(1,3,.5),(2,2,.5)],
                    [[0,1],[1,2],[2,3],[10,11]], [10]*4, [.9,.3,.99,.9])
    assert result.point_arc_ids == [0,1,2]
    assert result.route_segments == [[0,1,2]]
    assert result.segment_ids == [0,0,0]


def test_reset_does_not_corrupt_the_best_preceding_path():
    result = decode([(0,0,.2),(0,1,.2),(1,0,.3),(1,1,.3),(2,2,.5)],
                    [[0,1],[10,11],[20,21]], [100]*3, [.1,.9,.1,.9,.9])
    assert result.point_arc_ids == [1,1,2]
    assert result.route_segments == [[1],[2]]
    assert result.segment_ids == [0,0,1]
    assert result.break_reasons == {2:'no_reachable_transition'}


def test_missing_observations_and_large_time_gaps_remain_explicit():
    result = decode([(0,0,.2),(2,0,.8)], [[0,1]], [100], [.9,.9])
    assert result.point_arc_ids == [0,-1,0]
    assert result.segment_ids == [0,-1,1]
    assert result.break_reasons == {1:'no_candidates'}
    result = decode([(0,0,.2),(1,0,.8)], [[0,1]], [100], [.9,.9], times=[0,100])
    assert result.route_segments == [[0],[0]]
    assert result.break_reasons == {1:'time_gap'}


def test_route_reconstruction_includes_unobserved_intermediate_arcs():
    result = decode([(0,0,.5),(1,2,.5)], [[0,1],[1,2],[2,3]], [10]*3, [.9,.9])
    assert result.point_arc_ids == [0,2]
    assert result.route_segments == [[0,1,2]]


def test_viterbi_matches_exhaustive_sequence_scoring():
    arcs = np.array([[0,1],[0,2],[1,3],[2,3],[3,4],[3,5]])
    lengths = [100]*6
    rows = [(point, arc, .5) for point, ids in enumerate([[0,1],[2,3],[4,5]]) for arc in ids]
    probabilities = [.6,.4,.2,.8,.7,.3]
    result = decode(rows, arcs, lengths, probabilities, xy=[[0,0],[100,0],[200,0]])
    graph = RoadGraph(arcs, lengths)
    ranked = []
    for choices in itertools.product([0,1], repeat=3):
        path = [2*i+choice for i,choice in enumerate(choices)]
        score = sum(np.log(probabilities[arc]) for arc in path)
        for previous,current in zip(path,path[1:]):
            distance,_ = graph.transition(previous,.5,current,.5,210)
            score -= abs(distance-100)/50
        ranked.append((score,path))
    expected = max(ranked,key=lambda value:value[0])[1]
    assert result.point_arc_ids == expected


def test_local_search_does_not_expand_the_whole_network():
    arcs = np.column_stack([np.arange(20000),np.arange(1,20001)])
    graph = RoadGraph(arcs,np.full(20000,10),cache_size=2)
    for source in [0,1,2]:
        distances,_ = graph.search(source,30)
        assert len(distances) <= 11  # Cache rounds to 100 m, not all 20,001 nodes.
    assert len(graph.cache) == 2


def test_no_candidates_preserves_every_gps_point():
    result = decode([],[[0,1]],[10],[])
    assert result.point_arc_ids == [-1,-1,-1]
    assert result.route_segments == []


@pytest.mark.parametrize('fraction,probability', [(np.nan,.5),(np.inf,.5),(-.1,.5),(.5,np.nan),(.5,1.1)])
def test_invalid_scores_and_fractions_fail(fraction,probability):
    with pytest.raises(ValueError):
        decode([(0,0,fraction)],[[0,1]],[10],[probability])
