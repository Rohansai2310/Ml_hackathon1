"""Post-inference evaluation and maps. Reference data never enter the decoder."""
import numpy as np
import pandas as pd
import folium

from . import config, data_io
from .evaluate_forest import reference_partition, pointwise_matches
from .metrics import IDS, evaluate


def arc_lines(trip, arc_ids):
    """Longitude/latitude lines with the original directed arc orientation."""
    return [trip.nodes[trip.arcs[int(arc_id)], :2].tolist() for arc_id in arc_ids]


def connected_reference(trip):
    """Split reference breaks too: no visual connector across disconnected roads."""
    segments = []
    for arc_id in trip.route:
        arc_id = int(arc_id)
        if not segments or trip.arcs[segments[-1][-1], 1] != trip.arcs[arc_id, 0]:
            segments.append([])
        segments[-1].append(arc_id)
    return [arc_lines(trip, segment) for segment in segments]


def folium_html(geometry, trip_id):
    gps = geometry['gps']
    latlon = [[lat, lon] for lon, lat in gps]
    m = folium.Map(location=latlon[0], zoom_start=15, tiles='OpenStreetMap', control_scale=True)
    def add_lines(name, lines, color, weight, show=True):
        layer = folium.FeatureGroup(name=name, show=show)
        for line in lines:
            folium.PolyLine([[lat, lon] for lon, lat in line], color=color,
                            weight=weight, opacity=.8).add_to(layer)
        layer.add_to(m)
    add_lines('Road network (grey)', geometry['roads'], '#8795a5', 2, False)
    add_lines('Raw GPS (blue)', [gps], '#2563eb', 2)
    points = folium.FeatureGroup(name='GPS observations (blue dots)')
    for i in range(0, len(gps), max(1, len(gps)//2000)):
        folium.CircleMarker(latlon[i], radius=2, color='#2563eb', weight=0,
                            fill=True, fill_opacity=.8, tooltip=f'GPS point {i}').add_to(points)
    points.add_to(m)
    for name, key, color, weight in [('Reference (green)', 'reference_segments', '#15803d', 7),
                                     ('RF + HMM (orange)', 'matched_segments', '#e56720', 4)]:
        # Separate line objects preserve every explicit segment break.
        add_lines(name, [line for segment in geometry[key] for line in segment], color, weight)
    folium.Marker(latlon[0], tooltip='Start', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(latlon[-1], tooltip='End', icon=folium.Icon(color='red')).add_to(m)
    m.fit_bounds(latlon)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = (f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
              f'padding:12px;border:1px solid #ddd;font:13px sans-serif">Trip {trip_id}<br>'
              '<span style="color:#2563eb">● GPS</span> · '
              '<span style="color:#15803d">━ Reference</span> · '
              '<span style="color:#e56720">━ RF + HMM</span></div>')
    m.get_root().html.add_child(folium.Element(legend))
    return m.get_root().render()


def present_result(response, trip, geom, candidates, probabilities, decoded):
    """All arguments are completed predictions. Only now open a reference file."""
    root = trip.path.parent.parent
    reference_trip = data_io.load_trip(trip.trip_id, trip.partition, with_route=True, root=root)
    if any(not np.array_equal(getattr(trip, name), getattr(reference_trip, name))
           for name in ('track', 'nodes', 'arcs')):
        raise ValueError('Trip inputs changed after prediction; retry with unchanged files')
    reference = reference_trip.route
    evaluation = {'status': 'unavailable', 'reason': 'No reference route is available for this trip.'}
    reference_segments = []
    if reference is not None and len(reference):
        if (reference < 0).any() or (reference >= len(trip.arcs)).any():
            raise ValueError('Reference route contains invalid arc IDs')
        partition = reference_partition(reference_trip, candidates)
        scores = candidates.rename(columns={'point_idx': 'observation_id', 'arc_id': 'candidate_id'})[IDS].copy()
        for name in IDS:
            scores[name] = scores[name].astype('string')
        scores['probability'] = np.asarray(probabilities)
        rf, _ = evaluate(partition.candidates, partition.inventory, scores)
        predictions = pd.DataFrame({'trip_id': trip.trip_id,
            'observation_id': pd.Series(np.arange(len(trip.track)), dtype='string'),
            'predicted_candidate_id': pd.Series(decoded.point_arc_ids, dtype='Int64').astype('string')})
        predictions['trip_id'] = predictions.trip_id.astype('string')
        hmm = pointwise_matches(partition.inventory, predictions)
        evaluation = {'status': 'complete', 'rf_top1_accuracy': rf['top1_accuracy'],
            'hmm_top1_accuracy': hmm['top1_accuracy'],
            'hmm_correct_points': hmm['correct_known_observations'],
            'known_points': hmm['known_observations'], 'unresolved_points': hmm['unresolved_observations'],
            'candidate_macro_f1': rf['macro_f1'], 'candidate_log_loss': rf['log_loss'],
            'note': 'Pointwise accuracy against valid derived references (within 50 m). '
                    'Repeat development evaluation; references are read only after inference.'}
        reference_segments = connected_reference(reference_trip)
    lo, hi = geom.P.min(0)-250, geom.P.max(0)+250
    near = ~((np.maximum(geom.A, geom.B) < lo).any(1) | (np.minimum(geom.A, geom.B) > hi).any(1))
    # Draw one geometry for opposite arcs; direction is retained in matched route IDs.
    keep = near & ((trip.arcs[:, 0] < trip.arcs[:, 1]) | ~geom.reverse_exists)
    geometry = {'gps': trip.track[:, :2].tolist(), 'roads': arc_lines(trip, np.flatnonzero(keep)),
                'matched_segments': [arc_lines(trip, segment) for segment in decoded.route_segments],
                'reference_segments': reference_segments}
    return {'match': response.model_dump(), 'evaluation': evaluation, 'geometry': geometry,
            'map_html': folium_html(geometry, trip.trip_id)}
