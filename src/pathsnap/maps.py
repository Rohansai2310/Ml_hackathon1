"""Interactive Folium map of one trip (Deliverable section 6).
Layers (toggle in the layer control, top-right):
  blue dots/line = raw GPS observations   grey = road network   green = reference route
  red = predicted route (only if `predicted_arcs` is passed - Person 3/4 reuse this)
"""
from pathlib import Path
import numpy as np
import folium
from . import config, data_io as io
from .network import build_geom

LEGEND = """
<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:10px 12px;
border:1px solid #888;border-radius:6px;font-size:13px;line-height:1.5">
<b>Trip {tid} ({part})</b><br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS observations ({n} pts)<br>
<span style="color:#888">&#9473;</span> Road network (arcs near trip)<br>
<span style="color:#2ca02c">&#9473;</span> Reference route (reference label)<br>
{pred}<span style="color:#2ca02c">&#9650;</span> start &nbsp; <span style="color:#d62728">&#9632;</span> end
</div>"""


def _lines(g, arc_ids):
    ll_a = g.proj.to_lonlat(g.A[arc_ids]); ll_b = g.proj.to_lonlat(g.B[arc_ids])
    return [[[a[1], a[0]], [b[1], b[0]]] for a, b in zip(ll_a, ll_b)]


def make_trip_map(trip_id, partition, root=None, out_file=None, predicted_arcs=None,
                  with_reference=True, margin_m=250.0, max_points=2500):
    tr = io.load_trip(trip_id, partition, with_route=with_reference, root=root)
    g = build_geom(tr)
    lon, lat = tr.track[:, 0], tr.track[:, 1]
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    # road network near the trip, one line per road (drop the duplicate reverse arc)
    lo, hi = g.P.min(0) - margin_m, g.P.max(0) + margin_m
    near = ~((np.maximum(g.A, g.B) < lo).any(1) | (np.minimum(g.A, g.B) > hi).any(1))
    a = tr.arcs
    keep = near & ((a[:, 0] < a[:, 1]) | ~g.reverse_exists)
    ids = np.where(keep)[0]
    fg = folium.FeatureGroup(name=f"Road network ({len(ids)} segments)", show=True)
    folium.PolyLine(_lines(g, ids), color="#888888", weight=1.5, opacity=.8).add_to(fg); fg.add_to(m)
    if with_reference and tr.route is not None:
        fr = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_lines(g, tr.route), color="#2ca02c", weight=6, opacity=.75, tooltip="reference route").add_to(fr); fr.add_to(m)
    pred_html = ""
    if predicted_arcs is not None and len(predicted_arcs):
        fp = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_lines(g, np.asarray(predicted_arcs)), color="#d62728", weight=3, opacity=.9, dash_array="6 6", tooltip="predicted route").add_to(fp); fp.add_to(m)
        pred_html = '<span style="color:#d62728">&#9473;&#9473;</span> Predicted route<br>'
    fo = folium.FeatureGroup(name="GPS observations", show=True)
    folium.PolyLine(list(zip(lat, lon)), color="#1f77b4", weight=1, opacity=.5).add_to(fo)
    step = max(1, len(lat) // max_points)
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2.5, color="#1f77b4", fill=True, fill_opacity=.9, weight=0,
                            popup=f"point {i}, t={tr.track[i,2]:.0f}s").add_to(fo)
    fo.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=trip_id, part=partition, n=len(lat), pred=pred_html)))
    out = Path(out_file or config.MAP_DIR / f"{partition}_{trip_id}.html"); out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out)); return out
