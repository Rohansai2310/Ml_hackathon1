"""Geometry in METRES. Lon/lat are angles, so Euclidean distance in degrees is wrong
(1 deg lon != 1 deg lat, and it changes with latitude). Each trip is projected to the
UTM zone of its own track centroid (trips lie in different countries); UTM is
conformal with <0.04% scale error inside a zone - far below GPS noise."""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lo, la = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lo, la])


def point_segment(P, A, B):
    """Vectorised: distance from points P to segments A->B (all (n,2)).
    Returns (dist, t in [0,1] along segment)."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    t = np.einsum("ij,ij->i", P - A, AB) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(np.where(L2 > 0, t, 0.0), 0.0, 1.0)
    C = A + AB * t[:, None]
    d = np.hypot(*(P - C).T)
    return d, t


def bearing_deg(dx, dy):
    """Compass bearing (0=N, clockwise) from planar deltas."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def ang_diff_deg(a, b):
    """Smallest absolute angle difference in [0,180]."""
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)
