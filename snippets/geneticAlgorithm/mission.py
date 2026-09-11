"""Mission plan parsing and the flight segments obstacles are placed against."""

import json
import math
from dataclasses import dataclass
from typing import List, Tuple

Waypoint = Tuple[float, float]

# QGC .plan command codes that carry a position: waypoint, takeoff, land.
POSITION_COMMANDS = (16, 22, 21)
EARTH_RADIUS_M = 6378137.0
# A wall is up to 20 m long and sits between 20% and 80% of its segment, so a segment
# shorter than this cannot hold one without reaching the neighbouring segments.
# This drops takeoff hops and the 15-18 m corner stubs of missions 3, 4 and 7.
MIN_SEGMENT_LENGTH_M = 25.0


def wgs84ToLocal(lat: float, lon: float, homeLat: float, homeLon: float):
    """Convert a plan coordinate from degrees to metres relative to home."""
    # Equirectangular approximation; the mission area is a few hundred metres.
    dLat = math.radians(lat - homeLat)
    dLon = math.radians(lon - homeLon)
    return EARTH_RADIUS_M * dLat, EARTH_RADIUS_M * dLon * math.cos(math.radians(homeLat))


def parseWaypoints(planPath: str):
    """Return the mission waypoints in local metres, starting at home (0, 0)."""
    try:
        with open(planPath) as fh:
            plan = json.load(fh)
        items = plan["mission"]["items"]
        home = plan["mission"].get("plannedHomePosition", [0.0, 0.0, 0.0])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError("cannot read mission plan %s: %s" % (planPath, e))
    waypoints: List[Waypoint] = [(0.0, 0.0)]
    for item in items:
        params = item.get("params", [])
        if item.get("command") in POSITION_COMMANDS and len(params) >= 6:
            if params[4] is not None and params[5] is not None:
                waypoints.append(wgs84ToLocal(params[4], params[5], home[0], home[1]))
    if len(waypoints) < 2:
        raise ValueError("mission plan %s has no positioned waypoints" % planPath)
    return waypoints


@dataclass(frozen=True)
class Segment:
    """One straight flight segment between two consecutive waypoints."""

    index: int
    a: Waypoint
    b: Waypoint

    def length(self):
        """Return the segment length in metres."""
        return math.dist(self.a, self.b)

    def direction(self):
        """Return the unit vector pointing from a to b."""
        length = self.length()
        return (self.b[0] - self.a[0]) / length, (self.b[1] - self.a[1]) / length

    def sideNormal(self):
        """Return the unit normal of the segment that side=+1 leans towards."""
        # Named by the gene, not by a compass direction. parseWaypoints puts
        # latitude in x and longitude in y, so the frame is (north, east) and
        # this normal is the drone's left; the search only ever needs the two
        # sides to be told apart, and one of them scores while the other never
        # has. See tests/test_mission.py for the pinned frame.
        ux, uy = self.direction()
        return uy, -ux

    def pointAt(self, t: float):
        """Return the point a fraction t along the segment (0 = a, 1 = b)."""
        return (
            self.a[0] + t * (self.b[0] - self.a[0]),
            self.a[1] + t * (self.b[1] - self.a[1]),
        )


def allSegments(waypoints: List[Waypoint]):
    """Return every segment of the mission with a non-zero length."""
    segments = [Segment(i, waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1)]
    return [segment for segment in segments if segment.length() > 1e-6]


def longSegments(waypoints: List[Waypoint], minLength: float = MIN_SEGMENT_LENGTH_M):
    """Return the segments long enough to carry a wall, in flight order."""
    return [segment for segment in allSegments(waypoints) if segment.length() >= minLength]
