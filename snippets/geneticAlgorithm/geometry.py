"""Ground-plane geometry of box obstacles: overlap, bounds, axes, distances.

An obstacle is anything with `size.l`, `size.w`, `position.x`, `position.y`
and `position.r` (yaw in degrees). Every function here is pure and needs no
simulator.
"""

import math
from typing import List, Tuple

Point = Tuple[float, float]
# (xMin, xMax, yMin, yMax) of the placement area.
Bounds = Tuple[float, float, float, float]

# Footprints that merely touch count as separate.
OVERLAP_EPS = 1e-9


def boxAxes(o):
    """Return the unit axes of the rotated footprint: u along l, v along w."""
    rad = math.radians(o.position.r)
    cosR, sinR = math.cos(rad), math.sin(rad)
    return (cosR, sinR), (-sinR, cosR)


def boxCorners(o):
    """Return the four footprint corners in order around the box."""
    (ux, uy), (vx, vy) = boxAxes(o)
    halfL, halfW = 0.5 * o.size.l, 0.5 * o.size.w
    return [
        (
            o.position.x + sL * halfL * ux + sW * halfW * vx,
            o.position.y + sL * halfL * uy + sW * halfW * vy,
        )
        for sL, sW in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0))
    ]


def projectionRadius(o, nx: float, ny: float):
    """Return the half-width of the footprint's shadow on the unit axis n."""
    (ux, uy), (vx, vy) = boxAxes(o)
    return 0.5 * o.size.l * abs(ux * nx + uy * ny) + 0.5 * o.size.w * abs(vx * nx + vy * ny)


def overlaps(a, b):
    """Return whether two rotated footprints overlap."""
    # Separating axis theorem: two rectangles are apart as soon as one of
    # their four edge normals shows a gap between the two shadows.
    dx = b.position.x - a.position.x
    dy = b.position.y - a.position.y
    for nx, ny in boxAxes(a) + boxAxes(b):
        gap = abs(dx * nx + dy * ny) - projectionRadius(a, nx, ny) - projectionRadius(b, nx, ny)
        if gap > -OVERLAP_EPS:
            return False
    return True


def hasOverlap(obstacles: List):
    """Return whether any two obstacles of the layout overlap."""
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            if overlaps(obstacles[i], obstacles[j]):
                return True
    return False


def rotatedHalfExtents(l: float, w: float, rDeg: float):
    """Return the half-extents of the axis-aligned box around a rotated l x w box."""
    cosR = abs(math.cos(math.radians(rDeg)))
    sinR = abs(math.sin(math.radians(rDeg)))
    return 0.5 * (l * cosR + w * sinR), 0.5 * (l * sinR + w * cosR)


def fitsInBounds(bounds: Bounds, obstacles: List):
    """Return whether every rotated footprint stays inside the placement area."""
    xMin, xMax, yMin, yMax = bounds
    for o in obstacles:
        hx, hy = rotatedHalfExtents(o.size.l, o.size.w, o.position.r)
        if o.position.x - hx < xMin or o.position.x + hx > xMax:
            return False
        if o.position.y - hy < yMin or o.position.y + hy > yMax:
            return False
    return True


def wallAxis(o):
    """Return the two end points of the long centre line of a wall."""
    # The competition keeps r in [0, 90], so a wall may be stored with l < w:
    # the long side is then along the v axis instead of the u axis.
    (ux, uy), (vx, vy) = boxAxes(o)
    if o.size.l >= o.size.w:
        half, (ax, ay) = 0.5 * o.size.l, (ux, uy)
    else:
        half, (ax, ay) = 0.5 * o.size.w, (vx, vy)
    x, y = o.position.x, o.position.y
    return (x - half * ax, y - half * ay), (x + half * ax, y + half * ay)


def cross(o: Point, a: Point, b: Point):
    """Return the z component of (a - o) x (b - o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segmentsIntersect(p1: Point, p2: Point, q1: Point, q2: Point):
    """Return whether the closed segments p1-p2 and q1-q2 share a point."""
    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    if (d1 * d2 < 0.0) and (d3 * d4 < 0.0):
        return True
    # Touching or collinear cases: an end point lies on the other segment.
    return (
        pointToSegmentDistance(p1, q1, q2) < OVERLAP_EPS
        or pointToSegmentDistance(p2, q1, q2) < OVERLAP_EPS
        or pointToSegmentDistance(q1, p1, p2) < OVERLAP_EPS
        or pointToSegmentDistance(q2, p1, p2) < OVERLAP_EPS
    )


def pointToSegmentDistance(p: Point, a: Point, b: Point):
    """Return the distance from p to the closed segment a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    lengthSquared = dx * dx + dy * dy
    if lengthSquared <= 1e-18:
        return math.dist(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lengthSquared
    t = max(0.0, min(1.0, t))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def segmentDistance(a1: Point, a2: Point, b1: Point, b2: Point):
    """Return the shortest distance between two closed segments."""
    if segmentsIntersect(a1, a2, b1, b2):
        return 0.0
    return min(
        pointToSegmentDistance(a1, b1, b2),
        pointToSegmentDistance(a2, b1, b2),
        pointToSegmentDistance(b1, a1, a2),
        pointToSegmentDistance(b2, a1, a2),
    )


def footprintGap(a, b):
    """Return the shortest ground distance between two box footprints."""
    # For convex polygons the closest points lie on the edges, so the gap is
    # the minimum over the edge pairs.
    if overlaps(a, b):
        return 0.0
    cornersA, cornersB = boxCorners(a), boxCorners(b)
    edgesA = list(zip(cornersA, cornersA[1:] + cornersA[:1]))
    edgesB = list(zip(cornersB, cornersB[1:] + cornersB[:1]))
    return min(
        segmentDistance(a1, a2, b1, b2) for a1, a2 in edgesA for b1, b2 in edgesB
    )
