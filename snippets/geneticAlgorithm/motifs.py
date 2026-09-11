"""Turn genes into obstacles: the single wall and the chevron gate.

Both motifs are built in the frame of the attacked segment (unit direction u,
side normal n). Everything the planner reacts to, the crossing point, the
short tip and the gate opening, is a gene; the box coordinates follow.
"""

import math
from typing import List

from aerialist.px4.obstacle import Obstacle

from geneticAlgorithm.genes import KIND_GATE, Genes
from geneticAlgorithm.geometry import (
    Bounds,
    fitsInBounds,
    footprintGap,
    hasOverlap,
    segmentsIntersect,
    wallAxis,
)
from geneticAlgorithm.mission import Segment

# The thinnest allowed wall: the drone reacts to the wall's axis, extra
# thickness only costs placement room. 25 m is far above the 10 m flight
# height, so the drone can never go over.
WALL_THICKNESS_M = 2.0
WALL_HEIGHT_M = 25.0
# The blocker of the gate leans back at this angle to the segment and starts this
# far ahead of the wall tip. Both values come from the gate_r7 probes (8/8
# runs in the 2-point tier); moving the blocker further ahead only got worse.
GATE_BLOCKER_DEG = 30.0
GATE_AHEAD_M = 3.0
# Bisection range and steps for the blocker offset: 30 m / 2^40 is far below
# a millimetre.
GATE_OFFSET_MAX_M = 30.0
GATE_SOLVE_STEPS = 40


def makeWall(start, axisDeg: float, length: float, thickness: float = WALL_THICKNESS_M):
    """Return a wall that begins at start and runs length metres along axisDeg."""
    ax, ay = math.cos(math.radians(axisDeg)), math.sin(math.radians(axisDeg))
    cx = start[0] + 0.5 * length * ax
    cy = start[1] + 0.5 * length * ay
    r = axisDeg % 180.0
    l, w = length, thickness
    if r > 90.0:
        # The competition keeps r in [0, 90]: the same box rotated by 90 deg
        # more is the box with l and w swapped.
        r, l, w = r - 90.0, w, l
    return Obstacle(Obstacle.Size(l=l, w=w, h=WALL_HEIGHT_M), Obstacle.Position(x=cx, y=cy, z=0, r=r))


def tiltedAxis(segment: Segment, side: int, angleDeg: float):
    """Return the unit vector at angleDeg from the segment, leaning towards side."""
    ux, uy = segment.direction()
    nx, ny = segment.sideNormal()
    c, s = math.cos(math.radians(angleDeg)), side * math.sin(math.radians(angleDeg))
    return c * ux + s * nx, c * uy + s * ny


def decodeWall(genes: Genes, segment: Segment):
    """Return the wall crossing the segment at t with genes.overhang past it on genes.side."""
    # Retrieve the unit vectors defining the obstacle direction along the larger edge
    ax, ay = tiltedAxis(segment, genes.side, genes.crossingDeg)
    crossX, crossY = segment.pointAt(genes.t)
    back = genes.length - genes.overhang
    start = (crossX - back * ax, crossY - back * ay)
    return makeWall(start, math.degrees(math.atan2(ay, ax)), genes.length)


def decodeGate(genes: Genes, segment: Segment):
    """Return the wall plus the blocker that closes its short-tip bypass."""
    wall = decodeWall(genes, segment)
    ax, ay = tiltedAxis(segment, genes.side, genes.crossingDeg)
    crossX, crossY = segment.pointAt(genes.t)
    tip = (crossX + genes.overhang * ax, crossY + genes.overhang * ay)
    ux, uy = segment.direction()
    nx, ny = segment.sideNormal()
    # The blocker leans the other way, so the two walls form a funnel whose
    # inclination has a fixed value of GATE_BLOCKER_DEG = 30 deg
    bx, by = tiltedAxis(segment, -genes.side, GATE_BLOCKER_DEG)
    blockerDeg = math.degrees(math.atan2(-by, -bx))

    def blockerAt(offset: float):
        """Return the blocker whose inner end sits offset metres off the tip."""
        start = (
            tip[0] + GATE_AHEAD_M * ux + offset * genes.side * nx,
            tip[1] + GATE_AHEAD_M * uy + offset * genes.side * ny,
        )
        return makeWall(start, blockerDeg, genes.length)

    # The gap grows with the offset, so bisection finds the offset whose
    # true footprint gap equals the gateWidth gene.
    lo, hi = 0.0, GATE_OFFSET_MAX_M
    for _ in range(GATE_SOLVE_STEPS):
        mid = 0.5 * (lo + hi)
        if footprintGap(wall, blockerAt(mid)) < genes.gateWidth:
            lo = mid
        else:
            hi = mid
    return [wall, blockerAt(0.5 * (lo + hi))]


def segmentByIndex(segments: List[Segment], index: int):
    """Return the segment with the given mission index."""
    for segment in segments:
        if segment.index == index:
            return segment
    raise ValueError("no segment with index %d" % index)


def decode(genes: Genes, segments: List[Segment]):
    """Return the obstacles described by the genes."""
    segment = segmentByIndex(segments, genes.segment)
    if genes.kind == KIND_GATE:
        return decodeGate(genes, segment)
    return [decodeWall(genes, segment)]


def crossesOtherSegment(obstacles: List, segments: List[Segment], ownSegmentIndex: int):
    """Return whether any wall axis cuts a segment other than the attacked one."""
    # A wall across a later segment turns a proximity test into a liveness test:
    # on mission 3 such layouts stalled 6 times out of 11 against 0 of 10.
    for o in obstacles:
        p, q = wallAxis(o)
        for segment in segments:
            if segment.index != ownSegmentIndex and segmentsIntersect(p, q, segment.a, segment.b):
                return True
    return False


def layoutProblem(bounds: Bounds, obstacles: List, segments: List[Segment], ownSegmentIndex: int, strict: bool = True):
    """Return why the layout is unusable, or None when it is fine."""
    if not fitsInBounds(bounds, obstacles):
        return "outside the placement area"
    if hasOverlap(obstacles):
        return "walls overlap"
    if strict and crossesOtherSegment(obstacles, segments, ownSegmentIndex):
        return "crosses another segment"
    return None
