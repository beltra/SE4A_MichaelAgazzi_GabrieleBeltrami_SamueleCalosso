"""Path-relative genome of a test: where and how a wall crosses a flight segment.

The genes never store map coordinates. They say which segment is attacked, how
far along it, on which side the short end of the wall lies, at which angle,
and for a gate how wide the opening is. `motifs.decode` turns them into
obstacles; nothing else does.
"""

import copy
import random
from dataclasses import dataclass
from typing import List

KIND_WALL = "wall"
KIND_GATE = "gate"
# The niche of an individual is its obstacle count; this maps it to a motif.
KIND_BY_COUNT = {1: KIND_WALL, 2: KIND_GATE}

# Value ranges, from the mission-3 probes (see ALGORITHM.md, section 7):
# - t: keep the wall away from the waypoints, where the drone turns and slows;
# - crossingDeg: 45 deg is the measured optimum, 40 and 50 are each ~0.3 m
#   worse, so the search stays in a band around it;
# - length: walls below ~15 m are simply rounded, 20 m is the maximum size;
# - overhang: for a wall, 6-10 m spans the validated crossings, from 1-2 m
#   off-centre to centred (the best measured wall, mean 0.77 m over 11 runs,
#   is centred); for a gate the short tip (3.5-6 m) is what commits the
#   planner to the bypass;
# - gateWidth: 7-10 m completed 13/13 runs in the 2-point tier, below ~6.3 m
#   the drone stalls in front of the gate.
GENE_RANGES = {
    KIND_WALL: {
        "t": (0.2, 0.8),
        "crossingDeg": (35.0, 55.0),
        "length": (17.0, 20.0),
        "overhang": (6.0, 10.0),
    },
    KIND_GATE: {
        "t": (0.2, 0.8),
        "crossingDeg": (40.0, 55.0),
        "length": (18.0, 20.0),
        "overhang": (3.5, 6.0),
        "gateWidth": (7.0, 10.0),
    },
}

# Mutation step per gene, matched to the scale at which the measured response
# changes: about 1 m in position and 5 deg in angle. One t unit is a whole segment
# (45-80 m), so 0.015 is roughly 1 m.
GENE_SIGMAS = {
    "t": 0.015,
    "crossingDeg": 2.0,
    "length": 1.0,
    "overhang": 0.5,
    "gateWidth": 0.5,
}
# Now and then take a step four times larger, to leave a local optimum.
COARSE_STEP_PROBABILITY = 0.1
COARSE_STEP_FACTOR = 4.0


@dataclass
class Genes:
    """The path-relative description of one wall or one gate."""

    kind: str
    segment: int
    # Which of the two sides of the segment the short end of the wall lies on:
    # +1 is the side sideNormal points to, -1 the other. Only +1 has ever scored.
    side: int
    t: float
    crossingDeg: float
    length: float
    overhang: float
    gateWidth: float = 0.0


def mutableGenes(kind: str):
    """Return the names of the genes mutation may touch for this kind."""
    return list(GENE_RANGES[kind])


def randomGenes(rng: random.Random, kind: str, segmentChoices: List[int], sides=(-1, 1)):
    """Return fresh genes with every value drawn uniformly from its range."""
    values = {name: rng.uniform(lo, hi) for name, (lo, hi) in GENE_RANGES[kind].items()}
    return Genes(kind=kind, segment=rng.choice(segmentChoices), side=rng.choice(sides), **values)


def mutateGenes(rng: random.Random, genes: Genes, scale: float = 1.0):
    """Return a copy with exactly one gene moved by a gaussian step of sigma * scale."""
    child = copy.copy(genes)
    name = rng.choice(mutableGenes(genes.kind))
    lo, hi = GENE_RANGES[genes.kind][name]
    sigma = GENE_SIGMAS[name] * scale
    if rng.random() < COARSE_STEP_PROBABILITY:
        sigma *= COARSE_STEP_FACTOR
    old = getattr(genes, name)
    # Clamping at a range end can hand back the old value; draw again so the
    # child really differs from its parent.
    for _ in range(10):
        new = min(hi, max(lo, old + rng.gauss(0.0, sigma)))
        if new != old:
            break
    setattr(child, name, new)
    return child


def crossoverGenes(rng: random.Random, a: Genes, b: Genes):
    """Return a child taking each gene from either parent."""
    # Mixing genes only makes sense inside one frame: two parents on
    # different segments or sides describe unrelated places, so one of them is
    # copied whole instead.
    if (a.kind, a.segment, a.side) != (b.kind, b.segment, b.side):
        return copy.copy(a if rng.random() < 0.5 else b)
    child = copy.copy(a)
    for name in mutableGenes(a.kind):
        if rng.random() < 0.5:
            setattr(child, name, getattr(b, name))
    return child
