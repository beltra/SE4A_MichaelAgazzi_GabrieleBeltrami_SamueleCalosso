"""Pick the tests to submit: best official score first, distinct trajectories.

The published competition reports sum the test scores without a similarity
penalty and rank output diversity separately, as the mean nearest-neighbour
DTW distance between the averaged trajectories. The competition wiki, which is
the call participants were given, instead states that the suite total is
multiplied by a similarity penalty term computed from the DTW distance
between average trajectories. This module implements the wiki, and holds
under the reports too, since a suite of indistinguishable flights loses
output-diversity rank there. The rule either way: never submit two tests
whose flights cannot be told apart.

The wiki gives the penalty as an image, so its cutoff cannot be read. The
simulator's own repeatability, the spread between reruns of one layout, is
the natural scale for "cannot be told apart", so the threshold is measured
rather than guessed.
"""

import math
import statistics
from typing import List, Tuple

Point = Tuple[float, float]

# Trajectories are resampled to this many points by arc length before they
# are compared, so runs of different duration line up point by point.
RESAMPLE_POINTS = 50
# Used until at least one layout has been run twice.
DEFAULT_THRESHOLD_M = 3.0
# Below this the threshold only removes exact duplicates; stop halving there.
MIN_THRESHOLD_M = 0.1


def trajectoryPoints(trajectory):
    """Return the (x, y) points of a simulator trajectory."""
    positions = getattr(trajectory, "positions", None) or []
    return [(p.x, p.y) for p in positions]


def resample(points: List[Point], n: int = RESAMPLE_POINTS):
    """Return n points spaced evenly along the polyline through points."""
    if not points:
        return []
    if len(points) == 1:
        return [points[0]] * n
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(a, b))
    total = cumulative[-1]
    if total <= 0.0:
        return [points[0]] * n
    result = []
    segment = 0
    for k in range(n):
        target = total * k / (n - 1)
        while segment < len(points) - 2 and cumulative[segment + 1] < target:
            segment += 1
        a, b = points[segment], points[segment + 1]
        span = cumulative[segment + 1] - cumulative[segment]
        f = (target - cumulative[segment]) / span if span > 0.0 else 0.0
        result.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    return result


def meanTrajectory(ind):
    """Return the point-wise average of the individual's resampled runs."""
    runs = [resample(trajectoryPoints(t)) for t in ind.trajectories]
    runs = [run for run in runs if run]
    if not runs:
        return []
    return [
        (statistics.fmean(run[k][0] for run in runs), statistics.fmean(run[k][1] for run in runs))
        for k in range(RESAMPLE_POINTS)
    ]


def trajectoryDistance(a: List[Point], b: List[Point]):
    """Return the mean point-to-point distance between two resampled paths."""
    if not a or not b:
        return float("inf")
    return statistics.fmean(math.dist(p, q) for p, q in zip(a, b))


def calibrateThreshold(evaluated: List):
    """Return the similarity threshold measured from same-layout reruns."""
    # Two tests closer than the distance between two runs of the same layout
    # are indistinguishable; the median keeps one stalled rerun from
    # dominating.
    spreads = []
    for ind in evaluated:
        runs = [resample(trajectoryPoints(t)) for t in ind.trajectories]
        runs = [run for run in runs if run]
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                spreads.append(trajectoryDistance(runs[i], runs[j]))
    if not spreads:
        return DEFAULT_THRESHOLD_M
    return statistics.median(spreads)


def layoutSignature(ind):
    """Return the exact geometry of a layout, to drop duplicates."""
    return tuple(sorted(
        (o.size.l, o.size.w, o.size.h, o.position.x, o.position.y, o.position.z, o.position.r)
        for o in ind.obstacles
    ))


def officialRank(ind):
    """Sort key: best official score first, then closest approach."""
    meanDistance = statistics.fmean(ind.distances) if ind.distances else float("inf")
    return -ind.officialScore, meanDistance


def frameOf(ind):
    """Return the (segment, side) a layout attacks; None for layouts without genes."""
    genes = getattr(ind, "genes", None)
    return (genes.segment, genes.side) if genes is not None else None


def greedyFill(ranked: List, topK: int, threshold: float, maxPerFrame: int = 0):
    """Return up to topK tests in rank order, skipping look-alike flights.

    With maxPerFrame > 0 at most that many tests may attack the same segment
    from the same side, so the flights differ in where they bend, not only in
    how far they bend.
    """
    selected = []
    paths = []
    perFrame = {}
    for ind in ranked:
        if len(selected) >= topK:
            break
        frame = frameOf(ind)
        if maxPerFrame and perFrame.get(frame, 0) >= maxPerFrame:
            continue
        path = meanTrajectory(ind)
        if all(trajectoryDistance(path, other) >= threshold for other in paths):
            selected.append(ind)
            paths.append(path)
            perFrame[frame] = perFrame.get(frame, 0) + 1
    return selected


def fillableCount(ranked: List, topK: int, maxPerFrame: int):
    """Return how many tests the frame quota lets the suite hold at most."""
    if not maxPerFrame:
        return min(topK, len(ranked))
    counts = {}
    for ind in ranked:
        counts[frameOf(ind)] = counts.get(frameOf(ind), 0) + 1
    return min(topK, sum(min(maxPerFrame, n) for n in counts.values()))


def selectSuite(cfg, evaluated: List, threshold: float):
    """Return (suite, threshold used): the tests worth submitting, best first."""
    ranked = []
    seen = set()
    for ind in sorted(evaluated, key=officialRank):
        # Stuck runs and zero-point layouts add nothing to the suite score.
        if not ind.valid or ind.stuck or ind.officialScore <= 0.0:
            continue
        signature = layoutSignature(ind)
        if signature in seen:
            continue
        seen.add(signature)
        ranked.append(ind)
    # A gate flies a different path from every single; keeping the best one
    # buys trajectory diversity even when singles outscore it.
    if cfg.keepBestGate:
        gates = [ind for ind in ranked if len(ind.obstacles) == 2]
        if gates:
            ranked.remove(gates[0])
            ranked.insert(0, gates[0])
    wanted = fillableCount(ranked, cfg.topK, cfg.maxPerFrame)
    # Relaxing the threshold fills the suite with look-alikes; whether an
    # empty slot or a similar test costs more depends on the examiners'
    # similarity penalty, so it is a rule of the config.
    while True:
        suite = greedyFill(ranked, cfg.topK, threshold, cfg.maxPerFrame)
        if len(suite) >= wanted or threshold <= MIN_THRESHOLD_M or not cfg.relaxSimilarity:
            return sorted(suite, key=officialRank), threshold
        threshold *= 0.5


def suiteDiversity(suite: List):
    """Return the mean nearest-neighbour trajectory distance of the suite."""
    paths = [meanTrajectory(ind) for ind in suite]
    if len(paths) < 2:
        return 0.0
    nearest = [
        min(trajectoryDistance(paths[i], paths[j]) for j in range(len(paths)) if j != i)
        for i in range(len(paths))
    ]
    return statistics.fmean(nearest)
