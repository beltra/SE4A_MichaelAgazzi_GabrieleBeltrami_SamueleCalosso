"""Minimalist single-file genetic algorithm for PX4 Aerialist obstacle tests."""

import copy
import glob
import json
import logging
import math
import os
import random
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from aerialist.px4.aerialist_test import AerialistTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase

logger = logging.getLogger(__name__)


# Official scoring tiers: TIER_POINTS[i] applies when distance < TIER_BOUNDS[i];
# the last entry covers >= TIER_BOUNDS[-1] (5/2/1/0 for <0.25 / <1.0 / <1.5 / >=1.5).
TIER_BOUNDS = (0.25, 1.0, 1.5)
TIER_POINTS = (5, 2, 1, 0)
OFFICIAL_EVALUATION_RUNS = 3

# Side of the trajectory an obstacle sits on. With the trajectory unit vector
# (ux, uy) the perpendicular (-uy, ux) is a 90deg CCW rotation, i.e. LEFT of
# the drone's forward direction; the opposite perpendicular is RIGHT.
SIDE_LEFT = "L"
SIDE_RIGHT = "R"

# Longitudinal half of the segment an obstacle sits on. F (front) places the
# obstacle closer to the segment destination (drone's forward direction);
# R (rear) places it closer to the origin.
LONG_FRONT = "F"
LONG_REAR = "R"

# Uniform lateral offset bound (metres) used when placing corridor obstacles.
LATERAL_MAX_M = 10.0


@dataclass
class GAConfig:
    seed: Optional[int] = None

    popSize: int = 20
    # The competition evaluates the first 20 tests returned by a tool.
    topK: int = 20
    maxObstacles: int = 3
    maxRetries: int = 20
    mutationRate: float = 0.2
    addProb: float = 0.15
    removeProb: float = 0.15
    sigmaFrac: float = 0.10
    tournamentK: int = 3
    eliteSize: int = 2

    xMin: float = -40.0
    xMax: float = 30.0
    yMin: float = 10.0
    yMax: float = 40.0
    lMin: float = 2.0
    lMax: float = 20.0
    wMin: float = 2.0
    wMax: float = 20.0
    rMin: float = 0.0
    rMax: float = 90.0
    hFixed: float = 25.0

    # Path-aware seeding.
    pathSigma: float = 8.0

    # Post-GA refinement.
    refineFraction: float = 0.10
    refineThreshold: float = 1.5
    # One initial execution plus two reruns matches the three-run protocol.
    nReruns: int = OFFICIAL_EVALUATION_RUNS - 1
    failSentinel: float = 5.0

    # Top-K obstacle-centroid diversity filter.
    diversityMinM: float = 10.0

    # Stuck-avoidance detection: drone is "stuck" if its final trajectory point
    # is farther than goalTol from the last mission waypoint. The watchdog
    # kills runs that exceed timeoutMul * (slowest reached-goal run so far),
    # with minTimeout as the floor before any good run has been observed.
    goalTol: float = 5.0
    timeoutMul: float = 2.5
    minTimeout: float = 300.0


Waypoint = Tuple[float, float]


@dataclass
class Individual:
    obstacles: List[Obstacle]
    fitness: float = float("inf")
    officialScore: float = 0.0
    testCase: Optional[TestCase] = None
    valid: bool = False
    distances: List[float] = field(default_factory=list)
    executionDurations: List[float] = field(default_factory=list)
    stuck: bool = False
    duration: float = 0.0


def clamp(v: float, lo: float, hi: float):
    return max(lo, min(hi, v))


def gaussianPerturb(rng: random.Random, v: float, lo: float, hi: float, sigmaFrac: float):
    sigma = sigmaFrac * (hi - lo)
    return clamp(v + rng.gauss(0.0, sigma), lo, hi)


def rotatedHalfExtents(l: float, w: float, rDeg: float):
    cosR = abs(math.cos(math.radians(rDeg)))
    sinR = abs(math.sin(math.radians(rDeg)))
    return (
        0.5 * (l * cosR + w * sinR),
        0.5 * (l * sinR + w * cosR),
    )


def hasOverlap(obstacles: List[Obstacle]):
    for i in range(len(obstacles)):
        a = obstacles[i]
        hxA, hyA = rotatedHalfExtents(a.size.l, a.size.w, a.position.r)
        for j in range(i + 1, len(obstacles)):
            b = obstacles[j]
            hxB, hyB = rotatedHalfExtents(b.size.l, b.size.w, b.position.r)
            dx = abs(a.position.x - b.position.x)
            dy = abs(a.position.y - b.position.y)
            if (hxA + hxB) >= dx and (hyA + hyB) >= dy:
                return True
    return False


def fitsInBounds(cfg: GAConfig, obstacles: List[Obstacle]):
    # Wiki says obstacles must fit in the case-study rectangle; clamping only
    # the centre lets a rotated 20m box at y=10 reach down to y~1, i.e. right
    # on the takeoff point, which traps the avoidance planner.
    for o in obstacles:
        hx, hy = rotatedHalfExtents(o.size.l, o.size.w, o.position.r)
        if (
            o.position.x - hx < cfg.xMin
            or o.position.x + hx > cfg.xMax
            or o.position.y - hy < cfg.yMin
            or o.position.y + hy > cfg.yMax
        ):
            return False
    return True


def invalidLayout(cfg: GAConfig, obstacles: List[Obstacle]):
    return hasOverlap(obstacles) or not fitsInBounds(cfg, obstacles)


def shiftObstacle(o: Obstacle, dx: float, dy: float):
    return Obstacle(
        Obstacle.Size(l=o.size.l, w=o.size.w, h=o.size.h),
        Obstacle.Position(
            x=o.position.x + dx, y=o.position.y + dy,
            z=o.position.z, r=o.position.r,
        ),
    )


def shrinkObstacle(cfg: GAConfig, o: Obstacle):
    # Halve the footprint, clamped to lMin/wMin so the obstacle stays valid
    # geometry even after several shrink steps.
    return Obstacle(
        Obstacle.Size(
            l=max(cfg.lMin, o.size.l * 0.5),
            w=max(cfg.wMin, o.size.w * 0.5),
            h=o.size.h,
        ),
        o.position,
    )


def mtv(a: Obstacle, b: Obstacle):
    # Minimum translation vector along the centre-to-centre line, computed on
    # the rotated AABBs. Returns (dx, dy) pointing from a to b with length
    # equal to the penetration; (0, 0) when the AABBs do not overlap.
    hxA, hyA = rotatedHalfExtents(a.size.l, a.size.w, a.position.r)
    hxB, hyB = rotatedHalfExtents(b.size.l, b.size.w, b.position.r)
    cx = b.position.x - a.position.x
    cy = b.position.y - a.position.y
    dist = math.hypot(cx, cy)
    if dist < 1e-9:
        # Coincident centres: pick +x as a stable separation direction.
        ux, uy = 1.0, 0.0
        dist = 0.0
    else:
        ux, uy = cx / dist, cy / dist
    needed = (hxA + hxB) * abs(ux) + (hyA + hyB) * abs(uy)
    penetration = needed - dist
    if penetration <= 0.0:
        return 0.0, 0.0
    return ux * penetration, uy * penetration


def findOverlapPair(obstacles: List[Obstacle]):
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            dx, dy = mtv(obstacles[i], obstacles[j])
            if dx != 0.0 or dy != 0.0:
                return i, j
    return None


def separatePair(cfg: GAConfig, a: Obstacle, b: Obstacle, delta: float = 1.0):
    # Push a and b apart along the centre-to-centre line by MTV/2 each, plus
    # delta/2 each of slack so the final gap is "penetration + delta" past
    # touching. Returns None when either obstacle would leave the play area.
    dx, dy = mtv(a, b)
    if dx == 0.0 and dy == 0.0:
        return a, b
    pen = math.hypot(dx, dy)
    ux, uy = dx / pen, dy / pen
    half = (pen + delta) * 0.5
    newA = shiftObstacle(a, -ux * half, -uy * half)
    newB = shiftObstacle(b, ux * half, uy * half)
    if not fitsInBounds(cfg, [newA, newB]):
        return None
    return newA, newB


def resolveOverlaps(cfg: GAConfig, obstacles: List[Obstacle], maxIter: int = 6):
    # Move overlapping pairs apart with separatePair; if a move would push an
    # obstacle out of bounds, shrink both obstacles (l, w *= 0.5) and try again.
    # Capped by maxIter to guarantee termination at min footprint.
    obs = list(obstacles)
    for _ in range(maxIter):
        pair = findOverlapPair(obs)
        if pair is None:
            return obs
        i, j = pair
        moved = separatePair(cfg, obs[i], obs[j])
        if moved is not None:
            obs[i], obs[j] = moved
        else:
            obs[i] = shrinkObstacle(cfg, obs[i])
            obs[j] = shrinkObstacle(cfg, obs[j])
    return obs


def tierPoints(d: float):
    for b, p in zip(TIER_BOUNDS, TIER_POINTS[:-1]):
        if d < b:
            return p
    return TIER_POINTS[-1]


def officialTestScore(
    distances: List[float],
    obstacleCount: int,
    executionDurations: List[float],
):
    """Return the official failure score for one test case."""
    if not distances or not executionDurations:
        return 0.0
    if len(distances) != len(executionDurations):
        raise ValueError("each distance must have a matching execution duration")
    if obstacleCount <= 0:
        raise ValueError("obstacleCount must be positive")
    if any(duration <= 0.0 for duration in executionDurations):
        raise ValueError("execution durations must be positive")

    avgPoint = statistics.fmean(tierPoints(distance) for distance in distances)
    avgTimeMinutes = statistics.fmean(executionDurations) / 60.0
    return (avgPoint * 10.0) / ((obstacleCount ** 2) * avgTimeMinutes)


def fitnessFor(
    distances: List[float],
    obstacleCount: int,
    executionDurations: List[float],
):
    return -officialTestScore(distances, obstacleCount, executionDurations)


def selectionKey(ind: Individual):
    """Rank by score, then deterministic search tie-breakers."""
    meanDistance = statistics.fmean(ind.distances) if ind.distances else float("inf")
    meanDuration = (
        statistics.fmean(ind.executionDurations)
        if ind.executionDurations
        else float("inf")
    )
    return ind.fitness, len(ind.obstacles), meanDistance, meanDuration


def wgs84ToLocal(lat: float, lon: float, homeLat: float, homeLon: float):
    # Equirectangular approximation; valid for the small mission area.
    R = 6378137.0
    dLat = math.radians(lat - homeLat)
    dLon = math.radians(lon - homeLon)
    return R * dLat, R * dLon * math.cos(math.radians(homeLat))


def parseWaypoints(planPath: str):
    # QGC .plan command codes: 22=takeoff, 16=waypoint, 21=land.
    try:
        with open(planPath) as fh:
            plan = json.load(fh)
        items = plan["mission"]["items"]
        home = plan["mission"].get("plannedHomePosition", [0.0, 0.0, 0.0])
        hLat, hLon = home[0], home[1]
        wps: List[Waypoint] = [(0.0, 0.0)]
        for item in items:
            if item.get("command") in (16, 22, 21):
                params = item.get("params", [])
                if len(params) >= 6 and params[4] is not None and params[5] is not None:
                    wps.append(wgs84ToLocal(params[4], params[5], hLat, hLon))
        return wps
    except Exception as e:
        logger.warning("waypoint parse failed: %s", e)
        return [(0.0, 0.0)]


def reachedGoal(trajectory, goalXY: Waypoint, tol: float):
    # Trajectory.positions may be empty when ALLIGN_ORIGIN runs against a
    # truncated log; treat missing data as "did not reach" so the watchdog
    # path and the unfinished-flight path collapse into the same flag.
    positions = getattr(trajectory, "positions", None) if trajectory is not None else None
    if not positions:
        return False
    last = positions[-1]
    return math.hypot(last.x - goalXY[0], last.y - goalXY[1]) <= tol


def pickSegment(rng: random.Random, waypoints: List[Waypoint]):
    # Weight by segment length so short segments (e.g. home->takeoff ~0.5 m)
    # are not sampled as often as the main flight leg (~50 m).
    lengths = [
        math.hypot(waypoints[i + 1][0] - waypoints[i][0], waypoints[i + 1][1] - waypoints[i][1])
        for i in range(len(waypoints) - 1)
    ]
    total = sum(lengths)
    if total < 1e-9:
        return rng.randrange(len(waypoints) - 1)
    return rng.choices(range(len(waypoints) - 1), weights=lengths, k=1)[0]


def anchorOnPath(rng: random.Random, waypoints: List[Waypoint]):
    i = pickSegment(rng, waypoints)
    a, b = waypoints[i], waypoints[i + 1]
    t = rng.random()
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def sizeTierBounds(lo: float, hi: float, idx: int):
    # Lower bound rises with lower idx; upper bound is always hi.
    # idx 0: [lo+2*step, hi], idx 1: [lo+step, hi], idx 2+: [lo, hi]
    step = (hi - lo) / 3.0
    return lo + (2 - min(idx, 2)) * step, hi


def randomObstacle(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: Optional[List[Waypoint]] = None,
    idx: int = 0,
):
    if waypoints is not None and len(waypoints) >= 2:
        ax, ay = anchorOnPath(rng, waypoints)
        x = clamp(ax + rng.gauss(0, cfg.pathSigma), cfg.xMin, cfg.xMax)
        y = clamp(ay + rng.gauss(0, cfg.pathSigma), cfg.yMin, cfg.yMax)
    else:
        x = rng.uniform(cfg.xMin, cfg.xMax)
        y = rng.uniform(cfg.yMin, cfg.yMax)
    lLo, lHi = sizeTierBounds(cfg.lMin, cfg.lMax, idx)
    wLo, wHi = sizeTierBounds(cfg.wMin, cfg.wMax, idx)
    size = Obstacle.Size(l=rng.uniform(lLo, lHi), w=rng.uniform(wLo, wHi), h=cfg.hFixed)
    position = Obstacle.Position(x=x, y=y, z=0, r=rng.uniform(cfg.rMin, cfg.rMax))
    return Obstacle(size, position)


def obstacleCountSchedule(popSize: int, maxObstacles: int):
    # Cycle 1..maxObstacles so each count appears at least popSize // maxObstacles
    # times in the initial population. Counters random.randint bias that lets
    # singleton layouts dominate when popSize is small.
    return [1 + (k % maxObstacles) for k in range(popSize)]


def randomIndividual(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: Optional[List[Waypoint]] = None,
    n: Optional[int] = None,
):
    if n is None:
        n = rng.randint(1, cfg.maxObstacles)
    obstacles: List[Obstacle] = []
    for _ in range(cfg.maxRetries):
        obstacles = [randomObstacle(rng, cfg, waypoints, idx) for idx in range(n)]
        if not invalidLayout(cfg, obstacles):
            return Individual(obstacles=obstacles)
        if n > 1:
            obstacles = resolveOverlaps(cfg, obstacles)
            if not invalidLayout(cfg, obstacles):
                return Individual(obstacles=obstacles)
    return Individual(obstacles=obstacles)


def oppositeSide(s: str):
    return SIDE_RIGHT if s == SIDE_LEFT else SIDE_LEFT


def sideSchedule(rng: random.Random, n: int):
    # First obstacle picks a random side; the second is forced opposite so
    # the GA does not pile both onto the same flank (which the logs showed
    # was the main failure mode of the previous seeding). The third, when
    # present, is random again so both sides keep getting explored.
    if n == 1:
        return [rng.choice([SIDE_LEFT, SIDE_RIGHT])]
    if n == 2:
        first = rng.choice([SIDE_LEFT, SIDE_RIGHT])
        return [first, oppositeSide(first)]
    if n == 3:
        first = rng.choice([SIDE_LEFT, SIDE_RIGHT])
        return [first, oppositeSide(first), rng.choice([SIDE_LEFT, SIDE_RIGHT])]
    return []


def oppositeLong(s: str):
    return LONG_REAR if s == LONG_FRONT else LONG_FRONT


def longitudinalSchedule(rng: random.Random, n: int):
    # First obstacle picks a random half; the second copies it so the pair
    # crowds the same half (forcing the planner to navigate around a cluster);
    # the third, when present, is forced opposite so the path has obstacles in
    # both halves and the drone cannot stay on one easy side of the segment.
    if n == 1:
        return [rng.choice([LONG_FRONT, LONG_REAR])]
    if n == 2:
        first = rng.choice([LONG_FRONT, LONG_REAR])
        return [first, first]
    if n == 3:
        first = rng.choice([LONG_FRONT, LONG_REAR])
        return [first, first, oppositeLong(first)]
    return []


def corridorObstacle(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: List[Waypoint],
    idx: int = 0,
    side: str = SIDE_LEFT,
    longitudinal: str = LONG_FRONT,
):
    # Place an obstacle alongside the drone's path: centre on a random point
    # in the chosen half of a path segment (front = closer to destination,
    # rear = closer to origin), push it laterally onto the chosen side with a
    # uniform offset, then rotate to face the approach direction.
    i = pickSegment(rng, waypoints)
    a, b = waypoints[i], waypoints[i + 1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    segLen = math.hypot(dx, dy)
    if segLen < 1e-6:
        return randomObstacle(rng, cfg, waypoints)
    ux, uy = dx / segLen, dy / segLen   # unit vector along segment
    px, py = -uy, ux                    # +perpendicular = LEFT of forward direction
    t = rng.uniform(0.5, 1.0) if longitudinal == LONG_FRONT else rng.uniform(0.0, 0.5)
    cx = a[0] + t * dx
    cy = a[1] + t * dy
    sign = 1.0 if side == SIDE_LEFT else -1.0
    lateralOffset = sign * rng.uniform(0.0, LATERAL_MAX_M)
    x = clamp(cx + lateralOffset * px, cfg.xMin, cfg.xMax)
    y = clamp(cy + lateralOffset * py, cfg.yMin, cfg.yMax)
    # Rotate so the obstacle face is perpendicular to the approach direction;
    # mod 90 keeps r within [0, 90) regardless of segment orientation.
    r = clamp(math.degrees(math.atan2(dy, dx)) % 90.0, cfg.rMin, cfg.rMax)
    lLo, lHi = sizeTierBounds(cfg.lMin, cfg.lMax, idx)
    wLo, wHi = sizeTierBounds(cfg.wMin, cfg.wMax, idx)
    size = Obstacle.Size(l=rng.uniform(lLo, lHi), w=rng.uniform(wLo, wHi), h=cfg.hFixed)
    return Obstacle(size, Obstacle.Position(x=x, y=y, z=0, r=r))


def seededIndividual(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: List[Waypoint],
    n: Optional[int] = None,
):
    # Uses corridorObstacle for placement; falls back to randomIndividual if
    # the corridor placement cannot pass the layout check after maxRetries.
    if n is None:
        n = rng.randint(1, cfg.maxObstacles)
    sides = sideSchedule(rng, n)
    longs = longitudinalSchedule(rng, n)
    for _ in range(cfg.maxRetries):
        obstacles = [
            corridorObstacle(rng, cfg, waypoints, idx, sides[idx], longs[idx])
            for idx in range(n)
        ]
        if not invalidLayout(cfg, obstacles):
            return Individual(obstacles=obstacles)
        if n > 1:
            obstacles = resolveOverlaps(cfg, obstacles)
            if not invalidLayout(cfg, obstacles):
                return Individual(obstacles=obstacles)
    logger.info("seeded retries exhausted for n=%d, falling back to randomIndividual", n)
    return randomIndividual(rng, cfg, waypoints, n)


def crossover(rng: random.Random, a: Individual, b: Individual):
    # Positional crossover: for each slot index pick the obstacle from parent A
    # or B at that position. This preserves structured arrangements (e.g. a
    # left+right flanking pair) instead of shuffling the merged pool arbitrarily.
    n = rng.choice([len(a.obstacles), len(b.obstacles)])
    result = []
    for i in range(n):
        if i < len(a.obstacles) and i < len(b.obstacles):
            src = a if rng.random() < 0.5 else b
        elif i < len(a.obstacles):
            src = a
        else:
            src = b
        result.append(copy.deepcopy(src.obstacles[i]))
    return Individual(obstacles=result)


def mutate(
    rng: random.Random,
    cfg: GAConfig,
    ind: Individual,
    waypoints: Optional[List[Waypoint]] = None,
):
    # Obstacle.Size and Obstacle.Position are NamedTuples (immutable);
    # mutate by constructing a fresh Obstacle so internal geometry is rebuilt.
    for idx, o in enumerate(ind.obstacles):
        l = (
            gaussianPerturb(rng, o.size.l, cfg.lMin, cfg.lMax, cfg.sigmaFrac)
            if rng.random() < cfg.mutationRate
            else o.size.l
        )
        w = (
            gaussianPerturb(rng, o.size.w, cfg.wMin, cfg.wMax, cfg.sigmaFrac)
            if rng.random() < cfg.mutationRate
            else o.size.w
        )
        x = (
            gaussianPerturb(rng, o.position.x, cfg.xMin, cfg.xMax, cfg.sigmaFrac)
            if rng.random() < cfg.mutationRate
            else o.position.x
        )
        y = (
            gaussianPerturb(rng, o.position.y, cfg.yMin, cfg.yMax, cfg.sigmaFrac)
            if rng.random() < cfg.mutationRate
            else o.position.y
        )
        r = (
            gaussianPerturb(rng, o.position.r, cfg.rMin, cfg.rMax, cfg.sigmaFrac)
            if rng.random() < cfg.mutationRate
            else o.position.r
        )
        ind.obstacles[idx] = Obstacle(
            Obstacle.Size(l=l, w=w, h=cfg.hFixed),
            Obstacle.Position(x=x, y=y, z=0, r=r),
        )
    if rng.random() < cfg.addProb and len(ind.obstacles) < cfg.maxObstacles:
        idx = len(ind.obstacles)
        if waypoints is not None and len(waypoints) >= 2:
            side = rng.choice([SIDE_LEFT, SIDE_RIGHT])
            longitudinal = rng.choice([LONG_FRONT, LONG_REAR])
            ind.obstacles.append(corridorObstacle(rng, cfg, waypoints, idx, side, longitudinal))
        else:
            ind.obstacles.append(randomObstacle(rng, cfg, waypoints, idx))
    if rng.random() < cfg.removeProb and len(ind.obstacles) > 1:
        ind.obstacles.pop(rng.randrange(len(ind.obstacles)))
    # Perturbed positions/sizes and the add step can introduce overlaps; clean
    # them up here so nextGeneration's invalidLayout retry loop is rarely hit.
    if len(ind.obstacles) > 1 and hasOverlap(ind.obstacles):
        ind.obstacles = resolveOverlaps(cfg, ind.obstacles)
    return ind


def tournament(rng: random.Random, cfg: GAConfig, pop: List[Individual]):
    contenders = rng.sample(pop, min(cfg.tournamentK, len(pop)))
    return min(contenders, key=selectionKey)


# Anything spawned by roslaunch inherits the pipe write end Aerialist polls on;
# if rosmaster / mavros / rosout survive a SIGKILL of roslaunch, sim_thread's
# readlines() on the child stdout blocks forever and the GA hangs. Mirror
# kill_simulations.sh (minus "python3 cli.py", which would kill us).
_CLEANUP_PATTERNS = (
    "mavsdk_server",
    "px4",
    "gzserver",
    "gzclient",
    "gazebo",
    "roslaunch",
    "rosmaster",
    "rosout",
    "robot_state_publisher",
    "static_transform_publisher",
    "nodelet",
    "mavros",
    "mavros_node",
)
_STALE_GLOBS = ("/tmp/px4-sock-*", "/tmp/px4_lock-*")


def cleanupSimState():
    # SIGTERM first so processes flush stdout and release pipes cleanly;
    # SIGKILL the survivors after a brief grace period.
    for signame in ("-TERM", "-KILL"):
        for pattern in _CLEANUP_PATTERNS:
            subprocess.run(
                ["pkill", signame, "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        time.sleep(1)
    # pkill -9 skips PX4's cleanup hook, so its socket and lock file in /tmp
    # survive the kill and make the next SITL instance exit with code 255.
    for pattern in _STALE_GLOBS:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass


def armWatchdog(killAfter: float):
    # Returns (timer, killedFlag). The flag is a 1-element list so the timer
    # callback can mutate it without a nonlocal binding; cleanupSimState pkills
    # PX4/Gazebo which unblocks agent.run() with an exception.
    if killAfter <= 0:
        return None, [False]
    killed = [False]
    def watchdog():
        killed[0] = True
        cleanupSimState()
    timer = threading.Timer(killAfter, watchdog)
    timer.daemon = True
    timer.start()
    return timer, killed


def evaluate(
    cfg: GAConfig,
    ind: Individual,
    caseStudy: AerialistTest,
    goalXY: Optional[Waypoint] = None,
    killAfter: float = 0.0,
):
    if invalidLayout(cfg, ind.obstacles):
        ind.fitness = float("inf")
        ind.valid = False
        return False
    timer, killed = armWatchdog(killAfter)
    t0 = time.monotonic()
    try:
        tc = TestCase(caseStudy, ind.obstacles)
        executionStarted = time.monotonic()
        tc.execute()
        executionDuration = time.monotonic() - executionStarted
        distances = tc.get_distances()
        if not distances:
            raise RuntimeError("no distances")
        minDist = float(min(distances))
        # "minimum_distance:" kept verbatim so external grep-based scorers keep matching.
        logger.info("minimum_distance:%s", minDist)
        tc.plot()
        ind.testCase = tc
        ind.distances.append(minDist)
        ind.executionDurations.append(max(executionDuration, 1e-9))
        if goalXY is not None and not reachedGoal(tc.trajectory, goalXY, cfg.goalTol):
            ind.stuck = True
            logger.info("stuck: final trajectory point not within %.1fm of goal", cfg.goalTol)
        ind.fitness = fitnessFor(
            ind.distances, len(ind.obstacles), ind.executionDurations
        )
        ind.officialScore = -ind.fitness
        ind.valid = True
    except Exception as e:
        if killed[0]:
            ind.stuck = True
            logger.info("stuck: watchdog killed run after %.1fs", time.monotonic() - t0)
        else:
            logger.warning("sim failed: %s", e)
        ind.fitness = float("inf")
        ind.valid = False
    finally:
        if timer is not None:
            timer.cancel()
        ind.duration = time.monotonic() - t0
        cleanupSimState()
    return True


def rerun(
    cfg: GAConfig,
    ind: Individual,
    caseStudy: AerialistTest,
    goalXY: Optional[Waypoint] = None,
    killAfter: float = 0.0,
):
    timer, killed = armWatchdog(killAfter)
    t0 = time.monotonic()
    executionStarted = t0
    executionDuration = 0.0
    try:
        tc = TestCase(caseStudy, copy.deepcopy(ind.obstacles))
        executionStarted = time.monotonic()
        tc.execute()
        executionDuration = time.monotonic() - executionStarted
        distances = tc.get_distances()
        if not distances:
            raise RuntimeError("no distances")
        minDist = float(min(distances))
        logger.info("rerun minimum_distance:%s", minDist)
        tc.plot()
        ind.distances.append(minDist)
        ind.testCase = tc
        if goalXY is not None and not reachedGoal(tc.trajectory, goalXY, cfg.goalTol):
            ind.stuck = True
            logger.info("stuck on rerun: final point not within %.1fm of goal", cfg.goalTol)
    except Exception as e:
        executionDuration = time.monotonic() - executionStarted
        if killed[0]:
            ind.stuck = True
            logger.info("stuck on rerun: watchdog killed run after %.1fs", time.monotonic() - t0)
        else:
            # A finite sentinel records the official zero-point tier.
            logger.warning("rerun failed (penalised with sentinel %s): %s", cfg.failSentinel, e)
        ind.distances.append(cfg.failSentinel)
    finally:
        ind.executionDurations.append(max(executionDuration, 1e-9))
        if timer is not None:
            timer.cancel()
        ind.duration = time.monotonic() - t0
        cleanupSimState()


def nextGeneration(
    rng: random.Random,
    cfg: GAConfig,
    pop: List[Individual],
    waypoints: Optional[List[Waypoint]] = None,
):
    eliteCount = min(cfg.eliteSize, len(pop))
    elites = sorted(pop, key=selectionKey)[:eliteCount]
    children: List[Individual] = list(elites)
    while len(children) < cfg.popSize:
        child: Optional[Individual] = None
        for _ in range(cfg.maxRetries):
            c = mutate(
                rng,
                cfg,
                crossover(rng, tournament(rng, cfg, pop), tournament(rng, cfg, pop)),
                waypoints,
            )
            if not invalidLayout(cfg, c.obstacles):
                child = c
                break
        if child is None:
            for _ in range(cfg.maxRetries):
                c = randomIndividual(rng, cfg, waypoints)
                if not invalidLayout(cfg, c.obstacles):
                    child = c
                    break
        if child is None:
            child = randomIndividual(rng, cfg, waypoints)
        children.append(child)
    return children


def obsSignature(ind: Individual):
    pts = [(o.position.x, o.position.y) for o in ind.obstacles]
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def isTooSimilar(cfg: GAConfig, cand: Individual, selected: List[Individual]):
    cx, cy = obsSignature(cand)
    for s in selected:
        sx, sy = obsSignature(s)
        if math.hypot(cx - sx, cy - sy) < cfg.diversityMinM:
            return True
    return False


class GeneticGenerator:
    def __init__(self, missionPath: str, config: Optional[GAConfig] = None):
        cfg = config or GAConfig()
        # Resolve the seed up front so it appears in logs and on cfg, then build
        # a private RNG instance: no calls into the process-wide random module.
        resolvedSeed = (
            cfg.seed if cfg.seed is not None else random.SystemRandom().randrange(2 ** 32)
        )
        self.config = replace(cfg, seed=resolvedSeed)
        self.rng = random.Random(resolvedSeed)
        logger.info("GA seed: %s", resolvedSeed)
        self.caseStudy = AerialistTest.from_yaml(missionPath)
        planPath = self.caseStudy.robot.mission_file if self.caseStudy.robot else None
        self.waypoints = parseWaypoints(planPath) if planPath else [(0.0, 0.0)]
        logger.info("waypoints: %s", self.waypoints)

    def generate(self, budget: int):
        cfg = self.config
        if budget <= 0:
            logger.error("budget must be > 0")
            return []

        # Shrink population when budget is too small to fill the default pop.
        # Below popSize sims the initial evaluation alone would exhaust the budget,
        # so scale down to sqrt(budget) to leave room for at least a few generations.
        if budget < cfg.popSize:
            effectivePopSize = max(2, round(math.sqrt(budget)))
            cfg = replace(cfg, popSize=effectivePopSize)

        # Reserve refinement budget only when phase 1 can still run one full population.
        phase1Budget = min(budget, max(cfg.popSize, int((1 - cfg.refineFraction) * budget)))
        phase2Budget = budget - phase1Budget

        # Max generations: how many full populations fit inside the phase-1 budget
        # (minus 1 for the initial population that is evaluated before evolution starts).
        maxGen = max(1, phase1Budget // cfg.popSize - 1)

        logger.info(
            "GA: budget=%s pop=%s maxGen=%s phase1=%s phase2=%s",
            budget, cfg.popSize, maxGen, phase1Budget, phase2Budget,
        )

        # Goal == last mission waypoint; parseWaypoints falls back to [(0,0)]
        # when the .plan is missing, in which case we cannot detect "stuck".
        goalXY = self.waypoints[-1] if len(self.waypoints) >= 2 else None
        maxGoodDuration = 0.0

        useCorridorSeed = len(self.waypoints) >= 2
        counts = obstacleCountSchedule(cfg.popSize, cfg.maxObstacles)
        logger.info("initial obstacle counts: %s", counts)
        pop = [
            seededIndividual(self.rng, cfg, self.waypoints, n) if useCorridorSeed
            else randomIndividual(self.rng, cfg, self.waypoints, n)
            for n in counts
        ]
        evaluated: List[Individual] = []
        simsUsed = 0
        gen = 0
        while simsUsed < phase1Budget and gen < maxGen:
            for ind in pop:
                if ind.valid:
                    continue
                killAfter = max(cfg.minTimeout, cfg.timeoutMul * maxGoodDuration)
                consumed = evaluate(cfg, ind, self.caseStudy, goalXY, killAfter)
                if ind.valid:
                    evaluated.append(ind)
                    # Only reached-goal runs feed the threshold; otherwise a slow
                    # stuck run would inflate the budget and disarm the watchdog.
                    if not ind.stuck and ind.duration > maxGoodDuration:
                        maxGoodDuration = ind.duration
                if consumed:
                    simsUsed += 1
                if simsUsed >= phase1Budget:
                    break
            best = min(pop, key=selectionKey)
            logger.info(
                "[gen %d] best official score=%.4f sims=%d/%d",
                gen,
                best.officialScore,
                simsUsed,
                phase1Budget,
            )
            if simsUsed >= phase1Budget:
                break
            pop = nextGeneration(self.rng, cfg, pop, self.waypoints)
            gen += 1

        # Phase 2: rerun promising single-shot candidates to filter flaky near-misses.
        promising = sorted(
            [i for i in evaluated if i.distances and i.distances[0] < cfg.refineThreshold],
            key=selectionKey,
        )
        simsLeft = phase2Budget
        for cand in promising:
            if simsLeft <= 0:
                break
            nMore = min(cfg.nReruns, simsLeft)
            killAfter = max(cfg.minTimeout, cfg.timeoutMul * maxGoodDuration)
            for _ in range(nMore):
                rerun(cfg, cand, self.caseStudy, goalXY, killAfter)
                simsLeft -= 1
                if simsLeft <= 0:
                    break
            cand.fitness = fitnessFor(
                cand.distances,
                len(cand.obstacles),
                cand.executionDurations,
            )
            cand.officialScore = -cand.fitness
            meanD = sum(cand.distances) / len(cand.distances)
            logger.info(
                "refined: runs=%d mean-distance=%.2f official-score=%.4f",
                len(cand.distances),
                meanD,
                cand.officialScore,
            )

        evaluated.sort(key=selectionKey)
        selected: List[Individual] = []
        for cand in evaluated:
            if len(selected) >= cfg.topK:
                break
            if not isTooSimilar(cfg, cand, selected):
                selected.append(cand)
        top = [s.testCase for s in selected]
        refinedCount = sum(1 for i in evaluated if len(i.distances) > 1)
        logger.info("GA done. returning %d tests (refined %d).", len(top), refinedCount)
        return top
