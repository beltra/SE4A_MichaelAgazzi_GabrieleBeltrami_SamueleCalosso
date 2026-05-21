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


@dataclass
class GAConfig:
    seed: Optional[int] = None

    popSize: int = 10
    topK: int = 10
    maxObstacles: int = 3
    maxRetries: int = 20
    maxGenerations: int = 200
    mutationRate: float = 0.2
    addProb: float = 0.15
    removeProb: float = 0.15
    sigmaFrac: float = 0.10
    tournamentK: int = 3
    eliteSize: int = 1

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

    # Fitness weights (additive, all small relative to tier point gap).
    meanWeight: float = 0.1
    countWeight: float = 0.05
    varianceWeight: float = 0.2
    # Continuous pull toward lower distances; provides gradient in the tier-0
    # zone (dist >= 1.5 m) where tierPoints is flat at 0.  Bounded in (-1, 0],
    # so it cannot outweigh a genuine tier promotion.
    continuousWeight: float = 1.0

    # Path-aware seeding.
    pathBiasProb: float = 0.8
    pathSigma: float = 8.0
    # Corridor seeding for initial population: lateral jitter around the
    # flight-path centreline when placing a wall-like obstacle head-on.
    corridorSigma: float = 2.0

    # Post-GA refinement.
    refineFraction: float = 0.10
    refineThreshold: float = 1.5
    nReruns: int = 4
    failSentinel: float = 5.0

    # Top-K obstacle-centroid diversity filter.
    diversityMinM: float = 10.0

    # Stuck-avoidance detection: drone is "stuck" if its final trajectory point
    # is farther than goalTol from the last mission waypoint. The watchdog
    # kills runs that exceed timeoutMul * (slowest reached-goal run so far),
    # with minTimeout as the floor before any good run has been observed.
    # stuckBonus is subtracted from fitness on stuck candidates: an SUT that
    # fails to complete the mission is itself a defect we want surfaced, so
    # we make these competitive with a tier-2 obstacle hit without dominating
    # a tier-5 one (the official scorer only rewards obstacle proximity).
    goalTol: float = 5.0
    timeoutMul: float = 2.5
    minTimeout: float = 300.0
    stuckBonus: float = 2.0


Waypoint = Tuple[float, float]


@dataclass
class Individual:
    obstacles: List[Obstacle]
    fitness: float = float("inf")
    testCase: Optional[TestCase] = None
    valid: bool = False
    distances: List[float] = field(default_factory=list)
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


def tierPoints(d: float):
    for b, p in zip(TIER_BOUNDS, TIER_POINTS[:-1]):
        if d < b:
            return p
    return TIER_POINTS[-1]


def fitnessFor(cfg: GAConfig, distances: List[float], obstacleCount: int, stuck: bool = False):
    meanD = sum(distances) / len(distances)
    stdD = statistics.stdev(distances) if len(distances) > 1 else 0.0
    return (
        -tierPoints(meanD)
        - cfg.continuousWeight / (1.0 + meanD)
        - (cfg.stuckBonus if stuck else 0.0)
        + cfg.meanWeight * meanD
        + cfg.countWeight * obstacleCount
        + cfg.varianceWeight * stdD
    )


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


def anchorOnPath(rng: random.Random, waypoints: List[Waypoint]):
    i = rng.randrange(len(waypoints) - 1)
    a, b = waypoints[i], waypoints[i + 1]
    t = rng.random()
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def randomObstacle(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: Optional[List[Waypoint]] = None,
):
    if waypoints is not None and len(waypoints) >= 2 and rng.random() < cfg.pathBiasProb:
        ax, ay = anchorOnPath(rng, waypoints)
        x = clamp(ax + rng.gauss(0, cfg.pathSigma), cfg.xMin, cfg.xMax)
        y = clamp(ay + rng.gauss(0, cfg.pathSigma), cfg.yMin, cfg.yMax)
    else:
        x = rng.uniform(cfg.xMin, cfg.xMax)
        y = rng.uniform(cfg.yMin, cfg.yMax)
    size = Obstacle.Size(
        l=rng.uniform(cfg.lMin, cfg.lMax),
        w=rng.uniform(cfg.wMin, cfg.wMax),
        h=cfg.hFixed,
    )
    position = Obstacle.Position(x=x, y=y, z=0, r=rng.uniform(cfg.rMin, cfg.rMax))
    return Obstacle(size, position)


def randomIndividual(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: Optional[List[Waypoint]] = None,
):
    n = rng.randint(1, cfg.maxObstacles)
    obstacles: List[Obstacle] = []
    for _ in range(cfg.maxRetries):
        obstacles = [randomObstacle(rng, cfg, waypoints) for _ in range(n)]
        if not invalidLayout(cfg, obstacles):
            break
    return Individual(obstacles=obstacles)


def corridorObstacle(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: List[Waypoint],
):
    # Place an obstacle that walls off the drone's path head-on: centre on a
    # random point along a path segment, rotate to face the approach direction,
    # add small lateral jitter so it is not always exactly on the centreline.
    i = rng.randrange(len(waypoints) - 1)
    a, b = waypoints[i], waypoints[i + 1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    segLen = math.hypot(dx, dy)
    if segLen < 1e-6:
        return randomObstacle(rng, cfg, waypoints)
    ux, uy = dx / segLen, dy / segLen   # unit vector along segment
    px, py = -uy, ux                    # unit vector perpendicular to segment
    t = rng.random()
    cx = a[0] + t * dx
    cy = a[1] + t * dy
    lateralOffset = rng.gauss(0.0, cfg.corridorSigma)
    x = clamp(cx + lateralOffset * px, cfg.xMin, cfg.xMax)
    y = clamp(cy + lateralOffset * py, cfg.yMin, cfg.yMax)
    # Rotate so the obstacle face is perpendicular to the approach direction;
    # mod 90 keeps r within [0, 90) regardless of segment orientation.
    r = clamp(math.degrees(math.atan2(dy, dx)) % 90.0, cfg.rMin, cfg.rMax)
    size = Obstacle.Size(l=rng.uniform(cfg.lMin, cfg.lMax), w=rng.uniform(cfg.wMin, cfg.wMax), h=cfg.hFixed)
    return Obstacle(size, Obstacle.Position(x=x, y=y, z=0, r=r))


def seededIndividual(
    rng: random.Random,
    cfg: GAConfig,
    waypoints: List[Waypoint],
):
    # Uses corridorObstacle for placement; falls back to randomIndividual if
    # the corridor placement cannot pass the layout check after maxRetries.
    n = rng.randint(1, cfg.maxObstacles)
    for _ in range(cfg.maxRetries):
        obstacles = [corridorObstacle(rng, cfg, waypoints) for _ in range(n)]
        if not invalidLayout(cfg, obstacles):
            return Individual(obstacles=obstacles)
    return randomIndividual(rng, cfg, waypoints)


def crossover(rng: random.Random, a: Individual, b: Individual):
    pool = a.obstacles + b.obstacles
    n = rng.choice([len(a.obstacles), len(b.obstacles)])
    chosen = rng.sample(pool, n)
    return Individual(obstacles=[copy.deepcopy(o) for o in chosen])


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
        ind.obstacles.append(randomObstacle(rng, cfg, waypoints))
    if rng.random() < cfg.removeProb and len(ind.obstacles) > 1:
        ind.obstacles.pop(rng.randrange(len(ind.obstacles)))
    return ind


def tournament(rng: random.Random, cfg: GAConfig, pop: List[Individual]):
    contenders = rng.sample(pop, min(cfg.tournamentK, len(pop)))
    return min(contenders, key=lambda i: i.fitness)


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
        tc.execute()
        distances = tc.get_distances()
        if not distances:
            raise RuntimeError("no distances")
        minDist = float(min(distances))
        # "minimum_distance:" kept verbatim so external grep-based scorers keep matching.
        logger.info("minimum_distance:%s", minDist)
        tc.plot()
        ind.testCase = tc
        ind.distances.append(minDist)
        if goalXY is not None and not reachedGoal(tc.trajectory, goalXY, cfg.goalTol):
            ind.stuck = True
            logger.info("stuck: final trajectory point not within %.1fm of goal", cfg.goalTol)
        ind.fitness = fitnessFor(cfg, ind.distances, len(ind.obstacles), ind.stuck)
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
    try:
        tc = TestCase(caseStudy, copy.deepcopy(ind.obstacles))
        tc.execute()
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
        if killed[0]:
            ind.stuck = True
            logger.info("stuck on rerun: watchdog killed run after %.1fs", time.monotonic() - t0)
        else:
            # A finite sentinel keeps mean/stdev defined, drops the run into the
            # worthless tier, and inflates variance so flaky candidates rank worse.
            logger.warning("rerun failed (penalised with sentinel %s): %s", cfg.failSentinel, e)
        ind.distances.append(cfg.failSentinel)
    finally:
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
    elite = min(pop, key=lambda i: i.fitness)
    children: List[Individual] = [elite]
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
        # Reserve refinement budget only when phase 1 can still run one full population.
        phase1Budget = min(budget, max(cfg.popSize, int((1 - cfg.refineFraction) * budget)))
        phase2Budget = budget - phase1Budget
        logger.info(
            "GA: budget=%s pop=%s phase1=%s phase2=%s",
            budget, cfg.popSize, phase1Budget, phase2Budget,
        )

        # Goal == last mission waypoint; parseWaypoints falls back to [(0,0)]
        # when the .plan is missing, in which case we cannot detect "stuck".
        goalXY = self.waypoints[-1] if len(self.waypoints) >= 2 else None
        maxGoodDuration = 0.0

        useCorridorSeed = len(self.waypoints) >= 2
        pop = [
            seededIndividual(self.rng, cfg, self.waypoints) if useCorridorSeed
            else randomIndividual(self.rng, cfg, self.waypoints)
            for _ in range(cfg.popSize)
        ]
        evaluated: List[Individual] = []
        simsUsed = 0
        gen = 0
        while simsUsed < phase1Budget and gen < cfg.maxGenerations:
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
            bestFitness = min(i.fitness for i in pop)
            logger.info("[gen %d] best=%.2f sims=%d/%d", gen, bestFitness, simsUsed, phase1Budget)
            if simsUsed >= phase1Budget:
                break
            pop = nextGeneration(self.rng, cfg, pop, self.waypoints)
            gen += 1

        # Phase 2: rerun promising single-shot candidates to filter flaky near-misses.
        promising = sorted(
            [i for i in evaluated if i.distances and i.distances[0] < cfg.refineThreshold],
            key=lambda i: i.fitness,
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
            cand.fitness = fitnessFor(cfg, cand.distances, len(cand.obstacles), cand.stuck)
            meanD = sum(cand.distances) / len(cand.distances)
            logger.info(
                "refined: runs=%d mean=%.2f fitness=%.2f",
                len(cand.distances), meanD, cand.fitness,
            )

        evaluated.sort(key=lambda i: i.fitness)
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
