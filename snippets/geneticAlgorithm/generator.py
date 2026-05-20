"""Minimalist single-file genetic algorithm for PX4 Aerialist obstacle tests."""

import copy
import json
import logging
import math
import random
import statistics
import subprocess
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

    # Path-aware seeding.
    pathBiasProb: float = 0.7
    pathSigma: float = 8.0

    # Post-GA refinement.
    refineFraction: float = 0.25
    refineThreshold: float = 1.5
    nReruns: int = 4
    failSentinel: float = 5.0

    # Top-K obstacle-centroid diversity filter.
    diversityMinM: float = 10.0


Waypoint = Tuple[float, float]


@dataclass
class Individual:
    obstacles: List[Obstacle]
    fitness: float = float("inf")
    testCase: Optional[TestCase] = None
    valid: bool = False
    distances: List[float] = field(default_factory=list)


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


def tierPoints(d: float):
    for b, p in zip(TIER_BOUNDS, TIER_POINTS[:-1]):
        if d < b:
            return p
    return TIER_POINTS[-1]


def fitnessFor(cfg: GAConfig, distances: List[float], obstacleCount: int):
    meanD = sum(distances) / len(distances)
    stdD = statistics.stdev(distances) if len(distances) > 1 else 0.0
    return (
        -tierPoints(meanD)
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
        if not hasOverlap(obstacles):
            break
    return Individual(obstacles=obstacles)


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


_CLEANUP_PATTERNS = ("mavsdk_server", "px4", "gzserver", "gzclient", "roslaunch", "nodelet")


def cleanupSimState():
    for pattern in _CLEANUP_PATTERNS:
        subprocess.run(
            ["pkill", "-9", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(1)


def evaluate(cfg: GAConfig, ind: Individual, caseStudy: AerialistTest):
    if hasOverlap(ind.obstacles):
        ind.fitness = float("inf")
        ind.valid = False
        return False
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
        ind.fitness = fitnessFor(cfg, ind.distances, len(ind.obstacles))
        ind.valid = True
    except Exception as e:
        logger.warning("sim failed: %s", e)
        ind.fitness = float("inf")
        ind.valid = False
    finally:
        cleanupSimState()
    return True


def rerun(cfg: GAConfig, ind: Individual, caseStudy: AerialistTest):
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
    except Exception as e:
        # A finite sentinel keeps mean/stdev defined, drops the run into the
        # worthless tier, and inflates variance so flaky candidates rank worse.
        logger.warning("rerun failed (penalised with sentinel %s): %s", cfg.failSentinel, e)
        ind.distances.append(cfg.failSentinel)
    finally:
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
            if not hasOverlap(c.obstacles):
                child = c
                break
        if child is None:
            for _ in range(cfg.maxRetries):
                c = randomIndividual(rng, cfg, waypoints)
                if not hasOverlap(c.obstacles):
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

        pop = [randomIndividual(self.rng, cfg, self.waypoints) for _ in range(cfg.popSize)]
        evaluated: List[Individual] = []
        simsUsed = 0
        gen = 0
        while simsUsed < phase1Budget and gen < cfg.maxGenerations:
            for ind in pop:
                if ind.valid:
                    continue
                consumed = evaluate(cfg, ind, self.caseStudy)
                if ind.valid:
                    evaluated.append(ind)
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
            for _ in range(nMore):
                rerun(cfg, cand, self.caseStudy)
                simsLeft -= 1
                if simsLeft <= 0:
                    break
            cand.fitness = fitnessFor(cfg, cand.distances, len(cand.obstacles))
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
