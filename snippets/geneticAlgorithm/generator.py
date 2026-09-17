"""Genetic algorithm for PX4 Aerialist obstacle tests.

Individuals carry path-relative genes (`genes.py`) that decode to one thin
wall or one chevron gate across a flight segment (`motifs.py`). The loop here
evaluates them on the simulator, breeds the next generation inside two
protected niches (one and two obstacles), reruns the promising ones and
hands the final choice to `suite.py`.
"""

import copy
import glob
import logging
import math
import os
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import List, Optional

from aerialist.px4.aerialist_test import AerialistTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase, is_isolated_execution

from geneticAlgorithm.genes import KIND_BY_COUNT, Genes, crossoverGenes, mutateGenes, randomGenes
from geneticAlgorithm.mission import Segment, Waypoint, allSegments, longSegments, parseWaypoints
from geneticAlgorithm.motifs import decode, layoutProblem
from geneticAlgorithm.suite import calibrateThreshold, officialRank, selectSuite, suiteDiversity

logger = logging.getLogger(__name__)


# Official scoring tiers: TIER_POINTS[i] applies when distance < TIER_BOUNDS[i];
# the last entry covers >= TIER_BOUNDS[-1] (5/2/1/0 for <0.25 / <1.0 / <1.5 / >=1.5).
TIER_BOUNDS = (0.25, 1.0, 1.5)
TIER_POINTS = (5, 2, 1, 0)
OFFICIAL_EVALUATION_RUNS = 3


@dataclass
class GAConfig:
    """All tunable parameters of the genetic algorithm."""
    seed: Optional[int] = None

    popSize: int = 20
    topK: int = 20  # Number of tests returned for the submission.
    # Draws allowed before a child is given up on.
    maxRetries: int = 20
    # Draws allowed for a fresh seed. Decoding is free, and on mission 4 the
    # segments leave the placement area for most of their length, so only a few
    # percent of gate draws fit and on mission 6 about one; 500 draws make a miss rare.
    seedDraws: int = 500
    tournamentK: int = 3
    # Elites are carried over per niche (obstacle count), not globally, so the
    # best pair survives even when every single scores higher than it.
    elitePerNiche: int = 1
    # Share of every generation that are gates (two obstacles): 6 of 20.
    gateShare: float = 0.3
    # Sides a seed may put the short end of its wall on: +1 is the side
    # sideNormal points to (the drone's left), -1 the other. Both are drawn.
    # A side = -1 wall has never scored (0 of 188 across the 21 speed-1
    # campaigns), so (1,) is measurably better on mission 3, but it is not
    # the default: the protocol asks for a second mission and mission 7 did
    # not reproduce the gain. Every confirmed suite score was measured here.
    seedSides: tuple = (-1, 1)
    # Mutation step scale of the first generation of children and its decay
    # per generation; 1.0 / 1.0 keeps the measured GENE_SIGMAS throughout.
    mutationScale: float = 1.0
    mutationDecay: float = 1.0
    # Share of every generation's non-elite slots filled with fresh seeds
    # instead of children. Selection concentrates the population on one
    # segment and side within two generations; immigrants keep other frames
    # in play, which is where a distinct trajectory has to come from.
    immigrantShare: float = 0.0
    # Give each niche's elite one more run at the start of every generation
    # and rank it on the mean. An elite is chosen on a single run, and a
    # layout's two runs disagree by 0.48 m on average, so an elite picked on
    # a lucky run otherwise breeds for the rest of the campaign.
    rerunElites: bool = False
    # False: every generation is fresh seeds (the seed-only arm of the A/B
    # measurement, GA_EVOLVE=0); selection and reruns stay the same.
    evolve: bool = True
    # Suite rules (suite.py): relax the similarity threshold to fill the
    # suite; at most maxPerFrame tests per (segment, side), 0 for no quota;
    # keep the best gate even when singles outscore it.
    relaxSimilarity: bool = True
    maxPerFrame: int = 0
    keepBestGate: bool = False
    # Continuous search-fitness shaping: shapeGain must stay below the
    # smallest tier gap (1 point) so a shaped score can never search-outrank
    # a genuinely better tier, only break ties within one. incompleteFactor
    # discourages breeding from stuck runs, which the suite excludes anyway.
    shapeGain: float = 0.25
    incompleteFactor: float = 0.25
    parallelWorkers: int = field(
        default_factory=lambda: max(1, int(os.environ.get("GA_WORKERS", "1")))
    )

    # Placement area, the same for every case study of the competition.
    xMin: float = -40.0
    xMax: float = 30.0
    yMin: float = 10.0
    yMax: float = 40.0

    # Post-GA refinement.
    refineFraction: float = 0.10
    refineThreshold: float = 1.5
    # The official score averages three executions, and the first one is the
    # initial evaluation, so only the remaining two are reruns.
    nReruns: int = OFFICIAL_EVALUATION_RUNS - 1
    failSentinel: float = 5.0

    # Stuck-avoidance detection: drone is "stuck" if its final trajectory point
    # is farther than goalTol from the last mission waypoint. The watchdog
    # kills runs that exceed timeoutMul * (slowest reached-goal run so far),
    # with minTimeout as the floor before any good run has been observed.
    goalTol: float = 5.0
    # A run that misses the goal without ever coming this close to an obstacle
    # is treated as a simulator flake, not as a scored measurement.
    flakeObstacleDistanceM: float = 5.0
    maxFlakeRetries: int = 2
    timeoutMul: float = 2.5
    minTimeout: float = 300.0
    # A layout that keeps failing to run has no
    # way to turn illegal, so nothing else retires it: without this cap it
    # would be retried with the same genes until the whole budget is gone.
    maxRunFailures: int = 2


@dataclass(eq=False)
class Individual:
    """One candidate layout plus the results of its simulator runs."""

    # eq=False: individuals are compared by identity, never by their fields.

    obstacles: List[Obstacle]
    # The path-relative description the obstacles were decoded from; None
    # only for layouts loaded from a YAML by the experiment scripts.
    genes: Optional[Genes] = None
    fitness: float = float("inf")
    officialScore: float = 0.0
    testCase: Optional[TestCase] = None
    valid: bool = False
    distances: List[float] = field(default_factory=list)
    # Simulated flight seconds per run (from the trajectory), not wall-clock:
    # feeds officialScore and fitness. `duration` below is wall-clock and
    # feeds only the watchdog.
    flightDurations: List[float] = field(default_factory=list)
    # One trajectory per run, for the suite's similarity check.
    trajectories: List = field(default_factory=list)
    stuck: bool = False
    # No legal layout could be drawn for this slot: it costs no run and must
    # not hold up the generation, or a niche with nothing legal (side +1
    # gates on mission 6) would freeze the search.
    illegal: bool = False
    duration: float = 0.0
    # Discovery generation the layout was evaluated in (0 for the seeds).
    generation: int = 0
    # Failed simulator attempts so far, while still unresolved (not valid,
    # not illegal). Reset is unnecessary: an individual stops accumulating
    # this the moment it turns valid, since it then leaves freeSlots for good.
    attempts: int = 0


class AttemptOutcome(IntEnum):
    """Result of one simulator attempt; nonzero outcomes consumed budget."""

    SKIPPED = 0
    CONSUMED = 1
    FLAKE = 2


def placementBounds(cfg: GAConfig):
    """Return the placement area as (xMin, xMax, yMin, yMax)."""
    return cfg.xMin, cfg.xMax, cfg.yMin, cfg.yMax


def invalidLayout(cfg: GAConfig, obstacles: List[Obstacle]):
    """Return whether a layout overlaps itself or leaves the placement area."""
    return layoutProblem(placementBounds(cfg), obstacles, [], -1, strict=False) is not None


def tierPoints(d: float):
    """Return the competition points awarded for one minimum distance."""
    for b, p in zip(TIER_BOUNDS, TIER_POINTS[:-1]):
        if d < b:
            return p
    return TIER_POINTS[-1]


def shapingTerm(d: float):
    """Return m(d): how close a distance is to the next better tier, in [0, 1].

    Inside each tier the value falls linearly from 1 at the tier's lower
    bound to 0 at its upper bound; beyond the last bound it decays
    exponentially. So two distances in the same tier still compare by how
    near they are to the next tier down.
    """
    if d >= TIER_BOUNDS[2]:
        return math.exp(-(d - TIER_BOUNDS[2]))
    if d >= TIER_BOUNDS[1]:
        return (TIER_BOUNDS[2] - d) / (TIER_BOUNDS[2] - TIER_BOUNDS[1])
    if d >= TIER_BOUNDS[0]:
        return (TIER_BOUNDS[1] - d) / (TIER_BOUNDS[1] - TIER_BOUNDS[0])
    return (TIER_BOUNDS[0] - d) / TIER_BOUNDS[0]


def shapedPoints(d: float, shapeGain: float):
    """Return g(d) = tierPoints(d) plus a bounded continuous nudge.

    shapeGain must stay below 1 (the smallest real tier gap) so shaping can
    never make a worse-tier candidate outscore a better-tier one; it only
    breaks ties within a tier by real clearance.
    """
    return tierPoints(d) + shapeGain * shapingTerm(d)


def scoreFromPoints(avgPoint: float, obstacleCount: int, flightDurations: List[float]):
    """Return the official-formula score for an already-averaged point value."""
    if not flightDurations:
        return 0.0
    if obstacleCount <= 0:
        raise ValueError("obstacleCount must be positive")
    if any(duration <= 0.0 for duration in flightDurations):
        raise ValueError("flight durations must be positive")
    avgTimeMinutes = statistics.fmean(flightDurations) / 60.0
    return (avgPoint * 10.0) / ((obstacleCount ** 2) * avgTimeMinutes)


def officialTestScore(
    distances: List[float],
    obstacleCount: int,
    flightDurations: List[float],
):
    """Return the official failure score for one test case.

    Distances are converted to points independently and then averaged.
    Runtime is the average simulated flight time in minutes, taken from the
    trajectory rather than wall-clock execution time, so the estimate is
    independent of how many simulator workers were contending for the host.
    The obstacle-count penalty is quadratic, exactly as specified by the
    competition report.
    """
    if not distances or not flightDurations:
        return 0.0
    if len(distances) != len(flightDurations):
        raise ValueError("each distance must have a matching flight duration")
    avgPoint = statistics.fmean(tierPoints(distance) for distance in distances)
    return scoreFromPoints(avgPoint, obstacleCount, flightDurations)


def searchTestScore(
    distances: List[float],
    obstacleCount: int,
    flightDurations: List[float],
    shapeGain: float,
    stuck: bool,
    incompleteFactor: float,
):
    """Return a continuous, search-only analogue of officialTestScore.

    Uses shapedPoints instead of raw tier points, so candidates tied on the
    discrete tier are still ranked by real clearance; this is what makes the
    fitness monotone instead of a staircase. Two tests in the same tier get
    the same official score but different search scores, which keeps the
    search pushing the distance down. Applies incompleteFactor when the run
    never reached the goal, since the suite excludes stuck candidates
    regardless of their distance, so a stuck layout should not out-breed a
    completing one of similar quality.
    """
    if not distances or not flightDurations:
        return 0.0
    if len(distances) != len(flightDurations):
        raise ValueError("each distance must have a matching flight duration")
    avgPoint = statistics.fmean(shapedPoints(distance, shapeGain) for distance in distances)
    completionFactor = incompleteFactor if stuck else 1.0
    return scoreFromPoints(avgPoint, obstacleCount, flightDurations) * completionFactor


def searchSelectionKey(ind: Individual):
    """Return the sort key of the search phase, most important element first.

    Ties on fitness go to the layout with fewer obstacles, then to the
    closer approach, then to the shorter flight. Lower is better throughout.
    """
    meanDistance = statistics.fmean(ind.distances) if ind.distances else float("inf")
    meanDuration = (
        statistics.fmean(ind.flightDurations) if ind.flightDurations else float("inf")
    )
    return ind.fitness, len(ind.obstacles), meanDistance, meanDuration


def flightSeconds(trajectory):
    """Return the simulated flight time in seconds, from the trajectory timestamps.

    Independent of wall-clock scheduling: two runs of the same layout on a
    busy vs. idle host still report the same flight time, unlike
    time.monotonic() around the simulator call.
    """
    positions = getattr(trajectory, "positions", None) if trajectory is not None else None
    if not positions:
        return 0.0
    return (positions[-1].timestamp - positions[0].timestamp) / 1_000_000.0


def reachedGoal(trajectory, goalXY: Waypoint, tol: float):
    """Return whether the flight ended within tol metres of the goal."""
    # Trajectory.positions may be empty when ALLIGN_ORIGIN runs against a
    # truncated log; treat missing data as "did not reach" so the watchdog
    # path and the unfinished-flight path collapse into the same flag.
    positions = getattr(trajectory, "positions", None) if trajectory is not None else None
    if not positions:
        return False
    last = positions[-1]
    return math.hypot(last.x - goalXY[0], last.y - goalXY[1]) <= tol


def isFlakeRun(cfg: GAConfig, trajectory, goalXY, distances: List[float], label: str):
    """Return whether a run missed the goal without ever coming near an obstacle."""
    # Such a flight says nothing about the layout: the drone stopped on its own,
    # so the run is treated as a simulator flake instead of as a measurement.
    if goalXY is None or reachedGoal(trajectory, goalXY, cfg.goalTol):
        return False
    if not distances or min(distances) <= cfg.flakeObstacleDistanceM:
        return False
    logger.warning(
        "%sflake: goal not reached and every obstacle stayed more than %.1fm away",
        label,
        cfg.flakeObstacleDistanceM,
    )
    return True


def obstacleCountSchedule(popSize: int, gateShare: float = 0.3):
    """Return the obstacle-count quota of one generation, gates spread evenly."""
    # One or two obstacles only: the score is divided by obstacles^2, so a
    # three-obstacle test cannot beat a mediocre single wall. Gates get a
    # fixed share of every generation (6 of 20 by default): a pair only beats
    # a single when it causes a crash, and free selection would wipe the
    # pairs out before they get that chance.
    gates = round(popSize * gateShare)
    return [2 if (k + 1) * gates // popSize > k * gates // popSize else 1 for k in range(popSize)]


def mutationScaleAt(cfg: GAConfig, gen: int):
    """Return the mutation step scale for the children of generation gen."""
    # Geometric decay from the first children on, never below a quarter of
    # the measured sigmas: below that a step no longer moves the response.
    # At the shipped 1.0 / 1.0 this is a constant 1.0; the knob exists for
    # the annealing arm of the tuning study.
    return max(0.25, cfg.mutationScale * cfg.mutationDecay ** max(0, gen - 1))


def nicheOf(ind: Individual):
    """Return the niche an individual belongs which is its obstacle count."""
    return len(ind.obstacles)


def buildIndividual(cfg: GAConfig, genes: Genes, segments: List[Segment], strict: bool):
    """Decode the genes; return the individual, or None if the layout is unusable."""
    obstacles = decode(genes, segments)
    problem = layoutProblem(placementBounds(cfg), obstacles, segments, genes.segment, strict)
    if problem is not None:
        logger.debug("rejected %s: %s", genes, problem)
        return None
    return Individual(obstacles=obstacles, genes=genes)


def freshIndividual(
    rng: random.Random,
    cfg: GAConfig,
    segments: List[Segment],
    segmentChoices: List[int],
    n: int,
):
    """Return a fresh seed with n obstacles (1 wall, 2 gate) whose layout is legal.

    Draws random genes until one decodes to a legal layout. The first pass
    also rejects walls that cut another segment; if seedDraws draws all
    fail (segments packed too close), a second pass allows the crossing
    rather than leaving the slot empty.
    """
    kind = KIND_BY_COUNT[n]
    for strict in (True, False):
        for _ in range(cfg.seedDraws):
            genes = randomGenes(rng, kind, segmentChoices, cfg.seedSides)
            ind = buildIndividual(cfg, genes, segments, strict)
            if ind is not None:
                if not strict:
                    logger.info("seed accepted although a wall crosses another segment: %s", genes)
                return ind
    logger.warning("no legal %s layout found on segments %s; the slot is skipped", kind, segmentChoices)
    return Individual(obstacles=decode(genes, segments), genes=genes, illegal=True)


def childOf(rng: random.Random, cfg: GAConfig, segments: List[Segment], a: Individual, b: Individual, scale: float = 1.0):
    """Return the child of two parents, or None if it decodes to an unusable layout."""
    # Uniform crossover inside the parents' frame, then exactly one gene is
    # moved, so every child differs from both parents by a measurable step.
    genes = mutateGenes(rng, crossoverGenes(rng, a.genes, b.genes), scale)
    return buildIndividual(cfg, genes, segments, strict=True)


def tournament(
    rng: random.Random,
    cfg: GAConfig,
    pop: List[Individual],
    niche: Optional[int] = None,
):
    """Return the best of tournamentK individuals sampled inside one niche."""
    # Restricting the sample to a single obstacle count is what protects the
    # niches: with a global tournament the pair quota gets refilled with
    # children of singles and the pair motif disappears in a few generations.
    pool = [ind for ind in pop if (niche is None or nicheOf(ind) == niche) and not ind.illegal]
    if not pool:
        pool = pop
    contenders = rng.sample(pool, min(cfg.tournamentK, len(pool)))
    return min(contenders, key=searchSelectionKey)


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
    """Kill leftover simulator processes and delete the stale PX4 files."""
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
    """Start a timer that kills a simulation still running after killAfter seconds."""
    # The returned flag is a 1-element list so the timer callback can mutate it
    # without a nonlocal binding; cleanupSimState pkills PX4/Gazebo, which
    # unblocks agent.run() with an exception.
    if killAfter <= 0:
        return None, [False]
    killed = [False]
    def watchdog():
        """Kill the simulation and record that the watchdog fired."""
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
    """Run one simulation for an individual and store its distance and fitness."""
    if invalidLayout(cfg, ind.obstacles):
        ind.fitness = float("inf")
        ind.valid = False
        ind.illegal = True
        return AttemptOutcome.SKIPPED
    ind.attempts += 1
    isolated = is_isolated_execution()
    timer, killed = (None, [False]) if isolated else armWatchdog(killAfter)
    t0 = time.monotonic()
    try:
        tc = TestCase(caseStudy, ind.obstacles)
        tc.execute()
        distances = tc.get_distances()
        if not distances:
            raise RuntimeError("no distances")
        minDist = float(min(distances))
        # "minimum_distance:" kept verbatim so external grep-based scorers keep
        # matching; the obstacle count feeds the lab's campaign summary plot.
        logger.info("minimum_distance:%s obstacle_count=%d", minDist, len(ind.obstacles))
        # Threads interleave log lines, so repeat the distance next to the genes.
        logger.info("evaluated %s -> %.3f m", ind.genes, minDist)
        if isFlakeRun(cfg, tc.trajectory, goalXY, distances, ""):
            return AttemptOutcome.FLAKE
        tc.plot()
        ind.testCase = tc
        ind.distances.append(minDist)
        ind.flightDurations.append(max(flightSeconds(tc.trajectory), 1e-9))
        ind.trajectories.append(tc.trajectory)
        if goalXY is not None and not reachedGoal(tc.trajectory, goalXY, cfg.goalTol):
            ind.stuck = True
            logger.info("stuck: final trajectory point not within %.1fm of goal", cfg.goalTol)
        ind.officialScore = officialTestScore(
            ind.distances, len(ind.obstacles), ind.flightDurations
        )
        ind.fitness = -searchTestScore(
            ind.distances, len(ind.obstacles), ind.flightDurations,
            cfg.shapeGain, ind.stuck, cfg.incompleteFactor,
        )
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
        if not isolated:
            cleanupSimState()
    return AttemptOutcome.CONSUMED


def rerun(
    cfg: GAConfig,
    ind: Individual,
    caseStudy: AerialistTest,
    goalXY: Optional[Waypoint] = None,
    killAfter: float = 0.0,
):
    """Run an already evaluated layout again and append the new measurement."""
    isolated = is_isolated_execution()
    timer, killed = (None, [False]) if isolated else armWatchdog(killAfter)
    t0 = time.monotonic()
    try:
        tc = TestCase(caseStudy, copy.deepcopy(ind.obstacles))
        tc.execute()
        distances = tc.get_distances()
        if not distances:
            raise RuntimeError("no distances")
        minDist = float(min(distances))
        logger.info("rerun minimum_distance:%s", minDist)
        if isFlakeRun(cfg, tc.trajectory, goalXY, distances, "rerun "):
            return AttemptOutcome.FLAKE
        tc.plot()
        ind.distances.append(minDist)
        ind.flightDurations.append(max(flightSeconds(tc.trajectory), 1e-9))
        ind.trajectories.append(tc.trajectory)
        ind.testCase = tc
        if goalXY is not None and not reachedGoal(tc.trajectory, goalXY, cfg.goalTol):
            ind.stuck = True
            logger.info("stuck on rerun: final point not within %.1fm of goal", cfg.goalTol)
    except Exception as e:
        if killed[0]:
            ind.stuck = True
            logger.info("stuck on rerun: watchdog killed run after %.1fs", time.monotonic() - t0)
        else:
            # A finite sentinel records this execution in the official zero-point
            # tier while keeping the three-run average well-defined.
            logger.warning("rerun failed (penalised with sentinel %s): %s", cfg.failSentinel, e)
        ind.distances.append(cfg.failSentinel)
        # No trajectory survives a failed run, so flight time cannot be
        # measured; the watchdog ceiling already passed to this call is a
        # conservative stand-in that keeps the failure's weight bounded.
        ind.flightDurations.append(max(killAfter, 1e-9))
    finally:
        if timer is not None:
            timer.cancel()
        ind.duration = time.monotonic() - t0
        if not isolated:
            cleanupSimState()
    return AttemptOutcome.CONSUMED


def runParallelBatch(cfg: GAConfig, jobs, operation, label: str):
    """Run independent simulator jobs and report aggregate throughput."""
    if not jobs:
        return []

    def timedOperation(job):
        """Run one job and return its result together with its duration."""
        started = time.monotonic()
        result = operation(*job)
        return result, time.monotonic() - started

    workers = min(cfg.parallelWorkers, len(jobs))
    started = time.monotonic()
    if workers == 1:
        completed = [timedOperation(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ga-sim") as pool:
            futures = [pool.submit(timedOperation, job) for job in jobs]
            # Preserve population order even though execution is concurrent.
            completed = [future.result() for future in futures]
    wall = time.monotonic() - started
    results = [result for result, _ in completed]
    simulatorWork = sum(duration for _, duration in completed)
    logger.info(
        "%s batch: workers=%d jobs=%d wall=%.1fs simulator-work=%.1fs throughput=%.2fx",
        label,
        workers,
        len(jobs),
        wall,
        simulatorWork,
        simulatorWork / wall if wall > 0 else 0.0,
    )
    return results


def runRetryingBatch(cfg: GAConfig, jobs, operation, label: str, attemptBudget: int):
    """Run the jobs in waves, retrying flakes, and return the attempts used."""
    pending = [(job, 0) for job in jobs]
    attemptsUsed = 0
    wave = 0
    while pending and attemptsUsed < attemptBudget:
        capacity = min(cfg.parallelWorkers, attemptBudget - attemptsUsed)
        batch = []
        deferred = []
        batched = set()
        for item in pending:
            # Two attempts on the same individual would append to the same
            # distance list, so they must not share a wave.
            individualId = id(item[0][1])
            if len(batch) < capacity and individualId not in batched:
                batched.add(individualId)
                batch.append(item)
            else:
                deferred.append(item)
        pending = deferred
        if not batch:
            break
        outcomes = runParallelBatch(
            cfg,
            [job for job, _ in batch],
            operation,
            f"{label} wave {wave}",
        )
        retries = []
        for (job, retryCount), outcome in zip(batch, outcomes):
            if outcome != AttemptOutcome.SKIPPED:
                attemptsUsed += 1
            if outcome == AttemptOutcome.FLAKE:
                if retryCount < cfg.maxFlakeRetries:
                    retries.append((job, retryCount + 1))
                else:
                    logger.warning(
                        "%s: dropping measurement after %d flake retries",
                        label,
                        retryCount,
                    )
        # Retry flakes before spending the remaining budget on new jobs.
        pending = retries + pending
        wave += 1
    return attemptsUsed


def nicheChild(
    rng: random.Random,
    cfg: GAConfig,
    pop: List[Individual],
    niche: int,
    segments: List[Segment],
    segmentChoices: List[int],
    scale: float = 1.0,
):
    """Return one new individual for a niche, bred from parents of that niche."""
    if any(nicheOf(ind) == niche and not ind.illegal for ind in pop):
        for _ in range(cfg.maxRetries):
            child = childOf(rng, cfg, segments, tournament(rng, cfg, pop, niche), tournament(rng, cfg, pop, niche), scale)
            if child is not None:
                return child
    # The niche is extinct, or breeding keeps producing illegal layouts:
    # reseed the slot from the motifs instead of borrowing another niche.
    return freshIndividual(rng, cfg, segments, segmentChoices, niche)


def nicheGeneration(
    rng: random.Random,
    cfg: GAConfig,
    pop: List[Individual],
    niche: int,
    quota: int,
    segments: List[Segment],
    segmentChoices: List[int],
    scale: float = 1.0,
):
    """Return the quota slots of one niche: its own elites plus new children."""
    # The elite count is capped one below the quota so a niche always keeps at
    # least one slot to explore with, however small the population is.
    pool = [ind for ind in pop if nicheOf(ind) == niche]
    eliteCount = min(cfg.elitePerNiche, len(pool), max(0, quota - 1))
    children = sorted(pool, key=searchSelectionKey)[:eliteCount]
    immigrants = int(round(cfg.immigrantShare * (quota - eliteCount)))
    while len(children) < quota:
        if len(children) >= quota - immigrants:
            children.append(freshIndividual(rng, cfg, segments, segmentChoices, niche))
        else:
            children.append(nicheChild(rng, cfg, pop, niche, segments, segmentChoices, scale))
    return children


def nextGeneration(
    rng: random.Random,
    cfg: GAConfig,
    pop: List[Individual],
    segments: List[Segment],
    segmentChoices: List[int],
    scale: float = 1.0,
):
    """Return the next generation, one protected niche per obstacle count."""
    counts = obstacleCountSchedule(cfg.popSize, cfg.gateShare)
    queues = {
        niche: nicheGeneration(rng, cfg, pop, niche, counts.count(niche), segments, segmentChoices, scale)
        for niche in set(counts)
    }
    # Emit in schedule order so the population keeps the layout of generation 0.
    return [queues[niche].pop(0) for niche in counts]


class GeneticGenerator:
    """Two-phase generator: discover layouts, then rerun the promising ones."""

    def __init__(self, missionPath: str, config: Optional[GAConfig] = None):
        """Load the mission, fix the random seed and parse the flight segments."""
        cfg = config or GAConfig()
        # GA_SEED in the environment repeats a run exactly; an explicit
        # cfg.seed still wins. Without either, a fresh seed is drawn and logged.
        envSeed = os.environ.get("GA_SEED")
        if cfg.seed is None and envSeed is not None:
            cfg = replace(cfg, seed=int(envSeed))
        # GA_EVOLVE=0 switches breeding off for the seed-only measurement arm.
        if os.environ.get("GA_EVOLVE", "1") == "0":
            cfg = replace(cfg, evolve=False)
        # Resolve the seed up front so it appears in logs and on cfg, then build
        # a private RNG instance: no calls into the process-wide random module.
        resolvedSeed = (
            cfg.seed if cfg.seed is not None else random.SystemRandom().randrange(2 ** 32)
        )
        self.config = replace(cfg, seed=resolvedSeed)
        self.rng = random.Random(resolvedSeed)
        logger.info("GA seed: %s", resolvedSeed)
        self.caseStudy = AerialistTest.from_yaml(missionPath)
        self.waypoints = parseWaypoints(self.caseStudy.robot.mission_file)
        self.segments = allSegments(self.waypoints)
        # Seeds go on the long segments; a mission made of short hops only would
        # fall back to every segment it has.
        self.seedSegments = [segment.index for segment in longSegments(self.waypoints)] or [segment.index for segment in self.segments]
        logger.info("waypoints: %s", self.waypoints)
        logger.info("seed segments: %s", self.seedSegments)

    def rescore(self, cfg: GAConfig, ind: Individual):
        """Recompute the two scores of an individual from all of its runs."""
        ind.officialScore = officialTestScore(ind.distances, len(ind.obstacles), ind.flightDurations)
        ind.fitness = -searchTestScore(
            ind.distances, len(ind.obstacles), ind.flightDurations,
            cfg.shapeGain, ind.stuck, cfg.incompleteFactor,
        )

    def generate(self, budget: int):
        """Return the best test cases found within a budget of simulator runs."""
        cfg = self.config
        if budget <= 0:
            logger.error("budget must be > 0")
            return []
        if cfg.parallelWorkers > 1 and not is_isolated_execution():
            logger.warning(
                "parallel workers require an isolated docker/k8s agent; using one local worker"
            )
            cfg = replace(cfg, parallelWorkers=1)

        # Shrink population when budget is too small to fill the default pop.
        # Below popSize sims the initial evaluation alone would exhaust the budget,
        # so scale down to sqrt(budget) to leave room for at least a few generations.
        if budget < cfg.popSize:
            effectivePopSize = max(2, round(math.sqrt(budget)))
            cfg = replace(cfg, popSize=effectivePopSize)

        # Reserve refinement budget only when discovery can still run one full population.
        # Any reserve that cannot be spent on useful reruns is returned to discovery.
        phase1Budget = min(budget, max(cfg.popSize, int((1 - cfg.refineFraction) * budget)))
        phase2Budget = budget - phase1Budget

        logger.info(
            "GA: budget=%s pop=%s workers=%s discovery=%s refinement-reserve=%s evolve=%s",
            budget,
            cfg.popSize,
            cfg.parallelWorkers,
            phase1Budget,
            phase2Budget,
            cfg.evolve,
        )

        # Goal == last mission waypoint, used for the stuck check.
        goalXY = self.waypoints[-1]
        maxGoodDuration = 0.0

        counts = obstacleCountSchedule(cfg.popSize, cfg.gateShare)
        logger.info("initial obstacle counts: %s", counts)
        pop = [freshIndividual(self.rng, cfg, self.segments, self.seedSegments, n) for n in counts]
        evaluated: List[Individual] = []
        discoveryAttempts = 0
        gen = 0
        noProgressLimit = max(3, cfg.maxRetries)

        def runDiscovery(attemptBudget: int):
            """Spend up to attemptBudget simulations on new layouts."""
            nonlocal pop, gen, maxGoodDuration
            attemptsUsed = 0
            noProgressRounds = 0

            while attemptsUsed < attemptBudget and noProgressRounds < noProgressLimit:
                remaining = attemptBudget - attemptsUsed
                freeSlots = [idx for idx, ind in enumerate(pop) if not ind.valid and not ind.illegal]
                if not freeSlots:
                    gen += 1
                    if cfg.evolve:
                        pop = nextGeneration(self.rng, cfg, pop, self.segments, self.seedSegments, mutationScaleAt(cfg, gen))
                    else:
                        pop = [freshIndividual(self.rng, cfg, self.segments, self.seedSegments, n) for n in counts]
                    freeSlots = [idx for idx, ind in enumerate(pop) if not ind.valid and not ind.illegal]
                candidateSlots = freeSlots[:remaining]
                candidates = [pop[idx] for idx in candidateSlots]

                killAfter = max(cfg.minTimeout, cfg.timeoutMul * maxGoodDuration)
                # Confirm the carried-over elites before breeding from them.
                elites = [ind for ind in pop if ind.valid] if cfg.rerunElites else []
                if elites and remaining > len(elites):
                    eliteJobs = [(cfg, ind, self.caseStudy, goalXY, killAfter) for ind in elites]
                    attemptsUsed += runRetryingBatch(
                        cfg, eliteJobs, rerun, "elite confirmation %d" % gen, len(elites))
                    for ind in elites:
                        self.rescore(cfg, ind)
                    remaining = attemptBudget - attemptsUsed
                    candidateSlots = candidateSlots[:remaining]
                    candidates = candidates[:remaining]
                jobs = [(cfg, ind, self.caseStudy, goalXY, killAfter) for ind in candidates]
                usedNow = runRetryingBatch(cfg, jobs, evaluate, f"generation {gen}", remaining)
                attemptsUsed += usedNow

                for slotIdx, ind in zip(candidateSlots, candidates):
                    if ind.valid and ind not in evaluated:
                        ind.generation = gen
                        evaluated.append(ind)
                        # Only reached-goal runs feed the threshold; otherwise a slow
                        # stuck run would inflate the budget and disarm the watchdog.
                        if not ind.stuck and ind.duration > maxGoodDuration:
                            maxGoodDuration = ind.duration
                    elif not ind.valid and not ind.illegal and ind.attempts >= cfg.maxRunFailures:
                        # A layout that keeps failing (e.g. traps the drone so
                        # it never lands) would otherwise be retried with the
                        # same genes until the budget runs out; reseed the slot.
                        logger.warning(
                            "giving up on %s after %d failed attempts; reseeding the slot",
                            ind.genes, ind.attempts,
                        )
                        pop[slotIdx] = freshIndividual(
                            self.rng, cfg, self.segments, self.seedSegments, nicheOf(ind))

                best = min(evaluated, key=officialRank) if evaluated else None
                logger.info(
                    "[gen %d] best official score=%.4f discovery attempts=%d/%d",
                    gen,
                    best.officialScore if best is not None else 0.0,
                    attemptsUsed,
                    attemptBudget,
                )
                if usedNow == 0:
                    noProgressRounds += 1
                else:
                    noProgressRounds = 0

            if attemptsUsed < attemptBudget:
                logger.warning(
                    "discovery stopped after %d no-progress rounds; %d attempts remain",
                    noProgressRounds,
                    attemptBudget - attemptsUsed,
                )
            return attemptsUsed

        discoveryAttempts += runDiscovery(phase1Budget)

        # Refine only the candidates that would provisionally be submitted. This
        # spends reruns on positive, distinct results rather than zero-point or
        # already-filtered layouts.
        provisional, _ = selectSuite(cfg, evaluated, calibrateThreshold(evaluated))
        promising = [
            ind
            for ind in provisional
            if ind.distances and ind.distances[0] < cfg.refineThreshold
        ]
        rerunJobs = []
        killAfter = max(cfg.minTimeout, cfg.timeoutMul * maxGoodDuration)
        # Round-robin ordering gives every provisional result a second run
        # before any result gets a third.
        for rerunIndex in range(cfg.nReruns):
            for cand in promising:
                existingReruns = max(0, len(cand.distances) - 1)
                if existingReruns <= rerunIndex and len(rerunJobs) < phase2Budget:
                    rerunJobs.append((cfg, cand, self.caseStudy, goalXY, killAfter))
            if len(rerunJobs) >= phase2Budget:
                break

        refinementAttempts = 0
        if rerunJobs:
            refinementAttempts = runRetryingBatch(cfg, rerunJobs, rerun, "refinement", phase2Budget)
        for cand in promising:
            if len(cand.distances) > 1:
                self.rescore(cfg, cand)
                meanD = sum(cand.distances) / len(cand.distances)
                logger.info(
                    "refined: runs=%d mean-distance=%.2f official-score=%.4f",
                    len(cand.distances),
                    meanD,
                    cand.officialScore,
                )

        # If there were too few useful reruns, spend the remainder discovering
        # new layouts. The same global ledger keeps total attempts <= budget.
        totalAttempts = discoveryAttempts + refinementAttempts
        if totalAttempts < budget:
            returnedBudget = budget - totalAttempts
            logger.info(
                "returning %d unused refinement attempts to discovery",
                returnedBudget,
            )
            discoveryAttempts += runDiscovery(returnedBudget)
            totalAttempts = discoveryAttempts + refinementAttempts

        zeroPointCount = sum(1 for cand in evaluated if cand.officialScore <= 0.0)
        # Re-select from the full positive pool after refinement. Candidates
        # demoted by reruns are automatically replaced when alternatives exist.
        threshold = calibrateThreshold(evaluated)
        selected, threshold = selectSuite(cfg, evaluated, threshold)
        logger.info(
            "suite: tests=%d sum-score=%.4f mean-trajectory-distance=%.2f threshold=%.2f",
            len(selected),
            sum(ind.officialScore for ind in selected),
            suiteDiversity(selected),
            threshold,
        )
        refinedCount = sum(1 for i in evaluated if len(i.distances) > 1)
        logger.info(
            "GA done. attempts=%d/%d returning %d positive-score tests "
            "(refined %d, dropped zero-point %d).",
            totalAttempts,
            budget,
            len(selected),
            refinedCount,
            zeroPointCount,
        )
        # Every evaluated layout stays available for inspection after the run.
        self.evaluated = evaluated
        return [s.testCase for s in selected]
