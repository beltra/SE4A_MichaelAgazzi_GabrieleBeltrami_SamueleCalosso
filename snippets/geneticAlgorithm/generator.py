"""Minimalist single-file genetic algorithm for PX4 Aerialist obstacle tests."""

import copy
import math
import random
from dataclasses import dataclass
from typing import List, Optional

from aerialist.px4.aerialist_test import AerialistTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase


SEED = None
POP_SIZE = 10
TOP_K = 10
MAX_OBSTACLES = 3
MAX_RETRIES = 20
MAX_GENERATIONS = 200
MUTATION_RATE = 0.2
ADD_PROB = 0.15
REMOVE_PROB = 0.15
SIGMA_FRAC = 0.10
COUNT_WEIGHT = 0.5
TOURNAMENT_K = 3
ELITE_SIZE = 1
X_MIN, X_MAX = -40.0, 30.0
Y_MIN, Y_MAX = 10.0, 40.0
L_MIN, L_MAX = 2.0, 20.0
W_MIN, W_MAX = 2.0, 20.0
R_MIN, R_MAX = 0.0, 90.0
H_FIXED = 25.0


@dataclass
class Individual:
    obstacles: List[Obstacle]
    fitness: float = float("inf")
    testCase: Optional[TestCase] = None
    valid: bool = False


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def gaussianPerturb(v: float, lo: float, hi: float) -> float:
    sigma = SIGMA_FRAC * (hi - lo)
    return clamp(v + random.gauss(0.0, sigma), lo, hi)


def rotatedHalfExtents(l: float, w: float, rDeg: float):
    # Conservative AABB half-extents of a rotated (l x w) box.
    cosR = abs(math.cos(math.radians(rDeg)))
    sinR = abs(math.sin(math.radians(rDeg)))
    return (
        0.5 * (l * cosR + w * sinR),
        0.5 * (l * sinR + w * cosR),
    )


def hasOverlap(obstacles: List[Obstacle]) -> bool:
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


def randomObstacle() -> Obstacle:
    size = Obstacle.Size(
        l=random.uniform(L_MIN, L_MAX),
        w=random.uniform(W_MIN, W_MAX),
        h=H_FIXED,
    )
    position = Obstacle.Position(
        x=random.uniform(X_MIN, X_MAX),
        y=random.uniform(Y_MIN, Y_MAX),
        z=0,
        r=random.uniform(R_MIN, R_MAX),
    )
    return Obstacle(size, position)


def randomIndividual() -> Individual:
    n = random.randint(1, MAX_OBSTACLES)
    obstacles: List[Obstacle] = []
    for _ in range(MAX_RETRIES):
        obstacles = [randomObstacle() for _ in range(n)]
        if not hasOverlap(obstacles):
            break
    return Individual(obstacles=obstacles)


def crossover(a: Individual, b: Individual) -> Individual:
    pool = a.obstacles + b.obstacles
    n = random.choice([len(a.obstacles), len(b.obstacles)])
    chosen = random.sample(pool, n)
    return Individual(obstacles=[copy.deepcopy(o) for o in chosen])


def mutate(ind: Individual) -> Individual:
    for o in ind.obstacles:
        if random.random() < MUTATION_RATE:
            o.size.l = gaussianPerturb(o.size.l, L_MIN, L_MAX)
        if random.random() < MUTATION_RATE:
            o.size.w = gaussianPerturb(o.size.w, W_MIN, W_MAX)
        if random.random() < MUTATION_RATE:
            o.position.x = gaussianPerturb(o.position.x, X_MIN, X_MAX)
        if random.random() < MUTATION_RATE:
            o.position.y = gaussianPerturb(o.position.y, Y_MIN, Y_MAX)
        if random.random() < MUTATION_RATE:
            o.position.r = gaussianPerturb(o.position.r, R_MIN, R_MAX)
    if random.random() < ADD_PROB and len(ind.obstacles) < MAX_OBSTACLES:
        ind.obstacles.append(randomObstacle())
    if random.random() < REMOVE_PROB and len(ind.obstacles) > 1:
        ind.obstacles.pop(random.randrange(len(ind.obstacles)))
    return ind


def tournament(pop: List[Individual]) -> Individual:
    contenders = random.sample(pop, min(TOURNAMENT_K, len(pop)))
    return min(contenders, key=lambda i: i.fitness)


def evaluate(ind: Individual, caseStudy: AerialistTest) -> bool:
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
        print(f"minimum_distance:{minDist}")
        tc.plot()
        ind.testCase = tc
        ind.fitness = minDist + COUNT_WEIGHT * len(ind.obstacles)
        ind.valid = True
    except Exception as e:
        print(f"sim failed: {e}")
        ind.fitness = float("inf")
        ind.valid = False
    return True


def nextGeneration(pop: List[Individual]) -> List[Individual]:
    elite = min(pop, key=lambda i: i.fitness)
    children: List[Individual] = [elite]
    while len(children) < POP_SIZE:
        child: Optional[Individual] = None
        for _ in range(MAX_RETRIES):
            c = mutate(crossover(tournament(pop), tournament(pop)))
            if not hasOverlap(c.obstacles):
                child = c
                break
        if child is None:
            for _ in range(MAX_RETRIES):
                c = randomIndividual()
                if not hasOverlap(c.obstacles):
                    child = c
                    break
        if child is None:
            child = randomIndividual()
        children.append(child)
    return children


class GeneticGenerator:
    def __init__(self, missionPath: str) -> None:
        seed = SEED if SEED is not None else random.randrange(2 ** 32)
        random.seed(seed)
        print(f"GA seed: {seed}")
        self.caseStudy = AerialistTest.from_yaml(missionPath)

    def generate(self, budget: int) -> List[TestCase]:
        if budget <= 0:
            print("budget must be > 0")
            return []
        print(f"GA: budget={budget} pop={POP_SIZE}")
        pop = [randomIndividual() for _ in range(POP_SIZE)]
        evaluated: List[Individual] = []
        simsUsed = 0
        gen = 0
        while simsUsed < budget and gen < MAX_GENERATIONS:
            for ind in pop:
                if ind.valid:
                    continue
                consumed = evaluate(ind, self.caseStudy)
                if ind.valid:
                    evaluated.append(ind)
                if consumed:
                    simsUsed += 1
                if simsUsed >= budget:
                    break
            bestFitness = min(i.fitness for i in pop)
            print(f"[gen {gen}] best={bestFitness:.2f} sims={simsUsed}/{budget}")
            if simsUsed >= budget:
                break
            pop = nextGeneration(pop)
            gen += 1
        evaluated.sort(key=lambda i: i.fitness)
        top = [i.testCase for i in evaluated[:TOP_K]]
        print(f"GA done. returning {len(top)} tests.")
        return top
