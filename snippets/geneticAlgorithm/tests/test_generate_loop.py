"""The whole generate() loop against a fake simulator: budget, elites, arms.

No Aerialist and no simulator: `evaluate` and `rerun` are replaced by
functions that invent a distance from the genes, so one campaign runs in
milliseconds. What is pinned here is the accounting the simulator cannot
check cheaply: the total number of attempts never exceeds the budget, the
seed-only arm never breeds, and elite confirmation spends its runs on the
elites and on nothing else.
"""

import os
import random
import unittest
from types import SimpleNamespace

from geneticAlgorithm.tests.stubs import installAerialistStubs

installAerialistStubs()

from geneticAlgorithm import generator as ga  # noqa: E402
from geneticAlgorithm import mission  # noqa: E402

CASE_STUDIES = os.path.join(os.path.dirname(__file__), "..", "..", "case_studies")


def trajectory(x):
    """A straight flight ending at the goal, so no run counts as stuck."""
    return SimpleNamespace(positions=[
        SimpleNamespace(timestamp=k * 1_000_000, x=x + 0.01 * k, y=float(k), z=10.0) for k in range(60)])


class FakeSimulator:
    """Deterministic stand-in for evaluate() and rerun(), counting attempts."""

    def __init__(self, goal):
        self.evaluations = 0
        self.reruns = 0
        self.rerunTargets = []
        self.goal = goal

    def distanceFor(self, ind):
        # A smooth function of the genes, so breeding can actually improve it.
        return abs(ind.genes.crossingDeg - 45.0) * 0.05 + abs(ind.genes.t - 0.5) + 0.4

    def record(self, cfg, ind):
        ind.distances.append(self.distanceFor(ind))
        ind.flightDurations.append(120.0)
        ind.trajectories.append(trajectory(ind.genes.t * 10.0))
        ind.officialScore = ga.officialTestScore(ind.distances, len(ind.obstacles), ind.flightDurations)
        ind.fitness = -ga.searchTestScore(ind.distances, len(ind.obstacles), ind.flightDurations,
                                          cfg.shapeGain, False, cfg.incompleteFactor)
        ind.valid = True
        ind.testCase = SimpleNamespace(test=None)

    def evaluate(self, cfg, ind, caseStudy, goalXY=None, killAfter=0.0):
        # Same guard as the real evaluate: an illegal layout costs no run.
        if ga.invalidLayout(cfg, ind.obstacles):
            return ga.AttemptOutcome.SKIPPED
        self.evaluations += 1
        self.record(cfg, ind)
        return ga.AttemptOutcome.CONSUMED

    def rerun(self, cfg, ind, caseStudy, goalXY=None, killAfter=0.0):
        self.reruns += 1
        self.rerunTargets.append(id(ind))
        self.record(cfg, ind)
        return ga.AttemptOutcome.CONSUMED


class GenerateLoopTests(unittest.TestCase):
    def runCampaign(self, budget=60, missionNumber=3, **overrides):
        waypoints = mission.parseWaypoints(os.path.join(CASE_STUDIES, "mission%d.plan" % missionNumber))
        generator = ga.GeneticGenerator.__new__(ga.GeneticGenerator)
        generator.config = ga.GAConfig(seed=4, parallelWorkers=1, **overrides)
        generator.rng = random.Random(4)
        generator.caseStudy = SimpleNamespace(simulation=SimpleNamespace(obstacles=[]))
        generator.waypoints = waypoints
        generator.segments = mission.allSegments(waypoints)
        generator.seedSegments = [s.index for s in mission.longSegments(waypoints)]
        fake = FakeSimulator(waypoints[-1])
        original = (ga.evaluate, ga.rerun)
        ga.evaluate, ga.rerun = fake.evaluate, fake.rerun
        try:
            generator.generate(budget)
        finally:
            ga.evaluate, ga.rerun = original
        return generator, fake

    def test_attempts_never_exceed_the_budget(self):
        for budget in (25, 60, 100):
            _, fake = self.runCampaign(budget)
            self.assertLessEqual(fake.evaluations + fake.reruns, budget, budget)

    def test_the_seed_only_arm_never_breeds(self):
        generator, fake = self.runCampaign(60, evolve=False)
        # Every layout is a fresh draw, so no two generations share an elite:
        # each generation costs the full population instead of popSize - 2.
        generations = {ind.generation for ind in generator.evaluated}
        perGeneration = [sum(1 for ind in generator.evaluated if ind.generation == g) for g in sorted(generations)]
        self.assertTrue(all(count == generator.config.popSize for count in perGeneration[:-1]), perGeneration)

    def test_elite_confirmation_reruns_only_elites_and_stays_in_budget(self):
        plain, plainSim = self.runCampaign(60)
        elite, eliteSim = self.runCampaign(60, rerunElites=True)
        self.assertLessEqual(eliteSim.evaluations + eliteSim.reruns, 60)
        # Two niches, so two extra runs per generation after the first.
        self.assertGreater(eliteSim.reruns, plainSim.reruns)
        confirmed = [ind for ind in elite.evaluated if len(ind.distances) > 1]
        self.assertTrue(confirmed)

    def test_a_gate_free_arm_returns_only_single_walls(self):
        generator, _ = self.runCampaign(60, gateShare=0.0)
        self.assertTrue(all(len(ind.obstacles) == 1 for ind in generator.evaluated))

    def test_a_niche_with_no_legal_layout_does_not_stall_the_search(self):
        # No side +1 gate fits mission 6, so every gate seed comes back
        # illegal. Those slots must not block the generation counter: the
        # budget goes to the walls instead of burning in empty rounds.
        generator, fake = self.runCampaign(40, missionNumber=6, seedDraws=5, seedSides=(1,))
        self.assertEqual(fake.evaluations + fake.reruns, 40)
        self.assertTrue(all(len(ind.obstacles) == 1 for ind in generator.evaluated))
        self.assertGreater(max(ind.generation for ind in generator.evaluated), 0)


if __name__ == "__main__":
    unittest.main()
