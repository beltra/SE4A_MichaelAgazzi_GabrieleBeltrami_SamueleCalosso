import copy
import os
import random
import unittest
from dataclasses import replace

from geneticAlgorithm.tests.stubs import installAerialistStubs

installAerialistStubs()

from geneticAlgorithm import generator as ga  # noqa: E402
from geneticAlgorithm import mission  # noqa: E402

CASE_STUDIES = os.path.join(os.path.dirname(__file__), "..", "..", "case_studies")


def missionSegments(number):
    waypoints = mission.parseWaypoints(os.path.join(CASE_STUDIES, "mission%d.plan" % number))
    return mission.allSegments(waypoints), [segment.index for segment in mission.longSegments(waypoints)]


def evaluatedPopulation(rng, cfg, segments, segmentChoices):
    """A generation-0 population with made-up fitness, as if it had been run."""
    pop = [ga.freshIndividual(rng, cfg, segments, segmentChoices, n) for n in ga.obstacleCountSchedule(cfg.popSize)]
    for ind in pop:
        ind.valid = True
        ind.distances = [rng.uniform(0.3, 3.0)]
        ind.flightDurations = [40.0]
        ind.fitness = -ga.searchTestScore(ind.distances, len(ind.obstacles), ind.flightDurations, 0.25, False, 0.25)
        ind.officialScore = ga.officialTestScore(ind.distances, len(ind.obstacles), ind.flightDurations)
    return pop


class SeedingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ga.GAConfig(seed=1, popSize=12)
        self.rng = random.Random(1)

    def test_seeds_are_legal_on_every_mission(self):
        for number in range(1, 8):
            segments, segmentChoices = missionSegments(number)
            for n in (1, 2):
                for _ in range(10):
                    ind = ga.freshIndividual(self.rng, self.cfg, segments, segmentChoices, n)
                    if ind.illegal:
                        # The one niche with nothing legal: side +1 gates on mission 6.
                        self.assertEqual((number, n), (6, 2))
                        continue
                    self.assertEqual(len(ind.obstacles), n)
                    self.assertIn(ind.genes.segment, segmentChoices)
                    self.assertFalse(ga.invalidLayout(self.cfg, ind.obstacles), (number, ind.genes))

    def test_no_side_plus_one_gate_fits_mission_6(self):
        # A geometric property of the mission, not of the shipped default:
        # restricted to side +1 there is no legal gate on mission 6 at all.
        # Pinned so a change of GENE_RANGES or of the placement rules that
        # makes such a gate possible again is noticed.
        segments, segmentChoices = missionSegments(6)
        cfg = replace(self.cfg, seedDraws=50, seedSides=(1,))
        ind = ga.freshIndividual(self.rng, cfg, segments, segmentChoices, 2)
        self.assertTrue(ind.illegal)
        self.assertEqual(len(ind.obstacles), 2)

    def test_children_keep_their_niche_and_decode_legally(self):
        segments, segmentChoices = missionSegments(3)
        pop = evaluatedPopulation(self.rng, self.cfg, segments, segmentChoices)
        for _ in range(50):
            for niche in (1, 2):
                child = ga.nicheChild(self.rng, self.cfg, pop, niche, segments, segmentChoices)
                self.assertEqual(len(child.obstacles), niche)
                self.assertIsNotNone(child.genes)
                self.assertFalse(child.valid)
                self.assertFalse(ga.invalidLayout(self.cfg, child.obstacles))

    def test_next_generation_keeps_the_quota_and_one_elite_per_niche(self):
        segments, segmentChoices = missionSegments(3)
        pop = evaluatedPopulation(self.rng, self.cfg, segments, segmentChoices)
        nextPop = ga.nextGeneration(self.rng, self.cfg, pop, segments, segmentChoices)

        self.assertEqual([len(ind.obstacles) for ind in nextPop], ga.obstacleCountSchedule(12))
        elites = [ind for ind in nextPop if ind.valid]
        self.assertEqual(sorted(len(ind.obstacles) for ind in elites), [1, 2])
        for niche in (1, 2):
            best = min((ind for ind in pop if len(ind.obstacles) == niche), key=ga.searchSelectionKey)
            self.assertIn(best, elites)


class InvariantTests(unittest.TestCase):
    """Properties the search relies on but no single function owns."""

    def test_elites_are_not_modified_when_the_next_generation_is_built(self):
        cfg = ga.GAConfig(seed=3, popSize=12)
        rng = random.Random(3)
        segments, segmentChoices = missionSegments(3)
        pop = evaluatedPopulation(rng, cfg, segments, segmentChoices)
        elites = [min((ind for ind in pop if len(ind.obstacles) == niche), key=ga.searchSelectionKey) for niche in (1, 2)]
        before = [(copy.copy(e.genes), list(e.distances), e.officialScore, [(o.position.x, o.position.y, o.position.r) for o in e.obstacles]) for e in elites]
        for _ in range(3):
            pop = ga.nextGeneration(rng, cfg, pop, segments, segmentChoices)
        # The elite objects are carried over by identity, so any in-place edit
        # by crossover or mutation would show up here.
        for elite, (genes, distances, score, boxes) in zip(elites, before):
            self.assertIn(elite, pop)
            self.assertEqual(elite.genes, genes)
            self.assertEqual(elite.distances, distances)
            self.assertEqual(elite.officialScore, score)
            self.assertEqual([(o.position.x, o.position.y, o.position.r) for o in elite.obstacles], boxes)

    def test_same_seed_produces_the_same_children(self):
        cfg = ga.GAConfig(seed=11, popSize=12)
        segments, segmentChoices = missionSegments(7)
        runs = []
        for _ in range(2):
            rng = random.Random(cfg.seed)
            pop = evaluatedPopulation(rng, cfg, segments, segmentChoices)
            for _ in range(2):
                pop = ga.nextGeneration(rng, cfg, pop, segments, segmentChoices)
            runs.append([ind.genes for ind in pop])
        # Only the private rng may be used; a stray call into the global random
        # module or an unordered iteration would make the two runs diverge.
        self.assertEqual(runs[0], runs[1])

    def test_shaping_never_lifts_a_worse_tier_above_a_better_one(self):
        gain = ga.GAConfig().shapeGain
        # The literal is pinned too: read from the config alone, this test
        # would follow any change made to the constant it is guarding.
        self.assertEqual(gain, 0.25)
        self.assertLess(gain, 1.0)
        # The worst point of a tier (just inside its far edge) must still beat
        # the best point of the next tier (right on the boundary).
        for bound in ga.TIER_BOUNDS:
            self.assertGreater(ga.shapedPoints(bound - 1e-6, gain), ga.shapedPoints(bound, gain))


class ScheduleTests(unittest.TestCase):
    def test_default_share_is_fourteen_walls_and_six_gates(self):
        counts = ga.obstacleCountSchedule(20)
        self.assertEqual(len(counts), 20)
        self.assertEqual(counts.count(2), 6)
        self.assertEqual(counts.count(1), 14)

    def test_gate_share_sets_the_quota_and_spreads_the_gates(self):
        self.assertEqual(ga.obstacleCountSchedule(20, 0.0).count(2), 0)
        self.assertEqual(ga.obstacleCountSchedule(20, 0.5).count(2), 10)
        self.assertEqual(ga.obstacleCountSchedule(12, 0.3).count(2), 4)
        # No two gates in a row at the default share, so a partial generation
        # still contains both niches.
        counts = ga.obstacleCountSchedule(20, 0.3)
        self.assertNotIn((2, 2), list(zip(counts, counts[1:])))

    def test_immigrants_replace_children_but_never_the_elite(self):
        cfg = ga.GAConfig(seed=5, popSize=12, immigrantShare=0.25)
        rng = random.Random(5)
        segments, segmentChoices = missionSegments(3)
        pop = evaluatedPopulation(rng, cfg, segments, segmentChoices)
        quota = ga.obstacleCountSchedule(cfg.popSize, cfg.gateShare).count(1)
        niche = ga.nicheGeneration(rng, cfg, pop, 1, quota, segments, segmentChoices)
        self.assertEqual(len(niche), quota)
        self.assertEqual(sum(1 for ind in niche if ind.valid), 1)
        # Default share: every non-elite slot is a child, as before.
        plain = ga.GAConfig(seed=5, popSize=12)
        self.assertEqual(plain.immigrantShare, 0.0)

    def test_rescore_uses_every_run_of_an_individual(self):
        cfg = ga.GAConfig()
        generator = ga.GeneticGenerator.__new__(ga.GeneticGenerator)
        ind = ga.Individual(obstacles=[object()], distances=[0.5, 1.2], flightDurations=[60.0, 60.0])
        generator.rescore(cfg, ind)
        # One run at 2 points and one at 1 point, one obstacle, one minute each.
        self.assertAlmostEqual(ind.officialScore, 15.0)
        self.assertLess(ind.fitness, 0.0)

    def test_mutation_scale_anneals_down_to_a_floor(self):
        cfg = ga.GAConfig(mutationScale=2.0, mutationDecay=0.5)
        self.assertEqual(ga.mutationScaleAt(cfg, 0), 2.0)
        self.assertEqual(ga.mutationScaleAt(cfg, 1), 2.0)
        self.assertEqual(ga.mutationScaleAt(cfg, 2), 1.0)
        self.assertEqual(ga.mutationScaleAt(cfg, 9), 0.25)
        flat = ga.GAConfig()
        self.assertEqual([ga.mutationScaleAt(flat, g) for g in range(5)], [1.0] * 5)


class ScoringTests(unittest.TestCase):
    def test_tier_points_at_the_boundaries(self):
        self.assertEqual([ga.tierPoints(d) for d in (0.0, 0.24, 0.25, 0.99, 1.0, 1.49, 1.5, 9.0)], [5, 5, 2, 2, 1, 1, 0, 0])

    def test_official_score_formula(self):
        # Two runs at 2 points, one obstacle, 30 s flights: 2 * 10 / (1 * 0.5 min).
        self.assertAlmostEqual(ga.officialTestScore([0.5, 0.9], 1, [30.0, 30.0]), 40.0)
        self.assertAlmostEqual(ga.officialTestScore([0.5, 0.9], 2, [30.0, 30.0]), 10.0)


if __name__ == "__main__":
    unittest.main()
