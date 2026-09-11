import unittest
from types import SimpleNamespace

from geneticAlgorithm import suite
from geneticAlgorithm.tests.stubs import Obstacle, Position, Size


def trajectory(points):
    return SimpleNamespace(positions=[SimpleNamespace(x=x, y=y) for x, y in points])


def individual(score, xOffset, runs=1, stuck=False, valid=True, distance=0.8, frame=None, obstacles=1):
    """A test whose flight is a straight line at x = xOffset, plus a bump."""
    path = [(xOffset, 0.0), (xOffset + 3.0, 25.0), (xOffset, 50.0)]
    genes = None
    if frame is not None:
        genes = SimpleNamespace(segment=frame[0], side=frame[1])
    return SimpleNamespace(
        obstacles=[Obstacle(Size(20.0, 2.0, 25.0), Position(xOffset + k, 25.0, 0.0, 45.0)) for k in range(obstacles)],
        genes=genes,
        officialScore=score,
        distances=[distance] * runs,
        trajectories=[trajectory(path)] * runs,
        stuck=stuck,
        valid=valid,
    )


class ResampleTests(unittest.TestCase):
    def test_two_point_line_is_spaced_evenly(self):
        points = suite.resample([(0.0, 0.0), (0.0, 10.0)], n=5)
        self.assertEqual([round(y, 6) for _, y in points], [0.0, 2.5, 5.0, 7.5, 10.0])

    def test_polyline_keeps_its_end_points(self):
        points = suite.resample([(0.0, 0.0), (3.0, 4.0), (3.0, 14.0)], n=11)
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertAlmostEqual(points[-1][1], 14.0)
        # Total length 15 m, so sample 5 sits 7.5 m along: 2.5 m into the second segment.
        self.assertAlmostEqual(points[5][0], 3.0)
        self.assertAlmostEqual(points[5][1], 6.5)

    def test_degenerate_inputs(self):
        self.assertEqual(suite.resample([], n=4), [])
        self.assertEqual(suite.resample([(1.0, 1.0)], n=3), [(1.0, 1.0)] * 3)


class ThresholdTests(unittest.TestCase):
    def test_default_without_reruns_and_median_with_them(self):
        single = individual(5.0, 0.0)
        # Pinned to the literal, not to DEFAULT_THRESHOLD_M: comparing the
        # constant against itself would pass for any value it was given.
        self.assertEqual(suite.DEFAULT_THRESHOLD_M, 3.0)
        self.assertEqual(suite.calibrateThreshold([single]), 3.0)

        rerun = individual(5.0, 0.0, runs=2)
        rerun.trajectories = [trajectory([(0.0, 0.0), (0.0, 50.0)]), trajectory([(1.0, 0.0), (1.0, 50.0)])]
        self.assertAlmostEqual(suite.calibrateThreshold([single, rerun]), 1.0)


class SelectSuiteTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(topK=3, relaxSimilarity=True, maxPerFrame=0, keepBestGate=False)

    def test_orders_by_score_and_excludes_stuck_and_zero_point(self):
        candidates = [
            individual(2.0, 0.0),
            individual(9.0, 10.0, stuck=True),
            individual(0.0, 20.0),
            individual(5.0, 30.0),
            individual(3.0, 40.0, valid=False),
        ]
        chosen, threshold = suite.selectSuite(self.cfg, candidates, 3.0)
        self.assertEqual([ind.officialScore for ind in chosen], [5.0, 2.0])
        self.assertEqual(threshold, 3.0)

    def test_look_alike_flights_are_skipped_while_the_suite_can_be_filled(self):
        candidates = [individual(5.0, 0.0), individual(4.0, 0.5), individual(3.0, 10.0), individual(2.0, 20.0)]
        chosen, _ = suite.selectSuite(self.cfg, candidates, 3.0)
        self.assertEqual([ind.officialScore for ind in chosen], [5.0, 3.0, 2.0])

    def test_threshold_halves_until_the_suite_is_full(self):
        candidates = [individual(5.0, 0.0), individual(4.0, 0.5), individual(3.0, 1.0)]
        chosen, threshold = suite.selectSuite(self.cfg, candidates, 3.0)
        self.assertEqual(len(chosen), 3)
        self.assertLess(threshold, 0.5)

    def test_exact_duplicate_layouts_are_dropped(self):
        candidates = [individual(5.0, 0.0), individual(4.0, 0.0)]
        chosen, _ = suite.selectSuite(self.cfg, candidates, 0.0)
        self.assertEqual(len(chosen), 1)

    def test_at_most_top_k(self):
        candidates = [individual(float(k), 10.0 * k) for k in range(1, 8)]
        chosen, _ = suite.selectSuite(self.cfg, candidates, 3.0)
        self.assertEqual([ind.officialScore for ind in chosen], [7.0, 6.0, 5.0])

    def test_without_relaxation_the_suite_stays_short_and_distinct(self):
        candidates = [individual(5.0, 0.0), individual(4.0, 0.5), individual(3.0, 1.0)]
        strict = SimpleNamespace(topK=3, relaxSimilarity=False, maxPerFrame=0, keepBestGate=False)
        chosen, threshold = suite.selectSuite(strict, candidates, 3.0)
        self.assertEqual([ind.officialScore for ind in chosen], [5.0])
        self.assertEqual(threshold, 3.0)

    def test_frame_quota_limits_tests_per_segment_and_side(self):
        candidates = [individual(9.0 - k, 10.0 * k, frame=(1, 1)) for k in range(4)]
        candidates += [individual(1.0, 100.0, frame=(3, -1))]
        cfg = SimpleNamespace(topK=4, relaxSimilarity=True, maxPerFrame=2, keepBestGate=False)
        chosen, _ = suite.selectSuite(cfg, candidates, 3.0)
        self.assertEqual([suite.frameOf(ind) for ind in chosen], [(1, 1), (1, 1), (3, -1)])

    def test_quota_does_not_trigger_endless_relaxation(self):
        # Only two frames exist, so a quota of one caps the suite at two; the
        # threshold must not be halved to fill the remaining slots.
        candidates = [individual(9.0 - k, 10.0 * k, frame=(1, 1)) for k in range(4)]
        candidates += [individual(1.0, 100.0, frame=(3, -1))]
        cfg = SimpleNamespace(topK=4, relaxSimilarity=True, maxPerFrame=1, keepBestGate=False)
        chosen, threshold = suite.selectSuite(cfg, candidates, 3.0)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(threshold, 3.0)

    def test_keep_best_gate_takes_the_best_pair_before_the_singles(self):
        candidates = [individual(9.0, 0.0, frame=(1, 1)), individual(8.0, 20.0, frame=(3, 1)),
                      individual(1.0, 40.0, frame=(1, -1), obstacles=2)]
        cfg = SimpleNamespace(topK=2, relaxSimilarity=True, maxPerFrame=0, keepBestGate=True)
        chosen, _ = suite.selectSuite(cfg, candidates, 3.0)
        self.assertEqual([len(ind.obstacles) for ind in chosen], [1, 2])
        # The suite is still returned in official-score order.
        self.assertEqual([ind.officialScore for ind in chosen], [9.0, 1.0])

    def test_suite_diversity_is_the_mean_nearest_neighbour_distance(self):
        chosen = [individual(5.0, 0.0), individual(4.0, 10.0), individual(3.0, 30.0)]
        self.assertAlmostEqual(suite.suiteDiversity(chosen), (10.0 + 10.0 + 20.0) / 3.0)


if __name__ == "__main__":
    unittest.main()
