"""How a finished flight is read: duration, goal arrival, and flake runs.

These three functions turn a raw trajectory into the numbers everything else
is built on. flightSeconds is the denominator of the official score, and
reachedGoal / isFlakeRun decide whether a run counts as a measurement at all,
so a silent unit or sign error here rescales every result a campaign reports
without any formula being wrong.
"""

import unittest
from types import SimpleNamespace

from geneticAlgorithm.tests.stubs import installAerialistStubs

installAerialistStubs()

from geneticAlgorithm import generator as ga  # noqa: E402

GOAL = (10.0, 50.0)


def trajectory(points, startUs=0, stepUs=1_000_000):
    """A trajectory over the given (x, y), timestamped in microseconds.

    PX4 logs microseconds; the default step of one million is one second per
    sample, so a caller can read the expected duration off the point count.
    """
    return SimpleNamespace(positions=[
        SimpleNamespace(timestamp=startUs + k * stepUs, x=x, y=y, z=10.0)
        for k, (x, y) in enumerate(points)
    ])


def endingAt(x, y):
    return trajectory([(0.0, 0.0), (x, y)])


class FlightSecondsTests(unittest.TestCase):
    def test_timestamps_are_read_as_microseconds(self):
        # 90 s of flight, so a wrong divisor shows up as 90000 or 0.09.
        flight = trajectory([(0.0, 0.0)] * 4, startUs=5_000_000, stepUs=30_000_000)
        self.assertAlmostEqual(ga.flightSeconds(flight), 90.0)

    def test_duration_spans_first_to_last_sample_not_the_sample_count(self):
        uneven = SimpleNamespace(positions=[
            SimpleNamespace(timestamp=t, x=0.0, y=0.0, z=0.0)
            for t in (1_000_000, 1_500_000, 61_000_000)
        ])
        self.assertAlmostEqual(ga.flightSeconds(uneven), 60.0)

    def test_a_missing_or_empty_trajectory_is_zero_seconds(self):
        # A killed worker yields no samples; the score must not divide by junk.
        self.assertEqual(ga.flightSeconds(None), 0.0)
        self.assertEqual(ga.flightSeconds(SimpleNamespace(positions=[])), 0.0)
        self.assertEqual(ga.flightSeconds(SimpleNamespace()), 0.0)

    def test_a_single_sample_lasts_no_time(self):
        self.assertEqual(ga.flightSeconds(trajectory([(0.0, 0.0)])), 0.0)


class ReachedGoalTests(unittest.TestCase):
    def test_a_flight_ending_on_the_goal_arrived(self):
        self.assertTrue(ga.reachedGoal(endingAt(*GOAL), GOAL, 5.0))

    def test_only_the_last_sample_counts(self):
        # Passing over the goal and drifting away is not an arrival.
        passedBy = trajectory([(0.0, 0.0), GOAL, (10.0, 90.0)])
        self.assertFalse(ga.reachedGoal(passedBy, GOAL, 5.0))

    def test_the_tolerance_is_inclusive_and_measured_in_metres(self):
        self.assertTrue(ga.reachedGoal(endingAt(13.0, 54.0), GOAL, 5.0))    # 5.0 m away
        self.assertFalse(ga.reachedGoal(endingAt(13.1, 54.0), GOAL, 5.0))   # just outside
        self.assertFalse(ga.reachedGoal(endingAt(13.0, 54.0), GOAL, 4.9))

    def test_altitude_is_ignored(self):
        high = trajectory([(0.0, 0.0), GOAL])
        high.positions[-1].z = 400.0
        self.assertTrue(ga.reachedGoal(high, GOAL, 5.0))

    def test_a_missing_or_empty_trajectory_did_not_arrive(self):
        self.assertFalse(ga.reachedGoal(None, GOAL, 5.0))
        self.assertFalse(ga.reachedGoal(SimpleNamespace(positions=[]), GOAL, 5.0))
        self.assertFalse(ga.reachedGoal(SimpleNamespace(), GOAL, 5.0))


class FlakeRunTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ga.GAConfig()
        self.arrived = endingAt(*GOAL)
        self.stopped = endingAt(0.0, 20.0)

    def flake(self, trajectoryValue, goalXY, distances):
        return ga.isFlakeRun(self.cfg, trajectoryValue, goalXY, distances, "")

    def test_stopping_short_with_every_obstacle_far_away_is_a_flake(self):
        # Nothing was near enough to have caused it, so the run measures the
        # simulator, not the layout.
        with self.assertLogs(ga.logger, "WARNING"):
            self.assertTrue(self.flake(self.stopped, GOAL, [30.0, 41.0]))

    def test_stopping_short_next_to_an_obstacle_is_a_real_measurement(self):
        self.assertFalse(self.flake(self.stopped, GOAL, [0.4]))

    def test_a_completed_flight_is_never_a_flake(self):
        self.assertFalse(self.flake(self.arrived, GOAL, [30.0]))

    def test_the_obstacle_distance_bound_is_inclusive(self):
        bound = self.cfg.flakeObstacleDistanceM
        self.assertFalse(self.flake(self.stopped, GOAL, [bound]))
        with self.assertLogs(ga.logger, "WARNING"):
            self.assertTrue(self.flake(self.stopped, GOAL, [bound + 0.01]))

    def test_without_a_goal_nothing_can_be_judged_a_flake(self):
        # Missions the plan could not be parsed for run without a goal check.
        self.assertFalse(self.flake(self.stopped, None, [30.0]))

    def test_a_run_that_measured_no_obstacle_is_not_a_flake(self):
        # No distances means the log gave nothing to judge, which is a failure
        # to record rather than a flight the simulator spoiled.
        self.assertFalse(self.flake(self.stopped, GOAL, []))


if __name__ == "__main__":
    unittest.main()
