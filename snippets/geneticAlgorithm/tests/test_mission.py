import math
import os
import unittest

from geneticAlgorithm import mission

CASE_STUDIES = os.path.join(os.path.dirname(__file__), "..", "..", "case_studies")


class ParseWaypointsTests(unittest.TestCase):
    def test_mission3_has_the_known_outbound_segment(self):
        waypoints = mission.parseWaypoints(os.path.join(CASE_STUDIES, "mission3.plan"))

        self.assertEqual(waypoints[0], (0.0, 0.0))
        self.assertEqual(len(waypoints), 5)
        self.assertAlmostEqual(waypoints[2][0], 3.75, places=2)
        self.assertAlmostEqual(waypoints[2][1], 53.02, places=2)

    def test_the_local_frame_is_north_east(self):
        # x is built from latitude and y from longitude, so +x is north and
        # +y is east. Mission 1 flies about 53 m east and barely moves north.
        # Every left/right statement in the project depends on this, and
        # nothing else in the suite would notice if the two were swapped.
        waypoints = mission.parseWaypoints(os.path.join(CASE_STUDIES, "mission1.plan"))
        north, east = waypoints[-1]

        self.assertLess(abs(north), 10.0)
        self.assertGreater(east, 45.0)

    def test_missing_plan_raises(self):
        with self.assertRaises(ValueError):
            mission.parseWaypoints(os.path.join(CASE_STUDIES, "no_such.plan"))


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.segment = mission.Segment(1, (0.0, 0.0), (0.0, 40.0))

    def test_direction_and_side_normal_are_perpendicular_units(self):
        ux, uy = self.segment.direction()
        nx, ny = self.segment.sideNormal()

        self.assertAlmostEqual(math.hypot(ux, uy), 1.0)
        self.assertAlmostEqual(math.hypot(nx, ny), 1.0)
        self.assertAlmostEqual(ux * nx + uy * ny, 0.0)

    def test_side_normal_is_the_drones_left(self):
        # The segment of setUp runs towards +y, which parseWaypoints builds
        # from longitude, so the drone is flying east and its left is north
        # (+x). Pinning the sign here stops the convention being renamed by
        # accident: perpendicularity alone is invariant under a sign flip.
        nx, ny = self.segment.sideNormal()

        self.assertAlmostEqual(nx, 1.0)
        self.assertAlmostEqual(ny, 0.0)

    def test_point_at_interpolates(self):
        self.assertEqual(self.segment.pointAt(0.25), (0.0, 10.0))
        self.assertAlmostEqual(self.segment.length(), 40.0)


class SegmentSelectionTests(unittest.TestCase):
    def test_long_segments_drop_takeoff_hops_and_corner_stubs(self):
        for number, expected in ((1, [1]), (3, [1, 3]), (4, [1, 3]), (5, [1, 2, 3]), (6, [2, 3, 4, 5]), (7, [1, 3, 5])):
            waypoints = mission.parseWaypoints(os.path.join(CASE_STUDIES, "mission%d.plan" % number))
            self.assertEqual([segment.index for segment in mission.longSegments(waypoints)], expected, number)

    def test_all_segments_skips_zero_length_segments(self):
        waypoints = [(0.0, 0.0), (0.0, 0.0), (0.0, 30.0)]
        self.assertEqual([segment.index for segment in mission.allSegments(waypoints)], [1])


if __name__ == "__main__":
    unittest.main()
