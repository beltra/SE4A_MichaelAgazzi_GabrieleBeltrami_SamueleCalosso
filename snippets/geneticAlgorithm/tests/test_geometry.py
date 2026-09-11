import math
import random
import unittest

from geneticAlgorithm import geometry
from geneticAlgorithm.tests.stubs import Obstacle, Position, Size

BOUNDS = (-40.0, 30.0, 10.0, 40.0)


def box(x, y, r, l=20.0, w=2.0):
    return Obstacle(Size(l, w, 25.0), Position(x, y, 0.0, r))


def insidePolygon(point, corners):
    """Return whether point is inside the convex polygon given by corners."""
    signs = [
        geometry.cross(corners[i], corners[(i + 1) % 4], point) for i in range(4)
    ]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


class SegmentTests(unittest.TestCase):
    def test_crossing_touching_and_missing_segments(self):
        self.assertTrue(geometry.segmentsIntersect((0, 0), (2, 2), (0, 2), (2, 0)))
        self.assertTrue(geometry.segmentsIntersect((0, 0), (2, 0), (1, 0), (1, 5)))
        self.assertFalse(geometry.segmentsIntersect((0, 0), (2, 0), (0, 1), (2, 1)))
        self.assertFalse(geometry.segmentsIntersect((0, 0), (1, 0), (2, 0), (3, 0)))

    def test_point_and_segment_distances(self):
        self.assertAlmostEqual(geometry.pointToSegmentDistance((1, 1), (0, 0), (2, 0)), 1.0)
        self.assertAlmostEqual(geometry.pointToSegmentDistance((5, 0), (0, 0), (2, 0)), 3.0)
        self.assertAlmostEqual(geometry.segmentDistance((0, 0), (2, 0), (0, 3), (2, 3)), 3.0)
        self.assertEqual(geometry.segmentDistance((0, 0), (2, 2), (0, 2), (2, 0)), 0.0)


class OverlapTests(unittest.TestCase):
    def test_separating_axis_agrees_with_corner_sampling(self):
        # Two rotated boxes overlap when a corner of one is inside the other or
        # an edge pair crosses; compare the SAT answer against that check.
        rng = random.Random(3)
        for _ in range(500):
            a = box(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 90), rng.uniform(2, 12), rng.uniform(2, 12))
            b = box(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 90), rng.uniform(2, 12), rng.uniform(2, 12))
            cornersA, cornersB = geometry.boxCorners(a), geometry.boxCorners(b)
            expected = (
                any(insidePolygon(p, cornersB) for p in cornersA)
                or any(insidePolygon(p, cornersA) for p in cornersB)
                or any(
                    geometry.segmentsIntersect(cornersA[i], cornersA[(i + 1) % 4], cornersB[j], cornersB[(j + 1) % 4])
                    for i in range(4) for j in range(4)
                )
            )
            self.assertEqual(geometry.overlaps(a, b), expected, (a, b))

    def test_thin_oblique_walls_close_together_do_not_overlap(self):
        # The old rotated-AABB check rejected this pair; the exact test accepts it.
        a = box(0.0, 20.0, 45.0)
        b = box(6.0, 20.0, 45.0)
        self.assertFalse(geometry.hasOverlap([a, b]))
        self.assertGreater(geometry.footprintGap(a, b), 2.0)

    def test_footprint_gap_of_parallel_walls(self):
        a = box(0.0, 20.0, 0.0)
        b = box(0.0, 25.0, 0.0)
        self.assertAlmostEqual(geometry.footprintGap(a, b), 3.0)
        self.assertEqual(geometry.footprintGap(a, box(0.0, 21.0, 0.0)), 0.0)


class BoundsAndAxisTests(unittest.TestCase):
    def test_fits_in_bounds_uses_the_rotated_footprint(self):
        self.assertTrue(geometry.fitsInBounds(BOUNDS, [box(-5.0, 25.0, 45.0)]))
        self.assertFalse(geometry.fitsInBounds(BOUNDS, [box(-5.0, 34.0, 45.0)]))
        self.assertTrue(geometry.fitsInBounds(BOUNDS, [box(-5.0, 34.0, 0.0)]))

    def test_wall_axis_is_the_same_after_the_l_w_swap(self):
        # A 20 x 2 wall at 120 deg is stored as a 2 x 20 wall at 30 deg.
        stored = box(0.0, 20.0, 30.0, l=2.0, w=20.0)
        p, q = geometry.wallAxis(stored)
        self.assertAlmostEqual(math.dist(p, q), 20.0)
        angle = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180.0
        self.assertAlmostEqual(angle, 120.0)


if __name__ == "__main__":
    unittest.main()
