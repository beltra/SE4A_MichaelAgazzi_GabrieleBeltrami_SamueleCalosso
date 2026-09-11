import math
import os
import random
import unittest

from geneticAlgorithm.tests.stubs import installAerialistStubs

installAerialistStubs()

from geneticAlgorithm import genes as g  # noqa: E402
from geneticAlgorithm import geometry, mission, motifs  # noqa: E402

CASE_STUDIES = os.path.join(os.path.dirname(__file__), "..", "..", "case_studies")
BOUNDS = (-40.0, 30.0, 10.0, 40.0)


def missionSegments(number):
    return mission.allSegments(mission.parseWaypoints(os.path.join(CASE_STUDIES, "mission%d.plan" % number)))


class MakeWallTests(unittest.TestCase):
    def test_wall_starts_at_the_given_end(self):
        wall = motifs.makeWall((0.0, 0.0), 90.0, 20.0)
        p, q = geometry.wallAxis(wall)
        self.assertAlmostEqual(min(p[1], q[1]), 0.0)
        self.assertAlmostEqual(max(p[1], q[1]), 20.0)
        self.assertEqual(wall.size.h, motifs.WALL_HEIGHT_M)

    def test_yaw_beyond_90_swaps_length_and_width(self):
        wall = motifs.makeWall((0.0, 0.0), 120.0, 20.0)
        self.assertAlmostEqual(wall.position.r, 30.0)
        self.assertEqual((wall.size.l, wall.size.w), (motifs.WALL_THICKNESS_M, 20.0))


class DecodeWallTests(unittest.TestCase):
    def test_wall_crosses_its_own_segment_with_the_short_end_on_side_plus_one(self):
        segment = mission.Segment(1, (0.0, 0.0), (0.0, 50.0))
        genes = g.Genes(g.KIND_WALL, 1, 1, 0.5, 45.0, 20.0, 6.0)
        wall = motifs.decodeWall(genes, segment)
        p, q = geometry.wallAxis(wall)

        self.assertTrue(geometry.segmentsIntersect(p, q, segment.a, segment.b))
        plusEnd = max(p, q, key=lambda point: point[0])
        minusEnd = min(p, q, key=lambda point: point[0])
        # 6 m of the 20 m axis lie on the side = +1 side of the path (x > 0),
        # 14 m on the other.
        self.assertAlmostEqual(math.dist(plusEnd, (0.0, 25.0)), 6.0)
        self.assertAlmostEqual(math.dist(minusEnd, (0.0, 25.0)), 14.0)

    def test_decoded_walls_cross_their_segment_and_fit_on_every_mission(self):
        rng = random.Random(7)
        for number in range(1, 8):
            segments = missionSegments(number)
            longSegments = [segment for segment in segments if segment.length() >= mission.MIN_SEGMENT_LENGTH_M]
            fitted = 0
            for _ in range(100):
                genes = g.randomGenes(rng, g.KIND_WALL, [segment.index for segment in longSegments])
                segment = motifs.segmentByIndex(segments, genes.segment)
                wall = motifs.decodeWall(genes, segment)
                p, q = geometry.wallAxis(wall)
                self.assertTrue(geometry.segmentsIntersect(p, q, segment.a, segment.b), (number, genes))
                fitted += geometry.fitsInBounds(BOUNDS, [wall])
            self.assertGreater(fitted, 10, number)


class DecodeGateTests(unittest.TestCase):
    def test_reproduces_the_measured_mission3_chevron(self):
        # chevron_stable.yaml: wall centre (-1.751, 21.691), blocker centre
        # (18.165, 21.863), footprint gap 7.0 m, 4/4 runs in the 2-point tier.
        segments = missionSegments(3)
        segment = motifs.segmentByIndex(segments, 1)
        t = (25.0 - segment.a[1]) / (segment.b[1] - segment.a[1])
        genes = g.Genes(g.KIND_GATE, 1, 1, t, 45.0, 20.0, 5.0, gateWidth=7.0)

        wall, blocker = motifs.decodeGate(genes, segment)

        self.assertAlmostEqual(wall.position.x, -1.751, places=3)
        self.assertAlmostEqual(wall.position.y, 21.691, places=3)
        self.assertAlmostEqual(blocker.position.x, 18.165, places=3)
        self.assertAlmostEqual(blocker.position.y, 21.863, places=3)
        self.assertAlmostEqual(geometry.footprintGap(wall, blocker), 7.0, places=6)

    def test_gap_matches_the_gate_width_gene(self):
        rng = random.Random(8)
        segments = missionSegments(5)
        for _ in range(50):
            genes = g.randomGenes(rng, g.KIND_GATE, [1, 2, 3])
            wall, blocker = motifs.decode(genes, segments)
            self.assertAlmostEqual(geometry.footprintGap(wall, blocker), genes.gateWidth, places=4)


class SegmentCrossingTests(unittest.TestCase):
    def test_mirrored_gate_on_mission3_crosses_the_return_segment(self):
        segments = missionSegments(3)
        segment = motifs.segmentByIndex(segments, 1)
        t = (25.0 - segment.a[1]) / (segment.b[1] - segment.a[1])
        right = g.Genes(g.KIND_GATE, 1, 1, t, 45.0, 20.0, 5.0, gateWidth=7.0)
        left = g.Genes(g.KIND_GATE, 1, -1, t, 45.0, 20.0, 5.0, gateWidth=7.0)

        self.assertIsNone(motifs.layoutProblem(BOUNDS, motifs.decode(right, segments), segments, 1))
        self.assertEqual(motifs.layoutProblem(BOUNDS, motifs.decode(left, segments), segments, 1), "crosses another segment")
        self.assertIsNone(motifs.layoutProblem(BOUNDS, motifs.decode(left, segments), segments, 1, strict=False))

    def test_out_of_area_and_overlap_are_reported(self):
        far = motifs.makeWall((0.0, 60.0), 0.0, 20.0)
        a = motifs.makeWall((0.0, 20.0), 0.0, 20.0)
        b = motifs.makeWall((5.0, 20.5), 0.0, 20.0)
        self.assertEqual(motifs.layoutProblem(BOUNDS, [far], [], 1), "outside the placement area")
        self.assertEqual(motifs.layoutProblem(BOUNDS, [a, b], [], 1), "walls overlap")


if __name__ == "__main__":
    unittest.main()
