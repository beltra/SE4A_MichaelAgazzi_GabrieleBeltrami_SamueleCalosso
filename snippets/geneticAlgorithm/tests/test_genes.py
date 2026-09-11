import dataclasses
import random
import unittest

from geneticAlgorithm import genes as g


def inRange(genes):
    for name, (lo, hi) in g.GENE_RANGES[genes.kind].items():
        value = getattr(genes, name)
        if not lo <= value <= hi:
            return False
    return True


class RandomGenesTests(unittest.TestCase):
    def test_values_are_inside_their_ranges(self):
        rng = random.Random(1)
        for _ in range(200):
            for kind in (g.KIND_WALL, g.KIND_GATE):
                genes = g.randomGenes(rng, kind, [1, 3])
                self.assertTrue(inRange(genes), genes)
                self.assertIn(genes.segment, (1, 3))
                self.assertIn(genes.side, (-1, 1))

    def test_seed_sides_restrict_the_drawn_side(self):
        rng = random.Random(1)
        sides = {g.randomGenes(rng, g.KIND_WALL, [1], (1,)).side for _ in range(50)}
        self.assertEqual(sides, {1})


class MutationScaleTests(unittest.TestCase):
    def test_a_larger_scale_takes_larger_steps(self):
        moves = {}
        for scale in (0.5, 4.0):
            rng = random.Random(7)
            parent = g.Genes(g.KIND_WALL, 1, 1, 0.5, 45.0, 18.0, 8.0)
            steps = []
            for _ in range(300):
                child = g.mutateGenes(rng, parent, scale)
                name = next(n for n in g.mutableGenes(parent.kind) if getattr(child, n) != getattr(parent, n))
                steps.append(abs(getattr(child, name) - getattr(parent, name)) / g.GENE_SIGMAS[name])
            moves[scale] = sum(steps) / len(steps)
        # The mean step is proportional to the scale, up to clamping at the
        # range ends, which only ever shortens a step.
        self.assertGreater(moves[4.0], 3 * moves[0.5])


class MutationTests(unittest.TestCase):
    def test_exactly_one_gene_changes_and_stays_in_range(self):
        rng = random.Random(2)
        parent = g.randomGenes(rng, g.KIND_GATE, [1])
        changed = 0
        for _ in range(1000):
            child = g.mutateGenes(rng, parent)
            differing = [
                f.name for f in dataclasses.fields(g.Genes)
                if getattr(child, f.name) != getattr(parent, f.name)
            ]
            self.assertLessEqual(len(differing), 1, differing)
            self.assertTrue(inRange(child), child)
            changed += len(differing)
        self.assertGreaterEqual(changed, 950)

    def test_mutation_does_not_touch_the_parent(self):
        rng = random.Random(3)
        parent = g.randomGenes(rng, g.KIND_WALL, [1])
        before = dataclasses.asdict(parent)
        for _ in range(50):
            g.mutateGenes(rng, parent)
        self.assertEqual(dataclasses.asdict(parent), before)


class CrossoverTests(unittest.TestCase):
    def test_parents_on_different_segments_are_copied_whole(self):
        rng = random.Random(4)
        a = g.randomGenes(rng, g.KIND_WALL, [1])
        b = g.randomGenes(rng, g.KIND_WALL, [3])
        for _ in range(20):
            child = g.crossoverGenes(rng, a, b)
            self.assertIn(dataclasses.asdict(child), (dataclasses.asdict(a), dataclasses.asdict(b)))

    def test_same_frame_parents_mix_genes(self):
        rng = random.Random(5)
        a = g.Genes(g.KIND_WALL, 1, 1, 0.3, 40.0, 18.0, 6.0)
        b = g.Genes(g.KIND_WALL, 1, 1, 0.7, 50.0, 20.0, 9.0)
        mixed = False
        for _ in range(50):
            child = g.crossoverGenes(rng, a, b)
            for name in g.mutableGenes(g.KIND_WALL):
                self.assertIn(getattr(child, name), (getattr(a, name), getattr(b, name)))
            if child.t == a.t and child.crossingDeg == b.crossingDeg:
                mixed = True
        self.assertTrue(mixed)


if __name__ == "__main__":
    unittest.main()
