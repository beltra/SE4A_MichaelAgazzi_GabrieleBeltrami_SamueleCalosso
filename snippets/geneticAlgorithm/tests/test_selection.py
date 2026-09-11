"""Selection pressure: the tournament must actually prefer better parents.

nextGeneration already pins the quota and the elites, but nothing there fails
if the tournament stops ranking its contenders. A parent picked uniformly at
random still fills every slot and keeps every niche, so the search would go on
looking healthy while breeding at random.
"""

import random
import unittest

from geneticAlgorithm.tests.stubs import installAerialistStubs

installAerialistStubs()

from geneticAlgorithm import generator as ga  # noqa: E402


def individual(fitness, obstacles=1, distance=0.8, duration=120.0):
    """A scored individual; lower fitness is better, as the GA minimises it."""
    return ga.Individual(
        obstacles=[object()] * obstacles,
        fitness=fitness,
        distances=[distance],
        flightDurations=[duration],
    )


class TournamentTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(1)
        # Fitness -5 is the best of the six, 0 the worst.
        self.pop = [individual(-float(k)) for k in range(6)]
        self.best, self.worst = self.pop[5], self.pop[0]

    def test_a_tournament_covering_the_whole_pool_always_returns_the_best(self):
        cfg = ga.GAConfig(tournamentK=len(self.pop))
        for _ in range(50):
            self.assertIs(ga.tournament(self.rng, cfg, self.pop), self.best)

    def test_the_worst_never_wins_and_the_best_wins_about_half_the_time(self):
        # With k=3 drawn from 6, the best wins exactly when it is sampled:
        # C(5,2)/C(6,3) = 0.5. The worst can never win a sample it is in.
        cfg = ga.GAConfig(tournamentK=3)
        wins = [0] * len(self.pop)
        for _ in range(2000):
            wins[self.pop.index(ga.tournament(self.rng, cfg, self.pop))] += 1
        self.assertEqual(wins[0], 0)
        self.assertGreater(wins[5] / 2000.0, 0.4)
        # Monotone pressure: a better individual is never picked less often.
        self.assertEqual(wins, sorted(wins))

    def test_a_tournament_of_one_applies_no_pressure(self):
        cfg = ga.GAConfig(tournamentK=1)
        winners = {id(ga.tournament(self.rng, cfg, self.pop)) for _ in range(200)}
        self.assertEqual(len(winners), len(self.pop))

    def test_equal_fitness_is_broken_by_the_smaller_layout_then_the_distance(self):
        cfg = ga.GAConfig(tournamentK=4)
        pair = individual(-3.0, obstacles=2, distance=0.1)
        single = individual(-3.0, obstacles=1, distance=0.9)
        closer = individual(-3.0, obstacles=1, distance=0.2)
        for _ in range(20):
            self.assertIs(ga.tournament(self.rng, cfg, [pair, single]), single)
            self.assertIs(ga.tournament(self.rng, cfg, [single, closer]), closer)


class TournamentNicheTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(2)
        self.cfg = ga.GAConfig(tournamentK=3)
        # The pairs are all worse than the singles, so a tournament that ignored
        # the niche would never return one of them.
        self.singles = [individual(-9.0 - k, obstacles=1) for k in range(4)]
        self.pairs = [individual(-1.0 - k, obstacles=2) for k in range(4)]
        self.pop = self.singles + self.pairs

    def test_sampling_stays_inside_the_requested_niche(self):
        for niche in (1, 2):
            for _ in range(100):
                parent = ga.tournament(self.rng, self.cfg, self.pop, niche)
                self.assertEqual(len(parent.obstacles), niche)

    def test_an_absent_niche_falls_back_to_the_whole_population(self):
        # A generation may hold no pair yet; breeding must not crash on it.
        parent = ga.tournament(self.rng, self.cfg, self.pop, 3)
        self.assertIn(parent, self.pop)

    def test_a_pool_smaller_than_the_tournament_is_not_oversampled(self):
        cfg = ga.GAConfig(tournamentK=10)
        self.assertIs(ga.tournament(self.rng, cfg, self.pop, 2), self.pairs[3])


if __name__ == "__main__":
    unittest.main()
