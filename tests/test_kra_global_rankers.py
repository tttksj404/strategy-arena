import unittest

from kra_global_rankers import GlobalFamily, SPECS


class KraGlobalRankersTestCase(unittest.TestCase):
    def test_world_benchmark_registry_covers_distinct_model_families(self):
        families = {spec.family for spec in SPECS}

        self.assertIn(GlobalFamily.WINNER_SVM, families)
        self.assertIn(GlobalFamily.NEURAL_NETWORK, families)
        self.assertIn(GlobalFamily.RANDOM_FOREST, families)
        self.assertIn(GlobalFamily.CONDITIONAL_LOGIT, families)


if __name__ == "__main__":
    unittest.main()
