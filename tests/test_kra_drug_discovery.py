import unittest

import numpy as np

from kra_drug_discovery import (
    AssayScore,
    Genome,
    generate_wide_library,
    generate_hybrid_library,
    market_parity_pass,
    select_frontier,
)
from kra_drug_models import BASE_MODEL_NAMES


class KraDrugDiscoveryTestCase(unittest.TestCase):
    def test_base_library_contains_full_order_rankers(self):
        self.assertIn("full_order_plackett_luce_top3_r2", BASE_MODEL_NAMES)
        self.assertIn("full_order_full_order_pairwise_d3", BASE_MODEL_NAMES)

    def test_wide_library_covers_every_base_and_caps_market_influence(self):
        library = generate_wide_library(
            base_count=5,
            requested=200,
            seed=20260712,
            maximum_market_weight=0.15,
            parents=(),
        )

        self.assertGreaterEqual(len(library), 200)
        self.assertTrue(all(genome.market_weight <= 0.15 for genome in library))
        self.assertTrue(all(abs(sum(genome.weights) + genome.market_weight - 1.0) < 1e-9 for genome in library))
        for base_index in range(5):
            self.assertTrue(any(genome.weights[base_index] >= 0.999 for genome in library))

    def test_hybrid_library_covers_pair_boundaries_and_market_boundaries(self):
        library = generate_hybrid_library(
            base_count=3,
            requested=80,
            seed=20260712,
            maximum_market_weight=0.15,
            parents=(),
        )

        self.assertGreaterEqual(len(library), 80)
        self.assertTrue(any(
            abs(genome.weights[0] - 0.5) < 1e-9
            and abs(genome.weights[1] - 0.5) < 1e-9
            and genome.market_weight == 0.0
            for genome in library
        ))
        self.assertTrue(any(abs(genome.market_weight - 0.15) < 1e-9 for genome in library))

    def test_frontier_selection_never_uses_confirmation_score(self):
        stable = AssayScore(
            genome=Genome((1.0, 0.0), 0.0),
            discovery_market_gaps_pp=(-1.0, -1.0, -1.0),
            discovery_v4_lifts_pp=(6.0, 6.0, 6.0),
            confirmation_market_gap_pp=-20.0,
            confirmation_v4_lift_pp=-10.0,
        )
        confirmation_lucky = AssayScore(
            genome=Genome((0.0, 1.0), 0.0),
            discovery_market_gaps_pp=(-2.0, -2.0, -2.0),
            discovery_v4_lifts_pp=(7.0, 7.0, 7.0),
            confirmation_market_gap_pp=10.0,
            confirmation_v4_lift_pp=20.0,
        )

        selected = select_frontier((stable, confirmation_lucky), beam_width=1)

        self.assertEqual(selected, (stable,))

    def test_market_parity_requires_every_fold_and_five_point_market_advantage(self):
        valid = AssayScore(
            genome=Genome((1.0,), 0.0),
            discovery_market_gaps_pp=(np.float64(5.1), np.float64(6.0), np.float64(5.4)),
            discovery_v4_lifts_pp=(np.float64(6.0), np.float64(7.0), np.float64(5.5)),
            confirmation_market_gap_pp=np.float64(5.0),
            confirmation_v4_lift_pp=np.float64(6.0),
        )
        weak_market = AssayScore(
            genome=Genome((1.0,), 0.0),
            discovery_market_gaps_pp=(5.1, 6.0, 4.9),
            discovery_v4_lifts_pp=(6.0, 7.0, 5.5),
            confirmation_market_gap_pp=5.0,
            confirmation_v4_lift_pp=6.0,
        )

        self.assertTrue(market_parity_pass(valid))
        self.assertIsInstance(market_parity_pass(valid), bool)
        self.assertFalse(market_parity_pass(weak_market))


if __name__ == "__main__":
    unittest.main()
