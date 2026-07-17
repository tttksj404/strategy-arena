import unittest

from tools.kra_drug_discovery_loop import SearchPolicy, should_continue


class KraDrugDiscoveryLoopTestCase(unittest.TestCase):
    def test_policy_rejects_narrow_cycles(self):
        with self.assertRaises(ValueError):
            SearchPolicy(candidates=19_999, generations=5, beam_width=64)
        with self.assertRaises(ValueError):
            SearchPolicy(candidates=20_000, generations=4, beam_width=64)

    def test_loop_only_stops_after_historical_parity_or_explicit_limit(self):
        searching = {"historical_market_parity_pass": False}
        parity = {"historical_market_parity_pass": True}

        self.assertTrue(should_continue(searching, completed_cycles=8, max_cycles=0))
        self.assertFalse(should_continue(parity, completed_cycles=1, max_cycles=0))
        self.assertFalse(should_continue(searching, completed_cycles=2, max_cycles=2))


if __name__ == "__main__":
    unittest.main()
