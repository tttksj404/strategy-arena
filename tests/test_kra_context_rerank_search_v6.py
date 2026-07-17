import unittest

from tools.kra_context_rerank_search_v6 import CandidateScore, select_candidate


class KraContextRerankSearchV6TestCase(unittest.TestCase):
    def test_select_candidate_breaks_equal_minimum_net_gain_by_mean_lift(self):
        baseline_pairwise = CandidateScore(
            context_weight=0.0,
            rerank_weight=0.5,
            fold_lifts_pp=(0.328, 0.725, 0.329, 0.841),
            fold_net_wins=(4, 9, 4, 10),
        )
        context_ensemble = CandidateScore(
            context_weight=0.6,
            rerank_weight=1.0,
            fold_lifts_pp=(0.820, 0.322, 0.411, 2.439),
            fold_net_wins=(10, 4, 5, 29),
        )

        selected = select_candidate((baseline_pairwise, context_ensemble))

        self.assertEqual(selected, context_ensemble)


if __name__ == "__main__":
    unittest.main()
