import unittest

from kra_max_winrate_models import ModelFamily, SPECS
from tools.kra_max_winrate_search import CandidateEvidence, select_candidate


class KraMaxWinrateSearchTestCase(unittest.TestCase):
    def test_selection_uses_only_prespecified_discovery_folds(self):
        stable = CandidateEvidence(
            name="stable",
            weight=1.0,
            top1_lifts_pp=(2.0, 2.0, 2.0, -10.0),
            top3_lifts_pp=(1.0, 1.0, 1.0, -10.0),
            logloss_deltas=(-0.1, -0.1, -0.1, 1.0),
        )
        confirmation_lucky = CandidateEvidence(
            name="lucky",
            weight=1.0,
            top1_lifts_pp=(1.0, 1.0, 1.0, 20.0),
            top3_lifts_pp=(1.0, 1.0, 1.0, 20.0),
            logloss_deltas=(-0.1, -0.1, -0.1, -1.0),
        )

        selected = select_candidate((stable, confirmation_lucky))

        self.assertEqual(selected.name, "stable")

    def test_pairwise_ranker_is_in_the_registered_search_space(self):
        families = {spec.family for spec in SPECS}

        self.assertIn(ModelFamily.PAIRWISE_HGB, families)
