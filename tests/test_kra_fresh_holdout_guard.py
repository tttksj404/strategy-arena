import unittest

from tools.kra_fresh_holdout_guard import HoldoutEvidence, promotion_passes


class KraFreshHoldoutGuardTestCase(unittest.TestCase):
    def test_promotion_requires_five_point_lift_positive_net_and_confidence_bound(self):
        passing = HoldoutEvidence(top1_lift_pp=5.0, net_wins=3, ci95_low_pp=0.1)
        subthreshold = HoldoutEvidence(top1_lift_pp=4.99, net_wins=3, ci95_low_pp=0.1)
        losing = HoldoutEvidence(top1_lift_pp=5.1, net_wins=-2, ci95_low_pp=-2.0)
        uncertain = HoldoutEvidence(top1_lift_pp=5.1, net_wins=1, ci95_low_pp=-0.5)

        self.assertTrue(promotion_passes(passing))
        self.assertFalse(promotion_passes(subthreshold))
        self.assertFalse(promotion_passes(losing))
        self.assertFalse(promotion_passes(uncertain))


if __name__ == "__main__":
    unittest.main()
