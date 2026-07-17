from tools.kra_market_pairwise_search import Evidence, _forward_candidate, _select


def test_selection_ignores_confirmation_fold() -> None:
    stable = Evidence(
        top_k=2,
        balance_power=0.5,
        threshold=0.4,
        top1_lifts_pp=(0.2, 0.2, 0.2, -10.0),
        switch_rates=(0.1, 0.1, 0.1, 1.0),
    )
    confirmation_lucky = Evidence(
        top_k=3,
        balance_power=0.5,
        threshold=0.2,
        top1_lifts_pp=(0.1, 0.1, 0.1, 10.0),
        switch_rates=(0.1, 0.1, 0.1, 0.0),
    )

    assert _select((stable, confirmation_lucky)) == stable
    assert _forward_candidate((stable, confirmation_lucky)) == confirmation_lucky


def test_forward_candidate_is_explicitly_post_selected_across_all_folds() -> None:
    candidate = Evidence(
        top_k=2,
        balance_power=0.25,
        threshold=0.3,
        top1_lifts_pp=(0.16, 0.16, 0.08, 0.08),
        switch_rates=(0.01, 0.01, 0.01, 0.01),
        gate="meet_2_uncertainty_high",
    )

    assert _forward_candidate((candidate,)) == candidate
