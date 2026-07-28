import numpy as np

from drms.ambiguity import (
    compute_decision_ambiguity,
    compute_robust_decision_ambiguity,
    select_memory,
    uniform_value_loss,
)
from drms.confidence import sample_tabular_dataset, tabular_q_confidence_intervals
from drms.mdp import evaluate_policy, optimal_q_values, uniform_policy
from drms.predictive import conditional_next_observation_entropy
from drms.toy_envs import make_delayed_cue_mdp, make_nuisance_memory_mdp


def test_delayed_cue_ambiguity_has_sharp_memory_transition() -> None:
    mdp = make_delayed_cue_mdp(horizon=4, reward_gap=1.0)
    optimal = optimal_q_values(mdp)
    ambiguities = [
        compute_decision_ambiguity(mdp, optimal.q, memory=m) for m in range(1, 5)
    ]
    totals = np.array([result.total_ambiguity for result in ambiguities])
    np.testing.assert_allclose(totals, [0.5, 0.5, 0.5, 0.0], atol=1e-9)

    short_eval = evaluate_policy(mdp, ambiguities[0].policy)
    full_eval = evaluate_policy(mdp, ambiguities[-1].policy)
    assert np.isclose(uniform_value_loss(optimal.v, short_eval.v), 0.5)
    assert np.isclose(uniform_value_loss(optimal.v, full_eval.v), 0.0)


def test_nuisance_history_is_predictive_but_not_decision_relevant() -> None:
    mdp = make_nuisance_memory_mdp(horizon=4, reward_gap=1.0)
    optimal = optimal_q_values(mdp)
    behavior = uniform_policy(mdp)

    decision = [
        compute_decision_ambiguity(mdp, optimal.q, memory=m).total_ambiguity
        for m in range(1, 5)
    ]
    predictive = [
        conditional_next_observation_entropy(mdp, behavior, memory=m).total_entropy
        for m in range(1, 5)
    ]

    np.testing.assert_allclose(decision, np.zeros(4), atol=1e-9)
    np.testing.assert_allclose(predictive[:3], np.ones(3), atol=1e-9)
    assert np.isclose(predictive[3], 0.0)


def test_delayed_cue_has_no_predictive_memory_signal() -> None:
    mdp = make_delayed_cue_mdp(horizon=4)
    behavior = uniform_policy(mdp)
    entropies = [
        conditional_next_observation_entropy(mdp, behavior, memory=m).total_entropy
        for m in range(1, 5)
    ]
    np.testing.assert_allclose(entropies, np.zeros(4), atol=1e-9)


def test_zero_width_robust_certificate_matches_population_ambiguity() -> None:
    mdp = make_delayed_cue_mdp(horizon=4)
    optimal = optimal_q_values(mdp)
    for memory in range(1, 5):
        exact = compute_decision_ambiguity(mdp, optimal.q, memory)
        robust = compute_robust_decision_ambiguity(
            mdp, optimal.q, optimal.q, memory
        )
        assert np.isclose(exact.total_ambiguity, robust.total_ambiguity)
        for p_exact, p_robust in zip(exact.policy, robust.policy, strict=True):
            np.testing.assert_allclose(p_exact, p_robust, atol=1e-9)


def test_robust_policy_value_loss_is_bounded_by_certificate() -> None:
    mdp = make_delayed_cue_mdp(horizon=4)
    optimal = optimal_q_values(mdp)
    width = 0.15
    lower = [np.maximum(0.0, q - width) for q in optimal.q]
    upper = [q + width for q in optimal.q]

    for memory in range(1, 5):
        robust = compute_robust_decision_ambiguity(mdp, lower, upper, memory)
        evaluation = evaluate_policy(mdp, robust.policy)
        loss = uniform_value_loss(optimal.v, evaluation.v)
        assert loss <= robust.total_ambiguity + 1e-9


def test_tabular_confidence_intervals_contain_true_q_in_large_dataset() -> None:
    mdp = make_delayed_cue_mdp(horizon=4, cue_probability=0.3)
    behavior = uniform_policy(mdp)
    dataset = sample_tabular_dataset(mdp, behavior, 20_000, seed=4)
    intervals = tabular_q_confidence_intervals(mdp, dataset, delta=0.05)
    optimal = optimal_q_values(mdp)

    for q, lower, upper in zip(
        optimal.q, intervals.lower_q, intervals.upper_q, strict=True
    ):
        assert np.all(q >= lower - 1e-12)
        assert np.all(q <= upper + 1e-12)

    results = [
        compute_robust_decision_ambiguity(
            mdp, intervals.lower_q, intervals.upper_q, memory=m
        )
        for m in range(1, 5)
    ]
    selection = select_memory(results, tolerance=0.1)
    assert selection.certified
    assert selection.selected_memory == 4


def test_lower_bound_pair_has_different_decision_memory_orders() -> None:
    from drms.lower_bound import memory_order_pair_kl
    from drms.toy_envs import make_memory_order_lower_bound_pair

    short_model, long_model = make_memory_order_lower_bound_pair(
        cue_probability=0.1, action_gap=0.2
    )
    short_optimal = optimal_q_values(short_model)
    long_optimal = optimal_q_values(long_model)
    short_scores = [
        compute_decision_ambiguity(short_model, short_optimal.q, memory=m)
        .total_ambiguity
        for m in [1, 2]
    ]
    long_scores = [
        compute_decision_ambiguity(long_model, long_optimal.q, memory=m)
        .total_ambiguity
        for m in [1, 2]
    ]
    np.testing.assert_allclose(short_scores, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(long_scores, [0.1, 0.0], atol=1e-9)
    assert memory_order_pair_kl(0.1, 0.2) <= 6.0 * 0.1 * 0.2**2
    assert memory_order_pair_kl(0.1, 0.5) <= 6.0 * 0.1 * 0.5**2


def test_value_loss_sandwich_holds_for_every_memory() -> None:
    mdp = make_delayed_cue_mdp(horizon=5, reward_gap=0.6)
    optimal = optimal_q_values(mdp)
    previous = float("inf")
    for memory in range(1, 6):
        result = compute_decision_ambiguity(mdp, optimal.q, memory=memory)
        evaluation = evaluate_policy(mdp, result.policy)
        loss = uniform_value_loss(optimal.v, evaluation.v)
        assert max(result.stage_ambiguities) <= loss + 1e-10
        assert loss <= result.total_ambiguity + 1e-10
        assert result.total_ambiguity <= previous + 1e-10
        previous = result.total_ambiguity


def test_matching_rate_pair_is_recovered_with_large_dataset() -> None:
    from drms.toy_envs import make_memory_order_lower_bound_pair

    for expected_memory, mdp in enumerate(
        make_memory_order_lower_bound_pair(cue_probability=0.2, action_gap=0.5),
        start=1,
    ):
        behavior = uniform_policy(mdp)
        dataset = sample_tabular_dataset(mdp, behavior, 20_000, seed=expected_memory)
        intervals = tabular_q_confidence_intervals(mdp, dataset, delta=0.025)
        results = [
            compute_robust_decision_ambiguity(
                mdp, intervals.lower_q, intervals.upper_q, memory=memory
            )
            for memory in [1, 2]
        ]
        selection = select_memory(results, tolerance=0.5 / 4.0)
        assert selection.certified
        assert selection.selected_memory == expected_memory
