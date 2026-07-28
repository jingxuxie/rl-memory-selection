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


def test_two_action_envelope_solver_matches_generic_lp() -> None:
    from scipy.optimize import linprog

    from drms.ambiguity import _solve_minimax_mixture

    rng = np.random.default_rng(17)
    for n_rows in [1, 2, 5, 20]:
        for _ in range(25):
            costs = rng.uniform(0.0, 3.0, size=(n_rows, 2))
            mixture, value = _solve_minimax_mixture(costs)
            objective = np.array([0.0, 0.0, 1.0])
            a_ub = np.hstack([costs, -np.ones((n_rows, 1))])
            solution = linprog(
                objective,
                A_ub=a_ub,
                b_ub=np.zeros(n_rows),
                A_eq=np.array([[1.0, 1.0, 0.0]]),
                b_eq=np.array([1.0]),
                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, None)],
                method="highs",
            )
            assert solution.success
            np.testing.assert_allclose(mixture.sum(), 1.0, atol=1e-12)
            np.testing.assert_allclose(value, solution.fun, atol=1e-9)


def test_weighted_ambiguity_ignores_a_rare_conflict_in_expectation() -> None:
    from drms.ambiguity import compute_weighted_decision_ambiguity
    from drms.mdp import state_occupancies

    probability = 0.1
    gap = 0.4
    mdp = make_delayed_cue_mdp(
        horizon=4, reward_gap=gap, cue_probability=probability
    )
    optimal = optimal_q_values(mdp)
    reference = state_occupancies(mdp, uniform_policy(mdp))[:-1]
    short = compute_weighted_decision_ambiguity(
        mdp, optimal.q, memory=1, reference_occupancies=reference
    )
    full = compute_weighted_decision_ambiguity(
        mdp, optimal.q, memory=4, reference_occupancies=reference
    )
    np.testing.assert_allclose(short.total_cost, probability * gap, atol=1e-12)
    np.testing.assert_allclose(full.total_cost, 0.0, atol=1e-12)


def test_weighted_value_certificate_holds_with_concentrability() -> None:
    from drms.ambiguity import (
        compute_weighted_decision_ambiguity,
        deployment_concentrability,
        weighted_value_certificate,
    )
    from drms.mdp import state_occupancies

    mdp = make_delayed_cue_mdp(horizon=4, reward_gap=0.6, cue_probability=0.2)
    optimal = optimal_q_values(mdp)
    reference = state_occupancies(mdp, uniform_policy(mdp))[:-1]
    result = compute_weighted_decision_ambiguity(
        mdp, optimal.q, memory=1, reference_occupancies=reference
    )
    evaluation = evaluate_policy(mdp, result.policy)
    deployment = state_occupancies(mdp, result.policy)[:-1]
    coefficients = deployment_concentrability(deployment, reference)
    loss = float(mdp.initial_distribution @ optimal.v[0]) - evaluation.initial_value
    assert loss <= weighted_value_certificate(result, coefficients) + 1e-12


def test_robust_weighted_certificate_holds_on_q_interval_event() -> None:
    from drms.ambiguity import (
        compute_robust_weighted_decision_ambiguity,
        deployment_concentrability,
        upper_advantages_from_q_intervals,
        weighted_value_certificate,
    )
    from drms.mdp import state_occupancies

    mdp = make_delayed_cue_mdp(horizon=4, reward_gap=0.5, cue_probability=0.15)
    optimal = optimal_q_values(mdp)
    width = 0.08
    lower = [np.maximum(0.0, q - width) for q in optimal.q]
    upper = [q + width for q in optimal.q]
    reference = state_occupancies(mdp, uniform_policy(mdp))[:-1]
    result = compute_robust_weighted_decision_ambiguity(
        mdp, lower, upper, memory=1, reference_occupancies=reference
    )
    evaluation = evaluate_policy(mdp, result.policy)
    deployment = state_occupancies(mdp, result.policy)[:-1]
    coefficients = deployment_concentrability(deployment, reference)
    loss = float(mdp.initial_distribution @ optimal.v[0]) - evaluation.initial_value
    assert loss <= weighted_value_certificate(result, coefficients) + 1e-12
    bounds = upper_advantages_from_q_intervals(lower, upper)
    assert all(np.all(bound >= -1e-12) for bound in bounds)


def test_empirical_state_occupancy_radius_contains_truth_in_large_sample() -> None:
    from drms.confidence import empirical_state_occupancies, state_occupancy_l1_radii
    from drms.mdp import state_occupancies

    mdp = make_delayed_cue_mdp(horizon=4, cue_probability=0.2)
    behavior = uniform_policy(mdp)
    dataset = sample_tabular_dataset(mdp, behavior, 20_000, seed=91)
    empirical = empirical_state_occupancies(dataset)
    truth = state_occupancies(mdp, behavior)[:-1]
    radii = state_occupancy_l1_radii(mdp, dataset.n_episodes, delta=0.05)
    for estimate, target, radius in zip(empirical, truth, radii, strict=True):
        assert np.abs(estimate - target).sum() <= radius + 1e-12


def test_random_history_mdps_satisfy_population_and_robust_bounds() -> None:
    from drms.toy_envs import make_random_history_mdp

    for seed in range(8):
        mdp = make_random_history_mdp(horizon=3, seed=seed)
        optimal = optimal_q_values(mdp)
        previous = float("inf")
        width = 0.05
        lower = [np.maximum(0.0, q - width) for q in optimal.q]
        upper = [q + width for q in optimal.q]
        for memory in range(1, mdp.horizon + 1):
            exact = compute_decision_ambiguity(mdp, optimal.q, memory)
            robust = compute_robust_decision_ambiguity(mdp, lower, upper, memory)
            evaluation = evaluate_policy(mdp, exact.policy)
            robust_evaluation = evaluate_policy(mdp, robust.policy)
            exact_loss = uniform_value_loss(optimal.v, evaluation.v)
            robust_loss = uniform_value_loss(optimal.v, robust_evaluation.v)
            assert max(exact.stage_ambiguities) <= exact_loss + 1e-10
            assert exact_loss <= exact.total_ambiguity + 1e-10
            assert exact.total_ambiguity <= robust.total_ambiguity + 1e-10
            assert robust_loss <= robust.total_ambiguity + 1e-10
            assert robust.total_ambiguity <= exact.total_ambiguity + 2 * mdp.horizon * width + 1e-10
            assert exact.total_ambiguity <= previous + 1e-10
            previous = exact.total_ambiguity


def test_weighted_ambiguity_is_monotone_on_random_history_mdps() -> None:
    from drms.ambiguity import compute_weighted_decision_ambiguity
    from drms.mdp import state_occupancies
    from drms.toy_envs import make_random_history_mdp

    for seed in range(5):
        mdp = make_random_history_mdp(horizon=3, seed=100 + seed)
        optimal = optimal_q_values(mdp)
        reference = state_occupancies(mdp, uniform_policy(mdp))[:-1]
        totals = [
            compute_weighted_decision_ambiguity(
                mdp, optimal.q, memory, reference
            ).total_cost
            for memory in range(1, mdp.horizon + 1)
        ]
        assert all(left >= right - 1e-10 for left, right in zip(totals, totals[1:]))
