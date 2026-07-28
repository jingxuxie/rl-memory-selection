#!/usr/bin/env python3
"""Validate the deployment-weighted DRMS extension on rare conflicts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drms.ambiguity import (
    compute_decision_ambiguity,
    compute_robust_decision_ambiguity,
    compute_robust_weighted_decision_ambiguity,
    compute_weighted_decision_ambiguity,
    select_memory,
    upper_advantages_from_q_intervals,
    weighted_value_certificate,
)
from drms.confidence import (
    empirical_state_occupancies,
    sample_tabular_dataset,
    state_occupancy_l1_radii,
    tabular_q_confidence_intervals,
)
from drms.mdp import evaluate_policy, optimal_q_values, state_occupancies, uniform_policy
from drms.toy_envs import make_delayed_cue_mdp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--reward-gap", type=float, default=0.4)
    parser.add_argument("--tolerance", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    population_rows: list[dict[str, float | int]] = []
    for cue_probability in np.geomspace(0.01, 0.5, 20):
        mdp = make_delayed_cue_mdp(
            horizon=args.horizon,
            reward_gap=args.reward_gap,
            cue_probability=float(cue_probability),
        )
        optimal = optimal_q_values(mdp)
        reference = state_occupancies(mdp, uniform_policy(mdp))[:-1]
        uniform_result = compute_decision_ambiguity(mdp, optimal.q, memory=1)
        weighted_result = compute_weighted_decision_ambiguity(
            mdp, optimal.q, memory=1, reference_occupancies=reference
        )
        weighted_evaluation = evaluate_policy(mdp, weighted_result.policy)
        expected_loss = (
            float(mdp.initial_distribution @ optimal.v[0])
            - weighted_evaluation.initial_value
        )
        population_rows.append(
            {
                "cue_probability": float(cue_probability),
                "reward_gap": args.reward_gap,
                "tolerance": args.tolerance,
                "uniform_ambiguity": uniform_result.total_ambiguity,
                "weighted_ambiguity": weighted_result.total_cost,
                "weighted_policy_loss": expected_loss,
                "uniform_short_certifiable": uniform_result.total_ambiguity
                <= args.tolerance,
                "weighted_short_certifiable": weighted_result.total_cost
                <= args.tolerance,
            }
        )
    pd.DataFrame(population_rows).to_csv(
        RESULTS / "weighted_population.csv", index=False
    )

    sample_sizes = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
    cue_probabilities = [0.05, 0.10, 0.20, 0.50]
    rows: list[dict[str, float | int | bool]] = []
    q_delta = args.delta / 2.0
    occupancy_delta = args.delta / 2.0

    for cue_probability in cue_probabilities:
        mdp = make_delayed_cue_mdp(
            horizon=args.horizon,
            reward_gap=args.reward_gap,
            cue_probability=cue_probability,
        )
        behavior = uniform_policy(mdp)
        optimal = optimal_q_values(mdp)
        optimal_initial = float(mdp.initial_distribution @ optimal.v[0])

        for n_episodes in sample_sizes:
            for seed in range(args.seeds):
                dataset = sample_tabular_dataset(
                    mdp,
                    behavior,
                    n_episodes,
                    seed=(
                        7_000_000 * seed
                        + 10_000 * int(100 * cue_probability)
                        + n_episodes
                    ),
                )
                intervals = tabular_q_confidence_intervals(
                    mdp, dataset, delta=q_delta
                )
                empirical_reference = empirical_state_occupancies(dataset)
                occupancy_radii = state_occupancy_l1_radii(
                    mdp, n_episodes, delta=occupancy_delta
                )
                upper_costs = upper_advantages_from_q_intervals(
                    intervals.lower_q, intervals.upper_q
                )
                stage_bounds = [float(cost.max()) for cost in upper_costs]

                weighted_results = []
                weighted_certificates = []
                for memory in range(1, args.horizon + 1):
                    result = compute_robust_weighted_decision_ambiguity(
                        mdp,
                        intervals.lower_q,
                        intervals.upper_q,
                        memory,
                        empirical_reference,
                    )
                    certificate = weighted_value_certificate(
                        result,
                        [1.0] * args.horizon,
                        occupancy_radii=occupancy_radii,
                        stage_cost_bounds=stage_bounds,
                    )
                    weighted_results.append(result)
                    weighted_certificates.append(certificate)

                passing = [
                    index
                    for index, certificate in enumerate(weighted_certificates)
                    if certificate <= args.tolerance + 1e-12
                ]
                if passing:
                    weighted_index = passing[0]
                    weighted_certified = True
                else:
                    weighted_index = args.horizon - 1
                    weighted_certified = False
                weighted_result = weighted_results[weighted_index]
                weighted_certificate = weighted_certificates[weighted_index]
                weighted_evaluation = evaluate_policy(mdp, weighted_result.policy)
                weighted_true_loss = optimal_initial - weighted_evaluation.initial_value

                uniform_results = [
                    compute_robust_decision_ambiguity(
                        mdp,
                        intervals.lower_q,
                        intervals.upper_q,
                        memory,
                    )
                    for memory in range(1, args.horizon + 1)
                ]
                uniform_selection = select_memory(
                    uniform_results, tolerance=args.tolerance
                )

                q_coverage = all(
                    np.all(q >= lower - 1e-12) and np.all(q <= upper + 1e-12)
                    for q, lower, upper in zip(
                        optimal.q, intervals.lower_q, intervals.upper_q, strict=True
                    )
                )
                true_reference = state_occupancies(mdp, behavior)[:-1]
                occupancy_coverage = all(
                    np.abs(estimate - truth).sum() <= radius + 1e-12
                    for estimate, truth, radius in zip(
                        empirical_reference,
                        true_reference,
                        occupancy_radii,
                        strict=True,
                    )
                )

                rows.append(
                    {
                        "cue_probability": cue_probability,
                        "n_episodes": n_episodes,
                        "seed": seed,
                        "selected_memory": weighted_index + 1,
                        "certificate": weighted_certificate,
                        "certified": weighted_certified,
                        "short_memory_certified": weighted_certified
                        and weighted_index == 0,
                        "true_expected_loss": weighted_true_loss,
                        "certificate_valid": weighted_true_loss
                        <= weighted_certificate + 1e-10,
                        "uniform_selected_memory": uniform_selection.selected_memory,
                        "uniform_certified": uniform_selection.certified,
                        "uniform_short_certified": uniform_selection.certified
                        and uniform_selection.selected_memory == 1,
                        "q_interval_coverage": q_coverage,
                        "occupancy_interval_coverage": occupancy_coverage,
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "weighted_finite_sample_raw.csv", index=False)
    summary = (
        raw.groupby(["cue_probability", "n_episodes"], as_index=False)
        .agg(
            weighted_certification_rate=("certified", "mean"),
            weighted_short_rate=("short_memory_certified", "mean"),
            uniform_short_rate=("uniform_short_certified", "mean"),
            certificate_valid_rate=("certificate_valid", "mean"),
            q_coverage_rate=("q_interval_coverage", "mean"),
            occupancy_coverage_rate=("occupancy_interval_coverage", "mean"),
            median_certificate=("certificate", "median"),
            mean_true_expected_loss=("true_expected_loss", "mean"),
        )
        .sort_values(["cue_probability", "n_episodes"])
    )
    summary.to_csv(RESULTS / "weighted_finite_sample_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {len(raw)} weighted finite-sample runs to {RESULTS}")


if __name__ == "__main__":
    main()
