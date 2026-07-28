#!/usr/bin/env python3
"""Stress-test population, robust, and weighted DRMS theorems on random MDPs."""

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
    deployment_concentrability,
    uniform_value_loss,
    weighted_value_certificate,
)
from drms.mdp import (
    evaluate_policy,
    optimal_q_values,
    state_occupancies,
    uniform_policy,
)
from drms.toy_envs import make_random_history_mdp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances-per-horizon", type=int, default=100)
    parser.add_argument("--width", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | bool]] = []

    for horizon in [3, 4]:
        for instance in range(args.instances_per_horizon):
            seed = 100_000 * horizon + instance
            mdp = make_random_history_mdp(horizon=horizon, seed=seed)
            optimal = optimal_q_values(mdp)
            optimal_initial = float(mdp.initial_distribution @ optimal.v[0])
            behavior = uniform_policy(mdp)
            reference = state_occupancies(mdp, behavior)[:-1]
            lower = [np.maximum(0.0, q - args.width) for q in optimal.q]
            upper = [q + args.width for q in optimal.q]

            previous_uniform = float("inf")
            previous_weighted = float("inf")
            for memory in range(1, horizon + 1):
                exact = compute_decision_ambiguity(mdp, optimal.q, memory)
                exact_eval = evaluate_policy(mdp, exact.policy)
                exact_loss = uniform_value_loss(optimal.v, exact_eval.v)

                robust = compute_robust_decision_ambiguity(
                    mdp, lower, upper, memory
                )
                robust_eval = evaluate_policy(mdp, robust.policy)
                robust_loss = uniform_value_loss(optimal.v, robust_eval.v)

                weighted = compute_weighted_decision_ambiguity(
                    mdp, optimal.q, memory, reference
                )
                weighted_eval = evaluate_policy(mdp, weighted.policy)
                weighted_deployment = state_occupancies(mdp, weighted.policy)[:-1]
                weighted_c = deployment_concentrability(
                    weighted_deployment, reference
                )
                weighted_certificate = weighted_value_certificate(
                    weighted, weighted_c
                )
                weighted_loss = optimal_initial - weighted_eval.initial_value

                robust_weighted = compute_robust_weighted_decision_ambiguity(
                    mdp, lower, upper, memory, reference
                )
                robust_weighted_eval = evaluate_policy(
                    mdp, robust_weighted.policy
                )
                robust_weighted_deployment = state_occupancies(
                    mdp, robust_weighted.policy
                )[:-1]
                robust_weighted_c = deployment_concentrability(
                    robust_weighted_deployment, reference
                )
                robust_weighted_certificate = weighted_value_certificate(
                    robust_weighted, robust_weighted_c
                )
                robust_weighted_loss = (
                    optimal_initial - robust_weighted_eval.initial_value
                )

                lower_stage = max(exact.stage_ambiguities, default=0.0)
                violations = {
                    "monotonicity_violation": exact.total_ambiguity
                    > previous_uniform + 1e-9,
                    "sandwich_violation": not (
                        lower_stage - 1e-9
                        <= exact_loss
                        <= exact.total_ambiguity + 1e-9
                    ),
                    "robust_certificate_violation": robust_loss
                    > robust.total_ambiguity + 1e-9,
                    "inflation_violation": robust.total_ambiguity
                    > exact.total_ambiguity
                    + 2.0 * horizon * args.width
                    + 1e-9,
                    "weighted_monotonicity_violation": weighted.total_cost
                    > previous_weighted + 1e-9,
                    "weighted_certificate_violation": weighted_loss
                    > weighted_certificate + 1e-9,
                    "robust_weighted_certificate_violation": robust_weighted_loss
                    > robust_weighted_certificate + 1e-9,
                }
                theorem_violation = any(violations.values())
                relative_slack = (
                    (exact.total_ambiguity - exact_loss) / exact_loss
                    if exact_loss > 1e-9
                    else np.nan
                )
                ratio = (
                    weighted_certificate / exact.total_ambiguity
                    if exact.total_ambiguity > 1e-9
                    else np.nan
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "instance": instance,
                        "memory": memory,
                        "uniform_ambiguity": exact.total_ambiguity,
                        "uniform_loss": exact_loss,
                        "robust_ambiguity": robust.total_ambiguity,
                        "robust_loss": robust_loss,
                        "robust_inflation": robust.total_ambiguity
                        - exact.total_ambiguity,
                        "weighted_ambiguity": weighted.total_cost,
                        "weighted_loss": weighted_loss,
                        "weighted_certificate": weighted_certificate,
                        "robust_weighted_certificate": robust_weighted_certificate,
                        "robust_weighted_loss": robust_weighted_loss,
                        "max_concentrability": max(
                            [*weighted_c, *robust_weighted_c]
                        ),
                        "relative_uniform_slack": relative_slack,
                        "weighted_to_uniform_ratio": ratio,
                        "theorem_violation": theorem_violation,
                        **violations,
                    }
                )
                previous_uniform = exact.total_ambiguity
                previous_weighted = weighted.total_cost

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "random_stress_raw.csv", index=False)
    configurations = len(raw)
    violations = int(raw["theorem_violation"].sum())
    if violations == 0:
        upper_violation_probability = 1.0 - 0.05 ** (1.0 / configurations)
    else:
        upper_violation_probability = np.nan
    finite_slack = raw["relative_uniform_slack"].dropna()
    finite_ratios = raw["weighted_to_uniform_ratio"].dropna()
    summary = pd.DataFrame(
        [
            {
                "instances": 2 * args.instances_per_horizon,
                "memory_configurations": configurations,
                "theorem_violations": violations,
                "zero_failure_95pct_upper_rate": upper_violation_probability,
                "median_relative_uniform_slack": float(finite_slack.median()),
                "p90_relative_uniform_slack": float(
                    finite_slack.quantile(0.9)
                ),
                "max_robust_inflation": float(raw["robust_inflation"].max()),
                "weighted_certificate_valid_rate": float(
                    (~raw["weighted_certificate_violation"]).mean()
                ),
                "robust_weighted_valid_rate": float(
                    (~raw["robust_weighted_certificate_violation"]).mean()
                ),
                "weighted_tighter_rate": float(
                    (raw["weighted_certificate"] < raw["uniform_ambiguity"]).mean()
                ),
                "median_weighted_certificate_ratio": float(
                    finite_ratios.median()
                ),
                "max_concentrability": float(raw["max_concentrability"].max()),
            }
        ]
    )
    summary.to_csv(RESULTS / "random_stress_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {configurations} random-MDP memory configurations to {RESULTS}")


if __name__ == "__main__":
    main()
