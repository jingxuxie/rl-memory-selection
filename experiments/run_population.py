#!/usr/bin/env python3
"""Run exact population experiments for the DRMS counterexamples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drms.ambiguity import compute_decision_ambiguity, uniform_value_loss
from drms.mdp import evaluate_policy, optimal_q_values, uniform_policy
from drms.predictive import conditional_next_observation_entropy
from drms.toy_envs import make_delayed_cue_mdp, make_nuisance_memory_mdp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    builders = {
        "delayed_cue": make_delayed_cue_mdp,
        "nuisance_memory": make_nuisance_memory_mdp,
    }
    for task, builder in builders.items():
        for horizon in [2, 3, 4, 5, 6, 8]:
            mdp = builder(horizon=horizon, reward_gap=1.0, cue_probability=0.5)
            optimal = optimal_q_values(mdp)
            behavior = uniform_policy(mdp)
            full_predictive_entropy = conditional_next_observation_entropy(
                mdp, behavior, memory=horizon
            ).total_entropy
            for memory in range(1, horizon + 1):
                ambiguity = compute_decision_ambiguity(mdp, optimal.q, memory)
                evaluation = evaluate_policy(mdp, ambiguity.policy)
                predictive = conditional_next_observation_entropy(
                    mdp, behavior, memory
                )
                rows.append(
                    {
                        "task": task,
                        "horizon": horizon,
                        "memory": memory,
                        "decision_ambiguity": ambiguity.total_ambiguity,
                        "uniform_policy_loss": uniform_value_loss(
                            optimal.v, evaluation.v
                        ),
                        "expected_policy_loss": float(
                            mdp.initial_distribution
                            @ (optimal.v[0] - evaluation.v[0])
                        ),
                        "predictive_entropy_bits": predictive.total_entropy,
                        "predictive_residual_bits": (
                            predictive.total_entropy - full_predictive_entropy
                        ),
                    }
                )

    population = pd.DataFrame(rows)
    population.to_csv(RESULTS / "population_results.csv", index=False)

    soft_rows: list[dict[str, float | int]] = []
    horizon = 5
    tolerances = [0.025, 0.05, 0.1, 0.2]
    for gap in np.linspace(0.02, 1.0, 50):
        mdp = make_delayed_cue_mdp(horizon=horizon, reward_gap=float(gap))
        optimal = optimal_q_values(mdp)
        ambiguities = [
            compute_decision_ambiguity(mdp, optimal.q, memory=m)
            for m in range(1, horizon + 1)
        ]
        for tolerance in tolerances:
            certifiable = [
                result.memory
                for result in ambiguities
                if result.total_ambiguity <= tolerance + 1e-12
            ]
            soft_rows.append(
                {
                    "horizon": horizon,
                    "reward_gap": float(gap),
                    "tolerance": tolerance,
                    "selected_memory": min(certifiable, default=horizon),
                    "short_memory_ambiguity": ambiguities[0].total_ambiguity,
                }
            )
    pd.DataFrame(soft_rows).to_csv(
        RESULTS / "soft_relevance_results.csv", index=False
    )

    key = population.query("horizon == 4")
    print(key.to_string(index=False))
    print(f"\nWrote {len(population)} population rows to {RESULTS}")


if __name__ == "__main__":
    main()
