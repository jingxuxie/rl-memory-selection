#!/usr/bin/env python3
"""Finite-sample certification under rare decision-relevant histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drms.ambiguity import (
    compute_robust_decision_ambiguity,
    select_memory,
    uniform_value_loss,
)
from drms.confidence import sample_tabular_dataset, tabular_q_confidence_intervals
from drms.mdp import evaluate_policy, optimal_q_values, uniform_policy
from drms.toy_envs import make_delayed_cue_mdp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--reward-gap", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample_sizes = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
    cue_probabilities = [0.50, 0.20, 0.10, 0.05]
    rows: list[dict[str, float | int | bool]] = []

    for cue_probability in cue_probabilities:
        mdp = make_delayed_cue_mdp(
            horizon=args.horizon,
            reward_gap=args.reward_gap,
            cue_probability=cue_probability,
        )
        behavior = uniform_policy(mdp)
        optimal = optimal_q_values(mdp)

        for n_episodes in sample_sizes:
            for seed in range(args.seeds):
                dataset = sample_tabular_dataset(
                    mdp,
                    behavior,
                    n_episodes,
                    seed=1_000_000 * seed + 10_000 * int(100 * cue_probability) + n_episodes,
                )
                intervals = tabular_q_confidence_intervals(
                    mdp, dataset, delta=args.delta
                )
                results = [
                    compute_robust_decision_ambiguity(
                        mdp,
                        intervals.lower_q,
                        intervals.upper_q,
                        memory=memory,
                    )
                    for memory in range(1, args.horizon + 1)
                ]
                selection = select_memory(results, tolerance=args.tolerance)
                evaluation = evaluate_policy(mdp, selection.result.policy)
                coverage = all(
                    np.all(q >= lower - 1e-12) and np.all(q <= upper + 1e-12)
                    for q, lower, upper in zip(
                        optimal.q,
                        intervals.lower_q,
                        intervals.upper_q,
                        strict=True,
                    )
                )
                true_loss = uniform_value_loss(optimal.v, evaluation.v)
                rows.append(
                    {
                        "cue_probability": cue_probability,
                        "n_episodes": n_episodes,
                        "seed": seed,
                        "delta": args.delta,
                        "tolerance": args.tolerance,
                        "reward_gap": args.reward_gap,
                        "selected_memory": selection.selected_memory,
                        "certificate": selection.certificate,
                        "certified": selection.certified,
                        "correct_memory_certified": (
                            selection.certified
                            and selection.selected_memory == args.horizon
                        ),
                        "false_short_certification": (
                            selection.certified
                            and selection.selected_memory < args.horizon
                        ),
                        "true_uniform_loss": true_loss,
                        "certificate_valid": true_loss
                        <= selection.certificate + 1e-10,
                        "q_interval_coverage": coverage,
                        "rare_final_min_count": int(
                            dataset.counts[-1][1].min()
                        ),
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "finite_sample_raw.csv", index=False)
    summary = (
        raw.groupby(["cue_probability", "n_episodes"], as_index=False)
        .agg(
            certification_rate=("certified", "mean"),
            correct_memory_rate=("correct_memory_certified", "mean"),
            false_short_rate=("false_short_certification", "mean"),
            certificate_valid_rate=("certificate_valid", "mean"),
            q_coverage_rate=("q_interval_coverage", "mean"),
            median_certificate=("certificate", "median"),
            mean_true_loss=("true_uniform_loss", "mean"),
            median_rare_count=("rare_final_min_count", "median"),
        )
        .sort_values(["cue_probability", "n_episodes"])
    )
    summary.to_csv(RESULTS / "finite_sample_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {len(raw)} finite-sample runs to {RESULTS}")


if __name__ == "__main__":
    main()
