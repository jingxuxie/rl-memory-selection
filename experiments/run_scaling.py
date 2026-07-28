#!/usr/bin/env python3
"""Validate the rare-context/action-gap scaling in the lower-bound family."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drms.ambiguity import compute_robust_decision_ambiguity, select_memory
from drms.confidence import sample_tabular_dataset, tabular_q_confidence_intervals
from drms.mdp import uniform_policy
from drms.toy_envs import make_memory_order_lower_bound_pair

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--delta", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample_sizes = [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]
    cue_probabilities = [0.05, 0.10, 0.20, 0.50]
    # All gaps lie in the range covered by the quadratic-KL lower bound.
    action_gaps = [0.20, 0.30, 0.50]
    rows: list[dict[str, float | int | bool]] = []

    for action_gap in action_gaps:
        # The one-memory ambiguity of the long-memory model is gamma / 2.
        # epsilon=gamma/4 leaves a fixed relative separation margin.
        tolerance = action_gap / 4.0
        for cue_probability in cue_probabilities:
            _, mdp = make_memory_order_lower_bound_pair(
                cue_probability=cue_probability,
                action_gap=action_gap,
            )
            behavior = uniform_policy(mdp)
            for n_episodes in sample_sizes:
                for seed in range(args.seeds):
                    dataset = sample_tabular_dataset(
                        mdp,
                        behavior,
                        n_episodes,
                        seed=(
                            1_000_000 * seed
                            + 10_000 * int(100 * action_gap)
                            + 100 * int(100 * cue_probability)
                            + n_episodes
                        ),
                    )
                    intervals = tabular_q_confidence_intervals(
                        mdp, dataset, delta=args.delta
                    )
                    certificates = [
                        compute_robust_decision_ambiguity(
                            mdp,
                            intervals.lower_q,
                            intervals.upper_q,
                            memory=memory,
                        )
                        for memory in [1, 2]
                    ]
                    selection = select_memory(certificates, tolerance=tolerance)
                    rows.append(
                        {
                            "action_gap": action_gap,
                            "cue_probability": cue_probability,
                            "n_episodes": n_episodes,
                            "seed": seed,
                            "tolerance": tolerance,
                            "information_scale": (
                                n_episodes * cue_probability * action_gap**2
                            ),
                            "correct_memory_certified": (
                                selection.certified
                                and selection.selected_memory == 2
                            ),
                            "false_short_certification": (
                                selection.certified
                                and selection.selected_memory < 2
                            ),
                        }
                    )

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "scaling_raw.csv", index=False)
    summary = (
        raw.groupby(
            ["action_gap", "cue_probability", "n_episodes", "information_scale"],
            as_index=False,
        )
        .agg(
            recovery_rate=("correct_memory_certified", "mean"),
            false_short_rate=("false_short_certification", "mean"),
        )
        .sort_values(["action_gap", "cue_probability", "n_episodes"])
    )
    summary.to_csv(RESULTS / "scaling_summary.csv", index=False)

    threshold_rows: list[dict[str, float]] = []
    for (gap, probability), subset in summary.groupby(
        ["action_gap", "cue_probability"], sort=True
    ):
        subset = subset.sort_values("n_episodes")
        reached = subset[subset["recovery_rate"] >= 0.5]
        n50 = float(reached["n_episodes"].iloc[0]) if len(reached) else np.nan
        threshold_rows.append(
            {
                "action_gap": float(gap),
                "cue_probability": float(probability),
                "inverse_information": 1.0 / (float(probability) * float(gap) ** 2),
                "n50": n50,
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(RESULTS / "scaling_thresholds.csv", index=False)

    valid = thresholds.dropna()
    if len(valid) >= 2:
        slope, intercept = np.polyfit(
            np.log(valid["inverse_information"]), np.log(valid["n50"]), 1
        )
        print(f"log-log n50 slope versus 1/(p gamma^2): {slope:.3f}")
        print(f"log-log intercept: {intercept:.3f}")
    print(thresholds.to_string(index=False))
    print(f"\nWrote {len(raw)} scaling runs to {RESULTS}")


if __name__ == "__main__":
    main()
