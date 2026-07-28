"""Population predictive-memory diagnostics used for controlled comparisons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .mdp import FloatArray, HistoryMDP, Policy, state_occupancies, validate_policy


@dataclass(frozen=True)
class PredictiveResult:
    memory: int
    stage_entropies: tuple[float, ...]
    total_entropy: float


def conditional_next_observation_entropy(
    mdp: HistoryMDP, policy: Policy, memory: int
) -> PredictiveResult:
    """Compute ``sum_h H(O_{h+1} | suffix_m, A_h)`` in bits.

    Conditioning and weighting use the full-state occupancy induced by
    ``policy``.  This is an oracle predictive diagnostic for the toy examples,
    not part of DRMS itself.
    """

    validate_policy(mdp, policy)
    occupancies = state_occupancies(mdp, policy)
    stage_entropies: list[float] = []

    for h in range(1, mdp.horizon + 1):
        joint: dict[tuple[tuple[object, ...], int], dict[object, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        context_mass: dict[tuple[tuple[object, ...], int], float] = defaultdict(float)

        for state_index in range(mdp.n_states(h)):
            suffix = mdp.suffix_key(h, state_index, memory)
            for action in range(mdp.n_actions(h)):
                state_action_mass = occupancies[h - 1][state_index] * policy[h - 1][
                    state_index, action
                ]
                if state_action_mass <= 0:
                    continue
                key = (suffix, action)
                for next_index, transition_prob in enumerate(
                    mdp.transitions[h - 1][state_index, action]
                ):
                    mass = state_action_mass * transition_prob
                    if mass <= 0:
                        continue
                    next_observation = mdp.histories[h][next_index][-1]
                    joint[key][next_observation] += mass
                    context_mass[key] += mass

        entropy = 0.0
        for key, outcomes in joint.items():
            mass = context_mass[key]
            if mass <= 0:
                continue
            probabilities = np.array(list(outcomes.values()), dtype=float) / mass
            positive = probabilities[probabilities > 0]
            local_entropy = float(-np.sum(positive * np.log2(positive)))
            entropy += mass * local_entropy
        stage_entropies.append(entropy)

    return PredictiveResult(
        memory=memory,
        stage_entropies=tuple(stage_entropies),
        total_entropy=float(sum(stage_entropies)),
    )
