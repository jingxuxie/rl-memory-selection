"""Small finite-horizon history MDP utilities.

The full state at stage ``h`` is an observable action-observation history.  A
candidate memory length ``m`` maps that state to the suffix containing the most
recent ``m`` observations and ``m-1`` intervening actions.  All guarantees are
proved and evaluated in the full-history MDP; the suffix itself is never
assumed to be Markov.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
History: TypeAlias = tuple[Hashable, ...]
Policy: TypeAlias = list[FloatArray]


@dataclass(frozen=True)
class HistoryMDP:
    """A finite-horizon MDP whose states carry observable histories.

    Parameters use zero-based Python lists: element ``t`` corresponds to stage
    ``h=t+1``.  ``histories`` additionally contains the terminal stage at index
    ``horizon``.
    """

    horizon: int
    histories: list[list[History]]
    rewards: list[FloatArray]
    transitions: list[FloatArray]
    initial_distribution: FloatArray

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(self.histories) != self.horizon + 1:
            raise ValueError("histories must contain H decision stages plus terminal stage")
        if len(self.rewards) != self.horizon or len(self.transitions) != self.horizon:
            raise ValueError("rewards and transitions must have length H")
        if self.initial_distribution.shape != (len(self.histories[0]),):
            raise ValueError("initial distribution has incompatible shape")
        if not np.isclose(self.initial_distribution.sum(), 1.0):
            raise ValueError("initial distribution must sum to one")
        if np.any(self.initial_distribution < -1e-12):
            raise ValueError("initial distribution must be nonnegative")

        for t in range(self.horizon):
            n_states = len(self.histories[t])
            n_next = len(self.histories[t + 1])
            reward = np.asarray(self.rewards[t], dtype=float)
            transition = np.asarray(self.transitions[t], dtype=float)
            if reward.ndim != 2:
                raise ValueError(f"reward at stage {t + 1} must be a matrix")
            if reward.shape[0] != n_states:
                raise ValueError(f"reward at stage {t + 1} has wrong state dimension")
            if transition.shape != (n_states, reward.shape[1], n_next):
                raise ValueError(f"transition at stage {t + 1} has incompatible shape")
            if np.any(reward < -1e-12) or np.any(reward > 1.0 + 1e-12):
                raise ValueError("the current implementation assumes rewards in [0, 1]")
            if np.any(transition < -1e-12):
                raise ValueError("transition probabilities must be nonnegative")
            if not np.allclose(transition.sum(axis=2), 1.0):
                raise ValueError(f"transition rows at stage {t + 1} must sum to one")

            expected_token_count = 2 * (t + 1) - 1
            if any(len(history) != expected_token_count for history in self.histories[t]):
                raise ValueError(
                    f"each stage-{t + 1} history must contain {expected_token_count} tokens"
                )
        terminal_token_count = 2 * (self.horizon + 1) - 1
        if any(len(history) != terminal_token_count for history in self.histories[-1]):
            raise ValueError(
                f"each terminal history must contain {terminal_token_count} tokens"
            )

    def n_states(self, h: int) -> int:
        """Return the number of full-history states at one-based stage ``h``."""

        self._check_stage(h, include_terminal=True)
        return len(self.histories[h - 1])

    def n_actions(self, h: int) -> int:
        """Return the number of actions at one-based decision stage ``h``."""

        self._check_stage(h, include_terminal=False)
        return int(self.rewards[h - 1].shape[1])

    def suffix_key(self, h: int, state_index: int, memory: int) -> History:
        """Map a full history to its most recent ``memory`` observations.

        The returned tuple contains ``2m-1`` interleaved observation/action
        tokens whenever at least ``m`` observations are available.
        """

        self._check_stage(h, include_terminal=True)
        if memory <= 0:
            raise ValueError("memory must be positive")
        history = self.histories[h - 1][state_index]
        effective_memory = min(memory, h)
        token_count = 2 * effective_memory - 1
        return history[-token_count:]

    def suffix_groups(self, h: int, memory: int) -> dict[History, list[int]]:
        """Partition full contexts by their ``memory``-suffix."""

        groups: dict[History, list[int]] = {}
        for state_index in range(self.n_states(h)):
            key = self.suffix_key(h, state_index, memory)
            groups.setdefault(key, []).append(state_index)
        return groups

    def history_to_index(self, h: int) -> Mapping[History, int]:
        """Return a lookup table for full histories at stage ``h``."""

        self._check_stage(h, include_terminal=True)
        return {history: i for i, history in enumerate(self.histories[h - 1])}

    def _check_stage(self, h: int, *, include_terminal: bool) -> None:
        upper = self.horizon + int(include_terminal)
        if h < 1 or h > upper:
            raise ValueError(f"stage h must lie in [1, {upper}]")


@dataclass(frozen=True)
class OptimalValues:
    q: list[FloatArray]
    v: list[FloatArray]


@dataclass(frozen=True)
class PolicyEvaluation:
    q: list[FloatArray]
    v: list[FloatArray]
    initial_value: float


def optimal_q_values(mdp: HistoryMDP) -> OptimalValues:
    """Compute exact optimal action values by backward induction."""

    q_values: list[FloatArray] = [np.empty((0, 0)) for _ in range(mdp.horizon)]
    values: list[FloatArray] = [np.empty(0) for _ in range(mdp.horizon + 1)]
    values[mdp.horizon] = np.zeros(mdp.n_states(mdp.horizon + 1), dtype=float)

    for t in range(mdp.horizon - 1, -1, -1):
        q_t = mdp.rewards[t] + np.einsum(
            "san,n->sa", mdp.transitions[t], values[t + 1], optimize=True
        )
        q_values[t] = q_t
        values[t] = q_t.max(axis=1)

    return OptimalValues(q=q_values, v=values)


def validate_policy(mdp: HistoryMDP, policy: Sequence[FloatArray]) -> None:
    if len(policy) != mdp.horizon:
        raise ValueError("policy must contain one matrix per decision stage")
    for t, probs in enumerate(policy):
        expected = (mdp.n_states(t + 1), mdp.n_actions(t + 1))
        if probs.shape != expected:
            raise ValueError(f"policy at stage {t + 1} has shape {probs.shape}, expected {expected}")
        if np.any(probs < -1e-12):
            raise ValueError("policy probabilities must be nonnegative")
        if not np.allclose(probs.sum(axis=1), 1.0):
            raise ValueError("each policy row must sum to one")


def evaluate_policy(mdp: HistoryMDP, policy: Sequence[FloatArray]) -> PolicyEvaluation:
    """Evaluate a full-state expansion of a history-suffix policy exactly."""

    validate_policy(mdp, policy)
    q_values: list[FloatArray] = [np.empty((0, 0)) for _ in range(mdp.horizon)]
    values: list[FloatArray] = [np.empty(0) for _ in range(mdp.horizon + 1)]
    values[mdp.horizon] = np.zeros(mdp.n_states(mdp.horizon + 1), dtype=float)

    for t in range(mdp.horizon - 1, -1, -1):
        q_t = mdp.rewards[t] + np.einsum(
            "san,n->sa", mdp.transitions[t], values[t + 1], optimize=True
        )
        q_values[t] = q_t
        values[t] = np.einsum("sa,sa->s", policy[t], q_t, optimize=True)

    initial_value = float(mdp.initial_distribution @ values[0])
    return PolicyEvaluation(q=q_values, v=values, initial_value=initial_value)


def uniform_policy(mdp: HistoryMDP) -> Policy:
    """Return the stagewise uniform behavior policy."""

    policy: Policy = []
    for h in range(1, mdp.horizon + 1):
        n_states = mdp.n_states(h)
        n_actions = mdp.n_actions(h)
        policy.append(np.full((n_states, n_actions), 1.0 / n_actions, dtype=float))
    return policy


def state_occupancies(mdp: HistoryMDP, policy: Sequence[FloatArray]) -> list[FloatArray]:
    """Compute stagewise full-state occupancies under ``policy``."""

    validate_policy(mdp, policy)
    occupancies: list[FloatArray] = [np.empty(0) for _ in range(mdp.horizon + 1)]
    occupancies[0] = mdp.initial_distribution.copy()
    for t in range(mdp.horizon):
        state_action = occupancies[t][:, None] * policy[t]
        occupancies[t + 1] = np.einsum(
            "sa,san->n", state_action, mdp.transitions[t], optimize=True
        )
    return occupancies
