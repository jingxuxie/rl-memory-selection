"""Tabular offline-data simulation and simultaneous Q-value confidence bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FloatArray, HistoryMDP, Policy, validate_policy


@dataclass(frozen=True)
class TabularDataset:
    n_episodes: int
    counts: list[np.ndarray]
    reward_sums: list[FloatArray]
    transition_counts: list[np.ndarray]


@dataclass(frozen=True)
class QConfidenceIntervals:
    lower_q: list[FloatArray]
    upper_q: list[FloatArray]
    lower_v: list[FloatArray]
    upper_v: list[FloatArray]
    reward_radii: list[FloatArray]
    transition_radii: list[FloatArray]


def sample_tabular_dataset(
    mdp: HistoryMDP,
    behavior_policy: Policy,
    n_episodes: int,
    *,
    seed: int,
) -> TabularDataset:
    """Sample independent episodes and retain sufficient tabular statistics.

    Episodes are simulated in vectorized stage batches, so even the full
    finite-sample sweep remains a seconds-to-minutes CPU experiment.
    """

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    validate_policy(mdp, behavior_policy)
    rng = np.random.default_rng(seed)

    counts: list[np.ndarray] = []
    reward_sums: list[FloatArray] = []
    transition_counts: list[np.ndarray] = []
    for h in range(1, mdp.horizon + 1):
        counts.append(np.zeros((mdp.n_states(h), mdp.n_actions(h)), dtype=np.int64))
        reward_sums.append(
            np.zeros((mdp.n_states(h), mdp.n_actions(h)), dtype=float)
        )
        transition_counts.append(
            np.zeros(
                (mdp.n_states(h), mdp.n_actions(h), mdp.n_states(h + 1)),
                dtype=np.int64,
            )
        )

    states = rng.choice(
        mdp.n_states(1), size=n_episodes, p=mdp.initial_distribution
    ).astype(np.int64)
    for t in range(mdp.horizon):
        action_probs = behavior_policy[t][states]
        action_uniforms = rng.random(n_episodes)
        actions = (
            action_uniforms[:, None] > np.cumsum(action_probs, axis=1)
        ).sum(axis=1)
        actions = np.minimum(actions, mdp.n_actions(t + 1) - 1).astype(np.int64)

        reward_means = mdp.rewards[t][states, actions]
        rewards = rng.binomial(1, reward_means).astype(float)

        next_probs = mdp.transitions[t][states, actions]
        next_uniforms = rng.random(n_episodes)
        next_states = (
            next_uniforms[:, None] > np.cumsum(next_probs, axis=1)
        ).sum(axis=1)
        next_states = np.minimum(next_states, mdp.n_states(t + 2) - 1).astype(
            np.int64
        )

        np.add.at(counts[t], (states, actions), 1)
        np.add.at(reward_sums[t], (states, actions), rewards)
        np.add.at(transition_counts[t], (states, actions, next_states), 1)
        states = next_states

    return TabularDataset(
        n_episodes=n_episodes,
        counts=counts,
        reward_sums=reward_sums,
        transition_counts=transition_counts,
    )


def _span(values: FloatArray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.max() - values.min())


def tabular_q_confidence_intervals(
    mdp: HistoryMDP,
    dataset: TabularDataset,
    *,
    delta: float = 0.05,
) -> QConfidenceIntervals:
    """Construct simultaneous model-based intervals for ``Q*``.

    Reward means use a union-bounded Hoeffding radius.  Transition kernels use
    a Weissman-style L1 radius, propagated through optimistic and pessimistic
    dynamic programming.  The construction is intentionally conservative but
    fully tabular and transparent.
    """

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if dataset.n_episodes <= 0:
        raise ValueError("dataset must contain at least one episode")
    if len(dataset.counts) != mdp.horizon:
        raise ValueError("dataset is incompatible with the MDP horizon")

    total_pairs = sum(
        mdp.n_states(h) * mdp.n_actions(h) for h in range(1, mdp.horizon + 1)
    )
    reward_log = np.log(4.0 * total_pairs / delta)

    reward_radii: list[FloatArray] = []
    transition_radii: list[FloatArray] = []
    empirical_rewards: list[FloatArray] = []
    empirical_transitions: list[np.ndarray] = []

    for t in range(mdp.horizon):
        counts = dataset.counts[t]
        safe_counts = np.maximum(counts, 1)
        reward_hat = dataset.reward_sums[t] / safe_counts
        reward_radius = np.sqrt(reward_log / (2.0 * safe_counts))
        reward_radius = np.where(counts > 0, reward_radius, 1.0)

        n_next = mdp.n_states(t + 2)
        # Weissman et al.: P(||P_hat-P||_1 >= eps) is bounded by an
        # exponential term with a 2^S prefactor.  We use a simple union bound.
        prefactor_log = max(n_next * np.log(2.0), 0.0)
        transition_log = np.log(2.0 * total_pairs / delta) + prefactor_log
        transition_radius = np.sqrt(2.0 * transition_log / safe_counts)
        transition_radius = np.where(counts > 0, transition_radius, 2.0)
        transition_radius = np.minimum(transition_radius, 2.0)

        transition_hat = dataset.transition_counts[t] / safe_counts[:, :, None]
        missing = counts == 0
        if np.any(missing):
            transition_hat[missing] = 1.0 / n_next

        reward_radii.append(np.asarray(reward_radius, dtype=float))
        transition_radii.append(np.asarray(transition_radius, dtype=float))
        empirical_rewards.append(np.asarray(reward_hat, dtype=float))
        empirical_transitions.append(np.asarray(transition_hat, dtype=float))

    lower_q: list[FloatArray] = [np.empty((0, 0)) for _ in range(mdp.horizon)]
    upper_q: list[FloatArray] = [np.empty((0, 0)) for _ in range(mdp.horizon)]
    lower_v: list[FloatArray] = [np.empty(0) for _ in range(mdp.horizon + 1)]
    upper_v: list[FloatArray] = [np.empty(0) for _ in range(mdp.horizon + 1)]
    lower_v[mdp.horizon] = np.zeros(mdp.n_states(mdp.horizon + 1), dtype=float)
    upper_v[mdp.horizon] = np.zeros(mdp.n_states(mdp.horizon + 1), dtype=float)

    for t in range(mdp.horizon - 1, -1, -1):
        remaining_return = float(mdp.horizon - t)
        reward_lower = np.clip(
            empirical_rewards[t] - reward_radii[t], 0.0, 1.0
        )
        reward_upper = np.clip(
            empirical_rewards[t] + reward_radii[t], 0.0, 1.0
        )

        transition_lower_correction = (
            0.5 * transition_radii[t] * _span(lower_v[t + 1])
        )
        transition_upper_correction = (
            0.5 * transition_radii[t] * _span(upper_v[t + 1])
        )
        lower_expectation = np.einsum(
            "san,n->sa", empirical_transitions[t], lower_v[t + 1], optimize=True
        ) - transition_lower_correction
        upper_expectation = np.einsum(
            "san,n->sa", empirical_transitions[t], upper_v[t + 1], optimize=True
        ) + transition_upper_correction

        lower_q[t] = np.clip(reward_lower + lower_expectation, 0.0, remaining_return)
        upper_q[t] = np.clip(reward_upper + upper_expectation, 0.0, remaining_return)
        lower_v[t] = lower_q[t].max(axis=1)
        upper_v[t] = upper_q[t].max(axis=1)

    return QConfidenceIntervals(
        lower_q=lower_q,
        upper_q=upper_q,
        lower_v=lower_v,
        upper_v=upper_v,
        reward_radii=reward_radii,
        transition_radii=transition_radii,
    )
