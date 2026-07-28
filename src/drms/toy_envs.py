"""Exact history-MDP counterexamples for decision- versus prediction-memory."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .mdp import History, HistoryMDP


def _build_binary_cue_process(
    *,
    horizon: int,
    cue_probability: float,
    final_reward: Callable[[int, int], float],
    terminal_observation: Callable[[int], str],
) -> HistoryMDP:
    if horizon < 2:
        raise ValueError("horizon must be at least two")
    if not 0.0 < cue_probability < 1.0:
        raise ValueError("cue_probability must lie strictly between zero and one")

    # Only the final stage presents a genuine decision.  Earlier stages use a
    # single dummy action, avoiding an artificial exponential collection of
    # action histories while preserving the temporal suffix structure.
    action_counts = [1] * (horizon - 1) + [2]
    histories: list[list[History]] = [[("cue-0",), ("cue-1",)]]
    latent_bits: list[list[int]] = [[0, 1]]
    rewards: list[np.ndarray] = []
    transitions: list[np.ndarray] = []

    for h in range(1, horizon + 1):
        current_histories = histories[h - 1]
        current_bits = latent_bits[h - 1]
        n_actions = action_counts[h - 1]
        reward = np.zeros((len(current_histories), n_actions), dtype=float)
        if h == horizon:
            for state_index, bit in enumerate(current_bits):
                for action in range(n_actions):
                    reward[state_index, action] = final_reward(bit, action)

        next_histories: list[History] = []
        next_bits: list[int] = []
        next_lookup: dict[History, int] = {}
        next_indices = np.zeros((len(current_histories), n_actions), dtype=int)

        for state_index, (history, bit) in enumerate(zip(current_histories, current_bits, strict=True)):
            for action in range(n_actions):
                if h < horizon:
                    next_observation = "blank"
                else:
                    next_observation = terminal_observation(bit)
                next_history = history + (action, next_observation)
                if next_history not in next_lookup:
                    next_lookup[next_history] = len(next_histories)
                    next_histories.append(next_history)
                    next_bits.append(bit)
                next_indices[state_index, action] = next_lookup[next_history]

        transition = np.zeros(
            (len(current_histories), n_actions, len(next_histories)), dtype=float
        )
        for state_index in range(len(current_histories)):
            for action in range(n_actions):
                transition[state_index, action, next_indices[state_index, action]] = 1.0

        rewards.append(reward)
        transitions.append(transition)
        histories.append(next_histories)
        latent_bits.append(next_bits)

    return HistoryMDP(
        horizon=horizon,
        histories=histories,
        rewards=rewards,
        transitions=transitions,
        initial_distribution=np.array(
            [1.0 - cue_probability, cue_probability], dtype=float
        ),
    )


def make_delayed_cue_mdp(
    *, horizon: int = 4, reward_gap: float = 1.0, cue_probability: float = 0.5
) -> HistoryMDP:
    """A task where memory affects action choice but not next-observation prediction.

    The bit shown at stage one determines the rewarding action at the final
    stage.  All intervening and terminal observations are constant.  Hence the
    shortest decision-sufficient memory is ``horizon``, while a one-step
    predictive diagnostic sees no gain from retaining the cue.
    """

    if not 0.0 <= reward_gap <= 1.0:
        raise ValueError("reward_gap must lie in [0, 1]")
    return _build_binary_cue_process(
        horizon=horizon,
        cue_probability=cue_probability,
        final_reward=lambda bit, action: reward_gap if action == bit else 0.0,
        terminal_observation=lambda _bit: "terminal",
    )


def make_nuisance_memory_mdp(
    *, horizon: int = 4, reward_gap: float = 1.0, cue_probability: float = 0.5
) -> HistoryMDP:
    """A task where old history predicts observations but not optimal actions.

    The stage-one bit reappears in the terminal observation, making it useful
    for prediction at the final stage.  Nevertheless action zero is optimal for
    both bit values, so memory one is already decision sufficient.
    """

    if not 0.0 <= reward_gap <= 1.0:
        raise ValueError("reward_gap must lie in [0, 1]")
    return _build_binary_cue_process(
        horizon=horizon,
        cue_probability=cue_probability,
        final_reward=lambda _bit, action: reward_gap if action == 0 else 0.0,
        terminal_observation=lambda bit: f"reveal-{bit}",
    )


def make_memory_order_lower_bound_pair(
    *, cue_probability: float = 0.1, action_gap: float = 0.2
) -> tuple[HistoryMDP, HistoryMDP]:
    """Return the two environments used in the memory-order lower bound.

    Both environments expose a cue at stage one and a blank observation at the
    final decision.  In ``model_short``, action zero is optimal after either
    cue, so one-step memory suffices.  In ``model_long``, the optimal action
    flips only after the rare cue, so two-step memory is necessary.  Under a
    uniform behavior policy the one-episode KL divergence is proportional to
    ``cue_probability * action_gap**2``.
    """

    if not 0.0 < action_gap <= 0.5:
        raise ValueError("action_gap must lie in (0, 0.5] for the stated KL bound")
    # Symmetric Bernoulli means keep both models away from the boundary while
    # making the difference between the two actions exactly ``action_gap``.
    high = 0.5 + 0.5 * action_gap
    low = 0.5 - 0.5 * action_gap

    model_short = _build_binary_cue_process(
        horizon=2,
        cue_probability=cue_probability,
        final_reward=lambda _bit, action: high if action == 0 else low,
        terminal_observation=lambda _bit: "terminal",
    )
    model_long = _build_binary_cue_process(
        horizon=2,
        cue_probability=cue_probability,
        final_reward=lambda bit, action: high if action == bit else low,
        terminal_observation=lambda _bit: "terminal",
    )
    return model_short, model_long


def make_random_history_mdp(
    *,
    horizon: int = 3,
    seed: int = 0,
    n_observations: int = 2,
    n_actions: int = 2,
) -> HistoryMDP:
    """Generate a small fully supported random observable-history MDP.

    Every action-observation extension is represented explicitly, so suffix
    partitions can alias histories even though the complete history is Markov.
    Rewards and observation kernels vary across complete histories.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_observations <= 0 or n_actions <= 0:
        raise ValueError("observation and action counts must be positive")
    rng = np.random.default_rng(seed)

    histories: list[list[History]] = [
        [(f"o{observation}",) for observation in range(n_observations)]
    ]
    rewards: list[np.ndarray] = []
    transitions: list[np.ndarray] = []

    for _h in range(1, horizon + 1):
        current = histories[-1]
        next_histories: list[History] = []
        next_lookup: dict[History, int] = {}
        for history in current:
            for action in range(n_actions):
                for observation in range(n_observations):
                    next_history = history + (action, f"o{observation}")
                    if next_history not in next_lookup:
                        next_lookup[next_history] = len(next_histories)
                        next_histories.append(next_history)

        reward = rng.beta(2.0, 2.0, size=(len(current), n_actions))
        transition = np.zeros(
            (len(current), n_actions, len(next_histories)), dtype=float
        )
        for state_index, history in enumerate(current):
            for action in range(n_actions):
                probabilities = rng.dirichlet(np.ones(n_observations))
                for observation, probability in enumerate(probabilities):
                    next_history = history + (action, f"o{observation}")
                    transition[
                        state_index, action, next_lookup[next_history]
                    ] = probability

        rewards.append(reward)
        transitions.append(transition)
        histories.append(next_histories)

    initial_distribution = rng.dirichlet(np.ones(n_observations))
    return HistoryMDP(
        horizon=horizon,
        histories=histories,
        rewards=rewards,
        transitions=transitions,
        initial_distribution=initial_distribution,
    )
