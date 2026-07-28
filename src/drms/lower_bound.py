"""Information quantities for the two-point memory-selection lower bound."""

from __future__ import annotations

import math


def bernoulli_kl(p: float, q: float) -> float:
    """Return ``KL(Ber(p) || Ber(q))`` with natural logarithms."""

    if not 0.0 < p < 1.0 or not 0.0 < q < 1.0:
        raise ValueError("Bernoulli parameters must lie strictly inside (0, 1)")
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def memory_order_pair_kl(cue_probability: float, action_gap: float) -> float:
    """One-episode KL for the symmetric lower-bound pair under uniform actions."""

    if not 0.0 < cue_probability < 1.0:
        raise ValueError("cue_probability must lie strictly inside (0, 1)")
    if not 0.0 < action_gap <= 0.5:
        raise ValueError("action_gap must lie in (0, 0.5]")
    high = 0.5 + 0.5 * action_gap
    low = 0.5 - 0.5 * action_gap
    return 0.5 * cue_probability * (
        bernoulli_kl(high, low) + bernoulli_kl(low, high)
    )


def le_cam_episode_lower_bound(
    *, cue_probability: float, action_gap: float, delta: float
) -> float:
    """Bretagnolle--Huber lower bound for error at most ``delta`` per model.

    Any selector whose probability of choosing the wrong memory order is at
    most ``delta`` in both members of the two-point construction must use at
    least the returned number of independent episodes.
    """

    if not 0.0 < delta < 0.25:
        raise ValueError("delta must lie in (0, 0.25)")
    per_episode_kl = memory_order_pair_kl(cue_probability, action_gap)
    return math.log(1.0 / (4.0 * delta)) / per_episode_kl
