"""Decision-relevant ambiguity and high-confidence memory certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from .mdp import FloatArray, History, HistoryMDP, Policy


@dataclass(frozen=True)
class AmbiguityResult:
    """Decision ambiguity and the suffix policy attaining its local bound."""

    memory: int
    stage_ambiguities: tuple[float, ...]
    total_ambiguity: float
    policy: Policy
    cell_policies: tuple[Mapping[History, FloatArray], ...]


@dataclass(frozen=True)
class MemorySelection:
    selected_memory: int
    certificate: float
    certified: bool
    result: AmbiguityResult


@dataclass(frozen=True)
class WeightedAmbiguityResult:
    """Occupancy-weighted ambiguity and its suffix policy."""

    memory: int
    stage_costs: tuple[float, ...]
    total_cost: float
    policy: Policy
    cell_policies: tuple[Mapping[History, FloatArray], ...]


def _solve_minimax_mixture(costs: FloatArray) -> tuple[FloatArray, float]:
    """Solve ``min_p max_i <costs[i], p>`` over the probability simplex."""

    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("costs must be a nonempty matrix")
    n_actions = costs.shape[1]
    if n_actions == 1:
        return np.ones(1, dtype=float), max(float(costs[:, 0].max()), 0.0)

    # With two actions the objective is the upper envelope of affine lines
    # b_i + s_i p, where p is the probability of action zero.  Constructing
    # that envelope is exact and O(n log n), avoiding a generic LP inside the
    # large stress tests.
    if n_actions == 2:
        slopes = costs[:, 0] - costs[:, 1]
        intercepts = costs[:, 1]

        # For duplicate slopes only the largest intercept can appear on the
        # upper envelope.
        order = np.lexsort((-intercepts, slopes))
        unique_lines: list[tuple[float, float]] = []
        for index in order:
            slope = float(slopes[index])
            intercept = float(intercepts[index])
            if unique_lines and abs(slope - unique_lines[-1][0]) <= 1e-14:
                continue
            unique_lines.append((slope, intercept))

        hull: list[tuple[float, float]] = []
        starts: list[float] = []
        for slope, intercept in unique_lines:
            start = -np.inf
            while hull:
                previous_slope, previous_intercept = hull[-1]
                start = (previous_intercept - intercept) / (slope - previous_slope)
                if start > starts[-1] + 1e-14:
                    break
                hull.pop()
                starts.pop()
            if not hull:
                start = -np.inf
            hull.append((slope, intercept))
            starts.append(float(start))

        candidates = [0.0, 1.0]
        candidates.extend(start for start in starts[1:] if 0.0 <= start <= 1.0)
        candidate_array = np.asarray(candidates, dtype=float)
        values = np.max(
            intercepts[:, None] + slopes[:, None] * candidate_array[None, :],
            axis=0,
        )
        best = int(np.argmin(values))
        p0 = float(candidate_array[best])
        mixture = np.array([p0, 1.0 - p0], dtype=float)
        return mixture, max(float(values[best]), 0.0)

    objective = np.zeros(n_actions + 1, dtype=float)
    objective[-1] = 1.0

    # costs @ p <= t
    a_ub = np.hstack([costs, -np.ones((costs.shape[0], 1), dtype=float)])
    b_ub = np.zeros(costs.shape[0], dtype=float)
    a_eq = np.zeros((1, n_actions + 1), dtype=float)
    a_eq[0, :n_actions] = 1.0
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, 1.0)] * n_actions + [(0.0, None)]

    solution = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"cellwise minimax LP failed: {solution.message}")

    mixture = np.asarray(solution.x[:n_actions], dtype=float)
    mixture = np.maximum(mixture, 0.0)
    mixture /= mixture.sum()
    value = float(np.max(costs @ mixture))
    return mixture, max(value, 0.0)


def optimal_advantages(q_values: list[FloatArray]) -> list[FloatArray]:
    """Return ``A*(x,a)=V*(x)-Q*(x,a)`` for every stage."""

    return [q.max(axis=1, keepdims=True) - q for q in q_values]


def upper_advantages_from_q_intervals(
    lower_q: list[FloatArray], upper_q: list[FloatArray]
) -> list[FloatArray]:
    """Convert simultaneous Q intervals into upper optimal-advantage bounds.

    For action ``a``,

    ``A*(x,a) = max_{a'} [Q*(x,a') - Q*(x,a)]``.

    The self-comparison is exactly zero, so it is kept as zero instead of the
    looser ``U(x,a)-L(x,a)``.  This makes the certificate exact at stages with
    only one available action and materially tightens it when Q intervals are
    wide but strongly correlated.
    """

    if len(lower_q) != len(upper_q):
        raise ValueError("lower_q and upper_q must have the same number of stages")
    upper_advantages: list[FloatArray] = []
    for lower, upper in zip(lower_q, upper_q, strict=True):
        if lower.shape != upper.shape:
            raise ValueError("lower and upper Q arrays must have matching shapes")
        if np.any(lower > upper + 1e-10):
            raise ValueError("lower Q bounds cannot exceed upper Q bounds")
        n_states, n_actions = lower.shape
        bounds = np.zeros_like(lower)
        if n_actions > 1:
            for action in range(n_actions):
                competitors = [a for a in range(n_actions) if a != action]
                pairwise = upper[:, competitors] - lower[:, [action]]
                bounds[:, action] = np.maximum(0.0, pairwise.max(axis=1))
        upper_advantages.append(bounds)
    return upper_advantages


def _ambiguity_from_costs(
    mdp: HistoryMDP, memory: int, costs_by_stage: list[FloatArray]
) -> AmbiguityResult:
    if memory <= 0:
        raise ValueError("memory must be positive")
    if len(costs_by_stage) != mdp.horizon:
        raise ValueError("one cost matrix is required per decision stage")

    policy: Policy = []
    stage_values: list[float] = []
    all_cell_policies: list[Mapping[History, FloatArray]] = []

    for h, costs in enumerate(costs_by_stage, start=1):
        expected_shape = (mdp.n_states(h), mdp.n_actions(h))
        if costs.shape != expected_shape:
            raise ValueError(
                f"stage-{h} cost matrix has shape {costs.shape}, expected {expected_shape}"
            )
        groups = mdp.suffix_groups(h, memory)
        expanded_policy = np.zeros(expected_shape, dtype=float)
        cell_policies: dict[History, FloatArray] = {}
        cell_values: list[float] = []

        for key, state_indices in groups.items():
            mixture, value = _solve_minimax_mixture(costs[state_indices, :])
            expanded_policy[state_indices, :] = mixture
            cell_policies[key] = mixture
            cell_values.append(value)

        policy.append(expanded_policy)
        all_cell_policies.append(cell_policies)
        stage_values.append(max(cell_values, default=0.0))

    return AmbiguityResult(
        memory=memory,
        stage_ambiguities=tuple(stage_values),
        total_ambiguity=float(sum(stage_values)),
        policy=policy,
        cell_policies=tuple(all_cell_policies),
    )


def compute_decision_ambiguity(
    mdp: HistoryMDP, q_values: list[FloatArray], memory: int
) -> AmbiguityResult:
    """Compute the population decision ambiguity ``G_m``."""

    return _ambiguity_from_costs(mdp, memory, optimal_advantages(q_values))


def compute_robust_decision_ambiguity(
    mdp: HistoryMDP,
    lower_q: list[FloatArray],
    upper_q: list[FloatArray],
    memory: int,
) -> AmbiguityResult:
    """Compute the high-confidence ambiguity certificate ``Gbar_m``."""

    costs = upper_advantages_from_q_intervals(lower_q, upper_q)
    return _ambiguity_from_costs(mdp, memory, costs)


def select_memory(
    results: list[AmbiguityResult], tolerance: float
) -> MemorySelection:
    """Select the shortest memory whose certificate is at most ``tolerance``.

    If none is certified, the longest supplied memory is returned with
    ``certified=False``.  This explicit abstention is important in low-coverage
    offline datasets.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if not results:
        raise ValueError("at least one ambiguity result is required")
    ordered = sorted(results, key=lambda result: result.memory)
    for result in ordered:
        if result.total_ambiguity <= tolerance + 1e-12:
            return MemorySelection(
                selected_memory=result.memory,
                certificate=result.total_ambiguity,
                certified=True,
                result=result,
            )
    fallback = ordered[-1]
    return MemorySelection(
        selected_memory=fallback.memory,
        certificate=fallback.total_ambiguity,
        certified=False,
        result=fallback,
    )


def uniform_value_loss(
    optimal_values: list[FloatArray], policy_values: list[FloatArray]
) -> float:
    """Maximum stage-state value loss used by the population theorem."""

    if len(optimal_values) != len(policy_values):
        raise ValueError("value sequences must have the same length")
    return float(
        max(
            np.max(v_star - v_policy)
            for v_star, v_policy in zip(optimal_values, policy_values, strict=True)
        )
    )

def _weighted_ambiguity_from_costs(
    mdp: HistoryMDP,
    memory: int,
    costs_by_stage: list[FloatArray],
    reference_occupancies: list[FloatArray],
) -> WeightedAmbiguityResult:
    """Minimize reference-occupancy-weighted one-step costs in each cell."""

    if memory <= 0:
        raise ValueError("memory must be positive")
    if len(costs_by_stage) != mdp.horizon:
        raise ValueError("one cost matrix is required per decision stage")
    if len(reference_occupancies) < mdp.horizon:
        raise ValueError("one reference occupancy is required per decision stage")

    policy: Policy = []
    stage_costs: list[float] = []
    all_cell_policies: list[Mapping[History, FloatArray]] = []

    for h, (costs, occupancy) in enumerate(
        zip(costs_by_stage, reference_occupancies, strict=False), start=1
    ):
        if h > mdp.horizon:
            break
        expected_shape = (mdp.n_states(h), mdp.n_actions(h))
        if costs.shape != expected_shape:
            raise ValueError(
                f"stage-{h} cost matrix has shape {costs.shape}, expected {expected_shape}"
            )
        occupancy = np.asarray(occupancy, dtype=float)
        if occupancy.shape != (mdp.n_states(h),):
            raise ValueError(f"stage-{h} occupancy has incompatible shape")
        if np.any(occupancy < -1e-12) or not np.isclose(occupancy.sum(), 1.0):
            raise ValueError("each reference occupancy must be a probability vector")

        groups = mdp.suffix_groups(h, memory)
        expanded_policy = np.zeros(expected_shape, dtype=float)
        cell_policies: dict[History, FloatArray] = {}
        stage_cost = 0.0
        for key, state_indices in groups.items():
            indices = np.asarray(state_indices, dtype=int)
            action_costs = occupancy[indices] @ costs[indices, :]
            best_value = float(np.min(action_costs))
            best_actions = np.flatnonzero(
                np.isclose(action_costs, best_value, atol=1e-12, rtol=1e-10)
            )
            mixture = np.zeros(mdp.n_actions(h), dtype=float)
            mixture[best_actions] = 1.0 / len(best_actions)
            expanded_policy[indices, :] = mixture
            cell_policies[key] = mixture
            stage_cost += max(best_value, 0.0)

        policy.append(expanded_policy)
        all_cell_policies.append(cell_policies)
        stage_costs.append(float(stage_cost))

    return WeightedAmbiguityResult(
        memory=memory,
        stage_costs=tuple(stage_costs),
        total_cost=float(sum(stage_costs)),
        policy=policy,
        cell_policies=tuple(all_cell_policies),
    )


def compute_weighted_decision_ambiguity(
    mdp: HistoryMDP,
    q_values: list[FloatArray],
    memory: int,
    reference_occupancies: list[FloatArray],
) -> WeightedAmbiguityResult:
    """Compute the population occupancy-weighted ambiguity ``G_m^nu``."""

    return _weighted_ambiguity_from_costs(
        mdp, memory, optimal_advantages(q_values), reference_occupancies
    )


def compute_robust_weighted_decision_ambiguity(
    mdp: HistoryMDP,
    lower_q: list[FloatArray],
    upper_q: list[FloatArray],
    memory: int,
    reference_occupancies: list[FloatArray],
) -> WeightedAmbiguityResult:
    """Compute an interval-robust occupancy-weighted ambiguity."""

    costs = upper_advantages_from_q_intervals(lower_q, upper_q)
    return _weighted_ambiguity_from_costs(
        mdp, memory, costs, reference_occupancies
    )


def deployment_concentrability(
    deployment_occupancies: list[FloatArray],
    reference_occupancies: list[FloatArray],
    *,
    atol: float = 1e-12,
) -> tuple[float, ...]:
    """Return minimal stagewise coefficients ``C_h`` with ``d_h <= C_h nu_h``."""

    if len(deployment_occupancies) < len(reference_occupancies):
        raise ValueError("deployment occupancies are missing stages")
    coefficients: list[float] = []
    for deployment, reference in zip(
        deployment_occupancies, reference_occupancies, strict=False
    ):
        deployment = np.asarray(deployment, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if deployment.shape != reference.shape:
            raise ValueError("deployment and reference occupancies must match")
        unsupported = (deployment > atol) & (reference <= atol)
        if np.any(unsupported):
            coefficients.append(float("inf"))
            continue
        supported = reference > atol
        coefficient = (
            float(np.max(deployment[supported] / reference[supported]))
            if np.any(supported)
            else 0.0
        )
        coefficients.append(max(coefficient, 0.0))
    return tuple(coefficients)


def weighted_value_certificate(
    result: WeightedAmbiguityResult,
    concentrability: tuple[float, ...] | list[float],
    *,
    occupancy_radii: list[float] | tuple[float, ...] | None = None,
    stage_cost_bounds: list[float] | tuple[float, ...] | None = None,
) -> float:
    """Evaluate the weighted value bound, optionally correcting empirical occupancies."""

    coefficients = np.asarray(concentrability, dtype=float)
    stage_costs = np.asarray(result.stage_costs, dtype=float)
    if coefficients.shape != stage_costs.shape:
        raise ValueError("one concentrability coefficient is required per stage")
    corrected = stage_costs.copy()
    if occupancy_radii is not None or stage_cost_bounds is not None:
        if occupancy_radii is None or stage_cost_bounds is None:
            raise ValueError("occupancy radii and stage cost bounds must be supplied together")
        radii = np.asarray(occupancy_radii, dtype=float)
        bounds = np.asarray(stage_cost_bounds, dtype=float)
        if radii.shape != stage_costs.shape or bounds.shape != stage_costs.shape:
            raise ValueError("occupancy corrections must have one entry per stage")
        corrected += 0.5 * radii * bounds
    return float(np.sum(coefficients * corrected))

