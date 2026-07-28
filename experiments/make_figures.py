#!/usr/bin/env python3
"""Create paper figures from the lightweight CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def population_figure() -> None:
    data = pd.read_csv(RESULTS / "population_results.csv")
    data = data[data["horizon"] == 4]
    tasks = [
        ("delayed_cue", "Delayed cue (decision-only memory)"),
        ("nuisance_memory", "Nuisance cue (prediction-only memory)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
    for axis, (task, title) in zip(axes, tasks, strict=True):
        subset = data[data["task"] == task].sort_values("memory")
        axis.plot(
            subset["memory"],
            subset["decision_ambiguity"],
            marker="o",
            label="Decision ambiguity $G_m$",
        )
        axis.plot(
            subset["memory"],
            subset["predictive_residual_bits"],
            marker="s",
            linestyle="--",
            label="Predictive residual (bits)",
        )
        axis.set_title(title, fontsize=10.5)
        axis.set_xlabel("History length $m$")
        axis.set_xticks(subset["memory"])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Diagnostic value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=1.1)
    save_figure(fig, "decision_vs_prediction")


def finite_sample_figure() -> None:
    data = pd.read_csv(RESULTS / "finite_sample_summary.csv")
    fig, axis = plt.subplots(figsize=(4.6, 3.1))
    for cue_probability, subset in data.groupby("cue_probability", sort=False):
        subset = subset.sort_values("n_episodes")
        axis.plot(
            subset["n_episodes"],
            subset["correct_memory_rate"],
            marker="o",
            label=f"rare-context prob. $p={cue_probability:g}$",
        )
    axis.set_xscale("log")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Offline episodes")
    axis.set_ylabel("Probability of certifying $m=4$")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "finite_sample_recovery")


def soft_relevance_figure() -> None:
    data = pd.read_csv(RESULTS / "soft_relevance_results.csv")
    fig, axis = plt.subplots(figsize=(4.6, 3.1))
    for tolerance, subset in data.groupby("tolerance", sort=True):
        subset = subset.sort_values("reward_gap")
        axis.step(
            subset["reward_gap"],
            subset["selected_memory"],
            where="post",
            label=f"tolerance $\\epsilon={tolerance:g}$",
        )
    axis.set_xlabel("Decision gap $\\gamma$")
    axis.set_ylabel("Shortest certified memory")
    axis.set_yticks([1, 5])
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "soft_relevance")



def scaling_figure() -> None:
    summary = pd.read_csv(RESULTS / "scaling_summary.csv")
    thresholds = pd.read_csv(RESULTS / "scaling_thresholds.csv").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))

    gaps = sorted(float(value) for value in summary["action_gap"].unique())
    marker_cycle = ["o", "s", "^", "D", "v"]
    markers = {gap: marker_cycle[i % len(marker_cycle)] for i, gap in enumerate(gaps)}
    for (gap, probability), subset in summary.groupby(
        ["action_gap", "cue_probability"], sort=True
    ):
        subset = subset.sort_values("information_scale")
        axes[0].plot(
            subset["information_scale"],
            subset["recovery_rate"],
            marker=markers[float(gap)],
            markersize=3,
            linewidth=1,
            alpha=0.72,
        )
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel(r"Effective information $np\gamma^2$")
    axes[0].set_ylabel("Memory-recovery probability")
    axes[0].set_title("Coverage--gap collapse", fontsize=10.5)
    axes[0].grid(alpha=0.25)

    for gap, subset in thresholds.groupby("action_gap", sort=True):
        axes[1].scatter(
            subset["inverse_information"],
            subset["n50"],
            marker=markers[float(gap)],
            label=rf"$\gamma={gap:g}$",
        )
    x = thresholds["inverse_information"].to_numpy(dtype=float)
    y = thresholds["n50"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    grid = np.geomspace(x.min(), x.max(), 200)
    axes[1].plot(
        grid,
        np.exp(intercept) * grid**slope,
        linestyle="--",
        label=rf"fit slope ${slope:.2f}$",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Inverse information $1/(p\gamma^2)$")
    axes[1].set_ylabel(r"Episodes for 50\% recovery")
    axes[1].set_title("Empirical sample complexity", fontsize=10.5)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "coverage_gap_scaling")



def weighted_figure() -> None:
    population = pd.read_csv(RESULTS / "weighted_population.csv")
    finite = pd.read_csv(RESULTS / "weighted_finite_sample_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))

    axes[0].plot(
        population["cue_probability"],
        population["uniform_ambiguity"],
        linestyle="--",
        label="Uniform $G_1$",
    )
    axes[0].plot(
        population["cue_probability"],
        population["weighted_ambiguity"],
        label="Weighted $G_1^\\nu$",
    )
    axes[0].axhline(
        float(population["tolerance"].iloc[0]),
        linestyle=":",
        label="Loss budget $\\epsilon$",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Rare conflicting-context probability $p$")
    axes[0].set_ylabel("Population certificate")
    axes[0].set_title("Uniform vs. deployment-weighted", fontsize=10.5)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)

    for cue_probability, subset in finite.groupby("cue_probability", sort=True):
        subset = subset.sort_values("n_episodes")
        axes[1].plot(
            subset["n_episodes"],
            subset["weighted_short_rate"],
            marker="o",
            label=rf"$p={cue_probability:g}$",
        )
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Offline episodes")
    axes[1].set_ylabel("Probability of certifying $m=1$")
    axes[1].set_title("Finite-sample weighted selection", fontsize=10.5)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2)

    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "weighted_memory")


def random_stress_figure() -> None:
    data = pd.read_csv(RESULTS / "random_stress_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))

    axes[0].scatter(
        data["uniform_ambiguity"],
        data["uniform_loss"],
        s=9,
        alpha=0.45,
    )
    upper = max(data["uniform_ambiguity"].max(), data["uniform_loss"].max())
    axes[0].plot([0, upper], [0, upper], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Decision ambiguity $G_m$")
    axes[0].set_ylabel("Uniform value loss")
    axes[0].set_title("Population sandwich", fontsize=10.5)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        data["robust_weighted_certificate"],
        data["robust_weighted_loss"],
        s=9,
        alpha=0.45,
    )
    upper = max(
        data["robust_weighted_certificate"].max(),
        data["robust_weighted_loss"].max(),
    )
    axes[1].plot([0, upper], [0, upper], linestyle="--", linewidth=1)
    axes[1].set_xlabel("Robust weighted certificate")
    axes[1].set_ylabel("Deployment value loss")
    axes[1].set_title("Weighted certificate", fontsize=10.5)
    axes[1].grid(alpha=0.25)

    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "random_stress")


def main() -> None:
    population_figure()
    finite_sample_figure()
    soft_relevance_figure()
    scaling_figure()
    weighted_figure()
    random_stress_figure()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
