# Research roadmap and claim audit

## Current paper thesis

A history window should be retained only when histories collapsed by that
window require meaningfully different actions. Next-observation prediction can
over-prescribe memory because of predictive nuisance variables and
under-prescribe it when an old cue changes the rewarding action without
changing later observations. Exact reward--transition Markov sufficiency is
stronger and implies decision sufficiency. DRMS directly certifies the shortest
suffix whose induced policy has small full-history optimality loss.

## Completed milestone 1: population theory

- [x] Full-history MDP and nested suffix-policy classes.
- [x] Cellwise minimax decision ambiguity.
- [x] Monotonicity in memory length.
- [x] Exact characterization of an optimal short-memory policy.
- [x] Uniform value-loss sandwich.
- [x] Observation-prediction/decision counterexamples.
- [x] Clarification that exact controlled Markov sufficiency implies zero
  decision ambiguity.

## Completed milestone 2: offline certificate

- [x] Pairwise upper-advantage construction from simultaneous Q intervals.
- [x] High-confidence policy-loss certificate without a short-history Markov
  assumption.
- [x] Certificate tightness under Q-interval error.
- [x] Exact-memory recovery under a separation margin.
- [x] Explicit abstention when no candidate memory is certified.
- [x] Clarification that the current rule is shortest-memory certification under a loss budget, not a separately fitted per-memory bias--variance selector.

## Completed milestone 3: statistical characterization

- [x] Count-based reward and transition confidence regions.
- [x] Optimistic/pessimistic full-context dynamic programming.
- [x] Rare-context/action-gap Le Cam lower bound
  `Omega(log(1/delta)/(p gamma^2))`.
- [x] Matching tabular upper rate in `p` and `gamma`, up to logarithms and
  constants.
- [x] Empirical coverage--gap scaling on the same two-point family.

## Completed milestone 4: lightweight validation

- [x] Exact delayed-cue population experiment.
- [x] Exact nuisance-memory population experiment.
- [x] Tolerance/action-gap phase transition.
- [x] Finite-sample calibration under four rare-context probabilities.
- [x] Coverage--gap scaling over 12 `(p, gamma)` combinations.
- [x] Unit tests for all central constructions.

## Highest-value next extensions

1. **Occupancy-weighted DRMS.** Replace the worst cell with a
   deployment-weighted objective and prove a concentrability-based value bound.
2. **Random small history-MDP stress test.** Randomize reward conflicts,
   transition noise, and irrelevant predictive modes. Compare certificate
   tightness with a prediction-likelihood selector.
3. **Partial-coverage characterization.** Identify the suffix cells that need
   support to certify initial-distribution value rather than uniform value over
   every feasible history.
4. **Representation-aware extension.** Select a fixed encoder and temporal
   suffix jointly using the same advantage certificate.
5. **Small continuous demonstration.** Use discretized flickering CartPole or a
   delayed-cue gridworld only after the theory is stable.

## Claim audit

Safe claims in the current draft:

- next-observation prediction and decision relevance can demand very different
  history lengths;
- exact reward--transition Markov sufficiency implies decision sufficiency;
- zero decision ambiguity exactly characterizes uniform optimality of a suffix
  policy;
- DRMS provides a simultaneous high-confidence value-loss certificate;
- the selected suffix need not be Markov;
- the canonical memory-identification problem has matching dependence on
  `1 / (p gamma^2)`, up to logarithms and constants;
- the current experiments support this coverage--gap scaling.

Claims to avoid until broader evidence is added:

- superiority on standard deep offline-RL benchmarks;
- practical scalability to long raw histories;
- novelty of decision-sufficient state abstraction in general;
- global minimax optimality beyond the stated two-point family;
- guarantees under arbitrary function approximation.

## Submission-readiness checklist

- [x] Main theorem statements and complete proof notes.
- [x] Reproducible CPU experiments and saved summaries.
- [x] Figures generated from scripts, not hand-edited.
- [x] Tests and CI configuration.
- [x] Anonymous two-column manuscript and supplement.
- [x] Explicit related-work positioning against state abstraction and
  observation-prediction diagnostics.
- [ ] Add occupancy-weighted theory or a random-MDP stress test for a stronger
  final empirical section.
- [ ] Obtain an independent proof review, especially for the value-loss
  sandwich and matching-rate corollary.
- [ ] Switch to the unmodified official AAAI author kit.
- [ ] Add the official reproducibility checklist and perform the final page-limit
  pass.
