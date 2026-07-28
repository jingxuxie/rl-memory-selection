# Internal reviewer audit

This document records the main reviewer-facing risks and the corresponding
checks in the submission.

## 1. Is a short suffix assumed Markov?

No. All values, advantages, transition models, and confidence intervals are
defined on complete observable histories. A candidate suffix only constrains
which histories must share an action rule. The value proofs recurse on the
full-history MDP.

## 2. Is zero ambiguity merely sufficient, or also necessary?

It is necessary and sufficient for a suffix policy that is optimal from every
full history at every stage. Necessity follows because any uniformly optimal
suffix policy must have zero expected optimal-advantage loss in every history
of each suffix cell.

## 3. Does the value-loss lower bound ignore future mistakes?

No. For every suffix policy,

`V*_h(x) - V^pi_h(x) >= E_pi A*_h(x,a)`

because replacing the policy continuation by `V*` can only increase its value.
Maximizing this one-step loss within each cell yields the stage ambiguity. The
upper bound separately accumulates current ambiguity and future value loss.

## 4. Is selecting memory on the same data a post-selection error?

The `Q*` event holds simultaneously over all full-history state-action-stage
triples. Every candidate memory certificate is a deterministic function of
that common event, so all memory lengths—and the data-selected one—are covered
simultaneously. No second split is required for the stated finite candidate
set.

## 5. Why omit self-comparison in the robust advantage?

For action `a`, the comparison of `Q*(x,a)` with itself is exactly zero. Using
`U(x,a)-L(x,a)` would add uncertainty that is unrelated to action regret and
would prevent exact certification even at one-action stages. The pairwise
formula retains all genuine competitors and is still a valid upper bound.

## 6. What exactly is weighted, and what is assumed?

The weighted criterion averages optimal-advantage loss under a reference
history occupancy. Deployment value is bounded only when the selected policy's
occupancy is dominated by the reference with known or conservatively bounded
stagewise constants `C_h`. The method does not claim to estimate these
constants automatically. Empirical reference occupancies receive a separate
simultaneous L1 correction.

## 7. Are the lower and upper rates genuinely matched?

Only in the stated two-model rare-cue family. The lower bound is
`Omega(log(1/delta)/(p gamma^2))`; the tabular DRMS upper result is
`O(log(M/delta)/(p gamma^2))`. The paper does not claim a global minimax
characterization beyond this family.

## 8. Are experiments just restating the construction?

The exact cue/nuisance experiments illustrate the separation, while a separate
suite samples 200 random history MDPs and checks 700 memory configurations.
The random suite includes nontrivial transition noise, reward conflicts, and
occupancy shift and validates both uniform and weighted inequalities. It is
still a theorem stress test, not evidence of deep-benchmark superiority.

## 9. Reproducibility checks

- 16 unit tests pass.
- Full experiment grids regenerate 4,840 finite-sample datasets.
- Main and supplement compile in the official AAAI-27 style.
- No unresolved citations or overfull boxes remain.
- PDFs are US letter, use embedded fonts, and contain no Type-3 fonts.
- The official reproducibility checklist is completed.
- `make reproduce` reruns tests, experiments, figures, papers, and preflight.

## 10. Remaining human checks

The authors must still enter the correct author metadata in the submission
system, confirm the current supplement-upload policy, and may wish to obtain an
external proof read. These are not silently represented as automated checks.
