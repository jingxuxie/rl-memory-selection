# How Much History Is Enough?

This repository develops **Decision-Relevant Memory Selection (DRMS)**, a
theory-first method for selecting the shortest temporal suffix whose policy
loss can be certified from offline reinforcement-learning data. The selected
suffix is **not required to be Markov**.

The central distinction is:

- **predictive memory:** older history improves prediction of future
  observations; and
- **decision-relevant memory:** forgetting older history forces one action rule
  to compromise across contexts that call for different actions.

These notions are incomparable for observation-only prediction. An old
nuisance bit can remain predictive while never affecting an optimal action;
conversely, an old cue can determine the rewarding action while leaving all
later observations unchanged.

## Main results

For full histories `x` merged by an `m`-step suffix, DRMS defines a cellwise
minimax optimal-advantage loss and its cumulative ambiguity `G_m`. The project
contains proofs that:

1. `G_m` decreases with memory length.
2. `G_m = 0` exactly characterizes the existence of a uniformly optimal
   `m`-memory policy.
3. The best uniform value loss lies between the largest stage ambiguity and
   `G_m`.
4. Simultaneous full-history `Q*` intervals yield a post-selection-valid
   certificate for every candidate memory, even when the suffix is non-Markov.
5. A deployment-weighted extension discounts rare conflicts under an explicit
   deployment/reference concentrability bound and an empirical-occupancy
   correction.
6. In a rare-conflict family with context probability `p` and action gap
   `gamma`, exact memory recovery has matching upper and lower dependence
   `1 / (p * gamma**2)`, up to constants and logarithms.

## Validation summary

All validation is tabular, CPU-only, and theorem-driven.

- **4,840 finite-sample datasets** across uniform calibration,
  coverage--gap scaling, and deployment-weighted certification.
- **200 random history MDPs** and **700 memory configurations**.
- **16 unit tests** covering all central constructions.
- No observed violation of the population sandwich, robust uniform
  certificate, interval-inflation bound, weighted population certificate, or
  weighted robust certificate in the randomized sweep.
- The empirical 50% recovery threshold has log--log slope **0.935** against
  `1 / (p * gamma**2)`.
- The exact one-sided 95% binomial upper bound on the randomized theorem-test
  violation rate is **0.427%** after 0 failures in 700 configurations.

These experiments validate the theoretical mechanisms; they do not claim
state-of-the-art continuous-control performance.

## Repository layout

```text
src/drms/                 MDP, ambiguity, confidence, occupancy, and lower-bound code
experiments/              Reproducible experiment and plotting scripts
results/                  Raw and summarized CSV outputs
figures/                  Generated PDF and PNG figures
paper/                    AAAI-27 manuscript, supplement, checklist, and bibliography
notes/                    Proof notes, claim audit, and research roadmap
tests/                    Theorem-driven unit tests
scripts/                  Reproduction, paper-build, and submission-preflight scripts
```

## Reproduce everything

Python 3.10 or newer is required. A reference dependency snapshot is in
`requirements-lock.txt`.

```bash
python -m pip install -e '.[dev]'
make reproduce
```

This command runs the tests, regenerates every independent experiment sweep, rebuilds every figure, builds
the main paper and supplement with the unmodified AAAI-27 style, and performs
PDF/result preflight checks.

Equivalent one-shot script:

```bash
bash scripts/reproduce.sh
```

Individual targets are also available:

```bash
make test
make population
make finite-sample
make scaling
make weighted
make random-stress
make figures
make paper
make package
make submission-check
```

## Paper artifacts

- `paper/main.pdf`: anonymous AAAI-27 manuscript with the official
  reproducibility checklist after the references.
- `paper/supplement.pdf`: complete proofs, all experiment grids, random stress
  plots, and computational details.
- `paper/main.tex` and `paper/supplement.tex`: submission sources.

The packaging target emits anonymous `submission/main_paper.pdf`,
`submission/technical_appendix.pdf`, and `submission/code_and_data.zip`, together
with SHA-256 manifests. The preflight script checks US-letter size, page count,
embedded non-Type-3 fonts, unresolved references, package anonymity and hashes,
complete result-grid row counts, and theorem-test violations.

## Scope and limitations

The tabular implementation enumerates a finite maximum history and therefore
inherits exponential growth in raw history length. The weighted certificate
requires a known or conservatively bounded deployment-to-reference density
ratio. Function approximation, continuous observations, and adaptive history
representations are outside the current theorem scope.

## License

Code and generated research artifacts are released under the MIT License.
