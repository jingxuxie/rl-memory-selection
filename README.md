# Decision-Relevant Memory Selection for Offline RL

This repository develops **DRMS**, a theory-first method for selecting the
shortest history window whose policy loss can be certified from offline data.
The selected suffix does **not** need to be Markov.

The key distinction is between:

- **next-observation predictive memory**: old history reduces uncertainty about
  future observations; and
- **decision-relevant memory**: forgetting old history forces one action rule
  to compromise across contexts that require different decisions.

Observation-prediction diagnostics can over-prescribe memory because of
predictive nuisance variables, or under-prescribe it when an old cue changes a
rewarding action without changing later observations. Exact reward--transition
Markov sufficiency is stronger and always implies decision sufficiency.

DRMS estimates full-history optimal action values, converts simultaneous
confidence intervals into upper bounds on optimal advantage loss, and solves
small cellwise minimax programs to certify each candidate suffix length.

## Current results

- A population characterization of decision-relevant temporal memory.
- A value-loss sandwich and an exact common-optimal-action condition.
- A high-confidence certificate valid for non-Markov suffix policies.
- Exact-memory recovery under an advantage-separation condition.
- Matching upper and lower rates, up to logarithms and constants, of
  `1 / (p * gamma^2)` in a rare-context/action-gap family.
- Exact delayed-cue and nuisance-memory counterexamples.
- CPU-only population, calibration, and sample-complexity experiments.
- An anonymous two-column manuscript and a full technical supplement.

## Repository layout

```text
src/drms/                 Core MDP, ambiguity, confidence, and lower-bound code
experiments/              Reproducible experiment and plotting scripts
results/                  Raw and summarized CSV outputs
figures/                  Generated PDF/PNG figures
paper/                    Main manuscript, supplement, and bibliography
notes/                    Detailed proof notes and research roadmap
tests/                    Theorem-driven unit tests
```

## Reproduce

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[dev]'
make test
make experiments
make figures
make paper
```

The complete suite is tabular and CPU-only. To run the scripts separately:

```bash
python experiments/run_population.py
python experiments/run_finite_sample.py
python experiments/run_scaling.py
python experiments/make_figures.py
```

## Paper

`paper/main.tex` uses a portable two-column fallback by default. The generated
PDF is `paper/main.pdf`; complete proofs are in `paper/supplement.pdf`.

For an actual AAAI submission, copy the unmodified `aaai2027.sty` and `aaai2027.bst` files from
the repository's bundled `AAAI_AuthorKit27/` directory into `paper/`, change
`\officialaaaifalse` to `\officialaaaitrue`, and rebuild. The portable fallback
is intentionally not presented as the official conference format.

## Status

This branch establishes the core theorem stack, matching-rate construction,
and lightweight validation. The highest-value next extension is an
occupancy-weighted certificate or a random-history-MDP stress test; see
`notes/roadmap.md` for the claim audit and remaining submission checks.
