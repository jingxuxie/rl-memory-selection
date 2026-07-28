# Paper build

The manuscript uses the unmodified official AAAI-27 style files committed in
this directory. From the repository root, build both anonymous PDFs with:

```bash
make paper
```

or directly:

```bash
bash scripts/build_paper.sh
```

Generated artifacts:

- `paper/main.pdf`: seven-page technical manuscript, references, and the
  required reproducibility checklist.
- `paper/supplement.pdf`: complete proofs and experimental details.

Run `make submission-check` to verify US-Letter dimensions, embedded non-Type-3
fonts, resolved citations, technical-page placement, checklist completion,
anonymity, result-grid sizes, and recorded theorem violations. Run
`make package` to create the anonymous submission files and SHA-256 manifest in
`submission/`.

Do not modify `aaai2027.sty` or `aaai2027.bst`.
