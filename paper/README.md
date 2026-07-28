# Paper build

Both `main.tex` and `supplement.tex` use the official AAAI 2027 anonymous
submission style from the bundled `AAAI_AuthorKit27/` directory. Build both
documents from the repository root:

```bash
make paper
```

or directly:

```bash
bash scripts/build_paper.sh
```

The generated files are `paper/main.pdf` and `paper/supplement.pdf`.

The build script configures TeX and BibTeX to find the unmodified
`aaai2027.sty` and `aaai2027.bst`; there is no need to copy them into
`paper/`. Add the official reproducibility checklist before the final
submission build if the conference submission instructions require it.
