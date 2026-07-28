#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT/paper"
AAAI_KIT_DIR="$ROOT/AAAI_AuthorKit27"

export TEXINPUTS="$PAPER_DIR:$AAAI_KIT_DIR:${TEXINPUTS:-}"
export BSTINPUTS="$PAPER_DIR:$AAAI_KIT_DIR:${BSTINPUTS:-}"
export BIBINPUTS="$PAPER_DIR:${BIBINPUTS:-}"

if command -v bibtex >/dev/null 2>&1; then
  BIBTEX_BIN="$(command -v bibtex)"
elif [[ -x /usr/bin/bibtex.original ]]; then
  BIBTEX_BIN=/usr/bin/bibtex.original
else
  echo "bibtex was not found" >&2
  exit 1
fi

build_one() {
  local stem="$1"
  (
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >/dev/null
    "$BIBTEX_BIN" "$stem" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >/dev/null
  )
}

build_one main
build_one supplement
printf 'Built %s and %s\n' "$PAPER_DIR/main.pdf" "$PAPER_DIR/supplement.pdf"
