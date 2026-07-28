#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf submission
mkdir -p submission
cp paper/main.pdf submission/main_paper.pdf
cp paper/supplement.pdf submission/technical_appendix.pdf

python - <<'PY'
from pathlib import Path
import zipfile

root = Path('.').resolve()
out = root / 'submission'

code_patterns = [
    'README.md', 'LICENSE', 'Makefile', 'pyproject.toml', 'requirements.txt',
    'requirements-lock.txt', '.gitignore',
]
code_paths = [root / item for item in code_patterns]
for directory in ['src', 'experiments', 'tests', 'results', 'figures', 'scripts']:
    code_paths.extend(path for path in (root / directory).rglob('*') if path.is_file())

excluded_suffixes = {'.pyc', '.aux', '.bbl', '.blg', '.log', '.fls', '.fdb_latexmk', '.synctex.gz'}
with zipfile.ZipFile(out / 'code_and_data.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(code_paths)):
        if not path.exists() or '__pycache__' in path.parts or '.pytest_cache' in path.parts:
            continue
        if any(str(path).endswith(suffix) for suffix in excluded_suffixes):
            continue
        archive.write(path, path.relative_to(root))

paper_paths = [
    root / 'paper/main.tex', root / 'paper/supplement.tex',
    root / 'paper/references.bib', root / 'paper/ReproducibilityChecklist.tex',
    root / 'paper/aaai2027.sty', root / 'paper/aaai2027.bst',
]
figure_paths = sorted((root / 'figures').glob('*.pdf'))
with zipfile.ZipFile(out / 'paper_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in [*paper_paths, *figure_paths]:
        archive.write(path, path.relative_to(root))
PY

(
  cd submission
  sha256sum main_paper.pdf technical_appendix.pdf code_and_data.zip paper_source.zip > SHA256SUMS
)

echo "Created anonymous submission package in $ROOT/submission"
