#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

required=(
  paper/main.pdf
  paper/supplement.pdf
  paper/main.tex
  paper/supplement.tex
  paper/references.bib
  paper/ReproducibilityChecklist.tex
  results/population_results.csv
  results/finite_sample_raw.csv
  results/scaling_raw.csv
  results/weighted_finite_sample_raw.csv
  results/random_stress_raw.csv
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "Missing or empty required artifact: $path" >&2; exit 1; }
done

for tool in pdfinfo pdffonts pdftotext; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "$tool is required for submission preflight" >&2
    exit 1
  }
done

check_letter_pdf() {
  local pdf="$1"
  local pages width height
  pages="$(pdfinfo "$pdf" | awk -F: '/^Pages/ {gsub(/ /,"",$2); print $2}')"
  width="$(pdfinfo "$pdf" | awk '/^Page size/ {print $3}')"
  height="$(pdfinfo "$pdf" | awk '/^Page size/ {print $5}')"
  [[ -n "$pages" && "$pages" -ge 1 ]] || { echo "Invalid page count for $pdf" >&2; exit 1; }
  [[ "$width" == "612" && "$height" == "792" ]] || {
    echo "$pdf is not US Letter (found ${width}x${height} pt)" >&2
    exit 1
  }

  # Every font must be embedded, and Type 3 fonts are forbidden.
  if pdffonts "$pdf" | tail -n +3 | awk '{if ($6 != "yes" || $4 == "Type 3") bad=1} END{exit bad}'; then
    :
  else
    echo "Font preflight failed for $pdf" >&2
    pdffonts "$pdf" >&2
    exit 1
  fi
}

check_letter_pdf paper/main.pdf
check_letter_pdf paper/supplement.pdf

main_pages="$(pdfinfo paper/main.pdf | awk -F: '/^Pages/ {gsub(/ /,"",$2); print $2}')"
reference_page=0
checklist_page=0
for ((page=1; page<=main_pages; page++)); do
  text="$(pdftotext -f "$page" -l "$page" -layout paper/main.pdf - 2>/dev/null || true)"
  if [[ "$reference_page" -eq 0 ]] && grep -Eq '^[[:space:]]*References([[:space:]]|$)' <<<"$text"; then
    reference_page="$page"
  fi
  if [[ "$checklist_page" -eq 0 ]] && grep -q 'Reproducibility Checklist' <<<"$text"; then
    checklist_page="$page"
  fi
done
[[ "$reference_page" -gt 0 && "$reference_page" -le 8 ]] || {
  echo "References must begin by page 8 so that technical content uses at most seven pages; found page $reference_page" >&2
  exit 1
}
if [[ "$reference_page" -eq 8 ]]; then
  page8_text="$(pdftotext -f 8 -l 8 -layout paper/main.pdf - 2>/dev/null)"
  first_line="$(sed -n '/[^[:space:]]/{p;q;}' <<<"$page8_text")"
  grep -Eq '^[[:space:]]*References([[:space:]]|$)' <<<"$first_line" || {
    echo "Technical content spills onto page 8 before the references" >&2
    exit 1
  }
fi
[[ "$checklist_page" -gt "$reference_page" ]] || {
  echo "Reproducibility checklist was not found after the references" >&2
  exit 1
}

for log in paper/main.log paper/supplement.log; do
  [[ -f "$log" ]] || { echo "Missing LaTeX log: $log" >&2; exit 1; }
  if grep -Eiq 'undefined citations|Citation .* undefined|Reference .* undefined|There were undefined references|Overfull \\hbox|Overfull \\vbox' "$log"; then
    echo "LaTeX preflight warning in $log" >&2
    grep -Ein 'undefined citations|Citation .* undefined|Reference .* undefined|There were undefined references|Overfull \\hbox|Overfull \\vbox' "$log" >&2
    exit 1
  fi
done

main_text="$(pdftotext paper/main.pdf -)"
for forbidden in 'jingxuxie' 'Eston Javas' 'UC Berkeley'; do
  if grep -Fiq "$forbidden" <<<"$main_text"; then
    echo "Anonymity or checklist placeholder failure: found '$forbidden' in main PDF" >&2
    exit 1
  fi
done

python - <<'PY'
from pathlib import Path
import pandas as pd

expected_rows = {
    "population_results.csv": 56,
    "soft_relevance_results.csv": 200,
    "finite_sample_raw.csv": 1600,
    "scaling_raw.csv": 2880,
    "weighted_population.csv": 20,
    "weighted_finite_sample_raw.csv": 360,
    "random_stress_raw.csv": 700,
}
root = Path("results")
for name, expected in expected_rows.items():
    frame = pd.read_csv(root / name)
    if len(frame) != expected:
        raise SystemExit(f"{name}: expected {expected} rows, found {len(frame)}")

random = pd.read_csv(root / "random_stress_raw.csv")
if int(random["theorem_violation"].sum()) != 0:
    raise SystemExit("random stress test contains a theorem violation")

finite = pd.read_csv(root / "finite_sample_raw.csv")
if bool(finite["false_short_certification"].any()):
    raise SystemExit("uniform calibration contains a false short-memory certification")
if not bool(finite["certificate_valid"].all()):
    raise SystemExit("uniform calibration contains an invalid certificate")

weighted = pd.read_csv(root / "weighted_finite_sample_raw.csv")
for column in ["certificate_valid", "q_interval_coverage", "occupancy_interval_coverage"]:
    if not bool(weighted[column].all()):
        raise SystemExit(f"weighted experiment failed {column}")

checklist = Path("paper/ReproducibilityChecklist.tex").read_text(encoding="utf-8").splitlines()
for index, line in enumerate(checklist[:-1]):
    if line.lstrip().startswith("\\question{"):
        response = checklist[index + 1].strip()
        if response not in {"yes", "no", "partial", "NA"}:
            raise SystemExit(
                f"invalid or missing checklist response after line {index + 1}: {response!r}"
            )
PY

printf 'Submission preflight passed: %s pages, references start on page %s, checklist starts on page %s.\n' \
  "$main_pages" "$reference_page" "$checklist_page"
