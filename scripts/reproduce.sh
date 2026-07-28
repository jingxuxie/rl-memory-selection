#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

make test
make experiments
make figures
make paper
make submission-check

echo "Reproduction completed successfully."
