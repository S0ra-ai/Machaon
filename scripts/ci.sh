#!/usr/bin/env bash
# Everything CI checks, runnable offline in one command.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== tests =="
python3 -m unittest discover -s tests

echo "== demo =="
python3 -m machaon.cli demo > /dev/null && echo "demo ok"

echo "== spec validation =="
python3 -m machaon.cli validate examples/edit_spec.json

echo "== gate accepts a good edit =="
python3 -m machaon.cli gate examples/metrics_pass.json

echo "== gate rejects a bad edit =="
if python3 -m machaon.cli gate examples/metrics_fail.json; then
  echo "FAIL: gate accepted a failing edit"
  exit 1
fi

echo
echo "ALL CHECKS PASSED"
