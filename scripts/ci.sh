#!/usr/bin/env bash
# Everything CI checks, runnable offline in one command.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== tests =="
python3 -m unittest discover -s tests

echo "== entry points =="
python3 -m machaon --version
python3 -m machaon doctor > /dev/null && echo "doctor ok"

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

if python3 -c "import mcp" 2>/dev/null; then
  echo "== mcp transport =="
  python3 scripts/mcp_smoke.py
else
  echo "== mcp transport skipped (mcp not installed) =="
fi

if python3 -c "import fastapi, httpx" 2>/dev/null; then
  echo "== web API =="
  python3 -m unittest tests.test_web -v
else
  echo "== web API skipped (fastapi/httpx not installed) =="
fi

if command -v ruff > /dev/null 2>&1; then
  echo "== lint =="
  ruff check machaon tests examples scripts
else
  echo "== lint skipped (ruff not installed) =="
fi

echo
echo "ALL CHECKS PASSED"
