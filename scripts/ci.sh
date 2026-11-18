#!/usr/bin/env bash
# Everything CI checks, runnable offline in one command.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== tests =="
python3 -m unittest discover -s tests

echo "== entry points =="
python3 -m machaon --version
python3 -m machaon doctor > /dev/null && echo "doctor ok"
python3 -m machaon.cli start > /dev/null && echo "start ok"
python3 -m machaon.cli web --help > /dev/null && echo "web launch help ok"
python3 -m machaon.cli intent "The Eiffel Tower is in the city of -> Rome" > /dev/null && echo "intent ok"
python3 -m machaon.cli fetch --list > /dev/null && echo "fetch list ok"
python3 -m machaon.cli teach-many --help > /dev/null && echo "teach-many help ok"
for program in examples/*.machaon; do
  python3 -m machaon.cli program "$program" --plan > /dev/null
done
echo "program starters ok"

echo "== demo =="
python3 -m machaon.cli demo > /dev/null && echo "demo ok"

echo "== constrained action =="
python3 examples/action_edit.py > /dev/null && echo "action ok"

echo "== symmetry alignment =="
python3 examples/symmetry_align.py > /dev/null && echo "symmetry ok"

echo "== task arithmetic =="
python3 examples/task_arithmetic.py > /dev/null && echo "task-arithmetic ok"

echo "== weight fingerprints =="
python3 examples/fingerprint_diff.py > /dev/null && echo "fingerprint ok"

echo "== spec validation =="
python3 -m machaon.cli validate examples/edit_spec.json

echo "== gate accepts a good edit =="
python3 -m machaon.cli gate examples/metrics_pass.json

echo "== gate rejects a bad edit =="
if python3 -m machaon.cli gate examples/metrics_fail.json; then
  echo "FAIL: gate accepted a failing edit"
  exit 1
fi

echo "== WSP accepts a repair report =="
python3 -m machaon.cli wsp examples/wsp_report.json > /dev/null && echo "wsp ok"

echo "== WSP benchmark harness =="
python3 -m machaon.cli wsp-bench examples/wsp_benchmark.json > /dev/null && echo "wsp-bench ok"
python3 -m machaon.cli wsp-bench examples/wsp_measured_smoke.json --require-measured > /dev/null && echo "measured-bench ok"

echo "== evidence-driven plan =="
python3 -m machaon.cli plan examples/plan_obstruction.json > /dev/null && echo "plan ok"

echo "== automatic diagnosis =="
python3 -m machaon.cli diagnose examples/diagnose_obstruction.json > /dev/null && echo "diagnose ok"

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
