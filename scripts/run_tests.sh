#!/usr/bin/env bash
# GeoVision canonical test runner.
#
#   scripts/run_tests.sh          # standard (fast) suite — deselects marked-slow
#   scripts/run_tests.sh --full   # everything incl. heavy CPU-pipeline integration
#
# The 'slow' marker covers tests that run the full heavy CPU pipeline (GeoCLIP
# inference ~minutes), which legitimately need a long wall-clock.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--full" ]]; then
  echo "== GeoVision FULL suite (incl. heavy CPU-pipeline integration; ~5+ min) =="
  python3 -m pytest tests/ -v --no-header -p no:cacheprovider
  STATUS=$?
else
  echo "== GeoVision standard suite (deselects marked-slow heavy integration tests) =="
  python3 -m pytest tests/ -q --no-header -p no:cacheprovider -m "not slow" \
    --deselect tests/test_visual_similarity.py::test_offline_visual_verify_matches_real_eiffel \
    2>/dev/null
  STATUS=$?
fi
echo ""
echo "== exit=$STATUS =="
exit $STATUS