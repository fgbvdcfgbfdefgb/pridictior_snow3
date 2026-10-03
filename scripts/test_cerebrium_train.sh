#!/usr/bin/env bash
set -euo pipefail
: "${CEREBRIUM_API_KEY:?Export CEREBRIUM_API_KEY first}"
: "${CEREBRIUM_PROJECT_ID:?Export CEREBRIUM_PROJECT_ID first}"
MAX_TICKS="${MAX_TICKS:-4000}"
python -m pip install 'cerebrium>=2.9,<3'
cerebrium project set "$CEREBRIUM_PROJECT_ID"
cerebrium --service-account-token "$CEREBRIUM_API_KEY" run main.py::run_training \
  --max_ticks "$MAX_TICKS"
