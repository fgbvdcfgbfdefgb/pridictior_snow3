#!/usr/bin/env bash
set -euo pipefail
: "${CEREBRIUM_API_KEY:?Export CEREBRIUM_API_KEY first}"
python -m pip install 'cerebrium>=1.39,<2'
cerebrium deploy
