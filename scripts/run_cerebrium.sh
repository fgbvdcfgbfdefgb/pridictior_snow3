#!/usr/bin/env bash
set -euo pipefail
: "${CEREBRIUM_API_KEY:?Export CEREBRIUM_API_KEY first}"
python -m pip install 'cerebrium>=2.9,<3'
cerebrium --service-account-token "$CEREBRIUM_API_KEY" deploy
