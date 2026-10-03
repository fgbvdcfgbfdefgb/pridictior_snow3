#!/usr/bin/env bash
set -euo pipefail
START="${START:-2017-08}"
# Last complete UTC month; pass END explicitly to override.
END="${END:-$(python -c 'import pandas as p; print((p.Timestamp.utcnow().to_period("M")-1))')}"
python -m bitcoin_predictor.data_download --start "$START" --end "$END" --output data/btcusdt_1s
