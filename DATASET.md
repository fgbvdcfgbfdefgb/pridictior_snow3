# Dataset specification

The canonical source is Binance Spot **BTCUSDT aggregate trades**, converted to UTC one-second OHLCV bars. It is an exchange-specific spot series, not a claim to represent every Bitcoin venue.

Schema: `timestamp, open, high, low, close, volume, trades`.

`python -m bitcoin_predictor.data_download --start 2017-08 --end YYYY-MM` downloads monthly public archives, aggregates trades, fills no-trade seconds with the previous close and zero volume, and writes Zstandard Parquet partitions.

The full history is intentionally not tracked as Git blobs: it is far beyond normal GitHub repository limits and is reproducibly generated from the source archives. For an offline environment, run the downloader on an internet-connected machine, then upload `data/btcusdt_1s/` to a Snowflake stage or include it in the Cerebrium persistent volume before training. SHA-256 manifests can be created with `find data/btcusdt_1s -name '*.parquet' -print0 | sort -z | xargs -0 sha256sum > data/SHA256SUMS`.

The included synthetic sample is only a smoke-test fixture and must not be used to assess trading performance.
