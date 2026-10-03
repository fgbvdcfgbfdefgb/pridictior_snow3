# Bitcoin Price Predictor

Leakage-safe BTCUSDT simulator and GPU predictor for a price **30 minutes ahead**, updated every second. The project runs without network access after its data and Python environment are prepared.

## Design

- **MarketSimulator (CPU):** replays ordered one-second OHLCV bars; `speed=1` is wall-clock and `speed=0` is maximum throughput.
- **MarketAnalyser (CPU):** causal return, volatility, range, volume, trade-count and EMA features.
- **PricePredictor (GPU):** compact GRU that predicts a bounded log return. A causal EMA stabilizes the displayed prediction.
- **Delayed supervision:** each prediction is queued until its exact `t+1800s` target exists. Every 1,800 simulated seconds, matured examples train the next cycle, all predictions are stored, and model/optimizer state is checkpointed.
- **No fake “accuracy”:** MAE, RMSE, MAPE, within-0.5%, and directional accuracy are reported. Financial forecasts require walk-forward validation before any trading use.

A single T4 is used directly; “distributed training within one GPU” has no useful meaning. DataLoader workers can parallelize CPU input work, but PyTorch DDP is for multiple processes/devices and would add overhead here.

## Quick smoke test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,dashboard]'
pytest -q
btc-train --data data/sample/btcusdt_1s_sample.csv --max-ticks 4000
streamlit run dashboard/app.py
```

The sample is synthetic and exists only to verify the pipeline.

## Complete Binance BTCUSDT history

```bash
source .venv/bin/activate
./scripts/download_all.sh
# or choose exact complete months
btc-download --start 2017-08 --end 2026-09 --output data/btcusdt_1s
btc-train --data data/btcusdt_1s --output artifacts
```

The downloader uses Binance public monthly aggregate-trade archives and emits one-second Parquet partitions. The current partial month is excluded by the shell script. Full history is not committed as Git blobs because it exceeds GitHub's practical limits; see [DATASET.md](DATASET.md). Copy the generated directory into the offline runtime.

## Cerebrium

The committed `cerebrium.toml` requests 1×T4, 8 vCPU, 8 GB RAM, one replica. It deploys `main.py` as the train module.

```bash
export CEREBRIUM_API_KEY='your-key'  # never commit it
export CEREBRIUM_PROJECT_ID='p-xxxxxxxx'
# One-off train-module GPU smoke test (4,000 ticks by default)
./scripts/test_cerebrium_train.sh
# Persistent endpoint deployment
./scripts/run_cerebrium.sh
```

Upload full data to `/persistent-storage/data/btcusdt_1s` and invoke `run_training` with that path. Checkpoints go to `/persistent-storage/artifacts`. `min_replicas=0` avoids idle GPU cost. Current Cerebrium configuration uses `compute = "TURING_T4"` and `gpu_count = 1`.

## Snowflake offline operation

1. On an internet-connected machine, download all Parquet partitions and create `SHA256SUMS` as described in DATASET.md.
2. Upload this repository and dataset to an internal Snowflake stage.
3. Copy/mount them into the notebook workspace. No runtime code downloads models, features, or market data.
4. Select a Snowflake runtime that already provides Python, PyTorch, pandas, NumPy, PyArrow and Plotly (or upload approved wheel files to an internal stage).
5. Open `notebooks/snowflake_training.ipynb` and run all cells.

The notebook falls back to the bundled sample when the full dataset is absent.

## Outputs

- `artifacts/checkpoints/epoch_XXXXXX.pt`: model, optimizer, epoch and metrics.
- `artifacts/predictions/epoch_XXXXXX.csv`: every second's actual-at-prediction-time and t+30m forecast.
- `artifacts/latest_metrics.json`: dashboard metrics.

## Stability and evaluation

Stability comes from bounded return output, gradient clipping, Huber loss, output EMA, and a smoothness regularizer. Tune stability only on a validation period. Use chronological train/validation/test splits; random splitting leaks market regimes. Include fees, spread, latency, slippage and execution constraints in any later trading simulation.

## Security

Credentials are read from environment variables and are excluded by `.gitignore`. No key is stored in this repository. Rotate any credential pasted into chat after testing.
