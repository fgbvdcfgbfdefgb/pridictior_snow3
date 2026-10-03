from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/aggTrades"
COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp_ms", "buyer_maker", "best_match"]


def month_range(start: str, end: str):
    current = pd.Period(start, freq="M"); stop = pd.Period(end, freq="M")
    while current <= stop:
        yield str(current)
        current += 1


def download_month(symbol: str, month: str, output: Path, force: bool = False) -> Path:
    target = output / f"year={month[:4]}" / f"month={month[5:]}" / "bars.parquet"
    if target.exists() and not force: return target
    name = f"{symbol}-aggTrades-{month}.zip"
    url = f"{BASE}/{symbol}/{name}"
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    parts = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as stream:
            # Chunking bounds RAM use; Binance archives may include or omit a header.
            for raw in pd.read_csv(stream, header=None, names=COLUMNS, chunksize=1_000_000, low_memory=False):
                raw["timestamp_ms"] = pd.to_numeric(raw["timestamp_ms"], errors="coerce")
                raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
                raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
                raw = raw.dropna(subset=["timestamp_ms", "price", "quantity"])
                raw["timestamp"] = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True).dt.floor("s")
                part = raw.groupby("timestamp", sort=True).agg(open=("price", "first"), high=("price", "max"),
                    low=("price", "min"), close=("price", "last"), volume=("quantity", "sum"), trades=("price", "size")).reset_index()
                parts.append(part)
    bars = pd.concat(parts).groupby("timestamp", sort=True).agg(open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), trades=("trades", "sum")).reset_index()
    # A no-trade second carries the last close and zero volume. This makes the forecast horizon exact.
    full_index = pd.date_range(bars.timestamp.min(), bars.timestamp.max(), freq="s", tz="UTC")
    bars = bars.set_index("timestamp").reindex(full_index)
    bars["close"] = bars["close"].ffill(); bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])
    bars[["volume", "trades"]] = bars[["volume", "trades"]].fillna(0); bars["trades"] = bars["trades"].astype("int32")
    bars.index.name = "timestamp"; bars = bars.reset_index()
    target.parent.mkdir(parents=True, exist_ok=True); bars.to_parquet(target, index=False, compression="zstd")
    print(f"Wrote {target} ({len(bars):,} seconds)")
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="Download Binance BTCUSDT aggTrades and create causal 1-second OHLCV bars")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2017-08")
    p.add_argument("--end", default=pd.Timestamp.utcnow().strftime("%Y-%m"))
    p.add_argument("--output", default="data/btcusdt_1s")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(); output = Path(args.output)
    failures = []
    for month in month_range(args.start, args.end):
        try: download_month(args.symbol, month, output, args.force)
        except RuntimeError as exc: print(exc); failures.append(month)
    if failures: raise SystemExit(f"Unavailable months: {', '.join(failures)} (current partial month is expected)")


if __name__ == "__main__": main()
