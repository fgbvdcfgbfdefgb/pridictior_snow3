from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume", "trades"}


@dataclass(frozen=True)
class Tick:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int


class MarketSimulator:
    """Replay immutable one-second bars without exposing future observations."""

    def __init__(self, source: str | Path, speed: float = 0.0):
        self.source = Path(source)
        self.speed = speed
        if speed < 0:
            raise ValueError("speed must be >= 0")

    def _files(self) -> list[Path]:
        if self.source.is_dir():
            files = sorted(self.source.rglob("*.parquet")) or sorted(self.source.rglob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No parquet/csv files under {self.source}")
            return files
        return [self.source]

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    def stream(self) -> Iterator[Tick]:
        # Files are processed one at a time, so a multi-year dataset never resides in RAM.
        previous = None
        for path in self._files():
            chunks = [pd.read_parquet(path)] if path.suffix == ".parquet" else pd.read_csv(path, chunksize=1_000_000)
            for chunk in chunks:
                for row in self._normalize(chunk).itertuples(index=False):
                    ts = pd.Timestamp(row.timestamp)
                    if previous is not None and ts <= previous:
                        continue
                    if self.speed and previous is not None:
                        delay = max(0.0, (ts - previous).total_seconds() / self.speed)
                        time.sleep(delay)
                    previous = ts
                    yield Tick(ts, float(row.open), float(row.high), float(row.low), float(row.close),
                               float(row.volume), int(row.trades))
