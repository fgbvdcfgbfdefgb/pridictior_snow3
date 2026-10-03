from __future__ import annotations

from collections import deque

import numpy as np

from .simulator import Tick

FEATURE_NAMES = [
    "return_1s", "return_5s", "return_30s", "return_60s", "range_bps",
    "volume_log", "volume_z", "trades_log", "realized_vol_30s", "realized_vol_120s",
    "ema_gap_fast", "ema_gap_slow",
]


class MarketAnalyser:
    """CPU-only causal feature extractor with bounded memory."""

    def __init__(self, max_history: int = 300):
        self.prices: deque[float] = deque(maxlen=max_history)
        self.volumes: deque[float] = deque(maxlen=max_history)
        self.returns: deque[float] = deque(maxlen=max_history)
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None

    @staticmethod
    def _safe_return(now: float, old: float) -> float:
        return float(np.log(max(now, 1e-12) / max(old, 1e-12)))

    def update(self, tick: Tick) -> np.ndarray:
        p = tick.close
        r1 = self._safe_return(p, self.prices[-1]) if self.prices else 0.0
        self.prices.append(p)
        self.volumes.append(tick.volume)
        self.returns.append(r1)
        self.ema_fast = p if self.ema_fast is None else 0.2 * p + 0.8 * self.ema_fast
        self.ema_slow = p if self.ema_slow is None else 0.02 * p + 0.98 * self.ema_slow

        def lag_return(lag: int) -> float:
            return self._safe_return(p, self.prices[-lag - 1]) if len(self.prices) > lag else 0.0

        vols = np.asarray(self.volumes, dtype=np.float64)
        rets = np.asarray(self.returns, dtype=np.float64)
        vol_std = float(vols[-120:].std()) if len(vols) >= 2 else 0.0
        vol_z = (tick.volume - float(vols[-120:].mean())) / (vol_std + 1e-9)
        result = np.array([
            r1, lag_return(5), lag_return(30), lag_return(60),
            (tick.high - tick.low) / max(p, 1e-12) * 1e4,
            np.log1p(tick.volume), vol_z, np.log1p(tick.trades),
            float(rets[-30:].std()) if len(rets) >= 2 else 0.0,
            float(rets[-120:].std()) if len(rets) >= 2 else 0.0,
            (p - self.ema_fast) / max(p, 1e-12),
            (p - self.ema_slow) / max(p, 1e-12),
        ], dtype=np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
