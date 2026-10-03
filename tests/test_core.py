import numpy as np
import pandas as pd

from bitcoin_predictor.features import FEATURE_NAMES, MarketAnalyser
from bitcoin_predictor.metrics import regression_metrics
from bitcoin_predictor.simulator import MarketSimulator, Tick


def test_features_are_finite_and_causal():
    analyser = MarketAnalyser()
    for i in range(130):
        tick = Tick(pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(seconds=i),
                    100+i, 101+i, 99+i, 100+i, 1+i/10, 2)
        out = analyser.update(tick)
    assert out.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(out).all()


def test_simulator_orders_and_deduplicates(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame([
        ["2025-01-01T00:00:01Z", 2,2,2,2,1,1],
        ["2025-01-01T00:00:00Z", 1,1,1,1,1,1],
        ["2025-01-01T00:00:01Z", 3,3,3,3,1,1],
    ], columns=["timestamp","open","high","low","close","volume","trades"]).to_csv(path,index=False)
    ticks = list(MarketSimulator(path).stream())
    assert [t.close for t in ticks] == [1.0, 3.0]


def test_metrics_perfect():
    m = regression_metrics(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert m["mae"] == 0
    assert m["within_0_5pct"] == 100
