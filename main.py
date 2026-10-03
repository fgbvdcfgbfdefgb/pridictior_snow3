"""Cerebrium train-module entrypoint. Long training is invoked explicitly via run_training."""
from bitcoin_predictor.train import run


def run_training(data_path: str = "data/sample/btcusdt_1s_sample.csv", max_ticks: int | None = None) -> dict:
    return run(data_path=data_path, output="/persistent-storage/artifacts", max_ticks=max_ticks)


def health() -> dict:
    return {"status": "ok", "module": "train", "offline_runtime": True}
