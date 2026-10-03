from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import FEATURE_NAMES, MarketAnalyser
from .metrics import regression_metrics
from .model import PricePredictor
from .simulator import MarketSimulator


@dataclass
class PendingExample:
    prediction_time_ns: int
    target_time_ns: int
    current_price: float
    predicted_price: float
    sequence: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_batch(model: nn.Module, optimizer: torch.optim.Optimizer,
                examples: list[tuple[PendingExample, float]], device: torch.device,
                batch_size: int, smoothness_weight: float) -> float:
    if not examples:
        return float("nan")
    x = torch.from_numpy(np.stack([e.sequence for e, _ in examples])).float()
    current = torch.tensor([e.current_price for e, _ in examples], dtype=torch.float32)
    target = torch.tensor([math.log(actual / e.current_price) for e, actual in examples], dtype=torch.float32)
    ds = TensorDataset(x, current, target)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model.train(); losses = []
    for xb, _, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        fit = nn.functional.smooth_l1_loss(pred, yb)
        # Penalize adjacent predictions inside the shuffled mini-batch conservatively.
        smooth = torch.mean(torch.abs(pred[1:] - pred[:-1])) if len(pred) > 1 else pred.new_tensor(0.0)
        loss = fit + smoothness_weight * smooth
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def run(data_path: str, output: str, horizon: int = 1800, sequence_length: int = 120,
        epoch_seconds: int = 1800, speed: float = 0.0, max_ticks: int | None = None,
        hidden_size: int = 96, num_layers: int = 2, dropout: float = 0.15,
        ema_alpha: float = 0.15, max_abs_return: float = 0.08, learning_rate: float = 3e-4,
        weight_decay: float = 1e-4, batch_size: int = 256, smoothness_weight: float = 0.1,
        seed: int = 42) -> dict:
    seed_everything(seed)
    out = Path(output); checkpoints = out / "checkpoints"; predictions_dir = out / "predictions"
    checkpoints.mkdir(parents=True, exist_ok=True); predictions_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PricePredictor(len(FEATURE_NAMES), hidden_size, num_layers, dropout, max_abs_return).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    analyser = MarketAnalyser(max_history=max(300, sequence_length))
    feature_window: deque[np.ndarray] = deque(maxlen=sequence_length)
    pending: deque[PendingExample] = deque()
    resolved: list[tuple[PendingExample, float]] = []
    window_records: list[dict] = []
    last_smoothed: float | None = None
    epoch = 0; count = 0; latest_metrics: dict = {}

    for count, tick in enumerate(MarketSimulator(data_path, speed=speed).stream(), start=1):
        feat = analyser.update(tick); feature_window.append(feat)
        now_ns = tick.timestamp.value
        while pending and pending[0].target_time_ns <= now_ns:
            example = pending.popleft()
            # Exact timestamps are expected after gap-filling; first later tick is fallback.
            resolved.append((example, tick.close))
        if len(feature_window) == sequence_length:
            sequence = np.stack(feature_window).astype(np.float32)
            model.eval()
            with torch.inference_mode():
                x = torch.from_numpy(sequence[None]).to(device)
                raw_price = float(model.predict_price(x, torch.tensor([tick.close], device=device)).item())
            predicted = raw_price if last_smoothed is None else ema_alpha * raw_price + (1 - ema_alpha) * last_smoothed
            last_smoothed = predicted
            example = PendingExample(now_ns, now_ns + horizon * 1_000_000_000, tick.close, predicted, sequence)
            pending.append(example)
            window_records.append({"timestamp": tick.timestamp.isoformat(), "actual_price": tick.close,
                                   "predicted_price_30m": predicted,
                                   "target_timestamp": (tick.timestamp + __import__('pandas').Timedelta(seconds=horizon)).isoformat()})

        if count % epoch_seconds == 0:
            epoch += 1
            loss = train_batch(model, optimizer, resolved, device, batch_size, smoothness_weight)
            if resolved:
                actual = np.array([a for _, a in resolved]); pred = np.array([e.predicted_price for e, _ in resolved])
                latest_metrics = regression_metrics(actual, pred)
            latest_metrics.update({"epoch": epoch, "train_loss": loss, "ticks_seen": count, "device": str(device)})
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "metrics": latest_metrics}, checkpoints / f"epoch_{epoch:06d}.pt")
            with (predictions_dir / f"epoch_{epoch:06d}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "actual_price", "predicted_price_30m", "target_timestamp"])
                writer.writeheader(); writer.writerows(window_records)
            (out / "latest_metrics.json").write_text(json.dumps(latest_metrics, indent=2, allow_nan=False) if not math.isnan(loss) else json.dumps({**latest_metrics, "train_loss": None}, indent=2))
            resolved.clear(); window_records.clear()
        if max_ticks and count >= max_ticks: break
    return latest_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Replay one-second BTC bars and train a 30-minute predictor")
    p.add_argument("--data", default="data/sample/btcusdt_1s_sample.csv")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--speed", type=float, default=0.0)
    p.add_argument("--max-ticks", type=int)
    p.add_argument("--epoch-seconds", type=int, default=1800)
    p.add_argument("--horizon", type=int, default=1800)
    args = p.parse_args()
    metrics = run(args.data, args.output, speed=args.speed, max_ticks=args.max_ticks,
                  epoch_seconds=args.epoch_seconds, horizon=args.horizon)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
