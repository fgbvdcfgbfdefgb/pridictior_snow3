from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BTC Predictor", layout="wide")
st.title("BTCUSDT · 30-minute predictor")
root = Path(__file__).resolve().parents[1]
files = sorted((root / "artifacts/predictions").glob("epoch_*.csv"))
if not files:
    st.info("No prediction windows yet. Start training first.")
    st.stop()
frame = pd.concat([pd.read_csv(p) for p in files[-4:]], ignore_index=True).tail(7200)
frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
fig = go.Figure()
fig.add_scatter(x=frame.timestamp, y=frame.actual_price, name="Actual at prediction time")
fig.add_scatter(x=frame.timestamp, y=frame.predicted_price_30m, name="Predicted price at t+30m")
st.plotly_chart(fig, use_container_width=True)
metrics_path = root / "artifacts/latest_metrics.json"
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    cols = st.columns(4)
    cols[0].metric("Epoch", m.get("epoch", "—")); cols[1].metric("MAE", f"{m.get('mae', 0):,.2f}")
    cols[2].metric("RMSE", f"{m.get('rmse', 0):,.2f}"); cols[3].metric("Within 0.5%", f"{m.get('within_0_5pct', 0):.1f}%")
    st.progress(min(1.0, max(0.0, m.get("within_0_5pct", 0) / 100)), text="Forecasts within 0.5%")
    st.progress(min(1.0, max(0.0, m.get("directional_accuracy_pct", 0) / 100)), text="Directional accuracy")
st.caption("Refreshes every second while open.")
st.html("<script>setTimeout(() => window.parent.location.reload(), 1000)</script>")
