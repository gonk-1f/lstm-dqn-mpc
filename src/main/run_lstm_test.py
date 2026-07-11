"""LSTM prediction plots for all test voyages. Uses full feature prep from training pipeline."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[1]; PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from forecasting.feature_pipeline import prepare_lstm_features as _prepare_features
from forecasting.lstm_load_predictor import load_checkpoint, inverse_target, transform

CKPT = PROJ / "outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/best_lstm_load_predictor.pt"
SPLIT = PROJ / "outputs/config/voyage_split_721.json"
CSV = PROJ / "data/processed/aligned_timeseries.csv"
OUT = PROJ / "outputs/lstm_test"

with open(SPLIT, "r", encoding="utf-8") as f:
    split = json.load(f)
test_voyages = split["test_voyages"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model, payload = load_checkpoint(CKPT, device=device)
cfg = payload["config"]
HL = int(cfg["history_len"]); PH = int(cfg["pred_horizon"])
features = list(payload["features"])
feature_set = payload.get("feature_set", "rolling")
model.eval()
print(f"LSTM: HL={HL}, PH={PH}, features={len(features)}, feature_set={feature_set}")

df_all = pd.read_csv(CSV)
OUT.mkdir(parents=True, exist_ok=True)
all_metrics = []

for vi, vname in enumerate(test_voyages):
    df_v = df_all[df_all["voyage_name"] == vname].reset_index(drop=True)
    if len(df_v) == 0:
        continue

    # Use full feature preparation from training pipeline
    pdf = _prepare_features(df_v, feature_set)
    ff = list(features)
    for col in ["time_sin", "time_cos"]:
        if col in pdf.columns and col not in ff: ff.append(col)
    for col in ff:
        if col not in pdf.columns: pdf[col] = 0.0

    vals = pdf[ff].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    vals = transform(vals, payload["feature_scaler"])
    actual = pdf["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    duration_h = len(pdf) * 30 / 3600

    preds = np.zeros((len(pdf), PH))
    with torch.no_grad():
        for idx in range(len(pdf)):
            s = max(0, idx - HL + 1); h = vals[s : idx + 1]
            if len(h) < HL:
                h = np.vstack([np.repeat(h[:1], HL - len(h), axis=0), h])
            x = torch.as_tensor(h[None, :, :], dtype=torch.float32, device=device)
            p = model(x).detach().cpu().numpy()[0]
            preds[idx] = np.maximum(inverse_target(p, payload["target_scaler"]), 0.0)

    valid = np.arange(HL, len(pdf))
    t = valid * 30 / 3600
    h1_p = preds[valid, 0]; h1_a = actual[valid]
    rmse = np.sqrt(np.mean((h1_p - h1_a) ** 2))
    mae = np.mean(np.abs(h1_p - h1_a))
    bias = np.mean(h1_p - h1_a)
    wape = np.sum(np.abs(h1_p - h1_a)) / (np.sum(np.abs(h1_a)) + 1e-6) * 100

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(t, h1_a, "k-", lw=0.6, alpha=0.7, label="Actual")
    ax.plot(t, h1_p, "#1f77b4", lw=0.5, alpha=0.85, label="Predicted (h=1)")
    ax.set_title(f"Voyage {vi+1}: {vname[:60]}  |  RMSE={rmse:.1f}kW  MAE={mae:.1f}kW  WAPE={wape:.1f}%  ({duration_h:.1f}h)", fontsize=9)
    ax.set_ylabel("Load (kW)"); ax.set_xlabel("Time (hours)")
    ax.legend(fontsize=8); ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(OUT / f"voyage_{vi+1}_pred.png", dpi=200); plt.close(fig)

    all_metrics.append({"voyage": vi+1, "rows": len(df_v), "hours": duration_h,
                         "rmse_h1": rmse, "mae_h1": mae, "wape_h1": wape, "bias_h1": bias})
    print(f"Voyage {vi+1}: {len(df_v)} rows ({duration_h:.1f}h)  RMSE={rmse:.1f}kW  MAE={mae:.1f}kW  WAPE={wape:.1f}%  Bias={bias:+.1f}kW")

# Summary
df_m = pd.DataFrame(all_metrics)
print(f"\nSummary: avg RMSE={df_m['rmse_h1'].mean():.1f}kW  avg MAE={df_m['mae_h1'].mean():.1f}kW")
df_m.to_csv(OUT / "metrics.csv", index=False)
print(f"Saved: {OUT}/")
