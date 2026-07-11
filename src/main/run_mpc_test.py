"""MPC standalone test on voyage test set. Uses current LSTM checkpoint."""

from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch

SRC = Path(__file__).resolve().parents[1]; PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from forecasting.feature_pipeline import prepare_lstm_features as _prepare_features
from forecasting.lstm_load_predictor import inverse_target, load_checkpoint, transform
from mpc.controllers.reference_generator import DualSideCasadiReferenceGenerator
from mpc.solvers.casadi_solver import CasadiMPCConfig

LSTM_CKPT = PROJ / "outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/best_lstm_load_predictor.pt"
SPLIT = PROJ / "outputs/config/voyage_split_721.json"
TRAIN_CSV = PROJ / "data/processed/aligned_timeseries.csv"
OUT_DIR = PROJ / "outputs/mpc_test"

DT = 30; HORIZON = 6

MPC_KWARGS = dict(
    prediction_horizon=HORIZON, dt_hours=DT/3600.0,
    battery_capacity_kwh=1806.0, fuel_cell_max_kw=560.0, fuel_cell_ramp_kw=48.0,
    soc_target=0.65, soc_min=0.2, soc_max=0.8,
    use_raw_objective=True, raw_soc_squared=False,
    q_soc=40.0, q_fc=0.001, q_batt=0.01, q_ramp=0.08,
    ipopt_max_iter=100, ipopt_tol=1e-4,
)


def forecast_voyage(df_voyage, model, payload, device):
    cfg = payload["config"]; HL = int(cfg["history_len"]); PH = int(cfg["pred_horizon"])
    features = list(payload["features"]); feature_set = payload.get("feature_set", "rolling")
    pdf = _prepare_features(df_voyage, feature_set)
    ff = list(features)
    for col in ["time_sin","time_cos"]:
        if col in pdf.columns and col not in ff: ff.append(col)
    for col in ff:
        if col not in pdf.columns: pdf[col] = 0.0
    vals = pdf[ff].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    vals = transform(vals, payload["feature_scaler"])
    actual = pdf["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    forecasts = np.zeros((len(pdf), PH), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for idx in range(len(pdf)):
            start = max(0, idx-HL+1); hist = vals[start:idx+1]
            if len(hist) < HL:
                hist = np.vstack([np.repeat(hist[:1], HL-len(hist), axis=0), hist])
            x = torch.as_tensor(hist[None,:,:], dtype=torch.float32, device=device)
            pred = model(x).detach().cpu().numpy()[0]
            pred = np.maximum(inverse_target(pred, payload["target_scaler"]), 0.0)
            forecasts[idx,0] = actual[idx]
            forecasts[idx,1:] = pred[:PH-1]
    return forecasts, actual


def run_mpc(df_voyage, forecasts, mpc_cfg, init_soc=0.55):
    generator = DualSideCasadiReferenceGenerator(mpc_cfg)
    dt_h = DT/3600.0; PH = mpc_cfg.prediction_horizon
    fcL=fcR=0.0; sL=sR=init_soc
    rows=[]; solve_times=[]
    for idx in range(len(df_voyage)):
        row = df_voyage.iloc[idx]; load = max(float(row["load_total_kw"]), 1e-6)
        ratio = min(0.95, max(0.05, float(row.get("load_left_kw",0.5*load))/load))
        fcst = forecasts[idx,:PH]; lf=fcst*ratio; rf=fcst*(1.0-ratio)
        t0 = time.perf_counter()
        res = generator.generate_dual(load_left_forecast_kw=lf, load_right_forecast_kw=rf,
                                       soc_left=sL, soc_right=sR, prev_fc_left_kw=fcL, prev_fc_right_kw=fcR)
        solve_times.append((time.perf_counter() - t0) * 1000)
        fcL=float(res.fuel_cell_ref_left_kw); fcR=float(res.fuel_cell_ref_right_kw)
        bL=float(res.battery_ref_left_kw); bR=float(res.battery_ref_right_kw)
        hc=903.0; sL-=bL*dt_h/hc; sR-=bR*dt_h/hc
        sL=max(0.05,min(1.0,sL)); sR=max(0.05,min(1.0,sR))
        rows.append({"fc":fcL+fcR,"batt":bL+bR,"soc":(sL+sR)/2.0,"solve_ms":solve_times[-1]})
    ts=pd.DataFrame(rows)
    # Print timing stats
    st=np.array(solve_times)
    print(f"  Solve time: mean={st.mean():.1f}ms  median={float(np.median(st)):.1f}ms  max={st.max():.1f}ms  min={st.min():.1f}ms  p99={float(np.percentile(st,99)):.1f}ms")
    return ts


def metrics(ts, load, label=""):
    fc=ts["fc"].values; batt=ts["batt"].values; soc=ts["soc"].values; r=np.diff(fc)
    return {
        "voyage": label,
        "fc_mean": float(np.mean(fc)), "fc_std": float(np.std(fc,ddof=1)),
        "fc_ramp_mean": float(np.mean(np.abs(r))), "fc_ramp_max": float(np.max(np.abs(r))) if len(r) else 0,
        "batt_std": float(np.std(batt,ddof=1)),
        "soc_start": float(soc[0]), "soc_end": float(soc[-1]),
        "soc_min": float(np.min(soc)), "soc_max": float(np.max(soc)),
        "fc_energy_kwh": float(np.sum(fc)*DT/3600),
        "batt_out_kwh": float(np.sum(np.maximum(batt,0))*DT/3600),
        "batt_in_kwh": float(np.sum(np.maximum(-batt,0))*DT/3600),
    }


def plot(ts, load, label, out_path):
    t=np.arange(len(ts))*DT/3600
    fig,axes=plt.subplots(2,1,figsize=(16,8),sharex=True)
    axes[0].plot(t,load,"k-",lw=0.8,alpha=0.9,label="Load")
    axes[0].plot(t,ts["fc"],"tab:red",lw=0.7,alpha=0.85,label="FC")
    axes[0].plot(t,ts["batt"],"tab:blue",lw=0.7,alpha=0.85,label="Batt")
    axes[0].set_ylabel("kW"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.15)
    axes[1].plot(t,ts["soc"],"tab:green",lw=0.8,label="SOC")
    axes[1].axhline(0.65,color="gray",ls="--",lw=0.5)
    axes[1].set_ylabel("SOC"); axes[1].set_xlabel("Hours"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.15)
    axes[0].set_title(label, fontsize=10)
    fig.tight_layout(); out_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out_path,dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--voyage",default=None,help="Specific voyage name to test")
    p.add_argument("--soc",type=float,default=0.55)
    p.add_argument("--q_soc",type=float,default=None)
    p.add_argument("--q_fc",type=float,default=None)
    args=p.parse_args()

    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"MPC: q_soc={args.q_soc or MPC_KWARGS['q_soc']}, q_fc={args.q_fc or MPC_KWARGS['q_fc']}, "
          f"q_batt={MPC_KWARGS['q_batt']}, q_ramp={MPC_KWARGS['q_ramp']}")

    if args.q_soc is not None: MPC_KWARGS["q_soc"] = args.q_soc
    if args.q_fc is not None: MPC_KWARGS["q_fc"] = args.q_fc

    print("Loading LSTM...")
    model, payload = load_checkpoint(LSTM_CKPT, device=device)
    print(f"  LSTM: history_len={payload['config']['history_len']}, pred_horizon={payload['config']['pred_horizon']}, "
          f"hidden={payload['config']['hidden_size']}, layers={payload['config']['num_layers']}")

    # Load test voyages
    with open(SPLIT,"r",encoding="utf-8") as f: split=json.load(f)
    test_voyages = [args.voyage] if args.voyage else split["test_voyages"]
    df_all = pd.read_csv(TRAIN_CSV)

    mpc_cfg = CasadiMPCConfig(**MPC_KWARGS)
    all_metrics = []

    for vname in test_voyages:
        df_v = df_all[df_all["voyage_name"]==vname].reset_index(drop=True)
        if len(df_v)==0: print(f"  {vname}: not found, skipping"); continue
        print(f"\n{vname}: {len(df_v)} rows ({len(df_v)*DT/3600:.1f}h)")

        forecasts, actual = forecast_voyage(df_v, model, payload, device)
        ts = run_mpc(df_v, forecasts, mpc_cfg, init_soc=args.soc)
        m = metrics(ts, actual, vname)
        all_metrics.append(m)
        print(f"  FC={m['fc_mean']:.1f}kW  SOC {m['soc_start']:.3f}->{m['soc_end']:.3f}  "
              f"Batt in={m['batt_in_kwh']:.0f}kWh out={m['batt_out_kwh']:.0f}kWh")

        label = vname.replace(" ","_")[:40]
        plot(ts, actual, vname, OUT_DIR/f"mpc_{label}.png")

    # Summary
    if len(all_metrics)>1:
        df_m = pd.DataFrame(all_metrics)
        print(f"\n=== Summary ({len(all_metrics)} voyages) ===")
        for col in ["fc_mean","fc_std","fc_ramp_mean","batt_std","soc_end","soc_min","fc_energy_kwh","batt_out_kwh","batt_in_kwh"]:
            vals=df_m[col]
            print(f"  {col}: {vals.mean():.1f} ± {vals.std():.1f}  [{vals.min():.1f}, {vals.max():.1f}]")
        df_m.to_csv(OUT_DIR/"metrics.csv",index=False)
        print(f"\nSaved: {OUT_DIR}/metrics.csv")

    print("Done.")


if __name__=="__main__": main()
