"""Event-triggered DQN test on real voyage segments. Compares DQN vs MPC-only."""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path; from collections import deque

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch

SRC = Path(__file__).resolve().parents[1]; PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from forecasting.lstm_load_predictor import load_checkpoint, add_time_features, inverse_target, transform
from mpc.controllers.reference_generator import DualSideCasadiReferenceGenerator
from mpc.solvers.casadi_solver import CasadiMPCConfig
from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig

LSTM_CKPT = PROJ / "outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/best_lstm_load_predictor.pt"
DQN_CKPT = PROJ / "outputs/checkpoints/dqn/best_ship_dqn.pt"
SPLIT = PROJ / "outputs/config/voyage_split_721.json"
TRAIN_CSV = PROJ / "data/processed/aligned_timeseries.csv"
OUT_DIR = PROJ / "outputs/dqn_test"

Q_SOC_BASE, Q_FC, Q_BATT, Q_RAMP_BASE = 40.0, 0.001, 0.01, 0.08
Q_SOC_MIN, Q_SOC_MAX = 20.0, 60.0; Q_RAMP_MIN, Q_RAMP_MAX = 0.03, 0.15
MPC_BASE = dict(prediction_horizon=6, dt_hours=30/3600.0, battery_capacity_kwh=1806.0,
                fuel_cell_max_kw=560.0, fuel_cell_ramp_kw=48.0, soc_target=0.65,
                soc_min=0.2, soc_max=0.8, use_raw_objective=True, raw_soc_squared=False,
                q_soc=Q_SOC_BASE, q_fc=Q_FC, q_batt=Q_BATT, q_ramp=Q_RAMP_BASE,
                ipopt_max_iter=100, ipopt_tol=1e-4)

DQ_SOC=[-15,-10,-5,0,5,10,15]; DQ_RAMP=[-0.04,-0.02,0,0.02,0.04]
ACTION_TABLE=[(ds,dr) for ds in DQ_SOC for dr in DQ_RAMP]
N_ACTIONS=len(ACTION_TABLE); STATE_DIM=8; MC_SAMPLES=20

TRIG_SOC_LOW=0.40; TRIG_SOC_HIGH=0.75; TRIG_SIGMA_MAX=12.0
TRIG_LOAD_TREND=15.0; TRIG_SOC_TREND=0.01


def prepare_features(df):
    out=add_time_features(df.copy()).reset_index(drop=True); gi=[list(out.index)]
    for rc,dc in[("load_total_kw","delta_load_total"),("load_left_kw","delta_load_left"),
                 ("load_right_kw","delta_load_right"),("speed_knots","delta_speed")]:
        out[dc]=0.0
        if rc in out.columns:
            for il in gi: out.loc[il,dc]=out.loc[il,rc].astype(float).diff().fillna(0.0)
    for w in[3,6]:
        for col in[f"rolling_mean_load_total_{w}",f"rolling_std_load_total_{w}"]: out[col]=0.0
        for il in gi:
            s=out.loc[il,"load_total_kw"].astype(float)
            out.loc[il,f"rolling_mean_load_total_{w}"]=s.rolling(w,min_periods=1).mean()
            out.loc[il,f"rolling_std_load_total_{w}"]=s.rolling(w,min_periods=1).std().fillna(0.0)
    return out


def lstm_mc_predict(lm,lp,prep_df,device):
    cfg=lp["config"]; HL=cfg["history_len"]; feats=list(lp["features"])
    ff=list(feats); sd=len(lp["feature_scaler"]["mean"])
    for col in["time_sin","time_cos"]:
        if len(ff)<sd and col in prep_df.columns and col not in ff: ff.append(col)
    vals=prep_df[ff].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    vals=transform(vals,lp["feature_scaler"])
    actual=prep_df["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    mu=np.zeros((len(prep_df),6)); sigma=np.zeros((len(prep_df),6))
    lm.train(); lm.input_dropout.p=0.2
    for idx in range(len(prep_df)):
        s=max(0,idx-HL+1); h=vals[s:idx+1]
        if len(h)<HL: h=np.vstack([np.repeat(h[:1],HL-len(h),axis=0),h])
        x=torch.as_tensor(h[None,:,:],dtype=torch.float32,device=device)
        samples=[]
        for _ in range(MC_SAMPLES):
            with torch.no_grad(): o=lm.forward(x).cpu().numpy()[0]
            samples.append(np.maximum(inverse_target(o,lp["target_scaler"]),0.0))
        st=np.stack(samples,axis=0); mu[idx]=st.mean(axis=0); sigma[idx]=st.std(axis=0)
    return mu,sigma,actual


def build_state(soc,fc,load,mu_pred,sigma_pred,st,lt):
    return np.array([soc,fc/560.0,load/120.0,
                     mu_pred.mean()/120.0,sigma_pred.mean()/20.0,sigma_pred.max()/20.0,
                     st/0.02,lt/20.0],dtype=np.float32)


def clamp_q(qs,qr):
    return max(Q_SOC_MIN,min(Q_SOC_MAX,qs)),max(Q_RAMP_MIN,min(Q_RAMP_MAX,qr))


def run_test(df_voyage,mu,sigma,actual,da,init_soc,use_dqn=True):
    sL=sR=init_soc; fcL=fcR=0.0; prev_soc=init_soc; prev_load=float(actual[18])
    records=[]; trigger_count=0
    for t in range(len(df_voyage)):
        if t+18>=len(actual): break
        idx=t+18; load_now=float(actual[idx]); mu_now=mu[idx]; sigma_now=sigma[idx]
        soc_now=(sL+sR)/2.0; fc_now=fcL+fcR
        st=soc_now-prev_soc; lt=load_now-prev_load

        triggered=False
        if use_dqn:
            triggered=(soc_now<TRIG_SOC_LOW or soc_now>TRIG_SOC_HIGH or
                       float(sigma_now.max())>TRIG_SIGMA_MAX or
                       abs(lt)>TRIG_LOAD_TREND or abs(st)>TRIG_SOC_TREND)
        if triggered:
            trigger_count+=1
            sv=build_state(soc_now,fc_now,load_now,mu_now,sigma_now,st,lt)
            dq_soc,dq_ramp=ACTION_TABLE[da.greedy_action(sv)]
        else:
            dq_soc,dq_ramp=0.0,0.0

        qs,qr=clamp_q(Q_SOC_BASE+dq_soc,Q_RAMP_BASE+dq_ramp)
        mpc_cfg=CasadiMPCConfig(**{**MPC_BASE,"q_soc":qs,"q_ramp":qr})
        fcst=np.zeros(6); fcst[0]=max(load_now,1e-6)
        for j in range(min(5,len(mu_now)-1)): fcst[j+1]=mu_now[j]
        gen=DualSideCasadiReferenceGenerator(mpc_cfg)
        res=gen.generate_dual(load_left_forecast_kw=fcst*0.5,load_right_forecast_kw=fcst*0.5,
                               soc_left=sL,soc_right=sR,prev_fc_left_kw=fcL,prev_fc_right_kw=fcR)
        fcL=float(res.fuel_cell_ref_left_kw); fcR=float(res.fuel_cell_ref_right_kw)
        bL=float(res.battery_ref_left_kw); bR=float(res.battery_ref_right_kw)
        records.append({"load":load_now,"fc":fcL+fcR,"batt":bL+bR,"soc":soc_now,
                        "dq_soc":dq_soc,"dq_ramp":dq_ramp,"q_soc":qs,"q_ramp":qr,
                        "sigma_max":float(sigma_now.max()),"triggered":triggered})
        dh,hc=30.0/3600.0,903.0; sL-=bL*dh/hc; sR-=bR*dh/hc
        sL=max(0.05,min(1.0,sL)); sR=max(0.05,min(1.0,sR))
        prev_soc=soc_now; prev_load=load_now
    ts=pd.DataFrame(records)
    return ts,trigger_count


def metrics(ts,label=""):
    fc=ts["fc"].values; batt=ts["batt"].values; soc=ts["soc"].values; r=np.diff(fc)
    return{"voyage":label,
           "fc_mean":float(np.mean(fc)),"fc_std":float(np.std(fc,ddof=1)),
           "batt_std":float(np.std(batt,ddof=1)),
           "fc_ramp_mean":float(np.mean(np.abs(r))),"fc_ramp_max":float(np.max(np.abs(r))) if len(r) else 0,
           "soc_start":float(soc[0]),"soc_end":float(soc[-1]),
           "soc_min":float(np.min(soc)),"soc_max":float(np.max(soc)),
           "fc_energy_kwh":float(np.sum(fc)*30/3600),
           "batt_out_kwh":float(np.sum(np.maximum(batt,0))*30/3600),
           "batt_in_kwh":float(np.sum(np.maximum(-batt,0))*30/3600),
           "triggers":int(ts["triggered"].sum()) if "triggered" in ts.columns else 0}


def plot_compare(ts_mpc,ts_dqn,load,label,out_path):
    t=np.arange(len(ts_mpc))*30/3600
    fig,axes=plt.subplots(2,2,figsize=(18,10))
    # SOC
    axes[0,0].plot(t,ts_mpc["soc"],"tab:blue",lw=0.8,label="MPC-only")
    axes[0,0].plot(t,ts_dqn["soc"],"tab:red",lw=0.8,label="DQN")
    axes[0,0].axhline(0.65,color="gray",ls="--",lw=0.5)
    axes[0,0].set_ylabel("SOC"); axes[0,0].set_title("SOC: MPC vs DQN")
    axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.15)
    # FC
    axes[0,1].plot(t,ts_mpc["fc"],"tab:blue",lw=0.7,alpha=0.7,label="MPC-only")
    axes[0,1].plot(t,ts_dqn["fc"],"tab:red",lw=0.7,alpha=0.7,label="DQN")
    axes[0,1].plot(t,load,"k-",lw=0.5,alpha=0.5,label="Load")
    axes[0,1].set_ylabel("FC (kW)"); axes[0,1].set_title("Fuel Cell Power")
    axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.15)
    # Batt
    axes[1,0].plot(t,ts_mpc["batt"],"tab:blue",lw=0.7,alpha=0.7,label="MPC-only")
    axes[1,0].plot(t,ts_dqn["batt"],"tab:red",lw=0.7,alpha=0.7,label="DQN")
    axes[1,0].set_ylabel("Batt (kW)"); axes[1,0].set_title("Battery Power")
    axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.15)
    # DQN weights
    axes[1,1].plot(t,ts_dqn["q_soc"],"tab:orange",lw=0.5,label="q_soc")
    ax2=axes[1,1].twinx(); ax2.plot(t,ts_dqn["q_ramp"],"tab:purple",lw=0.5,label="q_ramp")
    axes[1,1].set_ylabel("q_soc"); ax2.set_ylabel("q_ramp")
    axes[1,1].set_xlabel("Hours"); axes[1,1].set_title("DQN Weights")
    axes[1,1].grid(alpha=0.15)
    fig.suptitle(label,fontsize=12); fig.tight_layout()
    out_path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out_path,dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--voyage",default=None)
    p.add_argument("--soc",type=float,default=0.55)
    args=p.parse_args()

    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | SOC_init={args.soc}")

    print("Loading LSTM..."); lm,lp=load_checkpoint(LSTM_CKPT,device=device)
    lm.dropout_rate=0.2; lm.input_dropout=torch.nn.Dropout(0.2)

    print("Loading DQN...")
    dc=DQNTrainConfig(network_type="sine_kan",sine_kan_latent_dim=16,sine_kan_width=32,
                      sine_kan_grid_size=8,sine_kan_dueling=True,sine_kan_dropout=0.1,
                      double_dqn=True,dueling_dqn=True,device=device)
    da=DQNAgent(state_dim=STATE_DIM,action_dim=N_ACTIONS,config=dc)
    da.q_net.load_state_dict(torch.load(DQN_CKPT,map_location=device)); da.q_net.eval()

    with open(SPLIT,"r",encoding="utf-8") as f: split=json.load(f)
    test_voyages=[args.voyage] if args.voyage else split["test_voyages"]
    df_all=pd.read_csv(TRAIN_CSV)

    all_metrics_mpc=[]; all_metrics_dqn=[]

    for vname in test_voyages:
        df_v=df_all[df_all["voyage_name"]==vname].reset_index(drop=True)
        if len(df_v)==0: continue
        print(f"\n{vname}: {len(df_v)} rows ({len(df_v)*30/3600:.1f}h)")
        prep=prepare_features(df_v); mu,sigma,actual=lstm_mc_predict(lm,lp,prep,device)

        # MPC-only
        ts_mpc,_=run_test(df_v,mu,sigma,actual,da,args.soc,use_dqn=False)
        m_mpc=metrics(ts_mpc,vname); all_metrics_mpc.append(m_mpc)

        # DQN
        ts_dqn,ntrig=run_test(df_v,mu,sigma,actual,da,args.soc,use_dqn=True)
        m_dqn=metrics(ts_dqn,vname); m_dqn["triggers"]=int(ntrig); all_metrics_dqn.append(m_dqn)

        print(f"  MPC:  SOC {m_mpc['soc_start']:.3f}->{m_mpc['soc_end']:.3f}  "
              f"FC={m_mpc['fc_mean']:.1f}kW  Batt in={m_mpc['batt_in_kwh']:.0f} out={m_mpc['batt_out_kwh']:.0f}")
        print(f"  DQN:  SOC {m_dqn['soc_start']:.3f}->{m_dqn['soc_end']:.3f}  "
              f"FC={m_dqn['fc_mean']:.1f}kW  Batt in={m_dqn['batt_in_kwh']:.0f} out={m_dqn['batt_out_kwh']:.0f}  "
              f"triggers={ntrig}/{len(ts_dqn)}")

        label=vname.replace(" ","_")[:40]
        plot_compare(ts_mpc,ts_dqn,actual[18:18+len(ts_mpc)],vname,OUT_DIR/f"compare_{label}.png")

    if len(all_metrics_mpc)>1:
        for name,ml in[("MPC-only",all_metrics_mpc),("DQN",all_metrics_dqn)]:
            df_m=pd.DataFrame(ml)
            print(f"\n=== {name} Summary ({len(df_m)} voyages) ===")
            for col in["fc_mean","fc_std","batt_std","soc_end","soc_min","fc_energy_kwh","batt_out_kwh","batt_in_kwh"]:
                v=df_m[col]
                print(f"  {col}: {v.mean():.1f} ± {v.std():.1f}  [{v.min():.1f}, {v.max():.1f}]")
            df_m.to_csv(OUT_DIR/f"metrics_{name.lower().replace(' ','_')}.csv",index=False)
    print(f"\nSaved: {OUT_DIR}")


if __name__=="__main__": main()
