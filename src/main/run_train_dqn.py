"""Event-triggered DQN training on real voyage segments. DQN adjusts q_soc, q_ramp."""

from __future__ import annotations
import json, sys
from collections import deque
from pathlib import Path

import numpy as np, pandas as pd, torch

SRC = Path(__file__).resolve().parents[1]; PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from forecasting.lstm_load_predictor import load_checkpoint, add_time_features, inverse_target, transform
from mpc.controllers.reference_generator import DualSideCasadiReferenceGenerator
from mpc.solvers.casadi_solver import CasadiMPCConfig
from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig
from dqn.memory.replay_buffer import ReplayBuffer
from dqn.policies.epsilon_greedy import EpsilonGreedyPolicy

# ── Paths ─────────────────────────────────────────────────────────────
LSTM_CKPT = PROJ / "outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/best_lstm_load_predictor.pt"
SPLIT = PROJ / "outputs/config/voyage_split_721.json"
TRAIN_CSV = PROJ / "data/processed/aligned_timeseries.csv"
OUT_DIR = PROJ / "outputs/checkpoints/dqn"

# ── MPC baseline ──────────────────────────────────────────────────────
Q_SOC_BASE, Q_FC, Q_BATT, Q_RAMP_BASE = 40.0, 0.001, 0.01, 0.08
Q_SOC_MIN, Q_SOC_MAX = 20.0, 60.0
Q_RAMP_MIN, Q_RAMP_MAX = 0.03, 0.15
MPC_BASE = dict(prediction_horizon=6, dt_hours=30/3600.0, battery_capacity_kwh=1806.0,
                fuel_cell_max_kw=560.0, fuel_cell_ramp_kw=48.0, soc_target=0.65,
                soc_min=0.2, soc_max=0.8, use_raw_objective=True, raw_soc_squared=False,
                q_soc=Q_SOC_BASE, q_fc=Q_FC, q_batt=Q_BATT, q_ramp=Q_RAMP_BASE,
                ipopt_max_iter=100, ipopt_tol=1e-4)

# ── DQN actions ───────────────────────────────────────────────────────
DQ_SOC = [-15, -10, -5, 0, 5, 10, 15]
DQ_RAMP = [-0.04, -0.02, 0, 0.02, 0.04]
ACTION_TABLE = [(ds, dr) for ds in DQ_SOC for dr in DQ_RAMP]
N_ACTIONS = len(ACTION_TABLE)  # 35

# ── Triggers ──────────────────────────────────────────────────────────
TRIG_SOC_LOW=0.40; TRIG_SOC_HIGH=0.75; TRIG_SIGMA_MAX=12.0
TRIG_LOAD_TREND=15.0; TRIG_SOC_TREND=0.01

# ── DQN hyperparams ───────────────────────────────────────────────────
STATE_DIM=8; MC_SAMPLES=20
BATCH_SIZE=64; BUFFER_SIZE=100000; WARMUP=10000; TARGET_SYNC=500
LR=2e-4; GRAD_CLIP=0.5; GAMMA=0.95
EPS_START=1.0; EPS_MIN=0.05; EPS_DECAY=0.99995
NUM_EP=150; EP_HOURS=6.0
DT_ENV=5.0; DT_MPC=30.0; MPC_EVERY=int(DT_MPC/DT_ENV)

# ── Reward ────────────────────────────────────────────────────────────
R_SOC_SCALE=5.0; R_ECO_SCALE=0.002; R_SMOOTH_SCALE=0.005
R_ACTION_SOC_SCALE=0.001; R_ACTION_RAMP_SCALE=0.05


def prepare_features(df):
    out=add_time_features(df.copy()).reset_index(drop=True); gi=[list(out.index)]
    for rc,dc in [("load_total_kw","delta_load_total"),("load_left_kw","delta_load_left"),
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


def lstm_mc_predict(lm, lp, prep_df, device):
    cfg=lp["config"]; HL=cfg["history_len"]; feats=list(lp["features"])
    ff=list(feats); sd=len(lp["feature_scaler"]["mean"])
    for col in["time_sin","time_cos"]:
        if len(ff)<sd and col in prep_df.columns and col not in ff: ff.append(col)
    vals=prep_df[ff].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    vals=transform(vals,lp["feature_scaler"])
    actual=prep_df["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    mu,sigma=np.zeros((len(prep_df),6)),np.zeros((len(prep_df),6))
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


def build_state(soc,fc_power,load,mu_pred,sigma_pred,soc_trend,load_trend):
    return np.array([soc,fc_power/560.0,load/120.0,
                     mu_pred.mean()/120.0,sigma_pred.mean()/20.0,sigma_pred.max()/20.0,
                     soc_trend/0.02,load_trend/20.0],dtype=np.float32)


def decode_action(i): return ACTION_TABLE[i]


def clamp_q(qs,qr):
    return max(Q_SOC_MIN,min(Q_SOC_MAX,qs)),max(Q_RAMP_MIN,min(Q_RAMP_MAX,qr))


def check_trigger(soc,sigma_max,load_trend,soc_trend):
    if soc<TRIG_SOC_LOW: return True
    if soc>TRIG_SOC_HIGH: return True
    if sigma_max>TRIG_SIGMA_MAX: return True
    if abs(load_trend)>TRIG_LOAD_TREND: return True
    if abs(soc_trend)>TRIG_SOC_TREND: return True
    return False


def compute_reward(soc,fc_power,delta_fc,dq_soc,dq_ramp):
    rs=-R_SOC_SCALE*(soc-0.65)**2
    re=-R_ECO_SCALE*fc_power
    rsmooth=-R_SMOOTH_SCALE*abs(delta_fc)
    ra=-R_ACTION_SOC_SCALE*dq_soc**2-R_ACTION_RAMP_SCALE*dq_ramp**2
    return rs+re+rsmooth+ra


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Event-triggered DQN | {NUM_EP}ep x {EP_HOURS}h")
    print(f"MPC: q_soc={Q_SOC_BASE} q_fc={Q_FC} q_batt={Q_BATT} q_ramp={Q_RAMP_BASE}")
    print(f"Actions: {len(DQ_SOC)}x{len(DQ_RAMP)}={N_ACTIONS} | gamma={GAMMA} | eps {EPS_START}->{EPS_MIN}")

    print("Loading LSTM...")
    lm,lp=load_checkpoint(LSTM_CKPT,device=device)
    lm.dropout_rate=0.2; lm.input_dropout=torch.nn.Dropout(0.2)

    with open(SPLIT,"r",encoding="utf-8") as f: split=json.load(f)
    train_voyages=split["train_voyages"]
    df_all=pd.read_csv(TRAIN_CSV)
    print(f"Training voyages: {len(train_voyages)} ({sum((df_all['voyage_name']==v).sum()*30/3600 for v in train_voyages):.0f}h total)")

    # Build episode start indices per voyage
    voyage_episodes={}
    for v in train_voyages:
        df_v=df_all[df_all["voyage_name"]==v]
        ep_len=int(EP_HOURS*3600/30)
        n_ep=max(1,(len(df_v)-ep_len-18)//ep_len)
        voyage_episodes[v]=[(v,i*ep_len) for i in range(n_ep)]
    all_eps=[e for vl in voyage_episodes.values() for e in vl]
    print(f"Available episodes: {len(all_eps)}")

    dqn_cfg=DQNTrainConfig(
        network_type="sine_kan",sine_kan_latent_dim=16,sine_kan_width=32,
        sine_kan_grid_size=8,sine_kan_dueling=True,sine_kan_dropout=0.1,
        lr=LR,batch_size=BATCH_SIZE,warmup_steps=WARMUP,buffer_size=BUFFER_SIZE,
        loss_type="huber",target_sync_interval=TARGET_SYNC,grad_clip_norm=GRAD_CLIP,
        epsilon_start=EPS_START,epsilon_min=EPS_MIN,epsilon_decay=EPS_DECAY,
        double_dqn=True,dueling_dqn=True,device=device)
    dqn_agent=DQNAgent(state_dim=STATE_DIM,action_dim=N_ACTIONS,config=dqn_cfg)
    replay_buffer=ReplayBuffer(BUFFER_SIZE)
    policy=EpsilonGreedyPolicy(EPS_START,EPS_MIN,EPS_DECAY)

    ep_rewards,loss_log=[],[]
    action_counts=np.zeros(N_ACTIONS,dtype=int)
    trigger_counts={"soc_low":0,"soc_high":0,"sigma":0,"load":0,"soc_trend":0}
    best_reward=-float("inf"); step=0

    for ep in range(NUM_EP):
        vname,si=all_eps[np.random.randint(0,len(all_eps))]
        df_v=df_all[df_all["voyage_name"]==vname].iloc[si:si+int(EP_HOURS*3600/30)+18].reset_index(drop=True)
        if len(df_v)<int(EP_HOURS*3600/30)+18: continue
        prep=prepare_features(df_v)
        mu,sigma,actual=lstm_mc_predict(lm,lp,prep,device)

        sL=sR=np.random.uniform(0.30,0.75); fcL=fcR=0.0; prev_fc=0.0
        current_action=DQ_SOC.index(0)*len(DQ_RAMP)+DQ_RAMP.index(0)
        dq_soc=dq_ramp=0.0; ep_reward=0.0; t_5s=0
        prev_soc=(sL+sR)/2.0; prev_load=float(actual[18])

        # Bootstrap
        idx0=18; mu0=mu[idx0]; load0=float(actual[idx0])
        mpc_cfg0=CasadiMPCConfig(**MPC_BASE)
        fcst0=np.zeros(6); fcst0[0]=max(load0,1e-6)
        for j in range(min(5,len(mu0)-1)): fcst0[j+1]=mu0[j]
        gen0=DualSideCasadiReferenceGenerator(mpc_cfg0)
        res0=gen0.generate_dual(load_left_forecast_kw=fcst0*0.5,load_right_forecast_kw=fcst0*0.5,
                                 soc_left=sL,soc_right=sR,prev_fc_left_kw=fcL,prev_fc_right_kw=fcR)
        fcL=float(res0.fuel_cell_ref_left_kw); fcR=float(res0.fuel_cell_ref_right_kw)

        ep_len=min(int(EP_HOURS*3600/30),len(mu)-18)
        for t_30s in range(ep_len):
            idx=t_30s+18; mu_now=mu[idx]; sigma_now=sigma[idx]
            soc_now=(sL+sR)/2.0; fc_now=fcL+fcR; load_now=float(actual[idx])
            soc_trend=soc_now-prev_soc; load_trend=load_now-prev_load

            sigma_max_now=float(sigma_now.max())
            triggered=check_trigger(soc_now,sigma_max_now,load_trend,soc_trend)

            if triggered:
                if soc_now<TRIG_SOC_LOW: trigger_counts["soc_low"]+=1
                if soc_now>TRIG_SOC_HIGH: trigger_counts["soc_high"]+=1
                if sigma_max_now>TRIG_SIGMA_MAX: trigger_counts["sigma"]+=1
                if abs(load_trend)>TRIG_LOAD_TREND: trigger_counts["load"]+=1
                if abs(soc_trend)>TRIG_SOC_TREND: trigger_counts["soc_trend"]+=1
                sv=build_state(soc_now,fc_now,load_now,mu_now,sigma_now,soc_trend,load_trend)
                greedy=dqn_agent.greedy_action(sv)
                current_action=policy.select_action(greedy,N_ACTIONS,warmup=step<WARMUP)
                dq_soc,dq_ramp=decode_action(current_action)
            else:
                dq_soc,dq_ramp=0.0,0.0
                current_action=DQ_SOC.index(0)*len(DQ_RAMP)+DQ_RAMP.index(0)

            action_counts[current_action]+=1
            if step>=WARMUP: policy.step()

            qs,qr=clamp_q(Q_SOC_BASE+dq_soc,Q_RAMP_BASE+dq_ramp)
            mpc_cfg=CasadiMPCConfig(**{**MPC_BASE,"q_soc":qs,"q_ramp":qr})
            fcst=np.zeros(6); fcst[0]=max(load_now,1e-6)
            for j in range(min(5,len(mu_now)-1)): fcst[j+1]=mu_now[j]
            gen=DualSideCasadiReferenceGenerator(mpc_cfg)
            res=gen.generate_dual(load_left_forecast_kw=fcst*0.5,load_right_forecast_kw=fcst*0.5,
                                   soc_left=sL,soc_right=sR,prev_fc_left_kw=fcL,prev_fc_right_kw=fcR)
            fcL=float(res.fuel_cell_ref_left_kw); fcR=float(res.fuel_cell_ref_right_kw)
            bL=float(res.battery_ref_left_kw); bR=float(res.battery_ref_right_kw)

            for k in range(MPC_EVERY):
                fc_total=fcL+fcR; soc_now_sub=(sL+sR)/2.0
                sv=build_state(soc_now_sub,fc_total,load_now,mu_now,sigma_now,soc_trend,load_trend)
                dqn_agent.observe_states_for_normalization(sv)
                dh,hc=DT_ENV/3600.0,903.0
                sL-=bL*dh/hc; sR-=bR*dh/hc
                sL=max(0.05,min(1.0,sL)); sR=max(0.05,min(1.0,sR))
                soc_new=(sL+sR)/2.0
                delta_fc=fc_total-prev_fc
                reward=compute_reward(soc_new,fc_total,delta_fc,dq_soc,dq_ramp)
                ns=build_state(soc_new,fc_total,load_now,mu_now,sigma_now,soc_trend,load_trend)
                done=(ep>=NUM_EP-1 and t_30s>=ep_len-1 and k>=MPC_EVERY-1)
                replay_buffer.push(sv,current_action,reward,done,ns)
                dqn_agent.observe_states_for_normalization(ns)
                if len(replay_buffer)>=BATCH_SIZE and step>=WARMUP:
                    batch=replay_buffer.sample(BATCH_SIZE)
                    loss=dqn_agent.update(batch)
                    loss_log.append(float(loss))
                    if len(loss_log)%TARGET_SYNC==0:
                        dqn_agent.target_q_net.load_state_dict(dqn_agent.q_net.state_dict())
                prev_fc=fc_total; step+=1; t_5s+=1; ep_reward+=reward
            prev_soc=soc_now; prev_load=load_now

        mean_r=ep_reward/max(1,t_5s)
        ep_rewards.append(float(mean_r))
        if mean_r>best_reward: best_reward=mean_r; dqn_agent.save(OUT_DIR/"best_ship_dqn.pt")
        if ep%10==0 or ep<5:
            nz=np.count_nonzero(action_counts)
            top3=np.argsort(action_counts)[-3:][::-1]
            top3_str="  ".join(f"a{a}={action_counts[a]}" for a in top3)
            trig_total=sum(trigger_counts.values())
            print(f"  ep {ep:3d}/{NUM_EP}: reward={mean_r:.4f}  eps={policy.epsilon:.4f}  "
                  f"updates={len(loss_log)}  actions={nz}/35  triggers={trig_total}  top={top3_str}")

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    dqn_agent.save(OUT_DIR/"last_ship_dqn.pt")
    summary={"episodes":NUM_EP,"total_steps":step,"loss_updates":len(loss_log),
             "best_reward":float(best_reward),"final_epsilon":policy.epsilon,"device":device,
             "q_soc_base":Q_SOC_BASE,"q_fc":Q_FC,"q_batt":Q_BATT,"q_ramp_base":Q_RAMP_BASE,
             "trigger_stats":trigger_counts,
             "action_distribution":{int(k):int(v) for k,v in enumerate(action_counts) if v>0}}
    with(OUT_DIR/"train_summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    with(OUT_DIR/"train_logs.json").open("w",encoding="utf-8") as f:
        json.dump({"episode_rewards":ep_rewards,"loss":loss_log,
                   "action_counts":[int(x) for x in action_counts],
                   "trigger_counts":trigger_counts},f,ensure_ascii=False)
    print(f"\nDone. Best: {best_reward:.4f}  Triggers: {trigger_counts}")


if __name__=="__main__": main()
