"""Plot economic TD3 training curves from saved episode records only."""
import argparse, csv, math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = args.run_dir / "training_episodes.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle: rows = list(csv.DictReader(handle))
    if not rows: return
    ep = [int(r["episode"]) for r in rows]
    reward = [float(r["raw_reward_sum"]) for r in rows]
    reward_step = [float(r["mean_reward_per_step"]) for r in rows]
    physical_cost = [float(r["total_cost_cny"]) for r in rows]
    reward_cost = [float(r["reward_total_cost_cny"]) for r in rows]
    soc = [float(r["soc_final"]) for r in rows]
    def rolling(v, n=20):
        return [math.nan if i+1<n else sum(v[i+1-n:i+1])/n for i in range(len(v))]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax[0,0].plot(ep, reward, alpha=.35, label="Episode raw return")
    ax[0,0].plot(ep, rolling(reward), label="20-episode mean")
    ax[0,0].set_title("Raw episode return (higher is better)")
    ax[0,1].plot(ep, reward_step, alpha=.35, label="Mean return per step")
    ax[0,1].plot(ep, rolling(reward_step), label="20-episode mean")
    ax[0,1].set_title("Length-normalized reward")
    ax[1,0].plot(ep, physical_cost, alpha=.35, label="Physical CNY ledger")
    ax[1,0].plot(ep, rolling(physical_cost), label="20-episode mean")
    ax[1,0].plot(ep, reward_cost, alpha=.35, label="Reward objective cost")
    ax[1,0].set_title("Voyage costs (lower is better)")
    ax[1,1].plot(ep, soc, label="Terminal SOC")
    ax[1,1].axhline(.5, linestyle="--", label="Target SOC 0.5")
    ax[1,1].axhline(.2, linestyle=":", label="Hard lower bound")
    ax[1,1].axhline(.8, linestyle=":", label="Hard upper bound")
    ax[1,1].set_title("Terminal SOC")
    for axis in ax.flat:
        axis.set_xlabel("Training episode"); axis.grid(alpha=.2); axis.legend()
    fig.savefig(args.run_dir / "economic_ems_reward_curves.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__": main()