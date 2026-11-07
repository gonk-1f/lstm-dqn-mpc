# TD3 Economic EMS v2

## Reward

This SOC5 experiment uses the negative sum of actual hydrogen cost, FC transient/low/high variable degradation, battery degradation, and per-step SOC deviation.

r_raw = -(C_H2 + C_FC_transient + C_FC_low + C_FC_high + C_LB_deg + 5.0*(SOC_next-0.5)^2).

TD3 receives 0.001*r_raw. The complete CNY ledger includes all four FC degradation terms and the signed battery-equivalent hydrogen cost for reporting. Discharge is a cost and charge is a credit in that ledger. Training reward includes action-dependent shutdown transient degradation. The environment settles the fixed start-stop cycle separately.

The step SOC coefficient is 5.0 CNY per squared deviation, as in the fixed-voyage run without battery-equivalent hydrogen in the reward. The last physical step uses the same formula as any other step. Terminal SOC is logged separately and used for Validation8 selection. A physical failure receives a raw reward of −20,000 (−20 to TD3).

## Action parameterization

For each state, the normalized Actor action a in [-1,1] maps affinely to the intersection of FC rating, battery charge/discharge power, and next-step SOC constraints:

P_FC = P_FC,min(s) + (a+1)/2 * (P_FC,max(s)-P_FC,min(s)).

The physical environment retains its fail-closed check. A degenerate feasible interval still has only one attainable power. Logs distinguish the normalized action, mapped power, feasible interval width, and executed power. The first four observations retain the original scaling. The fifth is elapsed steps divided by voyage length, from 0 at reset to 1 after a complete voyage.

## TD3 configuration

- Huber TD3, delta=1; twin critics; policy delay=2.
- batch=32; replay=1,000,000; gamma=1; tau=0.001.
- Actor/Critic learning rate=1e-4.
- train_freq=32 environment steps; gradient_steps=1.
- learning_starts=30,000 environment steps. SB3 uses random actions and applies its configured exploration noise during warm-up. No gradient update occurs during this phase.
- Gaussian exploration noise std=0.05 in normalized action coordinates. Away from the action boundaries, this is a power standard deviation of 2.5% of the current feasible interval width.
- Target policy smoothing sigma=0.2, clip=0.5.
- One environment and one PyTorch CPU thread, with CUDA as the training device. The episode limit is 500 and the environment-step safety ceiling is 20,000,000. Train35 lengths are 3,272–36,660 steps, so even 500 repeats of the longest voyage fit under the ceiling.
- With uninterrupted default training, the first Critic update occurs at step 30,016 and the first Actor and target updates at step 30,048.

## Voyage protocol

- Train: uniformly sample one of the 35 Train voyages at each reset, with replacement. Each voyage starts at t=0 and SOC=0.6 and executes its own length on success. A genuine physical failure terminates the attempt and is reported separately from successful completions. Replay accumulates across episodes.
- Fixed diagnostic: deterministic, full `parent_005_seg01` rollout every 20 episode endings, without exploration noise.
- Model selection: full deterministic Validation8 every 100 episodes.
- Test4 remains untouched until final frozen-checkpoint evaluation.

The fixed-voyage run provides a comparison for the sampling change. Training includes load profiles with different lengths and patterns. Independent Validation8 and the frozen Test4 quantify transfer.

## Research basis

- Wu et al. (2025) use Huber loss, an update every 32 steps, a policy delay of two, and learning rate 0.0001 for uniform FC control. Their increment-based action and historical voyage distribution differ from this protocol. Figure 5 aggregates 21 converged seeds out of 28. [Published paper, Table 1 and §§4.2–4.3](https://discovery.ucl.ac.uk/10204302/1/1-s2.0-S0360319925004239-main.pdf)
- Huang et al. (2023) compare action representations and train on one driving cycle with independent evaluation cycles. Liu et al. (2025) combine economic and degradation objectives with SOC management using MPC and TD3. These results motivate checking the action representation and independent economics. [Huang et al.](https://www.sciencedirect.com/science/article/pii/S0360544223025719) · [Liu et al.](https://www.sciencedirect.com/science/article/pii/S0196890425006053)
- The reference implementation of TD3 defaults to 25,000 random-action steps. The 30,000-step warm-up and exploration standard deviation of 0.05 are project settings. [Fujimoto et al. reference code](https://github.com/sfujim/TD3/blob/master/main.py) · [SB3 rollout and sampling code](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/off_policy_algorithm.html)

## Run

Use the existing SB3/RL Zoo 2.9.2 interpreter in PyCharm and run `run_td3_train_economic_ems.py` from this worktree. The launcher starts a fresh run under `outputs/td3_economic_ems_train35_soc5_no_bat_eq_500ep`. To pause, create an empty `pause_after_episode.request` in the specific run directory. Zoo saves the model and Replay after the current episode. For independent evaluation from PowerShell in this worktree, supply the versioned checkpoint explicitly.

```powershell
python .\run_td3_fixed_eval.py --reward-mode economic_ems --model-path '<checkpoint.zip>'
python .\run_td3_validate.py --reward-mode economic_ems --model-path '<checkpoint.zip>'
python .\run_td3_test.py --reward-mode economic_ems --model-path '<frozen-final-checkpoint.zip>'
```

Select the checkpoint after Validation8 has eight successes, all final SOC values within 0.05 of 0.5, and all minimum SOC values at least 0.2. Among eligible candidates, minimize mean full-voyage physical CNY cost. If no checkpoint qualifies, the run produces no best-validation model.

## Acceptance criteria

Do not select a checkpoint from training reward alone. Require full-voyage completion, hard SOC compliance, terminal SOC close to 0.5, no persistent Actor boundary saturation, stable FC output with transient degradation measured, battery response to load residuals, and separate reporting of H2, FC degradation, LB equivalent hydrogen, LB degradation, and total physical CNY cost.

For each attempt, divide the raw cumulative reward by the actual number of environment steps. Plot each attempt, highlight physical failures, and show a 20-episode moving mean after warm-up. Use the noise-free fixed-voyage diagnostics and independent Validation8 economics alongside that curve. Assess economic advantage at comparable final SOC and completion rates.
