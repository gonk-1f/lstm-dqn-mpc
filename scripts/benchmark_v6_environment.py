"""Bounded real-Train Queue/direct equivalence and CPU microbenchmark.

This does not launch a training run. At most 256 steps per prefix, 10 repeats,
and 4096 timed transitions per adapter are accepted. The fixed historical
reward is used for old/new equivalence because the frozen Queue adapter only
implements that mode. Optional Adam timing uses a disposable model and 32
real economic-mode transitions, never a formal training checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def validate_budget(sizes, *, repeats):
    if (not sizes or len(sizes)>8 or any(s<1 or s>256 for s in sizes)
            or repeats<1 or repeats>10 or sum(sizes)*repeats>4096):
        raise ValueError("bounded benchmark needs 1..8 prefix sizes in [1,256], repeats in [1,10], total <=4096 per adapter")


def _compare(left, right, path="root"):
    import numpy as np
    if isinstance(left, dict):
        assert left.keys() == right.keys(), f"{path}: field names differ"
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list,tuple,np.ndarray)):
        assert len(left)==len(right), f"{path}: length differs"
        for index,(a,b) in enumerate(zip(left,right)):
            _compare(a,b,f"{path}[{index}]")
    elif isinstance(left, (float,np.floating)):
        assert math.isclose(left,right,rel_tol=1e-11,abs_tol=1e-8), f"{path}: {left!r} != {right!r}"
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


def assert_equivalent(left, right):
    """Compare observations, ledgers, masks, reward, done and reason fields."""
    _compare(left, right)


def capture_step(env, result):
    observation, reward, terminated, truncated, info = result
    return dict(observation=observation.tolist(), reward=reward,
        terminated=terminated, truncated=truncated, mask=env.action_masks().tolist(),
        transition=asdict(info["transition"]))


def fixed_legal_prefix(episode, size):
    import numpy as np
    from v6_sb3.environment import VesselEnergyEnv
    env = VesselEnergyEnv(episode)
    actions = []
    try:
        env.reset(seed=42)
        for _ in range(size):
            legal = list(np.flatnonzero(env.action_masks()))
            # Hold when legal; otherwise the smallest legal absolute increment.
            action = min(legal,key=lambda index:(abs(int(index)-6),int(index)))
            actions.append(int(action))
            _,_,terminated,truncated,_ = env.step(int(action))
            if terminated or truncated:
                break
    finally:
        env.close()
    return actions


def run_prefix(env_class, episode, actions, *, reward_mode=None):
    kwargs = {} if reward_mode is None else dict(reward_mode=reward_mode)
    before_construct = time.perf_counter()
    env = env_class(episode, **kwargs)
    construct_s = time.perf_counter()-before_construct
    traces = []
    try:
        started = time.perf_counter()
        initial, info = env.reset(seed=42)
        reset_s = time.perf_counter()-started
        initial = dict(observation=initial.tolist(), mask=env.action_masks().tolist())
        prefix_started = time.perf_counter()
        step_times = []
        for action in actions:
            started = time.perf_counter()
            result = env.step(action)
            step_times.append(time.perf_counter()-started)
            traces.append(capture_step(env,result))
            if result[2] or result[3]:
                break
        prefix_wall_s = time.perf_counter()-prefix_started
    finally:
        started = time.perf_counter()
        env.close()
        close_s = time.perf_counter()-started
    return dict(initial=initial, traces=traces), dict(steps=len(traces), construct_s=construct_s,
        reset_s=reset_s, step_total_s=math.fsum(step_times),
        step_mean_s=statistics.mean(step_times), prefix_including_capture_wall_s=prefix_wall_s,
        close_s=close_s)


def economic_microbenchmark(episode, actions, *, adam_repeats):
    """32 real steps and 1..10 single Adam calls, with no learn() or checkpoint."""
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure
    from v6_sb3.direct_environment import DirectVesselEnv
    from v6_sb3.masked_dqn import MaskedDQN
    env = DirectVesselEnv(episode,reward_mode="zhou_economic_only")
    model = MaskedDQN(env=env,buffer_size=256,learning_starts=50000,batch_size=64,
        train_freq=32,gradient_steps=1,device="cpu",td_diagnostics=False)
    model.set_logger(configure(None, []))
    # This workbench does not collect formal training steps or call learn().
    # Samples are with replacement from <=32 genuine executed transitions.
    timings = []
    step_times = []
    costs, returns = [], []
    components = []
    try:
        observation,_ = env.reset(seed=42)
        for action in actions[:32]:
            started = time.perf_counter()
            next_observation,reward,terminated,truncated,info = env.step(action)
            step_times.append(time.perf_counter()-started)
            model.replay_buffer.add(observation[None],next_observation[None],np.array([action]),
                np.array([reward]),np.array([terminated or truncated]),[info])
            t = info["transition"]
            economic = math.fsum((t.executed_ledger.h2_cost_cny,
                t.executed_ledger.fuel_cell_degradation_cost_cny,
                t.executed_ledger.battery_degradation_cost_cny,
                t.soc_equivalent_hydrogen_cost_cny))
            costs.append(economic)
            returns.append(reward)
            components.append(dict(time_s=t.relative_time_s, economic_cost_cny=economic,
                h2_cost_cny=t.executed_ledger.h2_cost_cny,
                fc_degradation_cost_cny=t.executed_ledger.fuel_cell_degradation_cost_cny,
                battery_degradation_cost_cny=t.executed_ledger.battery_degradation_cost_cny,
                signed_soc_cost_cny=t.soc_equivalent_hydrogen_cost_cny,
                reward=reward,failure_penalty_cny=t.failure_penalty_equivalent_cny))
            observation = next_observation
            if terminated or truncated:
                break
        for _ in range(adam_repeats):
            started = time.perf_counter()
            model.train(gradient_steps=1,batch_size=64)
            timings.append(time.perf_counter()-started)
        assert model._n_updates == adam_repeats
        assert abs(math.fsum(returns)+math.fsum(costs)+math.fsum(r["failure_penalty_cny"] for r in components))<1e-8
        return dict(real_steps=len(costs), step_total_s=math.fsum(step_times),
            step_mean_s=statistics.mean(step_times), adam_updates=adam_repeats,
            adam_single_update_seconds=timings, adam_median_s=statistics.median(timings),
            replay_size=model.replay_buffer.size(), batch_size=64, torch_threads=torch.get_num_threads(),
            cost_cny=math.fsum(costs), return_cny=math.fsum(returns),
            return_identity_residual_cny=math.fsum(returns)+math.fsum(costs), step_evidence=components,
            note="Short fixed-policy prefix only. Tiny replay and first-update overhead differ from a mature 1M replay; no 500-episode runtime prediction.")
    finally:
        env.close()


def estimate_work_budget(manifest, measured, *, seed=42):
    """Metadata-only arithmetic; no environment, policy, or Test payload read."""
    from v6_sb3.episode_schedule import EpisodeSchedule
    train = sorted((r for r in manifest if r["split"]=="train"), key=lambda r:r["candidate_id"])
    validation = [r for r in manifest if r["split"]=="validation"]
    duration = {r["candidate_id"]:int(float(r["duration_s"])) for r in train}
    if len(duration)!=35 or len(validation)!=8:
        raise ValueError("budget requires frozen 35 Train / 8 Validation identities")
    micro = measured["economic_adam_microbenchmark"]
    raw_adam = micro["adam_single_update_seconds"]
    if len(raw_adam)<2:
        raise ValueError("at least two Adam timings needed to exclude first initialization")
    adam_s = statistics.median(raw_adam[1:])
    step_s = micro["step_mean_s"]
    val_steps = sum(int(float(r["duration_s"])) for r in validation)
    scenarios = []
    for episodes in (100,150,500):
        schedule = EpisodeSchedule(tuple(duration),episodes=episodes,seed=seed)
        order = [schedule.draw() for _ in range(episodes)]
        sampled_steps = sum(duration[sid] for sid in order)
        expected_steps = episodes*sum(duration.values())/len(duration)
        validation_calls = episodes//50
        def estimate(steps):
            updates = math.floor(steps/32)
            collection = steps*step_s
            adam = updates*adam_s
            return dict(formal_steps=steps,formal_adam_updates_or_expected_approximation=updates,
                collection_seconds=collection,adam_seconds=adam,
                collection_plus_adam_seconds=collection+adam,
                collection_plus_adam_hours=(collection+adam)/3600)
        warmup_s = 50000*step_s
        val_s = validation_calls*val_steps*step_s
        sampled = estimate(sampled_steps)
        scenarios.append(dict(episodes=episodes,seed=seed,full_completion_required_for_step_assumptions=True,
            extreme_all_shortest=estimate(episodes*min(duration.values())),
            extreme_all_longest=estimate(episodes*max(duration.values())),
            uniform_replacement_expected=estimate(expected_steps),
            fixed_seed42_sampled=sampled,seed42_voyage_counts=schedule.counts,
            warmup_real_steps=50000,warmup_physics_only_seconds=warmup_s,
            validation_calls=validation_calls,validation_full_completion_steps=validation_calls*val_steps,
            validation_physics_only_seconds=val_s,
            sampled_formal_plus_warmup_plus_validation_physics_adam_seconds=sampled["collection_plus_adam_seconds"]+warmup_s+val_s,
            sampled_formal_plus_warmup_plus_validation_physics_adam_hours=(sampled["collection_plus_adam_seconds"]+warmup_s+val_s)/3600))
    return dict(source="metadata durations plus saved short Train timings; no new step execution",
        step_mean_s=step_s,adam_raw_single_update_seconds=raw_adam,
        adam_excluded_first_observation=True,adam_remaining_median_s=adam_s,
        amortized_adam_seconds_per_formal_step=adam_s/32,
        assumed_formal_physics_plus_amortized_adam_seconds_per_step=step_s+adam_s/32,
        Train_duration_steps=dict(min=min(duration.values()),max=max(duration.values()),
            mean=sum(duration.values())/len(duration),sum=sum(duration.values())),
        Validation_metadata_only_duration_sum_steps=val_steps,scenarios=scenarios,
        limitations=["Scenario arithmetic is not a confidence interval or a reliable total-time forecast.",
            "No measured policy prediction, mature Replay, report, checkpoint, or validation inference overhead.",
            "Validation cost uses only manifest durations and Train step time, not measured Validation execution.",
            "Failures shorten episodes; formal step range is conditional on full completion.",
            "Expected-update count floors expected steps as an approximation; actual sampled count uses exact integer steps.",
            "Shared-machine short timing and fixed-FC behavior may not represent full trajectories."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/processed/operating_dataset_independent_v2")
    parser.add_argument("--output",type=Path,default=ROOT/"docs/audits/v6_episode500_20261029/environment_benchmark.json")
    parser.add_argument("--steps",type=int,nargs="+",default=[32,64,128])
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--sample-id",default=None,help="Train ID only; default is the shortest Train voyage")
    parser.add_argument("--measure-adam",action="store_true",help="Disposable CPU model, 32 real economic steps, no learn()")
    parser.add_argument("--adam-repeats",type=int,default=3)
    parser.add_argument("--torch-threads",type=int,default=1)
    parser.add_argument("--budget-only",action="store_true",help="Update saved --output with metadata-only budget; executes no steps or Adam")
    args = parser.parse_args(argv)
    validate_budget(args.steps,repeats=args.repeats)
    if not 1<=args.adam_repeats<=10 or not 1<=args.torch_threads<=16:
        parser.error("adam repeats must be [1,10]; torch threads [1,16]")
    if args.budget_only:
        import csv
        metadata = json.loads((args.dataset/"metadata/active_split.json").read_text(encoding="utf-8"))
        with (args.dataset/metadata["manifest_relative_path"]).open(encoding="utf-8-sig",newline="") as handle:
            manifest = list(csv.DictReader(handle))
        result = json.loads(args.output.read_text(encoding="utf-8"))
        result["budget_scenarios"] = estimate_work_budget(manifest,result)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps(dict(output=str(args.output),budget_only=True,new_real_steps=0,
            validation_metadata_duration_steps=result["budget_scenarios"]["Validation_metadata_only_duration_sum_steps"])))
        return
    import torch
    torch.set_num_threads(args.torch_threads)
    from v4.v6_dataset import V6TrainingDataset
    from v6_sb3.environment import VesselEnergyEnv
    from v6_sb3.direct_environment import DirectVesselEnv
    dataset = V6TrainingDataset.open(args.dataset)
    train = dataset._manifest[dataset._manifest["split"].eq("train")]
    if args.sample_id is None:
        selected = train.sort_values(["duration_s","candidate_id"]).iloc[0]
    else:
        matches = train[train["candidate_id"].eq(args.sample_id)]
        if len(matches)!=1:
            parser.error("--sample-id must identify one of the frozen Train voyages")
        selected = matches.iloc[0]
    # Load exactly one Train payload; do not load_train() all 35, Val, or Test.
    episode = dataset._load_episode(selected)
    if episode.split != "train":
        raise ValueError("only Train can be benchmarked")
    payload = args.dataset/str(selected.relative_path)
    actions = fixed_legal_prefix(episode,max(max(args.steps),32 if args.measure_adam else 0))
    rows = []
    compared_steps = 0
    for size in args.steps:
        for repeat in range(args.repeats):
            measurements = {}
            traces = {}
            # Alternate order to expose/reduce persistent first-run bias.
            order = ["queue","direct"] if repeat%2==0 else ["direct","queue"]
            for name in order:
                cls = VesselEnergyEnv if name=="queue" else DirectVesselEnv
                traces[name],measurements[name] = run_prefix(cls,episode,actions[:size],
                    reward_mode="v6_energy_soc_ramp" if name=="direct" else None)
            assert_equivalent(traces["queue"],traces["direct"])
            compared_steps += len(traces["queue"]["traces"])
            rows.append(dict(requested_steps=size,repeat=repeat+1,order=order,
                queue=measurements["queue"],direct=measurements["direct"],
                speedup_step_total_ratio=measurements["queue"]["step_total_s"]/measurements["direct"]["step_total_s"],
                equivalent=True))
    summaries = []
    for size in args.steps:
        subset = [r for r in rows if r["requested_steps"]==size]
        queue_median = statistics.median(r["queue"]["step_total_s"] for r in subset)
        direct_median = statistics.median(r["direct"]["step_total_s"] for r in subset)
        summaries.append(dict(requested_steps=size,repeats=args.repeats,
            queue_median_step_total_s=queue_median,direct_median_step_total_s=direct_median,
            ratio_of_median_step_totals=queue_median/direct_median,
            queue_median_reset_s=statistics.median(r["queue"]["reset_s"] for r in subset),
            direct_median_reset_s=statistics.median(r["direct"]["reset_s"] for r in subset)))
    result = dict(sample_id=episode.sample_id,split=episode.split,voyage_duration_s=episode.duration_s,
        payload_path=payload.resolve().relative_to(ROOT).as_posix(),
        payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),python=sys.executable,
        platform=platform.platform(),torch_threads=torch.get_num_threads(),
        benchmark_reward_mode="v6_energy_soc_ramp", action_policy="hold_if_legal_else_smallest_abs_increment; pre-generated via frozen Queue",
        action_prefix=actions,action_generation_real_steps=len(actions),
        compared_step_pairs=compared_steps,total_timed_real_steps=2*compared_steps,
        summaries=summaries,runs=rows,validation_payloads_opened=0,test_payloads_opened=0,
        formal_training_started=False,disposable_adam_timing_performed=args.measure_adam,
        limitations=["One short Train prefix and fixed policy, no full-voyage equivalence or throughput inference.",
            "Step timer excludes trace serialization and assertions; separate capture-inclusive wall time is also recorded.",
            "Queue equivalence uses its frozen historical reward. Optional economic/Adam workbench is measured separately.",
            "OS scheduling, shared machine load and cold caches affect short timings; report measured ratios only."])
    if args.measure_adam:
        result["economic_adam_microbenchmark"] = economic_microbenchmark(episode,actions,adam_repeats=args.adam_repeats)
        result["total_additional_microbenchmark_real_steps"] = result["economic_adam_microbenchmark"]["real_steps"]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output),sample_id=episode.sample_id,
        compared_step_pairs=compared_steps,summaries=summaries),ensure_ascii=False))


if __name__=="__main__":
    main()
