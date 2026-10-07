"""Two-stage beta and replay-cadence study with per-round greedy selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC

from .dqn import DirectPowerDDQN
from .monitored_training import _write_json, run_monitored_training
from .review import _manifest_hashes, _trajectory_plot
from .soc_beta_study import _profile
from .train import _default_data_root


BETAS = (250.0, 500.0, 1000.0, 2000.0)
CADENCES = {"A": "episode16", "B": "replay16", "C": "replay8"}


def select_beta(reports: dict[str, dict]) -> str | None:
    eligible = {label: report for label, report in reports.items() if report["best_checkpoint"] is not None}
    return min(eligible, key=lambda label: eligible[label]["best_checkpoint"]["validation_comparable_cost_cny"]) if eligible else None


def _csv_history(report: dict, destination: Path) -> None:
    fields = ["round", "epsilon", "exploratory_completed", "greedy_train_completed", "greedy_validation_completed",
              "validation_cost_cny", "qualified", "economic_optimizer_updates", "economic_replay_insertions",
              "environment_transitions"]
    metrics = ("fc_zero_fraction", "onboard_soc_mean", "onboard_soc_min_including_failed_prefix", "terminal_onboard_soc_mean")
    bins = ("below_0p4", "0p4_to_0p6", "above_0p6_below_0p79", "at_least_0p79")
    fields += [f"{split}_{key}" for split in ("train", "validation") for key in (*metrics, *bins)]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["rounds"]:
            validation = row["greedy_validation"]
            result = {
                "round":row["round"], "epsilon":row["epsilon"],
                "exploratory_completed":row["exploratory_train"]["completed"],
                "greedy_train_completed":row["greedy_train"]["completed"],
                "greedy_validation_completed":None if validation is None else validation["completed"],
                "validation_cost_cny":None if validation is None else validation["cost_cny"],
                "qualified":row["qualified_checkpoint"],
                "economic_optimizer_updates":row["economic_optimizer_updates_cumulative"],
                "economic_replay_insertions":row["economic_replay_insertions_cumulative"],
                "environment_transitions":row["training_environment_transitions_cumulative"],
            }
            for split, summary in (("train", row["greedy_train"]), ("validation", validation)):
                for key in metrics:
                    result[f"{split}_{key}"] = None if summary is None else summary[key]
                for key in bins:
                    result[f"{split}_{key}"] = None if summary is None else summary["soc_time_occupancy"]["fractions"][key]
            writer.writerow(result)


def _worker(args) -> int:
    roots = tuple(_default_data_root(name) for name in (
        "operating_dataset_zero_boundary_v2", "operating_dataset_zero_boundary_v2_ais", "operating_dataset_zero_boundary_v2_modes",
    ))
    before = _manifest_hashes(roots)
    dataset = FormalTrainingDataset.open(*roots)
    if len(dataset.load_train()) != 30 or len(dataset.load_validation()) != 8:
        raise ValueError("formal study requires exactly 30 Train and 8 Validation samples")
    _, report = run_monitored_training(
        dataset, output_dir=args.output_dir, rounds=args.rounds, beta_soc=args.beta,
        cadence=args.cadence, target_mode=args.target_mode, target_interval=1000,
        seed=42, batch_size=64, epsilon_start=1.0, epsilon_end=0.05, progress_every_steps=50,
    )
    report["manifest_sha256"] = before
    report["best_checkpoint_replay_verified"] = False
    if report["best_checkpoint"] is not None:
        checkpoint = torch.load(args.output_dir/"best_agent.pt", map_location="cpu", weights_only=True)
        best_agent = DirectPowerDDQN(seed=42)
        best_agent.online.load_state_dict(checkpoint["model_state"])
        accountant = EconomicMPC(nominal_cost_cny=1.0)
        profiles = {}
        profile_steps = 0
        for split, episodes in (("train",dataset.load_train()),("validation",dataset.load_validation())):
            profiles[split] = {str(ep.sample_id):_profile(ep,best_agent,accountant,args.beta) for ep in episodes}
            rows = list(profiles[split].values())
            if not all(row["completed"] for row in rows):
                raise RuntimeError("saved qualified checkpoint failed independent greedy replay")
            profile_steps += sum(len(row["soc_after"]) for row in rows)
            cost = sum(row["comparable_cost_cny"] for row in rows)
            expected = report["best_checkpoint"][split]["cost_cny"]
            if not math.isclose(cost, expected, rel_tol=1e-10):
                raise RuntimeError("saved checkpoint cost differs from selection evidence")
        _write_json(args.output_dir/"best_profiles.json",profiles)
        for row in profiles["validation"].values():
            _trajectory_plot(row,args.output_dir/f"best_{row['sample_id']}_power_soc.png",args.beta)
        report["best_checkpoint_replay_verified"] = True
        report["execution_counts"]["best_profile_replay_environment_transitions"] = profile_steps
    if dataset.opened_test_payloads != 0 or _manifest_hashes(roots) != before:
        raise RuntimeError("Test opened or manifests changed during study")
    report["dataset_manifests_unchanged"] = True
    _write_json(args.output_dir/"report.json",report)
    _csv_history(report,args.output_dir/"round_metrics.csv")
    print(f"WORKER_DONE beta={args.beta:g} cadence={args.cadence} best={report['best_checkpoint'] and report['best_checkpoint']['round']} Test=0",flush=True)
    return 0


def _parallel_jobs(jobs: list[dict], workers: int, rounds: int, resume: bool) -> dict[str, dict]:
    pending = list(jobs)
    running = {}
    reports = {}
    errors = []
    worktree = Path(__file__).resolve().parents[2]
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            path = job["path"]
            if (path/"report.json").exists():
                old = json.loads((path/"report.json").read_text(encoding="utf-8"))
                if resume and old["completed_training"] and old.get("dataset_manifests_unchanged"):
                    hp=old["hyperparameters"]
                    if hp["beta_soc"] != job["beta"] or hp["cadence"] != job["cadence"] or hp["target_mode"] != job["target"] or hp["rounds"] != rounds:
                        raise ValueError("resume configuration differs")
                    reports[job["label"]] = old
                    print(f"RESUME completed={job['label']}",flush=True)
                    continue
                raise FileExistsError(f"existing result: {path}")
            if (path/"round_history.json").exists():
                raise FileExistsError(f"partial run cannot silently restart within the same budget: {path}")
            path.mkdir(parents=True,exist_ok=True)
            handle=(path/"train.log").open("w",encoding="utf-8")
            command=[sys.executable,"-X","utf8","-u","-m","v4.staged_study","--worker",
                     "--output-dir",str(path),"--beta",str(job["beta"]),"--cadence",job["cadence"],
                     "--target-mode",job["target"],"--rounds",str(rounds)]
            process=subprocess.Popen(command,cwd=worktree,stdout=handle,stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            running[job["label"]]={"process":process,"handle":handle,"job":job,"reported_round":0}
            print(f"START {job['label']} pid={process.pid}",flush=True)
        for label,item in list(running.items()):
            history_path=item["job"]["path"]/"round_history.json"
            try:
                partial=json.loads(history_path.read_text(encoding="utf-8"))
                for row in partial["rounds"][item["reported_round"]:]:
                    val=row["greedy_validation"]
                    print(f"STATUS {label} round={row['round']} exploratory={row['exploratory_train']['completed']}/30 greedy_train={row['greedy_train']['completed']}/30 validation={'SKIPPED' if val is None else str(val['completed'])+'/8'} cost={None if val is None else val['cost_cny']} updates={row['economic_optimizer_updates_cumulative']} best={partial['best_checkpoint'] and partial['best_checkpoint']['round']}",flush=True)
                item["reported_round"]=len(partial["rounds"])
            except (FileNotFoundError,PermissionError,json.JSONDecodeError):
                pass
            code=item["process"].poll()
            if code is None:
                continue
            item["handle"].close()
            if code != 0:
                errors.append(f"{label}: exit {code}; see {item['job']['path']/'train.log'}")
                print(f"ERROR {errors[-1]}",flush=True)
            else:
                reports[label]=json.loads((item["job"]["path"]/"report.json").read_text(encoding="utf-8"))
                print(f"DONE {label}",flush=True)
            del running[label]
        if pending or running:
            time.sleep(5)
    if errors:
        raise RuntimeError("; ".join(errors))
    return reports


def compact_report(report: dict) -> dict:
    rows=report["rounds"]
    val_rows=[row["greedy_validation"] for row in rows if row["greedy_validation"] is not None]
    return {
        "hyperparameters":report["hyperparameters"],"best_checkpoint":report["best_checkpoint"],
        "first_qualification":report["first_qualification"],"eligible_rounds":report["eligible_rounds"],
        "historical_best_greedy_train_completed":max(row["greedy_train"]["completed"] for row in rows),
        "historical_best_greedy_validation_completed":max((row["completed"] for row in val_rows),default=None),
        "final_greedy_train":rows[-1]["greedy_train"],"final_greedy_validation":rows[-1]["greedy_validation"],
        "late_regression":report["best_checkpoint"] is not None and not rows[-1]["qualified_checkpoint"],
        "execution_counts":report["execution_counts"],"elapsed_seconds":report["elapsed_seconds"],
        "test_payloads_opened":report["test_payloads_opened"],"dataset_manifests_unchanged":report["dataset_manifests_unchanged"],
        "best_checkpoint_replay_verified":report["best_checkpoint_replay_verified"],
    }


def _plots(reports: dict[str,dict], destination: Path, prefix: str) -> None:
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    diag,extra=plt.subplots(4,1,figsize=(10,10),sharex=True)
    updates,upaxes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for label,report in reports.items():
        rows=report["rounds"];indices=[row["round"] for row in rows]
        values=([row["exploratory_train"]["completed"] for row in rows],
                [row["greedy_train"]["completed"] for row in rows],
                [None if row["greedy_validation"] is None else row["greedy_validation"]["completed"] for row in rows])
        for axis,series in zip(axes,values):
            axis.plot(indices,series,label=label,linewidth=1.2)
        best=report["best_checkpoint"]
        if best is not None:
            axes[2].scatter([best["round"]],[8],marker="*",s=80)
        extra[0].plot(indices,[row["greedy_train"]["soc_time_occupancy"]["fractions"]["below_0p4"] for row in rows],label=label)
        extra[1].plot(indices,[row["greedy_train"]["soc_time_occupancy"]["fractions"]["at_least_0p79"] for row in rows],label=label)
        extra[2].plot(indices,[row["greedy_train"]["fc_zero_fraction"] for row in rows],label=label)
        extra[3].plot(indices,[None if row["greedy_validation"] is None else row["greedy_validation"]["cost_cny"] for row in rows],label=label)
        optimizer=[row["economic_optimizer_updates_cumulative"] for row in rows]
        upaxes[0].plot(optimizer,values[1],label=label)
        upaxes[1].plot(optimizer,values[2],label=label)
    for axis,label,limit in zip(axes,("Exploratory Train / 30","Greedy Train / 30","Greedy Validation / 8"),(31,31,8.5)):
        axis.set_ylabel(label);axis.set_ylim(-.2,limit);axis.grid(alpha=.2);axis.legend(fontsize=8)
    axes[-1].set_xlabel("Round; Validation is skipped unless Train=30/30")
    for axis,label in zip(extra,("Greedy Train SOC<0.4","Greedy Train SOC>=0.79","Greedy Train FC=0","Validation comparable CNY")):
        axis.set_ylabel(label);axis.grid(alpha=.2);axis.legend(fontsize=8)
    extra[-1].set_xlabel("Round; gaps mean no complete Validation cost")
    for axis,label in zip(upaxes,("Greedy Train / 30","Greedy Validation / 8")):
        axis.set_ylabel(label);axis.grid(alpha=.2);axis.legend(fontsize=8)
    upaxes[-1].set_xlabel("Actual cumulative economic optimizer updates")
    for figure,name in ((fig,"completion"),(diag,"soc_fc_cost"),(updates,"optimizer_axis")):
        figure.tight_layout();figure.savefig(destination/f"{prefix}_{name}.png",dpi=150);plt.close(figure)


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/v4_staged_beta_cadence_seed42"))
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--beta",type=float)
    parser.add_argument("--cadence",choices=tuple(CADENCES.values()),default="episode16")
    parser.add_argument("--target-mode",choices=("round","optimizer"),default="round")
    parser.add_argument("--rounds",type=int,default=40)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    args.output_dir=args.output_dir.resolve()
    if args.worker:
        if args.beta is None:
            parser.error("worker needs --beta")
        return _worker(args)
    if not 1<=args.workers<=4 or not 1<=args.rounds<=40:
        parser.error("workers must be 1..4; rounds must be 1..40")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    jobs=[{"label":f"beta_{int(beta)}","beta":beta,"cadence":"episode16","target":"round",
           "path":args.output_dir/"stage1"/f"beta_{int(beta)}"} for beta in BETAS]
    stage1=_parallel_jobs(jobs,args.workers,args.rounds,args.resume)
    _plots(stage1,args.output_dir,"stage1")
    chosen=select_beta(stage1)
    summary={"stage1":{key:compact_report(value) for key,value in stage1.items()},
             "chosen_beta_label":chosen,"stage2":None,
             "stage2_status":"not_started" if chosen else "SKIPPED_no_qualified_beta",
             "test_payloads_opened":0}
    _write_json(args.output_dir/"stage1_summary.json",summary)
    print(f"STAGE1_DONE chosen={chosen}",flush=True)
    if chosen is not None:
        beta=stage1[chosen]["hyperparameters"]["beta_soc"]
        jobs=[{"label":label,"beta":beta,"cadence":cadence,"target":"optimizer",
               "path":args.output_dir/"stage2"/f"{label}_{cadence}"} for label,cadence in CADENCES.items()]
        stage2=_parallel_jobs(jobs,min(args.workers,3),args.rounds,args.resume)
        _plots(stage2,args.output_dir,"stage2")
        summary["stage2"]={key:compact_report(value) for key,value in stage2.items()}
        summary["stage2_status"]="completed"
    _write_json(args.output_dir/"study_summary.json",summary)
    print(f"STUDY_DONE stage2={summary['stage2_status']} Test=0",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
