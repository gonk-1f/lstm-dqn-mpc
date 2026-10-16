"""Freeze the user-selected Test and a seeded grouped Train/Validation split.

Metadata and streaming hashes only. No payload parsing, environment or training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/processed/operating_dataset_independent_v2"
VERSION="splits_v1_seed42"
TEST=("parent_001_seg01","parent_004_seg01","parent_013_seg01","parent_032_seg01")
SEED=42
VALIDATION_PARENT_FRACTION=.2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(262144),b""):h.update(block)
    return h.hexdigest()


def save(frame,path):
    frame.to_csv(path,index=False,encoding="utf-8-sig",float_format="%.17g")


def month_allocation(counts,total):
    """One group per represented month, then proportional largest remainders."""
    if total<len(counts):raise ValueError("Not enough Validation groups to cover months")
    quota=pd.Series(1,index=counts.index,dtype=int)
    remaining=total-int(quota.sum())
    capacity=counts-quota
    if remaining:
        ideal=capacity/capacity.sum()*remaining
        extra=np.floor(ideal).astype(int)
        quota+=extra
        leftover=remaining-int(extra.sum())
        order=sorted(counts.index,key=lambda x:(-(ideal[x]-extra[x]),int(x)))
        for month in order[:leftover]:quota.loc[month]+=1
    assert int(quota.sum())==total and (quota<=counts).all()
    return quota


def main():
    output=DATA/"metadata"/VERSION
    if output.exists():raise FileExistsError(f"Frozen version already exists: {output}")
    source_manifest=DATA/"metadata/candidate_manifest.csv"
    m=pd.read_csv(source_manifest,float_precision="round_trip")
    assert len(m)==48 and m.candidate_id.is_unique
    assert set(TEST)<=set(m.candidate_id)
    assert not m.unknown_points.any() and not m.shore_points.any()
    test_parents=set(m.loc[m.candidate_id.isin(TEST),"parent_sample_id"])
    if set(m.loc[m.parent_sample_id.isin(test_parents),"candidate_id"])!=set(TEST):
        raise ValueError("User-selected Test has siblings: request a grouping decision first")
    # Check all candidate intervals before assigning roles. No Test data opened.
    temporal=m.assign(start=pd.to_datetime(m.start_timestamp),end=pd.to_datetime(m.end_timestamp))
    overlaps=[]
    for i,a in temporal.iterrows():
        for j,b in temporal.iloc[i+1:].iterrows():
            if min(a.end,b.end)>max(a.start,b.start):
                overlaps.append((a.candidate_id,b.candidate_id))
    if overlaps:raise ValueError(f"Overlapping source intervals require joint grouping: {overlaps}")
    remaining=m.loc[~m.parent_sample_id.isin(test_parents)].copy()
    remaining["month"]=pd.to_datetime(remaining.start_timestamp).dt.month
    groups=remaining.groupby("parent_sample_id",sort=True).agg(
        month=("month","first"),candidate_count=("candidate_id","size"))
    validation_group_count=int(round(len(groups)*VALIDATION_PARENT_FRACTION))
    month_counts=groups.groupby("month").size().sort_index()
    quota=month_allocation(month_counts,validation_group_count)
    rng=np.random.default_rng(SEED)
    validation_parents=[]
    for month,count in quota.items():
        pool=groups.index[groups.month.eq(month)].to_numpy()
        validation_parents.extend(rng.choice(pool,size=int(count),replace=False).tolist())
    m["split"]=np.where(m.parent_sample_id.isin(test_parents),"test",
        np.where(m.parent_sample_id.isin(validation_parents),"validation","train"))
    m["new_split"]=m["split"]
    m["group_id"]=m.parent_sample_id
    assert m.groupby("group_id").split.nunique().max()==1
    assert set(m.loc[m.split.eq("test"),"candidate_id"])==set(TEST)
    assert m.split.value_counts().to_dict()=={"train":36,"validation":8,"test":4}
    # Freeze bytes for both existing1s canonical curves and their30s previews.
    # Reading bytes to hash is not Test trajectory loading or policy evaluation.
    payloads=[]
    for row in m.itertuples(index=False):
        for kind,relative,expected in (("canonical_1s",row.relative_path,row.sha256),
                ("existing_30s_preview",row.preview_30s_relative_path,row.preview_30s_sha256)):
            path=(DATA/relative).resolve()
            if not path.is_relative_to(DATA.resolve()):raise ValueError(path)
            actual=sha(path)
            assert actual==expected,(row.candidate_id,kind)
            payloads.append(dict(candidate_id=row.candidate_id,split=row.split,kind=kind,
                                 relative_path=relative,sha256=actual))
    output.mkdir(parents=True)
    save(m,output/"sample_manifest.csv")
    for split in ("train","validation","test"):
        save(m.loc[m.split.eq(split)],output/f"{split}.csv")
        (output/f"{split}_ids.txt").write_text("\n".join(m.loc[m.split.eq(split),"candidate_id"])+"\n",encoding="utf-8")
    parent_roles=m.groupby(["parent_sample_id","split"],sort=True).agg(
        candidate_count=("candidate_id","size"),duration_s=("duration_s","sum")).reset_index()
    save(parent_roles,output/"parent_assignments.csv")
    save(pd.DataFrame(payloads),output/"payload_hashes.csv")
    summary=[]
    for split in ("train","validation","test"):
        f=m.loc[m.split.eq(split)]
        summary.append(dict(split=split,candidate_count=len(f),parent_count=f.group_id.nunique(),
            duration_h=f.duration_s.sum()/3600,minimum_duration_h=f.duration_s.min()/3600,
            maximum_duration_h=f.duration_s.max()/3600,peak_min_kw=f.load_peak_kw.min(),
            peak_max_kw=f.load_peak_kw.max(),nonzero_boundary_B_count=int(f.quality_class.eq("B").sum())))
    save(pd.DataFrame(summary),output/"split_summary.csv")
    assignment_map=dict(zip(m.candidate_id,m.split))
    policy=dict(schema="independent_v2_split_v1",dataset_version="operating_dataset_independent_v2",
        split_version=VERSION,seed=SEED,test_selected_by_user=list(TEST),
        validation_fraction_of_remaining_parent_groups=VALIDATION_PARENT_FRACTION,
        validation_group_count=validation_group_count,remaining_parent_count=len(groups),
        group_key="parent_sample_id",algorithm="seeded month-stratified group sampling, one parent group per month then proportional largest remainders; stable sorted IDs; numpy.default_rng(42)",
        month_group_counts={str(k):int(v) for k,v in month_counts.items()},
        validation_month_quota={str(k):int(v) for k,v in quota.items()},
        test_not_used_to_design_validation=True,no_training_or_controller_metrics_used=True,
        split_counts={s:int((m.split==s).sum()) for s in ("train","validation","test")},
        parent_split_counts={s:int(m.loc[m.split==s,"group_id"].nunique()) for s in ("train","validation","test")},
        source_candidate_manifest_sha256=sha(source_manifest),
        sample_manifest_sha256=sha(output/"sample_manifest.csv"),
        payload_hash_manifest_sha256=sha(output/"payload_hashes.csv"),
        cross_parent_time_overlap_pairs=overlaps,checked_payload_hashes=len(payloads),
        payload_path_base="dataset root, not split-manifest directory",numpy_version=np.__version__,
        environment_integration_status="PENDING_NO_FORMAL_ENTRY_CHANGED",
        economic_model_status="PENDING_ZHOU_EQUIVALENT_HYDROGEN_NOT_IMPLEMENTED",
        historical_checkpoints_valid_for_new_task=False,
        limitations=["Fixed user-selected Test must not be used for hyperparameter selection.",
            "Existing processed interpolation/deadband provenance is inherited unchanged.",
            "Sub-kW signed values and final30s partial-tail handling remain explicit environment integration items."])
    (output/"policy.json").write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding="utf-8")
    active=dict(split_version=VERSION,manifest_relative_path=f"metadata/{VERSION}/sample_manifest.csv",
        manifest_sha256=policy["sample_manifest_sha256"],policy_relative_path=f"metadata/{VERSION}/policy.json",
        split_counts=policy["split_counts"],status="SPLIT_FROZEN_ENV_INTEGRATION_PENDING")
    (DATA/"metadata/active_split.json").write_text(json.dumps(active,ensure_ascii=False,indent=2),encoding="utf-8")
    index_path=DATA/"metadata/review_index.csv"
    index=pd.read_csv(index_path)
    index["新划分"]=index["候选ID"].map(assignment_map)
    save(index,index_path)
    html_path=DATA/"index.html"
    html=html_path.read_text(encoding="utf-8")
    for cid,split in assignment_map.items():
        html=html.replace(f"<h2>{cid}</h2>",f"<h2>{cid} <small>[{split.upper()}]</small></h2>")
    banner=('<p><strong>划分已冻结：Train36 / Validation8 / Test4，seed42，同父子段不跨集合。</strong> '
            f'<a href="metadata/{VERSION}/sample_manifest.csv">最终划分清单</a>。尚未接入训练或新经济模型。</p>')
    html=html.replace("<h1>现有43样本裁剪后的48个候选</h1>","<h1>现有43样本裁剪后的48个候选</h1>"+banner)
    html=html.replace("等待用户选择Test，尚未正式划分。","Test已由用户选定，新划分已冻结；未启动训练。")
    html_path.write_text(html,encoding="utf-8")
    readme=DATA/"README.md"
    text=readme.read_text(encoding="utf-8")
    note=("## 已确认的最终划分\n\n"
        "用户固定Test：`parent_001_seg01`、`parent_004_seg01`、`parent_013_seg01`、`parent_032_seg01`。\n\n"
        "剩余39个父组按约8:2、seed42并按月份分层：Train31父组/36航段，Validation8父组/8航段，Test4父组/4航段。"
        "同父拆分段始终同组，未发现时间重叠。保留原候选清单作为生成记录，**新训练应以已冻结清单的split列为准，不能使用legacy_split**。\n\n"
        f"- [冻结的48条划分清单](metadata/{VERSION}/sample_manifest.csv)\n"
        f"- [Train清单](metadata/{VERSION}/train.csv)、[Validation清单](metadata/{VERSION}/validation.csv)、[Test清单](metadata/{VERSION}/test.csv)\n"
        f"- [划分方法、seed及哈希](metadata/{VERSION}/policy.json)、[分布摘要](metadata/{VERSION}/split_summary.csv)\n\n"
        "此划分尚未改变正式训练入口或接入Zhou经济模型。旧checkpoint曾接触重划分数据，不能在此新任务上续训或冒充未见Test的模型。\n\n")
    text=text.replace("## 快速查看",note+"## 快速查看",1)
    text=text.replace("选择Test。","审查数据。")
    text=text.replace("新划分尚未建立。","最终划分见上方冻结清单。")
    text=text.replace("**现在等待用户从这48个候选中选择Test ID。**","**Test已由用户指定，Train/Validation划分已冻结；正式环境与经济模型尚待接入。**")
    text=text.replace("没有划分新Train/Validation/Test，也没有接入新经济模型。","候选生成阶段未划分；现已按用户选择冻结分组，尚未接入新经济模型。")
    readme.write_text(text,encoding="utf-8")
    print(json.dumps(dict(output=str(output),counts=policy["split_counts"],
                         validation_ids=m.loc[m.split.eq("validation"),"candidate_id"].tolist(),
                         parent_counts=policy["parent_split_counts"]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
