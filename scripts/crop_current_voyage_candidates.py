"""Review-only crops of the EXISTING43 frozen processed curves; no raw rebuild."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from independent_candidate_figures import _font_settings, mark_pdf_operation_started

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
POWER = DATA / "operating_dataset_zero_boundary_v2"
MODES = DATA / "operating_dataset_zero_boundary_v2_modes_v3"
COMPONENTS = DATA / "operating_dataset_zero_boundary_v2_modes"
OUTPUT = DATA / "operating_dataset_independent_v2"

# 0-based rows of the current frozen30s decision grid, end exclusive.
# These are the user's named crops, not an automatic fragmentation algorithm.
CROPS = {
    "007": [(10, None, "REMOVE_OPENING_UNKNOWN")],
    "011": [(58, None, "AFTER_OPENING_SHORE")],
    "017": [(15, None, "REMOVE_OPENING_UNKNOWN")],
    "029": [(0, 355, "END_AT_START_OF_LONG_NEAR_ZERO_2_950H"),
            (1013, None, "AFTER_UNKNOWN_8_442H")],
    "033": [(0, 370, "END_AT_LONG_NEAR_ZERO_3_075H")],
    "034": [(858, 1183, "MAIN_ACTIVITY_7_150_TO9_850H_VERIFY_TIME_AXIS")],
    "037": [(0, 1223, "BEFORE_FIRST_SHORE_10_183H_NO_EXTRA_IDLE_SPLITS")],
    "041": [(13, None, "REMOVE_OPENING_UNKNOWN")],
    "042": [(10, None, "REMOVE_OPENING_UNKNOWN")],
    "046": [(14, 3128, "REMOVE_OPENING_UNKNOWN_END_BEFORE_INTERNAL_UNKNOWN"),
            (3135, None, "AFTER_INTERNAL_UNKNOWN_NO_STATE_BRIDGE")],
    "009": [(0, 317, "BEFORE_FIRST_SHORE_2_633H"),
            (771, None, "AFTER_LAST_SHORE_6_425H")],
    "010": [(0, 374, "BEFORE_FIRST_SHORE_3_108H"),
            (734, None, "AFTER_LAST_UNKNOWN_6_117H_NOT6_033H")],
    "019": [(0, 350, "BEFORE_SHORE_2_908H"),
            (659, None, "AFTER_SHORE_5_492H")],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def boundary(value, at_original_boundary):
    if abs(value) < 1e-12:
        return "INHERITED_PROCESSED_ZERO_NOT_NEW_RAW_OBSERVATION" if at_original_boundary else "EXISTING_PROCESSED_ZERO"
    return "NEAR_ZERO_VALUE_UNCHANGED" if abs(value) <= 1 else "NONZERO_VALUE_UNCHANGED"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-unpublished-review",action="store_true")
    args=parser.parse_args()
    if ((OUTPUT/"metadata/active_split.json").exists() or
            any((OUTPUT/"metadata").glob("splits_v*"))):
        raise FileExistsError("Dataset split is frozen; build a new version instead of refreshing it")
    if OUTPUT.exists() and not args.refresh_unpublished_review:
        raise FileExistsError(f"Never overwrite an existing candidate review: {OUTPUT}")
    if OUTPUT.exists():
        old_policy=json.loads((OUTPUT/"metadata/policy.json").read_text(encoding="utf-8"))
        assert old_policy["candidate_status"]=="REVIEW_ONLY_NO_FINAL_SPLIT" and old_policy["source_parent_count"]==43
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest = pd.read_csv(POWER / "metadata/sample_manifest.csv")
    mode_manifest = pd.read_csv(MODES / "metadata/sample_manifest.csv").set_index("sample_id")
    assert len(manifest) == 43
    records, sources, original_frames, plots, excluded = [], [], {}, {}, []
    zero_audit, checks = [], []
    for row in manifest.itertuples(index=False):
        source_path = POWER / row.relative_path
        mode_path = MODES / row.relative_path
        assert sha(source_path) == row.sha256
        assert sha(mode_path) == mode_manifest.loc[row.sample_id, "sha256"]
        source = pd.read_csv(source_path,float_precision="round_trip")
        source_load_text=pd.read_csv(source_path,usecols=["load_total_kw"],dtype=str).load_total_kw
        mode = pd.read_csv(mode_path)
        comp = pd.read_csv(COMPONENTS / row.relative_path)
        old_time = mode.time_s.to_numpy(dtype=float)
        old_indices = old_time.astype(int)
        assert np.array_equal(np.diff(old_time), np.full(len(old_time)-1, 30.))
        assert np.array_equal(source.time_s.iloc[old_indices].to_numpy(), old_time)
        # Select the existing processed values directly, not FC+BMS reconstruction.
        old_load = source.load_total_kw.iloc[old_indices].to_numpy(dtype=float)
        original_frames[row.sample_id] = source[["time_s","load_total_kw"]]
        sources.extend([dict(path=str(source_path),sha256=sha(source_path)),
                        dict(path=str(mode_path),sha256=sha(mode_path)),
                        dict(path=str(COMPONENTS/row.relative_path),sha256=sha(COMPONENTS/row.relative_path))])
        zero_audit.append(dict(parent_sample_id=row.sample_id,
            processed_1s_exact_zero_count=int(source.load_total_kw.eq(0).sum()),
            processed_1s_internal_exact_zero_count=int(source.load_total_kw.iloc[1:-1].eq(0).sum()),
            current_30s_exact_zero_count=int((old_load==0).sum()),
            current_30s_nonzero_abs_le1_count=int(((abs(old_load)<=1)&(old_load!=0)).sum()),
            current_30s_negative_subkw_count=int(((old_load<0)&(old_load>=-1)).sum())))
        suffix=row.sample_id[-3:]
        plan=CROPS.get(suffix,[(0,None,"UNCHANGED_EXISTING_SAMPLE")])
        kept=np.zeros(len(mode),dtype=bool)
        for number,(start,end,reason) in enumerate(plan,1):
            end=len(mode) if end is None else end
            kept[start:end]=True
            indices=old_indices[start:end]
            candidate=source.iloc[indices][["timestamp","time_s","load_total_kw"]].copy()
            stamps=pd.DatetimeIndex(pd.to_datetime(candidate.timestamp))
            candidate["source_time_s"]=candidate.time_s
            candidate["time_s"]-=candidate.time_s.iloc[0]
            candidate["source_1s_row"]=indices
            candidate["source_30s_row"]=np.arange(start,end)
            candidate["legacy_mode"]=mode["mode"].iloc[start:end].to_numpy()
            candidate["legacy_mode_reason"]=mode.mode_reason.iloc[start:end].to_numpy()
            candidate["source_fc_kw"]=comp.p_fc_total_kw.iloc[start:end].to_numpy()
            candidate["source_battery_bus_kw"]=comp.p_batt_bus_kw.iloc[start:end].to_numpy()
            candidate["speed_kn"]=comp.speed_kn.iloc[start:end].to_numpy()
            cid=f"parent_{suffix}_seg{number:02d}"
            candidate.insert(0,"candidate_id",cid)
            candidate.insert(1,"parent_sample_id",row.sample_id)
            first_1s=int(indices[0])
            final_1s=len(source)-1 if end==len(mode) else int(indices[-1])
            full=source.iloc[first_1s:final_1s+1][["timestamp","time_s","load_total_kw"]].copy()
            full["source_time_s"]=full.time_s
            full["time_s"]-=full.time_s.iloc[0]
            full["source_1s_row"]=np.arange(first_1s,final_1s+1)
            full.insert(0,"candidate_id",cid)
            full.insert(1,"parent_sample_id",row.sample_id)
            unknown=int(candidate.legacy_mode.eq("unknown").sum())
            shore=int(candidate.legacy_mode.isin(["shore_pending","shore_charging"]).sum())
            flags=[]
            if unknown: flags.append("UNRESOLVED_UNKNOWN_RETAINED_REVIEW_ONLY_NO_ENV_CONNECTION")
            if shore: flags.append("SHORE_STILL_PRESENT_REVIEW_ONLY")
            load=full.load_total_kw.to_numpy()
            if load.min() < -1: flags.append("NEGATIVE_BELOW_MINUS1_REVIEW_ONLY")
            if ((load<0)&(load>=-1)).any():flags.append("SMALL_NEGATIVE_RETAINED_NOT_SILENTLY_ZEROED")
            if abs(load[0])>1 or abs(load[-1])>1:flags.append("NONZERO_BOUNDARY_NOT_MODIFIED")
            if suffix=="034": flags.append("MAIN_ACTIVITY_TIME_AXIS_SELECTION_REQUIRES_USER_REVIEW")
            grade="C" if unknown or shore or load.min() < -1 else (
                "A" if abs(load[0])<=1 and abs(load[-1])<=1 else "B")
            preview_relative=f"candidates/{cid}.csv"
            save_csv(candidate,OUTPUT/preview_relative)
            # The complete existing1s curve is canonical. Never drop its tail
            # merely because a30s preview does not sample the old zero endpoint.
            relative=f"candidates/{cid}.csv.gz"
            full_csv=full.copy()
            full_csv["load_total_kw"]=source_load_text.iloc[first_1s:final_1s+1].to_numpy()
            save_csv(full_csv,OUTPUT/relative)
            reread=pd.read_csv(OUTPUT/relative,dtype={"load_total_kw":str})
            assert np.array_equal(reread.load_total_kw,full_csv.load_total_kw)
            delta=np.diff(candidate.load_total_kw.to_numpy())
            rec=dict(candidate_id=cid,parent_sample_id=row.sample_id,legacy_split=row.split,
                parent_name=row.parent,new_split="WAIT_USER_TEST_SELECTION",quality_class=grade,
                source_start_30s_row=start,source_end_30s_row_exclusive=end,
                source_start_1s_row=first_1s,source_end_1s_row_inclusive=final_1s,
                source_start_time_s=float(source.time_s.iloc[first_1s]),source_end_time_s=float(source.time_s.iloc[final_1s]),
                source_start_h=float(source.time_s.iloc[first_1s]/3600),source_end_h=float(source.time_s.iloc[final_1s]/3600),
                start_timestamp=str(full.timestamp.iloc[0]),end_timestamp=str(full.timestamp.iloc[-1]),
                point_count=len(full),preview_30s_point_count=len(candidate),
                preview_30s_end_source_s=float(old_time[end-1]),
                full_curve_tail_after_preview_s=float(source.time_s.iloc[final_1s]-old_time[end-1]),
                duration_s=float(full.time_s.iloc[-1]),load_peak_kw=float(load.max()),
                load_mean_kw=float(load.mean()),mean_abs_delta_kw=float(abs(delta).mean()),
                max_abs_delta_kw=float(abs(delta).max()),
                start_load_kw=float(load[0]),end_load_kw=float(load[-1]),
                start_boundary_kind=boundary(load[0],first_1s==0),
                end_boundary_kind=boundary(load[-1],final_1s==len(source)-1),
                unknown_points=unknown,shore_points=shore,
                source_original_1s_end_time_s=float(source.time_s.iloc[-1]),
                source_original_1s_end_load_kw=float(source.load_total_kw.iloc[-1]),
                source_file_relative_path=row.relative_path,crop_reason=reason,
                review_flags=";".join(flags),relative_path=relative,sha256=sha(OUTPUT/relative),
                preview_30s_relative_path=preview_relative,preview_30s_sha256=sha(OUTPUT/preview_relative))
            records.append(rec);plots[cid]=full
            assert np.array_equal(candidate.load_total_kw.to_numpy(),old_load[start:end])
            assert np.array_equal(np.diff(candidate.time_s),np.full(len(candidate)-1,30.))
            assert np.all(np.diff(stamps.as_unit("ns").asi8)==30_000_000_000)
            assert np.array_equal(np.diff(full.time_s),np.ones(len(full)-1))
            assert reason=="UNCHANGED_EXISTING_SAMPLE" or (unknown==0 and shore==0)
            checks.append(dict(candidate_id=cid,unchanged_load_values=True,strict_30s=True,
                               unknown_points=unknown,shore_points=shore))
        for start in np.flatnonzero(np.diff(np.r_[False,~kept,False].astype(int))==1):
            end=start
            while end<len(kept) and not kept[end]:end+=1
            excluded.append(dict(parent_sample_id=row.sample_id,source_start_30s_row=int(start),
                source_end_30s_row_exclusive=int(end),reason="OUTSIDE_EXPLICIT_USER_CROP",point_count=end-start))
        print(f"{row.sample_id}: {len(plan)} candidate(s)",flush=True)
    assert len(records)==48
    m=pd.DataFrame(records)
    save_csv(m,OUTPUT/"metadata/candidate_manifest.csv")
    save_csv(pd.DataFrame(zero_audit),OUTPUT/"metadata/zero_value_audit.csv")
    save_csv(pd.DataFrame(excluded),OUTPUT/"metadata/excluded_intervals.csv")
    save_csv(pd.DataFrame(sources),OUTPUT/"metadata/input_hashes.csv")
    # The inputs must still match after all candidate generation.
    assert all(sha(v["path"])==v["sha256"] for v in sources)
    (OUTPUT/"metadata/checks.json").write_text(json.dumps(dict(status="PASS",source_count=43,
        candidate_count=48,input_hashes_checked=len(sources),unknown_candidate_ids=m.loc[m.unknown_points>0,"candidate_id"].tolist(),
        no_load_value_changed=True,canonical_1s_load_text_preserved_and_reread=True,
        full_original_tail_preserved=True,no_raw_rebuild=True,no_new_split=True,checks=checks),indent=2),encoding="utf-8")
    policy=dict(source_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_power_root=str(POWER),source_mode_root=str(MODES),candidate_status="REVIEW_ONLY_NO_FINAL_SPLIT",
        old_raw_rebuild_v1_not_used=True,source_parent_count=43,candidate_count=48,
        initial_soc_for_future_task=.6,no_shore_control_for_future_task=True,no_terminal_charge_for_future_task=True,
        equivalent_hydrogen_economic_model="WAIT_USER_TEST_SELECTION_PHASE2_NOT_IMPLEMENTED",
        model_step_seconds=30,canonical_curve_seconds=1,
        preview_30s="Existing30sdecision grid is preserved separately; full1stail remains in canonical curve. Finalmodel partial-tail convention awaits phase2.",
        values="Canonical source load strings preserved verbatim and reread. Current1scurves and30spreview are cropped only; no new interpolation, clipping, zero forcing or raw data assembly")
    (OUTPUT/"metadata/policy.json").write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding="utf-8")

    mark_pdf_operation_started()
    fonts,_=_font_settings()
    plt.rcParams.update({"font.family":fonts,"axes.unicode_minus":False,"path.simplify":False})
    (OUTPUT/"figures").mkdir(exist_ok=True)
    for rec in records:
        c=plots[rec["candidate_id"]];parent=original_frames[rec["parent_sample_id"]]
        fig,(ax,context)=plt.subplots(2,1,figsize=(12,6),gridspec_kw={"height_ratios":[4,1]})
        fig.subplots_adjust(top=.83,bottom=.10,hspace=.6,left=.08,right=.98)
        ax.plot(c.time_s/3600,c.load_total_kw,color="#17466e",lw=1)
        ax.scatter([0,c.time_s.iloc[-1]/3600],[c.load_total_kw.iloc[0],c.load_total_kw.iloc[-1]],s=22,color="#b75c42",zorder=4)
        ax.set(xlabel="Cropped relative time (h)",ylabel="Total load (kW)")
        ax.grid(alpha=.25);ax.axhline(0,color="gray",lw=.6)
        context.plot(parent.time_s/3600,parent.load_total_kw,color="#919ba3",lw=.6)
        context.axvspan(rec["source_start_h"],rec["source_end_h"],color="#478773",alpha=.35)
        context.set(xlabel="Original43-sample relative time (h); green = retained interval",ylabel="Original kW")
        context.grid(alpha=.2)
        title=(f"{rec['candidate_id']} | {rec['legacy_split']} _{rec['parent_sample_id'][-3:]} | class {rec['quality_class']}\n"
               f"{rec['duration_s']/3600:.3f} h | mean {rec['load_mean_kw']:.1f} kW | peak {rec['load_peak_kw']:.1f} kW | "
               f"start {rec['start_load_kw']:.3f} -> end {rec['end_load_kw']:.3f} kW\n"
               f"Original time {rec['source_start_h']:.3f} to {rec['source_end_h']:.3f} h; UNKNOWN={rec['unknown_points']}; SHORE={rec['shore_points']}")
        fig.suptitle(title,x=.08,ha="left",fontsize=11,y=.98)
        fig.savefig(OUTPUT/f"figures/{rec['candidate_id']}.png",dpi=160)
        plt.close(fig)
    with PdfPages(OUTPUT/"figures/current43_cropped_overview.pdf") as pdf:
        for page_start in range(0,len(records),4):
            fig,axes=plt.subplots(4,1,figsize=(11.7,12.5))
            fig.subplots_adjust(top=.90,bottom=.05,left=.08,right=.98,hspace=.65)
            for ax,rec in zip(axes,records[page_start:page_start+4]):
                c=plots[rec["candidate_id"]]
                ax.plot(c.time_s/3600,c.load_total_kw,color="#17466e",lw=.85)
                ax.set(xlabel="Cropped relative time (h)",ylabel="Load (kW)")
                ax.set_title(f"{rec['candidate_id']} | {rec['point_count']} source1s points | "
                    f"{rec['duration_s']/3600:.2f}h | max {rec['load_peak_kw']:.0f} kW | "
                    f"ends {rec['start_load_kw']:.2f}, {rec['end_load_kw']:.2f} kW | "
                    f"UNKNOWN {rec['unknown_points']}",fontsize=10,loc="left")
                ax.grid(alpha=.25);ax.axhline(0,color="gray",lw=.5)
            for ax in axes[len(records[page_start:page_start+4]):]:ax.set_visible(False)
            fig.suptitle(f"Existing43 processed samples ->48 crop candidates | page {page_start//4+1}/12\n"
                         "Full EXISTING processed1s curves;30s previews retained separately; no final split",fontsize=11,y=.985)
            pdf.savefig(fig);plt.close(fig)
    # Small human index. The full manifest keeps exact source rows and hashes.
    columns={"candidate_id":"候选ID","parent_sample_id":"原样本ID","legacy_split":"旧来源划分",
        "source_start_h":"原图起点h","source_end_h":"原图终点h","duration_s":"时长秒",
        "start_load_kw":"起点负载kW","end_load_kw":"终点负载kW","load_peak_kw":"峰值kW",
        "load_mean_kw":"平均负载kW","quality_class":"审查类别","unknown_points":"UNKNOWN点数",
        "shore_points":"SHORE点数","crop_reason":"裁剪原因","review_flags":"待确认事项"}
    index=m[list(columns)].rename(columns=columns)
    index["总览PDF页码"]=np.arange(len(index))//4+1
    save_csv(index,OUTPUT/"metadata/review_index.csv")
    cards=[]
    for rec in records:
        cid=rec["candidate_id"]
        cards.append(f'<section><h2>{cid}</h2><p>原图{rec["source_start_h"]:.3f}–{rec["source_end_h"]:.3f}h；'
          f'首尾{rec["start_load_kw"]:.3f}/{rec["end_load_kw"]:.3f}kW；UNKNOWN={rec["unknown_points"]}</p>'
          f'<a href="figures/{cid}.png"><img loading="lazy" src="figures/{cid}.png"></a></section>')
    (OUTPUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>现有43样本裁剪后的48个候选</title>'
        '<style>body{font-family:Microsoft YaHei,sans-serif;max-width:1200px;margin:auto;padding:24px;background:#f5f7fa}'
        'section{background:white;padding:15px;margin:18px 0}img{width:100%}h2{font-size:19px}</style>'
        '<h1>现有43样本裁剪后的48个候选</h1><p>仅裁剪现有负载，不重建原始数据。Ctrl+F搜索ID。'
        '每图上方为裁后负载，下方灰线为原样本、绿色为保留范围。等待用户选择Test，尚未正式划分。</p>'+
        ''.join(cards),encoding="utf-8")
    print(json.dumps(dict(output=str(OUTPUT),parents=43,candidates=len(records),
        figures=48,overview_pages=12,quality_counts=m.quality_class.value_counts().to_dict(),
        unknown_candidates=m.loc[m.unknown_points>0,"candidate_id"].tolist()),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
