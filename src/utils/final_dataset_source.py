"""Original telemetry ingestion and auditable one-to-one asynchronous alignment.

No historical derived dataset or controller result is an input to this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

def align_ais_vectorized(power, ais, max_normal_gap_s, max_nearest_s):
    """Same bounded interpolation policy as the existing helper; nearest ties reject.

    AIS interpolation is semantic evidence, never a repair of missing power data.
    Vectorized assignments avoid slow per-cell pandas mutation on the raw corpus.
    """
    ref=pd.DatetimeIndex(power.timestamp);source=ais.dropna(subset=['timestamp','ais_speed_kn']).sort_values('timestamp')
    result=pd.DataFrame({'speed_aligned_kn':np.full(len(ref),np.nan),'speed_source':'unavailable'})
    if source.empty:return result
    t=ref.as_unit('ns').asi8;s=pd.DatetimeIndex(source.timestamp).as_unit('ns').asi8;v=source.ais_speed_kn.to_numpy(float)
    right=np.searchsorted(s,t);left=right-1
    ri=np.clip(right,0,len(s)-1);li=np.clip(left,0,len(s)-1)
    exact=(right<len(s))&(s[ri]==t)
    values=np.full(len(t),np.nan);kinds=np.full(len(t),'unavailable',dtype=object)
    values[exact]=v[ri[exact]];kinds[exact]='exact'
    interp=(~exact)&(left>=0)&(right<len(s))&((s[ri]-s[li])<=max_normal_gap_s*1e9)
    values[interp]=v[li[interp]]+(v[ri[interp]]-v[li[interp]])*(t[interp]-s[li[interp]])/(s[ri[interp]]-s[li[interp]])
    kinds[interp]='linear'
    ld=np.where(left>=0,np.abs(t-s[li]),np.inf);rd=np.where(right<len(s),np.abs(t-s[ri]),np.inf)
    near=(~exact)&(~interp)&(ld!=rd)&(np.minimum(ld,rd)<=max_nearest_s*1e9)
    chosen=np.where(ld<rd,li,ri);values[near]=v[chosen[near]];kinds[near]='nearest'
    result['speed_aligned_kn']=values;result['speed_source']=kinds
    return result


def parent_sort_key(path: Path):
    values = re.findall(r'(\d+)月(\d+)日(\d+)', path.name)
    if not values:
        raise ValueError(f'Unrecognized parent name: {path.name}')
    return tuple(map(int, values[0])) + (path.name,)


def collapse_channel(frame: pd.DataFrame, columns: list[str]):
    """Exact duplicates collapse; conflicting measurements remain unavailable."""
    data = frame.copy()
    data['timestamp'] = pd.to_datetime(data.timestamp, errors='coerce')
    invalid_times = int(data.timestamp.isna().sum())
    data = data.dropna(subset=['timestamp'])
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').replace([-9999.,np.inf,-np.inf], np.nan)
    exact = data.drop_duplicates(subset=['timestamp', *columns])
    groups = exact.groupby('timestamp', sort=True)
    unique = groups[columns].first()
    conflicts = groups.size().gt(1)
    unique.loc[conflicts, columns] = np.nan
    return unique.reset_index(), {'input_rows': len(frame), 'invalid_timestamp_rows': invalid_times,
        'exact_duplicate_rows': len(data) - len(exact), 'conflicting_timestamps': int(conflicts.sum()),
        'conflicting_timestamp_values':json.dumps([str(t) for t in conflicts.index[conflicts]]),
        'invalid_timestamp_input_row_indices':json.dumps(frame.index[pd.to_datetime(frame.timestamp,errors='coerce').isna()].tolist())}


def unique_nearest(reference, source: pd.DataFrame, tolerance_s: float):
    """Reject equidistant ties, source reuse, and nonmonotone assignments."""
    ref = pd.DatetimeIndex(reference)
    if not ref.is_monotonic_increasing or ref.has_duplicates:
        raise ValueError('Reference timestamps must be strictly increasing')
    src = source.sort_values('timestamp').reset_index(drop=True)
    st = pd.DatetimeIndex(src.timestamp)
    if st.has_duplicates or st.isna().any():
        raise ValueError('Source timestamps must be unique and finite')
    rn = ref.as_unit('ns').asi8
    sn = st.as_unit('ns').asi8
    selected = np.full(len(ref), -1, dtype=int)
    last = -1
    for i, (t, right) in enumerate(zip(rn, np.searchsorted(sn, rn))):
        cand = [j for j in (right - 1, right) if 0 <= j < len(sn)]
        if not cand:
            continue
        distances = [abs(int(sn[j]) - int(t)) for j in cand]
        if len(cand) == 2 and distances[0] == distances[1]:
            continue
        j = cand[int(np.argmin(distances))]
        if j <= last or abs(int(sn[j]) - int(t)) > tolerance_s * 1e9:
            continue
        selected[i] = j
        last = j
    result = pd.DataFrame({'timestamp': ref})
    ok = selected >= 0
    for col in src.columns.drop('timestamp'):
        values = np.full(len(ref), np.nan)
        values[ok] = src[col].to_numpy(dtype=float)[selected[ok]]
        result[col] = values
    offsets = np.full(len(ref), np.nan)
    offsets[ok] = (sn[selected[ok]] - rn[ok]) / 1e9
    result['offset_s'] = offsets
    matched = np.full(len(ref), np.datetime64('NaT'), dtype='datetime64[ns]')
    matched[ok] = sn[selected[ok]].astype('datetime64[ns]')
    result['source_timestamp'] = matched
    return result


def _read(directory, prefix, fields, file_rows):
    paths = sorted(p for p in directory.glob('*.csv') if p.name.startswith(prefix + '_'))
    if not paths:
        raise ValueError(f'Missing channel {directory}/{prefix}')
    frames = []
    for path in paths:
        content = path.read_bytes()
        file_rows.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
        from io import BytesIO
        frame = pd.read_csv(BytesIO(content), encoding='utf-8-sig', usecols=lambda c: c == 'Time' or c in fields)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True).rename(columns={'Time': 'timestamp', **fields})
    for col in fields.values():
        if col not in raw:
            raw[col] = np.nan
    return collapse_channel(raw, list(fields.values()))


def read_originals(raw_root: Path):
    parents = sorted([p for p in raw_root.iterdir() if p.is_dir() and
        all((p / sub).is_dir() for sub in ('BMS', 'EMS', '燃料电池系统'))], key=parent_sort_key)
    if not parents:
        raise ValueError('No original parent directories')
    cache, files, qa = {}, [], []
    for i, parent in enumerate(parents):
        channels = {}
        for side in ('左', '右'):
            for n in range(1, 5):
                name = f'{side}氢燃料电池#{n}'
                channels[name], stats = _read(parent/'燃料电池系统', name,
                    {'发电功率(kW)': 'power_kw', '停止状态': 'stopped', '待机状态': 'standby', '运行状态': 'running'}, files)
                qa.append({'parent': parent.name, 'channel': name, **stats})
            for n in range(1, 7):
                name = f'{side}电池簇{n}'
                raw, stats = _read(parent/'BMS', name,
                    {'总电压(V)': 'voltage_v', '总电流(A)': 'current_a', 'SOC(%)': 'soc_pct'}, files)
                raw['power_kw'] = -(raw.voltage_v * raw.current_a) / 1000.
                raw.loc[raw.voltage_v.le(0), 'power_kw'] = np.nan
                raw.loc[~raw.soc_pct.between(0, 100), 'soc_pct'] = np.nan
                channels[name] = raw
                qa.append({'parent': parent.name, 'channel': name, **stats})
            name = f'{side}逆变电源'
            try:
                channels[name], stats = _read(parent/'EMS', name, {'输出有功功率(kW)': 'inverter_kw'}, files)
                qa.append({'parent': parent.name, 'channel': name, **stats})
            except ValueError:
                channels[name] = pd.DataFrame(columns=['timestamp', 'inverter_kw'])
        # AIS text can contain a trailing kn suffix, unlike device scalar columns.
        ais_paths = sorted((parent/'推进系统').glob('AIS航速_*.csv'))
        if not ais_paths:
            channels['ais'] = pd.DataFrame(columns=['timestamp', 'ais_speed_kn'])
        else:
            from io import BytesIO
            ais_frames = []
            for path in ais_paths:
                content = path.read_bytes()
                files.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
                ais_frames.append(pd.read_csv(BytesIO(content), encoding='utf-8-sig'))
            ais = pd.concat(ais_frames, ignore_index=True).rename(columns={'Time': 'timestamp', '航速(节)': 'ais_speed_kn'})
            ais['ais_speed_kn'] = ais.ais_speed_kn.astype(str).str.replace(r'\s*kn$', '', regex=True)
            channels['ais'], stats = collapse_channel(ais, ['ais_speed_kn'])
            channels['ais'].loc[channels['ais'].ais_speed_kn.lt(0), 'ais_speed_kn'] = np.nan
            qa.append({'parent': parent.name, 'channel': 'ais', **stats})
        cache[parent.name] = channels
        if (i + 1) % 10 == 0:
            print(f'Read original parents {i+1}/{len(parents)}', flush=True)
    return cache, files, qa


def derive_timing_policy(cache):
    offsets, cadence, off_power, positive_power, ais_gaps = [], [], [], [], []
    for channels in cache.values():
        ref = channels['左氢燃料电池#1'].timestamp
        cadence.extend(ref.diff().dt.total_seconds().dropna().tolist())
        for name, ch in channels.items():
            if '氢燃料' in name or '电池簇' in name:
                found = unique_nearest(ref, ch[['timestamp', 'power_kw']], 10.)
                offsets.extend(found.offset_s.abs().dropna().tolist())
            if '氢燃料' in name:
                off_power.extend(ch.loc[ch.stopped.eq(1) & ch.running.ne(1), 'power_kw'].dropna().abs().tolist())
                positive_power.extend(ch.loc[ch.power_kw.gt(0), 'power_kw'].tolist())
        ais_gaps.extend(pd.to_datetime(channels['ais'].timestamp).diff().dt.total_seconds().dropna().tolist())
    cadence = np.asarray(cadence); normal = cadence[(cadence >= 20) & (cadence <= 40)]
    if not len(normal) or not off_power or not positive_power:
        raise ValueError('Insufficient timing or FC stopped-state evidence')
    dt = float(np.median(normal)); cap = dt / 6.
    offsets = np.asarray(offsets)
    # The micro-jitter cluster is bounded to 1/6 cadence to avoid adjacent-cycle aliases.
    micro = offsets[offsets <= cap]
    tol = float(max(1., np.ceil(micro.max())))
    off_q99 = float(np.quantile(off_power, .99))
    resolution = float(np.min(positive_power))
    noise = max(off_q99, resolution / 2.)
    ag = np.asarray(ais_gaps); an = ag[(ag >= 5) & (ag <= 40)]
    ais_cad = float(np.median(an)) if len(an) else 20.
    return {'power_cadence_s': dt, 'alignment_tolerance_s': min(tol, cap),
        'alignment_alias_cap_s': cap, 'offset_quantiles_s': {str(q): float(np.quantile(offsets, q)) for q in [0,.95,.99,.999,.9999,1]},
        'offset_count': len(offsets), 'micro_cluster_count': len(micro),
        'offset_histogram_s': {str(v): int((offsets == v).sum()) for v in np.unique(offsets)},
        'tolerance_basis': 'Entire observed micro-offset cluster within cadence/6; do not discard rare 2-3 s offsets by percentile truncation',
        'power_gap_limit_s': 1.5 * dt, 'ais_gap_limit_s': 2. * ais_cad,
        'ais_nearest_limit_s': ais_cad, 'fc_noise_per_channel_kw': noise,
        'fc_noise_total_kw': noise * 8., 'fc_stopped_abs_q99_kw': off_q99,
        'fc_min_positive_resolution_kw': resolution, 'fc_stopped_observations': len(off_power),
        'basis': 'Original telemetry only; unique nearest offsets; 1/6 cadence alias cap; 1.5 cadence rejects a missing full power cycle; stopped status FC q99 and half positive resolution.'}


def align_originals(cache, policy):
    frames, repairs = {}, []
    for parent, channels in cache.items():
        ref = channels['左氢燃料电池#1'].timestamp
        out = pd.DataFrame({'timestamp': ref, 'parent': parent})
        fc_cols, batt_cols, soc_cols, stopped_cols = [], [], [], []
        for name, ch in channels.items():
            if name == 'ais':
                continue
            match = unique_nearest(ref, ch, policy['alignment_tolerance_s'])
            if '氢燃料' in name:
                col = name+'_kw'; out[col] = match.power_kw; fc_cols.append(col)
                col2 = name+'_stopped'; out[col2] = match.stopped; stopped_cols.append(col2)
            elif '电池簇' in name:
                col = name+'_kw'; out[col] = match.power_kw; batt_cols.append(col)
                col2 = name+'_soc'; out[col2] = match.soc_pct; soc_cols.append(col2)
            else:
                out[name+'_kw'] = match.inverter_kw
            measured_col = 'power_kw' if 'power_kw' in match else 'inverter_kw'
            repaired=match.loc[match.offset_s.abs().gt(0.) & match[measured_col].notna(),['timestamp','source_timestamp','offset_s']]
            for timestamp,source_timestamp,offset in repaired.itertuples(index=False,name=None):
                repairs.append({'parent': parent, 'timestamp': timestamp, 'channel': name,
                    'source_timestamp': source_timestamp, 'original_offset_s': float(offset),
                    'repair_method': 'unique_monotone_nearest_no_reuse', 'tolerance_s': policy['alignment_tolerance_s'],
                    'power_channel': measured_col == 'power_kw', 'exceeded_legacy_1s_tolerance': bool(abs(offset)>1.)})
        out['fc_total_kw'] = out[fc_cols].sum(axis=1, min_count=8)
        out['battery_total_kw'] = out[batt_cols].sum(axis=1, min_count=12)
        out['load_total_kw'] = out.fc_total_kw + out.battery_total_kw
        out['soc_mean_pct'] = out[soc_cols].mean(axis=1)
        out['soc_channel_count'] = out[soc_cols].notna().sum(axis=1)
        out['fc_all_stopped'] = out[stopped_cols].eq(1).all(axis=1)
        out['inverter_total_kw'] = out[['左逆变电源_kw','右逆变电源_kw']].sum(axis=1, min_count=2)
        out['aligned'] = out[['fc_total_kw','battery_total_kw']].notna().all(axis=1)
        speed = align_ais_vectorized(out, channels['ais'], max_normal_gap_s=policy['ais_gap_limit_s'], max_nearest_s=policy['ais_nearest_limit_s'])
        out['speed_kn'] = pd.to_numeric(speed.speed_aligned_kn, errors='coerce')
        out['speed_source'] = speed.speed_source
        frames[parent] = out
        if len(frames)%10==0:print(f'Aligned original parents {len(frames)}/{len(cache)}',flush=True)
    return frames, repairs
