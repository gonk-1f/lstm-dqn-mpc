"""Necessary energy/power conditions, not executable trajectories or sufficiency."""
import json
import csv
from math import fsum
from audit_support import OUT, load_episode, EconomicMPC
from v2.models.battery_energy import next_soc

episode, _ = load_episode('parent_015_seg01', 'train')
accountant = EconomicMPC(nominal_cost_cny=1.)
baseline = list(csv.DictReader((OUT/'counterfactual_parent_015_seg01_greedy_r100.csv').open(encoding='utf-8-sig')))
blocks = []
begin = None
for i in range(episode.step_count + 1):
    over = i < episode.step_count and episode.load_kw[i] > 600
    if over and begin is None:
        begin = i
    if not over and begin is not None:
        soc = .8
        for j in range(begin, i):
            soc = next_soc(soc, float(episode.load_kw[j])-600., float(episode.dt_s[j]),
                           624., efficiency=accountant.efficiency)
        blocks.append({'start_step_1based': begin+1, 'end_step_1based': i,
                       'duration_s': fsum(float(x) for x in episode.dt_s[begin:i]),
                       'max_load_kw': float(max(episode.load_kw[begin:i])),
                       'min_SOC_required_at_block_start_even_with_instant_FC600': .8-soc+.2,
                       'end_SOC_if_start_0_8_and_FC600_entire_block': soc,
                       'battery_stored_energy_required_kwh': (.8-soc)*624,
                       'strict_energy_impossibility_found': soc < .2})
        if begin < len(baseline):
            actual_start_soc = float(baseline[begin]['soc_before'])
            optimistic_end_soc = actual_start_soc - (.8-soc)
            blocks[-1].update(
                actual_greedy_soc_at_block_start=actual_start_soc,
                optimistic_end_soc_from_actual_start_with_instant_FC600=optimistic_end_soc,
                additional_SOC_required_before_block_even_without_ramp=max(0., .2-optimistic_end_soc),
                additional_stored_kwh_required_before_block_even_without_ramp=max(0., .2-optimistic_end_soc)*624,
                no_possible_completion_from_actual_block_start_even_without_ramp=optimistic_end_soc<.2)
        begin = None
payload = {
    'sample_id': episode.sample_id,
    'FC_upper_kw': 600, 'battery_energy_kwh': 624,
    'battery_charge_min_kw': accountant.plant.battery_charge_min_kw,
    'battery_discharge_max_kw': accountant.plant.battery_discharge_max_kw,
    'max_load_kw': float(max(episode.load_kw)),
    'max_battery_discharge_needed_even_at_FC600_kw': max(0., float(max(episode.load_kw))-600.),
    'total_load_bus_energy_kwh': fsum(float(p)*float(dt)/3600 for p,dt in zip(episode.load_kw,episode.dt_s)),
    'total_duration_s': episode.duration_s,
    'all_FC600_energy_upper_kwh': 600*episode.duration_s/3600,
    'battery_initial_0_6_to_0_2_bus_energy_kwh': .4*624*.95,
    'continuous_load_above_FC_rated_blocks': blocks,
    'conclusion': 'These optimistic bounds ignore FC ramp reachability and starting FC state. '
                  'Starting the high-load block from SOC 0.8 passes a necessary energy condition, '
                  'which is not proof of whole-voyage feasible completion. The actual Greedy '
                  'SOC at high-load-block entry is below the optimistic minimum required: '
                  'completion from that actual block-entry state is impossible even with immediate FC600. '
                  'The specified greedy-next-SOC intervention fails by upper-SOC increment reachability '
                  'before the original high-load window; it does not diagnose unavoidable SOC depletion.'}
(OUT/'parent015_energy_necessary_conditions.json').write_text(
    json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2))
