"""Small synthetic migration checks; no training, route data or saved run is loaded."""
import hashlib
import importlib.util
from pathlib import Path
import random

import pytest
import torch


def migration_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/migrate_v4_soc_mask_checkpoint.py'
    assert path.is_file(), 'the standalone SOC-mask checkpoint migration is required'
    spec = importlib.util.spec_from_file_location('soc_mask_migration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_state():
    from v2.models.battery_energy import next_soc
    from v3.control import EconomicMPC
    from v4.dqn import ReplayItem
    accountant = EconomicMPC(nominal_cost_cny=1.)
    step = next_soc(.5, 10., 30., accountant.plant.battery_nominal_energy_kwh,
                    efficiency=accountant.efficiency) - .5
    # The saved physical state is legal; the old zero-FC mask projects just below SOC_MIN.
    boundary = (.2 - step - 5e-10, 0., 0., 0., 10. / 600., 0., 0.)
    terminal = (.6, 0., 0., 0., 0., 0., 0.)
    items = (
        ReplayItem(terminal, 1, -1., boundary, False, (0, 1), bootstrap_steps=8,
                   replay_id=101, source_sample_id='synthetic', source_round=30,
                   source_row_index=1),
        ReplayItem(terminal, 0, -2., terminal, True, (), bootstrap_steps=1,
                   terminal_reason='completed', replay_id=102, source_sample_id='synthetic',
                   source_round=30, source_row_index=2),
    )
    return {
        'format_version': 2, 'replay': items, 'replay_ids': (101, 102),
        'failure_pool_ids': (), 'ordinary_pool_ids': (102, 101), 'next_replay_id': 103,
        'online': {'weight': torch.tensor([1.])}, 'target': {'weight': torch.tensor([2.])},
        'optimizer': {'state': {1: {'exp_avg': torch.tensor([.25])}}},
        'random_state': random.Random(42).getstate(),
        'torch_rng_state': torch.random.get_rng_state().clone(), 'counts': {'updates': 17},
    }


def test_two_item_migration_only_narrows_masks_preserving_order_state_and_rng():
    module = migration_module()
    from v3.control import EconomicMPC
    before = synthetic_state()
    global_python_rng = random.getstate()
    global_torch_rng = torch.random.get_rng_state().clone()
    after, counts = module.migrate_replay_state(before, EconomicMPC(nominal_cost_cny=1.))
    assert before['replay'][0].next_feasible_indices == (0, 1)
    assert after['replay'][0].next_feasible_indices == (1,)
    assert after['replay'][1] is before['replay'][1]
    from dataclasses import replace
    assert after['replay'][0] == replace(before['replay'][0], next_feasible_indices=(1,))
    for key in before.keys() - {'replay'}:
        assert after[key] is before[key]
    by_id = dict(zip(after['replay_ids'], after['replay']))
    restored_pool = [(key, by_id[key]) for key in after['ordinary_pool_ids']]
    assert [key for key, _ in restored_pool] == [102, 101]
    assert restored_pool[1][1] is after['replay'][0]
    assert counts['narrowed_nonterminal_masks'] == 1 and counts['removed_action_indices'] == 1
    assert counts['nonterminal_items'] == counts['terminal_items'] == 1
    assert counts['added_action_indices'] == 0
    assert random.getstate() == global_python_rng
    assert torch.equal(torch.random.get_rng_state(), global_torch_rng)


def test_source_inverse_hash_matches_original_crlf_contract_and_rejects_other_edits(tmp_path):
    module = migration_module()
    control = tmp_path / 'src/v4/control.py'
    control.parent.mkdir(parents=True)
    old = b'def mask():\n        if SOC_MIN - 1e-9 <= soc <= SOC_MAX + 1e-9:\n            pass\n'
    control.write_bytes(old.replace(b'\n', b'\r\n'))
    other = tmp_path / 'src/other.json'
    other.write_bytes(b'{"kept": true}\r\n')
    digest = hashlib.sha256()
    for relative, content in [('src/other.json', b'{"kept": true}\n'), ('src/v4/control.py', old)]:
        digest.update(relative.encode() + b'\0' + content + b'\0')
    saved = {'source_commit': 'synthetic-head', 'source_tree_sha256': digest.hexdigest()}
    control.write_bytes(old.replace(module.OLD_MASK_LINE, module.STRICT_MASK_LINE).replace(b'\n', b'\r\n'))
    current, reconstructed = module.verify_source_contract(tmp_path, saved, 'synthetic-head')
    assert reconstructed == saved['source_tree_sha256'] and current != reconstructed
    other.write_bytes(b'{"kept": false}\n')
    with pytest.raises(ValueError, match='only the strict SOC-mask line'):
        module.verify_source_contract(tmp_path, saved, 'synthetic-head')


def test_nonterminal_that_becomes_empty_is_rejected_without_inventing_terminal():
    module = migration_module()
    from dataclasses import replace
    from v3.control import EconomicMPC
    before = synthetic_state()
    item = replace(before['replay'][0], next_feasible_indices=(0,))
    before['replay'] = (item, before['replay'][1])
    with pytest.raises(ValueError, match='empty strict nonterminal mask'):
        module.migrate_replay_state(before, EconomicMPC(nominal_cost_cny=1.))
    assert not item.done and item.reward_cny == -1. and item.next_feasible_indices == (0,)


def test_terminal_mask_must_already_be_empty():
    module = migration_module()
    from dataclasses import replace
    from v3.control import EconomicMPC
    before = synthetic_state()
    before['replay'] = (before['replay'][0], replace(before['replay'][1], next_feasible_indices=(0,)))
    with pytest.raises(ValueError, match='terminal replay mask must be empty'):
        module.migrate_replay_state(before, EconomicMPC(nominal_cost_cny=1.))


def test_nested_historical_diagnostic_and_trajectory_links_use_original_output(tmp_path):
    module = migration_module()
    source, output = tmp_path / 'original', tmp_path / 'new'
    source.mkdir()
    for name in ('round_030.pt', 'round_030_decisions.json', 'round_030_train.json'):
        (source / name).touch()
    artifact = {'round': 30, 'checkpoint': 'round_030.pt', 'decisions': 'round_030_decisions.json'}
    before = {'diagnostic_artifacts': [artifact], 'rounds': [{'round': 30,
              'diagnostic_artifacts': dict(artifact),
              'greedy_train': {'trajectory_file': 'round_030_train.json'}}]}
    after = module.link_historical_artifacts(before, source, output)
    assert before['diagnostic_artifacts'][0]['checkpoint'] == 'round_030.pt'
    assert after['diagnostic_artifacts'][0] == after['rounds'][0]['diagnostic_artifacts']
    assert after['rounds'][0]['diagnostic_artifacts']['checkpoint'] == '../original/round_030.pt'
    assert after['rounds'][0]['diagnostic_artifacts']['decisions'] == '../original/round_030_decisions.json'
    assert after['rounds'][0]['greedy_train']['trajectory_file'] is None
    assert after['rounds'][0]['greedy_train']['historical_trajectory_file'] == '../original/round_030_train.json'
    assert before['rounds'][0]['greedy_train']['trajectory_file'] == 'round_030_train.json'
