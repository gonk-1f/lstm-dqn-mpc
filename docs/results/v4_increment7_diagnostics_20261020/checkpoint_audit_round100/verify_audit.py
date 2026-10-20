"""Verify local audit invariants without writing original artifacts."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import audit_support as a
from v4.dqn import masked_double_dqn_targets

policy, cp, _ = a.load_policy()
with (a.OUT / 'q_action_audit_parent020.csv').open(encoding='utf-8-sig', newline='') as f:
    row = next(csv.DictReader(f))
profile = a.load_profile('parent_020_seg01', 'validation')
i = int(row['archive_transition_index'])
t, nt = profile['transitions'][i:i+2]
s = (t['soc_before'], t['load_kw']/600, (t['fc_kw']-t['action_delta_kw'])/600)
assert policy.q(s)[int(row['actual_action_index'])] == float(row['actual_q'])
ns = (t['soc_after'], nt['load_kw']/600, t['fc_kw']/600)
with a.torch.random.fork_rng(devices=[]):
    target = a.MLPQNetwork(3, 7, (128, 128))
target.load_state_dict(cp['target_state'], strict=True)
target.eval().requires_grad_(False)
acc, cfg = a.accountant_config(cp)
_, candidates = a.physical_mask(a.AccountState(soc=ns[0], previous_fc_kw=t['fc_kw']), nt['load_kw'], acc, nt['duration_s'])
mask = [[t['fc_kw']+d in candidates for d in a.FC_INCREMENT_KW]]
with a.torch.inference_mode():
    x = a.torch.tensor([ns], dtype=a.torch.float32)
    y = masked_double_dqn_targets(
        a.torch.tensor([[t['scaled_training_reward']]], dtype=a.torch.float32),
        policy.model(x), target(x), a.torch.zeros((1,1)),
        next_action_masks=a.torch.tensor(mask, dtype=a.torch.bool), gamma=1.0)
official_target_difference = abs(y.item()-float(row['fixed_ddqn_target']))
assert official_target_difference < 1e-7
baseline = json.loads((a.OUT/'readonly_baseline.json').read_text(encoding='utf-8'))
for item in baseline['protected_files']:
    assert a.sha256(a.RUN/item['relative_path']) == item['sha256'], item['relative_path']
for item in baseline['source_files']:
    assert a.sha256(a.ROOT/item['file']) == item['sha256'], item['file']
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=a.ROOT, text=True).strip() == baseline['head']
result = {'checkpoint_and_original_files_unchanged': True, 'formal_source_files_unchanged': True,
          'git_head_unchanged_since_readonly_task_start': True,
          'official_masked_double_dqn_target_difference': official_target_difference,
          'independent_single_state_q_matches_csv': True, 'optimizer_or_backward_calls': 0}
(a.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
