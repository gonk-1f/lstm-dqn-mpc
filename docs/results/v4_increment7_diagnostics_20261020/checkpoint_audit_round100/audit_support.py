"""Read-only R100 audit helpers. This file belongs only to local output."""
from pathlib import Path
import hashlib
import json
import sys

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
from dqn.networks.mlp_qnet import MLPQNetwork
from v3.control import EconomicMPC, AccountState
from v4.actions import FC_INCREMENT_KW, candidate_fc_actions, previous_grid_power_kw
from v4.control import feasible_fc_actions
from v4.independent_dataset import IndependentTrainingDataset
from v4.independent_economics import ZhouEvaluationConfig
from v4.trajectory_io import iter_trajectory_records


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_policy():
    metadata = json.loads((RUN / 'run_metadata.json').read_text(encoding='utf-8'))
    cp = torch.load(RUN / 'diagnostic_checkpoints/round_100.pt', map_location='cpu', weights_only=False)
    assert cp['round'] == 100 and cp['purpose'] == 'diagnostic_only'
    for key in ('source_commit', 'experiment_id', 'manifest_sha256', 'hyperparameters'):
        assert cp[key] == metadata[key], key
    hp = cp['hyperparameters']
    assert hp['network_dims'] == [3, 128, 128, 7]
    assert hp['state_mode'] == 'minimal3' and hp['action_mode'] == 'increment7'
    assert tuple(hp['action_kw']) == FC_INCREMENT_KW
    torch.set_num_threads(hp['torch_threads'])
    with torch.random.fork_rng(devices=[]):
        online = MLPQNetwork(3, 7, (128, 128))
    online.load_state_dict(cp['online_state'], strict=True)
    online.eval().requires_grad_(False)
    return FrozenPolicy(online), cp, metadata


class FrozenPolicy:
    def __init__(self, model):
        self.model = model

    def q(self, state):
        with torch.inference_mode():
            return self.model(torch.tensor([state], dtype=torch.float32))[0].tolist()

    def __call__(self, state, feasible_powers):
        previous = previous_grid_power_kw(state[2] * 600)
        q = self.q(state)
        indices = [i for i, delta in enumerate(FC_INCREMENT_KW)
                   if previous + delta in feasible_powers]
        if not indices:
            raise ValueError('no legal increment')
        index = max(indices, key=lambda i: q[i])
        return previous + FC_INCREMENT_KW[index]


def load_episode(sample_id, split):
    dataset = IndependentTrainingDataset.open(ROOT / 'data/processed/operating_dataset_independent_v2')
    dataset._training_split(split)
    rows = dataset._manifest[(dataset._manifest['candidate_id'] == sample_id)
                             & (dataset._manifest['split'] == split)]
    if len(rows) != 1:
        raise ValueError('sample is not exactly one authorized Train/Validation episode')
    return dataset._load_episode(next(rows.itertuples(index=False))), dataset.identity_hashes()


def load_profile(sample_id, split):
    filename = f'round_100_{split}_trajectories.json'
    records = iter_trajectory_records(RUN / filename, sample_id=sample_id)
    try:
        return next(records)[1]
    finally:
        records.close()


def accountant_config(cp):
    return (EconomicMPC(nominal_cost_cny=1.0),
            ZhouEvaluationConfig(cp['hyperparameters']['economic_evaluation']['fc_average_efficiency']))


def physical_mask(physical, load, accountant, duration):
    full = feasible_fc_actions(physical, load, accountant, duration_s=duration)
    return full, candidate_fc_actions(full, physical.previous_fc_kw, action_mode='increment7')
