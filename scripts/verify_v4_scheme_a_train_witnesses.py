"""Replay five P0 Train feasibility witnesses through the Scheme A gate.

Run from the repository root with PYTHONPATH=src. No Test payload is opened.
The rule sees only current measured load, SOC and supplied candidate actions.
"""

from __future__ import annotations

from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC
from v4.control import replay_episode
from v4.train import _default_data_root


WITNESSES = {
    "zero_boundary_015": (337, 1),
    "zero_boundary_017": (868, 7),
    "zero_boundary_044": (808, 2),
    "zero_boundary_008": (752, 2),
    "zero_boundary_046": (1349, 8),
}


def main() -> None:
    roots = tuple(_default_data_root(name) for name in (
        "operating_dataset_zero_boundary_v2", "operating_dataset_zero_boundary_v2_ais",
        "operating_dataset_zero_boundary_v2_modes"))
    dataset = FormalTrainingDataset.open(*roots)
    episodes = {item.sample_id: item for item in dataset.load_train()}
    accountant = EconomicMPC(nominal_cost_cny=1.)

    def witness(state, actions):
        positive = tuple(action for action in actions if action > 0)
        assert positive, "P0 positive-power witness unexpectedly became infeasible"
        load_kw = state[1] * 600.
        target = load_kw + 3000. * (.70 - state[0])
        return min(positive, key=lambda action: (abs(action - target), action))

    for sample_id, expected in WITNESSES.items():
        replay = replay_episode(episodes[sample_id], witness, accountant=accountant)
        transitions = replay.transitions
        voyages = sum(item.done for item in transitions)
        assert (len(transitions), voyages) == expected
        assert all(item.action_kw > 0 for item in transitions)
        assert all(item.action_kw in item.policy_candidate_actions for item in transitions)
        assert all(item.action_kw in item.physical_feasible_actions for item in transitions)
        print(f"{sample_id}: {len(transitions)} steps, {voyages} voyages, "
              f"min SOC={min(item.actual_soc for item in transitions):.6f}")
    assert dataset.opened_test_payloads == 0
    print("Test payloads opened: 0")


if __name__ == "__main__":
    main()
