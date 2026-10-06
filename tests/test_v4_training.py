from types import SimpleNamespace

from v4.train import run_train_validation


class FakeDataset:
    opened_test_payloads = 0

    def __init__(self):
        self.calls = []

    def load_train(self):
        self.calls.append("train")
        return (SimpleNamespace(
            sample_id="train_1", split="train", operating_mode=("onboard", "shore_charging"),
            load_kw=(100.0, 0.0), battery_bus_kw=(0.0, -20.0),
        ),)

    def load_validation(self):
        self.calls.append("validation")
        return (SimpleNamespace(
            sample_id="val_1", split="validation", operating_mode=("onboard",),
            load_kw=(80.0,), battery_bus_kw=(0.0,),
        ),)

    def load_final_test(self, *_args):
        raise AssertionError("Test must remain closed")


def test_training_uses_only_train_validation_and_reports_raw_costs():
    dataset = FakeDataset()
    _, report = run_train_validation(
        dataset, rounds=1, batch_size=1, updates_per_episode=1, seed=1,
    )
    assert dataset.calls == ["train", "validation"]
    assert report["test_payloads_opened"] == 0
    assert report["bootstrap"]["completed"] == 1
    assert report["validation"]["episodes"] == 1
    assert report["validation"]["cost_cny"] > 0
    assert report["reward_definition"] == "negative_actual_four_component_CNY"
    assert report["unsettled_terminal_onboard_episodes"] == {"train": 0, "validation": 1}
    assert report["selection_eligible"] is False


def test_any_incomplete_training_episode_blocks_checkpoint_selection():
    class InfeasibleDataset(FakeDataset):
        def load_train(self):
            source = super().load_train()[0]
            source.load_kw = (2000.0, 0.0)
            return (source,)

    _, report = run_train_validation(
        InfeasibleDataset(), rounds=1, batch_size=1, updates_per_episode=1,
    )
    assert report["bootstrap"]["failed"]
    assert report["training_rounds"][0]["failed"]
    assert report["selection_eligible"] is False
