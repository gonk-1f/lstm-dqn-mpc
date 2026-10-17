"""Output serialization must not retain an entire round of rendered trajectories."""

import json
from types import SimpleNamespace
import weakref

import pytest

from v4 import monitored_training


def test_report_writer_streams_json_without_building_one_large_string(tmp_path, monkeypatch):
    expected = {"round": 100, "samples": ["航段", {"cost": 1.25}], "best": None}
    path = tmp_path / "report.json"
    path.write_text('{"round": 99}', encoding="utf-8")

    def prohibit_full_string(*args, **kwargs):
        raise AssertionError("whole-document json.dumps increases memory peaks")

    monkeypatch.setattr(json, "dumps", prohibit_full_string)
    monitored_training._write_json(path, expected)
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert not path.with_suffix(".json.tmp").exists()


def test_completed_trajectories_are_rendered_and_released_one_at_a_time(tmp_path, monkeypatch):
    results = [SimpleNamespace(sample_id=f"sample_{index}", transitions=(index,),
                               economic_evaluation={"total": index}) for index in range(3)]
    references = []
    calls = []

    class Record(dict):
        pass

    def render(sample_id, transitions, **kwargs):
        assert all(reference() is None for reference in references)
        calls.append((sample_id, transitions, kwargs))
        record = Record(sample_id=sample_id, steps=list(transitions), **kwargs)
        references.append(weakref.ref(record))
        return record

    monkeypatch.setattr(monitored_training, "trajectory_record", render)
    path = tmp_path / "trajectories.json"
    failed = [{"sample_id": "失败样本", "completed": False}]
    monitored_training._write_trajectory_json(path, iter(results), failed,
        reward_scale=0.001, reward_cost_mode="control_only")
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert list(decoded) == ["completed", "failed"]
    assert decoded["failed"] == failed
    assert [record["sample_id"] for record in decoded["completed"]] == [r.sample_id for r in results]
    assert path.read_text(encoding="utf-8") == json.dumps(decoded, ensure_ascii=False, indent=2)
    assert all(kwargs == {"completed": True, "reward_scale": 0.001,
                          "reward_cost_mode": "control_only",
                          "economic_evaluation": {"total": index}}
               for index, (_, _, kwargs) in enumerate(calls))
    assert all(reference() is None for reference in references)


def test_failed_stream_preserves_previous_complete_output(tmp_path, monkeypatch):
    path = tmp_path / "trajectories.json"
    previous = '{"completed": [], "failed": []}'
    path.write_text(previous, encoding="utf-8")

    def failed_render(*args, **kwargs):
        raise MemoryError("serialization interrupted")

    monkeypatch.setattr(monitored_training, "trajectory_record", failed_render)
    result = SimpleNamespace(sample_id="sample", transitions=(), economic_evaluation=None)
    with pytest.raises(MemoryError, match="serialization interrupted"):
        monitored_training._write_trajectory_json(path, [result], [], reward_scale=0.001,
                                                 reward_cost_mode="control_only")
    assert path.read_text(encoding="utf-8") == previous


def test_empty_trajectory_groups_keep_existing_schema(tmp_path):
    path = tmp_path / "trajectories.json"
    monitored_training._write_trajectory_json(path, (), [], reward_scale=0.001,
                                             reward_cost_mode="control_only")
    assert json.loads(path.read_text(encoding="utf-8")) == {"completed": [], "failed": []}
