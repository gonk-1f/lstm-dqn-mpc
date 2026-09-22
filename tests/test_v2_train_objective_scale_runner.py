from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
MAIN = SRC / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))


class TrainObjectiveScaleRunnerTests(unittest.TestCase):
    def test_entrypoint_json_conversion_recurses_through_mappings_and_lists(self) -> None:
        from run_v2_objective_scale_audit import _jsonable
        from v2.analysis.objective_scale_audit import TermStatistics

        statistic = TermStatistics("base", 1, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        converted = _jsonable({"statistics": [statistic]})

        self.assertEqual(converted["statistics"][0]["term"], "base")
        json.dumps(converted, allow_nan=False)

    def test_representative_selection_is_small_deterministic_and_covers_available_strata(self) -> None:
        from v2.analysis.train_objective_scale_runner import (
            AuditReadyState,
            select_representative_cases,
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        specifications = (
            (80.0, 0.01, 0.50),
            (100.0, 90.0, 0.52),
            (250.0, 0.02, 0.55),
            (300.0, 130.0, 0.65),
            (500.0, 0.03, 0.70),
            (650.0, 180.0, 0.75),
            (450.0, 20.0, 0.81),
        )
        states = tuple(
            AuditReadyState(
                parent_id="train-parent",
                timestamp=start + timedelta(seconds=30 * index),
                p_load_kw=load,
                soc_system=soc,
                previous_p_fc_total_kw=200.0,
                history_loads_kw=(load - delta,),
                absolute_load_change_kw=delta,
            )
            for index, (load, delta, soc) in enumerate(specifications)
        )

        first = select_representative_cases(states)
        second = select_representative_cases(tuple(reversed(states)))

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 7)
        selected_loads = tuple(case.payload[0] for case in first)
        self.assertTrue(any(load < 200.0 for load in selected_loads))
        self.assertTrue(any(200.0 <= load < 400.0 for load in selected_loads))
        self.assertTrue(any(load >= 400.0 for load in selected_loads))
        selected_socs = tuple(case.payload[1] for case in first)
        self.assertTrue(any(0.4 <= soc <= 0.6 for soc in selected_socs))
        self.assertTrue(any(0.6 < soc <= 0.8 for soc in selected_socs))
        self.assertTrue(all(0.2 <= soc <= 0.8 for soc in selected_socs))
        self.assertEqual(tuple(case.order for case in first), tuple(range(len(first))))


if __name__ == "__main__":
    unittest.main()
