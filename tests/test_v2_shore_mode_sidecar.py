from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestShoreModeSidecar(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, object]:
        from v2.data.segment_power_source import ParentPowerSeries

        power = root / "power"
        ais = root / "ais"
        for base in (power, ais):
            (base / "train").mkdir(parents=True)
            (base / "metadata").mkdir()

        origin = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)
        timestamps = tuple(origin + timedelta(seconds=30 * index) for index in range(8))
        load = np.asarray([0.0, 60.0, -20.0, -30.0, -40.0, 50.0, 0.0, 0.0])
        power_frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "time_s": np.arange(8, dtype=float) * 30.0,
                "load_total_kw": load,
            }
        )
        power_path = power / "train" / "sample.csv"
        power_frame.to_csv(power_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [{
                "parent": "parent",
                "sample_id": "sample",
                "relative_path": "train/sample.csv",
                "split": "train",
                "duration_s": 210.0,
                "sha256": _sha256(power_path),
            }]
        ).to_csv(power / "metadata" / "sample_manifest.csv", index=False)

        ais_frame = pd.DataFrame(
            {
                "timestamp": timestamps[:-1],
                "time_s": np.arange(7, dtype=float) * 30.0,
                "speed_kn": [0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                "speed_provenance": ["UNIQUE_NEAREST_RAW_WITHIN_10S"] * 7,
            }
        )
        ais_path = ais / "train" / "sample.csv"
        ais_frame.to_csv(ais_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [{
                "parent": "parent",
                "sample_id": "sample",
                "relative_path": "train/sample.csv",
                "split": "train",
                "supervisory_row_count": 7,
                "sha256": _sha256(ais_path),
            }]
        ).to_csv(ais / "metadata" / "sample_manifest.csv", index=False)

        fc = np.asarray([0.0, 100.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0])
        battery_raw = fc - load
        series = ParentPowerSeries(
            parent="parent",
            timestamps=timestamps,
            fc_total_kw=fc,
            battery_raw_total_kw=battery_raw,
            source_total_kw=load,
            ais_present=np.ones(8, dtype=bool),
            duplicate_count=0,
            duplicate_conflict_count=0,
            channel_span_violation_count=0,
        )
        return power, ais, series

    def test_builder_preserves_axes_evidence_modes_and_hashes(self) -> None:
        from v2.data.shore_mode_sidecar import build_shore_mode_sidecar

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            power, ais, series = self._fixture(root)
            output = root / "modes"

            summary = build_shore_mode_sidecar(
                power,
                ais,
                root / "raw",
                output,
                load_parent=lambda _root, parent: series,
                expected_split_counts={"train": 1, "validation": 0, "test": 0},
            )

            manifest = pd.read_csv(output / "metadata" / "sample_manifest.csv")
            payload = pd.read_csv(output / "train" / "sample.csv")
            self.assertEqual(
                tuple(payload.columns),
                (
                    "timestamp", "time_s", "load_total_kw", "p_fc_total_kw",
                    "p_batt_bus_kw", "component_balance_residual_kw",
                    "power_provenance", "speed_kn", "speed_provenance",
                    "channels_complete", "freshness_valid",
                    "duplicate_conflict", "mode", "mode_reason",
                ),
            )
            self.assertEqual(
                payload["mode"].tolist(),
                [
                    "onboard", "onboard", "shore_pending", "shore_pending",
                    "shore_charging", "onboard", "onboard",
                ],
            )
            np.testing.assert_allclose(payload["p_batt_bus_kw"], -series.battery_raw_total_kw[:-1])
            np.testing.assert_allclose(payload["component_balance_residual_kw"], 0.0)
            self.assertTrue(payload["channels_complete"].all())
            self.assertTrue(payload["freshness_valid"].all())
            self.assertFalse(payload["duplicate_conflict"].any())
            self.assertEqual(manifest.loc[0, "sha256"], _sha256(output / "train" / "sample.csv"))
            self.assertEqual(summary["mode_counts"]["shore_pending"], 2)
            self.assertEqual(summary["mode_counts"]["shore_charging"], 1)
            self.assertEqual(
                summary["shore_candidate_evidence"],
                "quality-valid AIS speed and battery-bus charging power",
            )
            self.assertEqual(
                summary["recorded_fc_role"],
                "diagnostic only; ignored for shore classification",
            )

    def test_builder_rejects_existing_destination_and_identity_mismatch(self) -> None:
        from v2.data.shore_mode_sidecar import build_shore_mode_sidecar

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            power, ais, series = self._fixture(root)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                build_shore_mode_sidecar(
                    power, ais, root / "raw", existing,
                    load_parent=lambda _root, parent: series,
                    expected_split_counts={"train": 1, "validation": 0, "test": 0},
                )

            manifest_path = ais / "metadata" / "sample_manifest.csv"
            manifest = pd.read_csv(manifest_path)
            manifest.loc[0, "parent"] = "other"
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "identities"):
                build_shore_mode_sidecar(
                    power, ais, root / "raw", root / "modes",
                    load_parent=lambda _root, parent: series,
                    expected_split_counts={"train": 1, "validation": 0, "test": 0},
                )

    def test_mode_payload_hash_tamper_is_detectable(self) -> None:
        from v2.data.shore_mode_sidecar import verify_mode_payload

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.csv"
            path.write_text("a\n1\n", encoding="utf-8")
            expected = _sha256(path)
            verify_mode_payload(path, expected)
            path.write_text("a\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_mode_payload(path, expected)


if __name__ == "__main__":
    unittest.main()
