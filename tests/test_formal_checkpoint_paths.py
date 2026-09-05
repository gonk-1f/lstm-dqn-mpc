from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = REPO_ROOT / "src" / "main"

if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))


from formal_paths import (  # noqa: E402
    formal_checkpoint_path,
    formal_output_dir,
    require_formal_checkpoint,
)
import run_dqn_mpc_causal_training as formal_training  # noqa: E402
import test_dqn_mpc_causal as formal_test  # noqa: E402


class FormalCheckpointPathTests(unittest.TestCase):
    def test_formal_entrypoints_use_new_mlp_namespace(self) -> None:
        expected_output = formal_output_dir("mlp")
        expected_checkpoint = formal_checkpoint_path("mlp")
        self.assertEqual(formal_training.NETWORK_TYPE, "mlp")
        self.assertEqual(formal_training.FORMAL_OUTPUT_DIR, expected_output)
        self.assertEqual(formal_test.NETWORK_TYPE, "mlp")
        self.assertEqual(formal_test.MODEL_ROOT, expected_output)
        self.assertEqual(formal_test.MODEL_PATH, expected_checkpoint)

    def test_mlp_and_kan_outputs_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlp = formal_output_dir("mlp", repo_root=root)
            kan = formal_output_dir("kan", repo_root=root)
            self.assertEqual(
                mlp.name,
                "dqn_mpc_mlp_causal_soc_deadband_formal_rounds",
            )
            self.assertEqual(
                kan.name,
                "dqn_mpc_kan_causal_soc_deadband_formal_rounds",
            )
            self.assertNotEqual(mlp, kan)

    def test_default_checkpoint_is_round_two_in_matching_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for backend in ("mlp", "kan"):
                path = formal_checkpoint_path(backend, repo_root=root)
                self.assertEqual(path.parent.name, "round_2")
                self.assertEqual(path.name, "model_round2.pt")
                self.assertTrue(
                    path.is_relative_to(formal_output_dir(backend, repo_root=root))
                )

    def test_missing_checkpoint_fails_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = formal_checkpoint_path("mlp", repo_root=root)
            with self.assertRaisesRegex(FileNotFoundError, "model_round2.pt"):
                require_formal_checkpoint("mlp", repo_root=root)
            self.assertFalse(expected.exists())

    def test_cross_backend_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlp_checkpoint = formal_checkpoint_path("mlp", repo_root=root)
            mlp_checkpoint.parent.mkdir(parents=True)
            mlp_checkpoint.write_bytes(b"placeholder")
            with self.assertRaisesRegex(ValueError, "KAN checkpoint"):
                require_formal_checkpoint(
                    "kan",
                    checkpoint_path=mlp_checkpoint,
                    repo_root=root,
                )

    def test_existing_matching_checkpoint_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = formal_checkpoint_path("kan", repo_root=root)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"placeholder")
            self.assertEqual(
                require_formal_checkpoint("kan", repo_root=root),
                checkpoint.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
