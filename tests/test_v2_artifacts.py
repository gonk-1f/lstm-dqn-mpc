from __future__ import annotations

from dataclasses import FrozenInstanceError
import base64
import hashlib
import json
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class V2ArtifactTests(unittest.TestCase):
    @staticmethod
    def _catalog():
        from v2.dqn.action_space import ActionCandidate

        return (ActionCandidate(1, 4, 5), ActionCandidate(2, 3, 5))

    @staticmethod
    def _rng_state_json(seed=1234):
        import numpy as np

        return json.dumps(
            np.random.PCG64(seed).state,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _metadata(self, kind="checkpoint", **changes):
        from v2.config import TimeScaleConfig
        from v2.training.artifacts import make_artifact_metadata

        values = {
            "kind": kind,
            "timescale": TimeScaleConfig(30.0, 2, 3),
            "action_catalog": self._catalog(),
            "state_schema_version": "candidate_operating_state_v1",
            "state_dimension": 10,
            "reward_scaling_identity": "raw_cny_unscaled",
            "episode_index": 7,
            "macro_index": 11,
            "seed": 1234,
            "rng_algorithm": "PCG64",
            "rng_state_json": self._rng_state_json(),
        }
        values.update(changes)
        return make_artifact_metadata(**values)

    def test_checkpoint_roundtrip_is_canonical_and_deterministic(self) -> None:
        from v2.training.artifacts import decode_checkpoint, encode_checkpoint

        metadata = self._metadata()
        first = encode_checkpoint(metadata, b"opaque-model-state")
        second = encode_checkpoint(metadata, b"opaque-model-state")
        decoded = decode_checkpoint(first, expected_metadata=metadata)

        self.assertEqual(first, second)
        self.assertEqual(decoded.metadata, metadata)
        self.assertEqual(decoded.payload, b"opaque-model-state")
        self.assertEqual(encode_checkpoint(decoded.metadata, decoded.payload), first)
        self.assertEqual(metadata.n_mpc, 2)
        self.assertEqual(metadata.dqn_switch_steps, 3)
        self.assertEqual(json.loads(metadata.rng_state_json)["bit_generator"], "PCG64")
        self.assertEqual(len(metadata.rng_state_digest), 64)

    def test_legacy_mismatch_cross_kind_and_extra_metadata_reject_before_payload_access(self) -> None:
        from v2.contracts import IncompatibleArtifactError
        from v2.training.artifacts import decode_artifact, encode_artifact

        expected = self._metadata()
        called = []

        def consumer(payload):
            called.append(payload)
            return payload

        incompatible = (
            json.dumps({"model": {"legacy": True}}).encode(),
            encode_artifact(self._metadata(kind="replay"), b"replay"),
            encode_artifact(self._metadata(seed=99), b"wrong-seed"),
            encode_artifact(self._metadata(episode_index=8), b"wrong-index"),
            encode_artifact(
                self._metadata(state_schema_version="candidate_operating_state_v0"),
                b"wrong-state",
            ),
        )
        for value in incompatible:
            with self.subTest(value=value[:30]), self.assertRaises(IncompatibleArtifactError):
                decode_artifact(
                    value,
                    expected_metadata=expected,
                    payload_consumer=consumer,
                )
        self.assertEqual(called, [])

        from v2.training.artifacts import encode_checkpoint

        with self.assertRaises(TypeError):
            encode_checkpoint(self._metadata(kind="replay"), b"wrong-kind")

        document = json.loads(encode_artifact(expected, b"opaque"))
        document["metadata"]["legacy_extra"] = 1
        tampered = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaises(IncompatibleArtifactError):
            decode_artifact(tampered, expected_metadata=expected, payload_consumer=consumer)
        self.assertEqual(called, [])

    def test_changed_timescale_catalog_digest_and_bool_int_alias_are_rejected(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.contracts import IncompatibleArtifactError
        from v2.dqn.action_space import ActionCandidate
        from v2.training.artifacts import decode_artifact, encode_artifact

        expected = self._metadata()
        variants = (
            self._metadata(timescale=TimeScaleConfig(30.0, 3, 3)),
            self._metadata(
                action_catalog=(
                    self._catalog()[0],
                    ActionCandidate(3, 3, 4),
                )
            ),
            self._metadata(rng_state_json=self._rng_state_json(seed=99)),
        )
        for metadata in variants:
            with self.assertRaises(IncompatibleArtifactError):
                decode_artifact(
                    encode_artifact(metadata, b"payload"),
                    expected_metadata=expected,
                )

        document = json.loads(encode_artifact(expected, b"opaque"))
        document["metadata"]["episode_index"] = True
        tampered = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaises(IncompatibleArtifactError):
            decode_artifact(tampered, expected_metadata=expected)

    def test_payload_tamper_is_detected_and_metadata_is_immutable(self) -> None:
        from v2.contracts import IncompatibleArtifactError
        from v2.training.artifacts import decode_artifact, encode_artifact

        metadata = self._metadata()
        document = json.loads(encode_artifact(metadata, b"payload"))
        document["payload_base64"] = "dGFtcGVyZWQ="
        tampered = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

        with self.assertRaises(IncompatibleArtifactError):
            decode_artifact(tampered, expected_metadata=metadata)
        with self.assertRaises(FrozenInstanceError):
            metadata.seed = 5  # type: ignore[misc]

    def test_noncanonical_envelope_is_rejected_before_base64_decode(self) -> None:
        import base64

        from v2.contracts import IncompatibleArtifactError
        from v2.training.artifacts import decode_artifact, encode_artifact

        metadata = self._metadata()
        document = json.loads(encode_artifact(metadata, b"payload"))
        pretty = json.dumps(document, indent=2).encode("ascii")
        real_decode = base64.b64decode

        with mock.patch(
            "v2.training.artifacts.base64.b64decode",
            wraps=real_decode,
        ) as decoder:
            with self.assertRaises(IncompatibleArtifactError):
                decode_artifact(pretty, expected_metadata=metadata)
        decoder.assert_not_called()

    def test_untrusted_json_integer_limit_errors_are_wrapped(self) -> None:
        from v2.contracts import IncompatibleArtifactError
        from v2.training.artifacts import decode_artifact, decode_replay, encode_artifact

        huge_integer = b"1" * 5000
        with self.assertRaises(IncompatibleArtifactError):
            decode_artifact(
                b"[" + huge_integer + b"]",
                expected_metadata=self._metadata(),
            )

        replay_metadata = self._metadata(kind="replay", state_dimension=2)
        replay_envelope = encode_artifact(
            replay_metadata,
            b"[" + huge_integer + b"]",
        )
        with self.assertRaises(IncompatibleArtifactError):
            decode_replay(replay_envelope, expected_metadata=replay_metadata)

    def test_legacy_action_identity_cannot_form_v2_metadata(self) -> None:
        from v2.training.artifacts import ArtifactMetadata

        values = dict(self._metadata().__dict__)
        values["action_catalog_identity"] = ("A0",)
        encoded = json.dumps(
            {"action_ids": ["A0"]}, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        values["action_catalog_digest"] = hashlib.sha256(encoded).hexdigest()

        with self.assertRaises(ValueError):
            ArtifactMetadata(**values)

    def test_action_identity_requires_canonical_v2_ids_and_catalog_order(self) -> None:
        from v2.training.artifacts import ArtifactMetadata

        base = dict(self._metadata().__dict__)
        invalid_identities = (
            ("w_0_5_5",),
            ("w_2_3_4",),
            ("w_02_3_5",),
            ("w_2.0_3_5",),
            ("w_2_3_5_v1",),
            tuple(reversed(base["action_catalog_identity"])),
        )
        for identity in invalid_identities:
            with self.subTest(identity=identity):
                values = dict(base)
                values["action_catalog_identity"] = identity
                encoded = json.dumps(
                    {"action_ids": list(identity)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                values["action_catalog_digest"] = hashlib.sha256(encoded).hexdigest()
                with self.assertRaises(ValueError):
                    ArtifactMetadata(**values)

    def test_metadata_rejects_action_candidate_with_injected_method(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights

        candidate = self._catalog()[0]
        object.__setattr__(
            candidate,
            "to_mpc_weights",
            lambda: MPCWeights(0.1, 0.4, 0.5),
        )
        with self.assertRaises(ValueError):
            self._metadata(action_catalog=(candidate, self._catalog()[1]))

    def test_rng_state_must_match_algorithm_and_be_canonically_recoverable(self) -> None:
        with self.assertRaises(ValueError):
            self._metadata(rng_algorithm="MT19937")

        state = json.loads(self._rng_state_json())
        del state["state"]["inc"]
        malformed = json.dumps(state, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ValueError):
            self._metadata(rng_state_json=malformed)

        state = json.loads(self._rng_state_json())
        state["state"]["state"] = True
        coercing = json.dumps(state, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ValueError):
            self._metadata(rng_state_json=coercing)

    def test_replay_transition_roundtrip_is_exact_and_immutable(self) -> None:
        from v2.dqn.action_space import ActionCandidate
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.training.artifacts import decode_replay, encode_replay

        transition = MacroTransition(
            state=(1.0, 2.0),
            action=ActionCandidate(2, 3, 5),
            learning_reward=-10.0,
            next_state=(3.0, 4.0),
            done=False,
            executed_mpc_steps=3,
            ledger=RawCnyIntervalLedger(1.0, 2.0, 3.0, 4.0),
        )
        metadata = self._metadata(kind="replay", state_dimension=2)
        encoded = encode_replay(metadata, (transition,))
        decoded = decode_replay(encoded, expected_metadata=metadata)

        self.assertEqual(decoded, (transition,))
        self.assertIs(type(decoded), tuple)
        self.assertIs(type(decoded[0].state), tuple)
        self.assertEqual(encode_replay(metadata, decoded), encoded)
        with self.assertRaises(FrozenInstanceError):
            decoded[0].done = True  # type: ignore[misc]

    def test_replay_transitions_must_match_metadata_on_encode_and_decode(self) -> None:
        from v2.contracts import IncompatibleArtifactError
        from v2.dqn.action_space import ActionCandidate
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.training.artifacts import decode_replay, encode_artifact, encode_replay

        metadata = self._metadata(kind="replay", state_dimension=2)
        dimension_metadata = self._metadata(kind="replay", state_dimension=10)
        ledger = RawCnyIntervalLedger(1.0, 2.0, 3.0, 4.0)

        def transition(*, state=(1.0, 2.0), action=None, steps=3, done=False):
            return MacroTransition(
                state=state,
                action=action or self._catalog()[0],
                learning_reward=-10.0,
                next_state=tuple(value + 1.0 for value in state),
                done=done,
                executed_mpc_steps=steps,
                ledger=ledger,
            )

        invalid = (
            (dimension_metadata, transition(state=(1.0, 2.0))),
            (metadata, transition(action=ActionCandidate(3, 3, 4))),
            (metadata, transition(steps=99)),
            (metadata, transition(steps=2, done=False)),
        )
        for case_metadata, value in invalid:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    encode_replay(case_metadata, (value,))

                payload = json.dumps(
                    [
                        {
                            "action_numerators": list(value.action.numerators),
                            "done": value.done,
                            "executed_mpc_steps": value.executed_mpc_steps,
                            "failure_kind": value.failure_kind,
                            "failure_penalty_score": value.failure_penalty_score,
                            "ledger_components_cny": list(value.ledger.components_cny),
                            "learning_reward": value.learning_reward,
                            "next_state": list(value.next_state),
                            "raw_economic_cost_cny": value.raw_economic_cost_cny,
                            "episode_completed": value.episode_completed,
                            "state": list(value.state),
                        }
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                envelope = encode_artifact(case_metadata, payload)
                with self.assertRaises(IncompatibleArtifactError):
                    decode_replay(envelope, expected_metadata=case_metadata)

        early_terminal = transition(steps=2, done=True)
        encoded = encode_replay(metadata, (early_terminal,))
        self.assertEqual(
            decode_replay(encoded, expected_metadata=metadata),
            (early_terminal,),
        )

    def test_failed_replay_transition_roundtrip_preserves_separate_penalty(self) -> None:
        from v2.dqn.action_space import ActionCandidate
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.failure_policy import FORMAL_FAILURE_KIND
        from v2.training.artifacts import decode_replay, encode_replay

        transition = MacroTransition(
            state=(1.0, 2.0),
            action=ActionCandidate(2, 3, 5),
            learning_reward=-50_010.0,
            next_state=(3.0, 4.0),
            done=True,
            executed_mpc_steps=0,
            ledger=RawCnyIntervalLedger(1.0, 2.0, 3.0, 4.0),
            failure_penalty_score=50_000.0,
            failure_kind=FORMAL_FAILURE_KIND,
        )
        metadata = self._metadata(kind="replay", state_dimension=2)

        encoded = encode_replay(metadata, (transition,))

        self.assertEqual(decode_replay(encoded, expected_metadata=metadata), (transition,))
        payload = json.loads(encoded.decode("ascii"))
        document = json.loads(base64.b64decode(payload["payload_base64"]).decode("ascii"))[0]
        self.assertEqual(document["raw_economic_cost_cny"], 10.0)
        self.assertEqual(document["failure_penalty_score"], 50_000.0)
        self.assertEqual(document["learning_reward"], -50_010.0)
        self.assertFalse(document["episode_completed"])

    def test_legacy_reward_only_replay_payload_is_rejected(self) -> None:
        from v2.contracts import IncompatibleArtifactError
        from v2.training.artifacts import decode_replay, encode_artifact

        metadata = self._metadata(kind="replay", state_dimension=2)
        legacy = [{
            "state": [1.0, 2.0],
            "action_numerators": [1, 4, 5],
            "reward_cny": -10.0,
            "next_state": [2.0, 3.0],
            "done": False,
            "executed_mpc_steps": 3,
            "ledger_components_cny": [1.0, 2.0, 3.0, 4.0],
        }]
        payload = json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("ascii")
        envelope = encode_artifact(metadata, payload)

        with self.assertRaises(IncompatibleArtifactError):
            decode_replay(envelope, expected_metadata=metadata)

    def test_replay_revalidates_tampered_exact_fields(self) -> None:
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.training.artifacts import encode_replay

        class IntSubclass(int):
            pass

        metadata = self._metadata(kind="replay", state_dimension=2)
        ledger = RawCnyIntervalLedger(1.0, 2.0, 3.0, 4.0)
        for forged_steps in (True, IntSubclass(2)):
            value = MacroTransition(
                state=(1.0, 2.0),
                action=self._catalog()[0],
                learning_reward=-10.0,
                next_state=(2.0, 3.0),
                done=False,
                executed_mpc_steps=2,
                ledger=ledger,
            )
            object.__setattr__(value, "executed_mpc_steps", forged_steps)
            with self.subTest(forged_steps=forged_steps):
                with self.assertRaises(TypeError):
                    encode_replay(metadata, (value,))


if __name__ == "__main__":
    unittest.main()
